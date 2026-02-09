use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::paddleocr_vl::{Config, PaddleOCRVLModel};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::llm_util::get_hugging_face_device;

pub const MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL";
const REVISION: &str = "main";

/// Task type for PaddleOCR-VL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // All variants are part of the public API
pub enum PaddleOcrTask
{
   /// Text recognition (OCR)
   Ocr,
   /// Table recognition
   Table,
   /// Formula recognition
   Formula,
   /// Chart recognition
   Chart,
}

impl PaddleOcrTask
{
   pub fn prompt(&self) -> &'static str
   {
      match self
      {
         | PaddleOcrTask::Ocr => "OCR:",
         | PaddleOcrTask::Table => "Table Recognition:",
         | PaddleOcrTask::Formula => "Formula Recognition:",
         | PaddleOcrTask::Chart => "Chart Recognition:",
      }
   }
}

/// Smart resize algorithm matching PyTorch's PaddleOCRVLImageProcessor.
///
/// Rescales the image so that:
/// 1. Both dimensions are divisible by `factor` (patch_size × merge_size = 28)
/// 2. Total pixels are within [min_pixels, max_pixels] range
/// 3. Aspect ratio is maintained as closely as possible
pub fn paddle_smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)>
{
   let mut h = height;
   let mut w = width;

   // Handle tiny images by scaling up to minimum factor
   if h < factor
   {
      w = (w * factor + h / 2) / h;
      h = factor;
   }
   if w < factor
   {
      h = (h * factor + w / 2) / w;
      w = factor;
   }

   // Check aspect ratio constraint
   let aspect = if h > w { h as f64 / w as f64 } else { w as f64 / h as f64 };
   if aspect > 200.0
   {
      return Err(anyhow!("Aspect ratio {:.1} exceeds maximum of 200", aspect));
   }

   // Round to nearest multiple of factor
   let mut h_bar = ((h + factor / 2) / factor) * factor;
   let mut w_bar = ((w + factor / 2) / factor) * factor;

   let total_pixels = h_bar * w_bar;

   if total_pixels > max_pixels
   {
      // Scale down to fit within max_pixels
      let beta = ((h * w) as f64 / max_pixels as f64).sqrt();
      h_bar = ((h as f64 / beta / factor as f64).floor() as usize) * factor;
      w_bar = ((w as f64 / beta / factor as f64).floor() as usize) * factor;
   }
   else if total_pixels < min_pixels
   {
      // Scale up to meet min_pixels
      let beta = (min_pixels as f64 / (h * w) as f64).sqrt();
      h_bar = ((h as f64 * beta / factor as f64).ceil() as usize) * factor;
      w_bar = ((w as f64 * beta / factor as f64).ceil() as usize) * factor;
   }

   Ok((h_bar, w_bar))
}

/// Preprocess image bytes for PaddleOCR-VL.
///
/// Returns (pixel_values, grid_thw) tensors ready for the model.
fn preprocess_image(image_bytes: &[u8], device: &Device, dtype: DType) -> Result<(Tensor, Tensor)>
{
   let img = image::load_from_memory(image_bytes)?;
   let img = img.to_rgb8();
   let (width, height) = (img.width() as usize, img.height() as usize);

   // PaddleOCR-VL uses dynamic resolution with patch size 14
   let patch_size = 14;
   let spatial_merge = 2;
   let factor = patch_size * spatial_merge; // 28
   let min_pixels = 147384; // from preprocessor_config.json
   let max_pixels = 2822400; // from preprocessor_config.json

   let (new_height, new_width) = paddle_smart_resize(height, width, factor, min_pixels, max_pixels)?;

   // Resize image
   let resized = image::imageops::resize(&img, new_width as u32, new_height as u32, image::imageops::FilterType::CatmullRom);

   // Normalize to [-1, 1] range (matching PyTorch processor output)
   let mut normalized = vec![0f32; 3 * new_height * new_width];

   for c in 0..3
   {
      for y in 0..new_height
      {
         for x in 0..new_width
         {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = c * new_height * new_width + y * new_width + x;
            // Simple [-1, 1] normalization: 2 * (x/255) - 1
            normalized[idx] = pixel[c] as f32 / 255.0 * 2.0 - 1.0;
         }
      }
   }

   // Create tensor: (1, 3, H, W)
   let pixel_values = Tensor::from_vec(normalized, (1, 3, new_height, new_width), device)?.to_dtype(dtype)?;

   // Grid THW: (temporal, height_patches, width_patches)
   let h_patches = (new_height / patch_size) as u32;
   let w_patches = (new_width / patch_size) as u32;
   let grid_thw = Tensor::new(&[[1u32, h_patches, w_patches]], device)?;

   Ok((pixel_values, grid_thw))
}

/// Build input tokens with proper chat format.
/// Format: <|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>...<|IMAGE_END|>[task]\nAssistant:
fn build_input_tokens(tokenizer: &Tokenizer, task: PaddleOcrTask, num_image_tokens: usize, image_token_id: u32, vision_start_token_id: u32,
                      vision_end_token_id: u32, device: &Device)
                      -> Result<Tensor>
{
   // Get BOS token
   let bos_token_id = tokenizer.token_to_id("<|begin_of_sentence|>").unwrap_or(1);

   // Build prompt parts
   let user_prefix = "User: ";
   let task_text = task.prompt();
   let assistant_prefix = "\nAssistant: ";

   // Tokenize parts
   let user_encoding = tokenizer.encode(user_prefix, false)
                                .map_err(|e| anyhow!("Tokenization error: {}", e))?;
   let task_encoding = tokenizer.encode(task_text, false)
                                .map_err(|e| anyhow!("Tokenization error: {}", e))?;
   let assistant_encoding = tokenizer.encode(assistant_prefix, false)
                                     .map_err(|e| anyhow!("Tokenization error: {}", e))?;

   // Build full input:
   // <BOS> + "User: " + <IMAGE_START> + <IMAGE_PLACEHOLDER>... + <IMAGE_END> + task + "\nAssistant: "
   let mut input_ids: Vec<u32> = vec![bos_token_id];
   input_ids.extend(user_encoding.get_ids());
   input_ids.push(vision_start_token_id);
   input_ids.extend(vec![image_token_id; num_image_tokens]);
   input_ids.push(vision_end_token_id);
   input_ids.extend(task_encoding.get_ids());
   input_ids.extend(assistant_encoding.get_ids());

   let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
   Ok(tensor)
}

/// PaddleOCR-VL model wrapper for inference
pub struct PaddleOCRVL
{
   model:                 PaddleOCRVLModel,
   tokenizer:             Tokenizer,
   config:                Config,
   device:                Device,
   dtype:                 DType,
   eos_token_id:          u32,
   max_generation_length: usize,
}

impl PaddleOCRVL
{
   pub fn load(device_id: i32) -> Result<Self>
   //-----------------------------------------
   {
      let device: Device = match get_hugging_face_device(device_id)
      {
         Ok(d) => d,
         Err(e) => return Err(e),
      };

      let dtype = DType::F32; // Use F32 for stability; BF16 can be used on supported hardware

      crate::status::set_status(&format!("Loading PaddleOCR-VL on device {}: {:?}", device_id, device));

      let api = Api::new()?;
      let repo = api.repo(hf_hub::Repo::with_revision(MODEL_ID.to_string(), hf_hub::RepoType::Model, REVISION.to_string()));

      // Load config
      crate::status::set_status("Loading PaddleOCR-VL: downloading config...");
      let config_file = repo.get("config.json")?;
      let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;

      // Load tokenizer
      crate::status::set_status("Loading PaddleOCR-VL: downloading tokenizer...");
      let tokenizer_file = repo.get("tokenizer.json")?;
      let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

      // Load model weights
      crate::status::set_status("Loading PaddleOCR-VL: downloading model weights...");
      let model_file = match repo.get("model.safetensors")
      {
         | Ok(f) => f,
         | Err(_) => repo.get("pytorch_model.bin")?,
      };

      crate::status::set_status("Loading PaddleOCR-VL: building model...");

      let vb = if model_file.extension().is_some_and(|ext| ext == "bin")
      {
         VarBuilder::from_pth(&model_file, dtype, &device)?
      }
      else
      {
         unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)? }
      };

      let model = PaddleOCRVLModel::new(&config, vb)?;

      // Get EOS token ID
      let eos_token_id = tokenizer.token_to_id("</s>")
                                  .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
                                  .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                                  .unwrap_or(2);

      crate::status::set_status("PaddleOCR-VL: model loaded, running inference...");

      Ok(Self { model,
                tokenizer,
                config,
                device,
                dtype,
                eos_token_id,
                max_generation_length: 1024 })
   }

   /// Run OCR on image bytes
   pub fn recognize(&mut self, image_bytes: &[u8], task: PaddleOcrTask) -> Result<String>
   {
      // Preprocess image
      let (pixel_values, grid_thw) = preprocess_image(image_bytes, &self.device, self.dtype)?;

      // Calculate number of image tokens after spatial merge
      let grid_vec: Vec<Vec<u32>> = grid_thw.to_vec2()?;
      let g = &grid_vec[0];
      let spatial_merge = self.config.vision_config.spatial_merge_size;
      let num_image_tokens = (g[1] as usize / spatial_merge) * (g[2] as usize / spatial_merge);

      // Build input tokens
      let input_ids = build_input_tokens(&self.tokenizer, task, num_image_tokens, self.config.image_token_id, self.config.vision_start_token_id,
                                         self.config.vision_end_token_id, &self.device)?;

      // Clear KV cache for fresh generation
      self.model.clear_kv_cache();

      // Generate output
      let generated_tokens = self.model
                                 .generate(&input_ids, &pixel_values, &grid_thw, self.max_generation_length, self.eos_token_id)?;

      // Decode tokens (filter out EOS)
      let output_tokens: Vec<u32> = generated_tokens.into_iter()
                                                    .take_while(|&t| t != self.eos_token_id)
                                                    .collect();

      let output_text = self.tokenizer
                            .decode(&output_tokens, true)
                            .map_err(|e| anyhow!("Decoding error: {}", e))?;

      Ok(output_text.trim().to_string())
   }
}

/// Global model cache to avoid reloading
use std::sync::Mutex;
static MODEL_CACHE: Mutex<Option<(i32, PaddleOCRVL)>> = Mutex::new(None);

/// Drop the cached model, freeing GPU memory.
pub fn clear_cache()
{
   if let Ok(mut cache) = MODEL_CACHE.lock()
   {
      *cache = None;
   }
}

/// Run PaddleOCR-VL inference on an image
/// This caches the model to avoid reloading on subsequent calls
/// device_id: -1 for CPU, 0+ for GPU device index
pub fn paddleocr_vl_recognize(image_bytes: &[u8], device_id: i32, task: PaddleOcrTask) -> Result<String>
{
   let mut cache = MODEL_CACHE.lock()
                              .map_err(|e| anyhow!("Failed to acquire model lock: {}", e))?;

   // Check if we need to load/reload the model (different device)
   let need_load = match &*cache
   {
      | Some((cached_device, _)) => *cached_device != device_id,
      | None => true,
   };

   if need_load
   {
      crate::status::set_status(&format!("Loading PaddleOCR-VL on device {}...", device_id));
      let model = PaddleOCRVL::load(device_id)?;
      *cache = Some((device_id, model));
   }

   // Run inference
   if let Some((_, model)) = cache.as_mut()
   {
      model.recognize(image_bytes, task)
   }
   else
   {
      Err(anyhow!("Model not loaded"))
   }
}

#[cfg(test)]
mod tests
{
   use super::*;

   #[test]
   fn test_smart_resize()
   {
      // Test basic resize
      let (h, w) = paddle_smart_resize(100, 100, 28, 147384, 2822400).unwrap();
      assert!(h % 28 == 0);
      assert!(w % 28 == 0);
      assert!(h * w >= 147384);
      assert!(h * w <= 2822400);
   }

   #[test]
   fn test_task_prompts()
   {
      assert_eq!(PaddleOcrTask::Ocr.prompt(), "OCR:");
      assert_eq!(PaddleOcrTask::Formula.prompt(), "Formula Recognition:");
   }
}
