//! Qwen3-VL implementation using Candle for local HuggingFace model inference.
//!
//! This module provides Vision-Language OCR functionality using Alibaba's Qwen3-VL model.

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

// Available model variants
pub const MODEL_ID_2B: &str = "Qwen/Qwen3-VL-2B-Instruct";
pub const MODEL_ID_4B: &str = "Qwen/Qwen3-VL-4B-Instruct";
const REVISION: &str = "main";

const MAX_GENERATION_LENGTH: usize = 512;
const REPETITION_PENALTY: f64 = 1.2;

/// Qwen3-VL model wrapper for inference
pub struct Qwen3VL
{
   model:                 Qwen3VLModel,
   tokenizer:             Tokenizer,
   config:                Config,
   device:                Device,
   dtype:                 DType,
   eos_token_id:          u32,
   max_generation_length: usize,
}

/// Qwen3-VL model variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3VLVariant
//======================
{
   /// 2B parameter model - faster, less VRAM
   TwoB,
   /// 4B parameter model - more capable
   FourB,
}

impl Qwen3VLVariant
//------------------
{
   pub fn model_id(&self) -> &'static str
   {
      match self
      {
         | Qwen3VLVariant::TwoB => MODEL_ID_2B,
         | Qwen3VLVariant::FourB => MODEL_ID_4B,
      }
   }

   pub fn from_model_id(id: &str) -> Option<Self>
   {
      match id
      {
         | MODEL_ID_2B  => Some(Self::TwoB),
         | MODEL_ID_4B  => Some(Self::FourB),
         | _ => None,
      }
   }
}

fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) ->
   Result<(usize, usize)>
//----------------------------------------------------------------------------------------------
{
   let mut h = height;
   let mut w = width;

   // Handle tiny images
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

   // Round to nearest multiple of factor
   let mut h_bar = ((h + factor / 2) / factor) * factor;
   let mut w_bar = ((w + factor / 2) / factor) * factor;

   let total_pixels = h_bar * w_bar;

   if total_pixels > max_pixels
   {
      let beta = ((h * w) as f64 / max_pixels as f64).sqrt();
      h_bar = ((h as f64 / beta / factor as f64).floor() as usize) * factor;
      w_bar = ((w as f64 / beta / factor as f64).floor() as usize) * factor;
   }
   else if total_pixels < min_pixels
   {
      let beta = (min_pixels as f64 / (h * w) as f64).sqrt();
      h_bar = ((h as f64 * beta / factor as f64).ceil() as usize) * factor;
      w_bar = ((w as f64 * beta / factor as f64).ceil() as usize) * factor;
   }

   // Ensure minimum size
   h_bar = h_bar.max(factor);
   w_bar = w_bar.max(factor);

   Ok((h_bar, w_bar))
}

/// Preprocess image bytes for Qwen3-VL
fn preprocess_image(image_bytes: &[u8], config: &Config, device: &Device, dtype: DType) ->
   Result<(Tensor, Tensor, usize, usize)>
//---------------------------------------------------------------------------------------
{
   let img = image::load_from_memory(image_bytes)?;
   let img = img.to_rgb8();
   let (width, height) = (img.width() as usize, img.height() as usize);

   let patch_size = config.vision_config.patch_size;
   let spatial_merge = config.vision_config.spatial_merge_size;
   let temporal_patch = config.vision_config.temporal_patch_size;
   let factor = patch_size * spatial_merge;

   let min_pixels = 256 * 28 * 28; // ~200k
   let max_pixels = 1280 * 28 * 28; // ~1M

   let (new_height, new_width) = smart_resize(height, width, factor, min_pixels, max_pixels)?;

   // Resize image
   let resized = image::imageops::resize(&img, new_width as u32, new_height as u32, image::imageops::FilterType::CatmullRom);

   // Qwen3-VL uses simple [0.5, 0.5, 0.5] normalization (from preprocessor_config.json)
   let mean = [0.5f32, 0.5, 0.5];
   let std = [0.5f32, 0.5, 0.5];

   let mut normalized = vec![0f32; 3 * new_height * new_width];

   for c in 0..3
   {
      for y in 0..new_height
      {
         for x in 0..new_width
         {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = c * new_height * new_width + y * new_width + x;
            normalized[idx] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
         }
      }
   }

   // For Qwen3-VL, we need to reshape for the vision encoder
   // The vision model expects patches, not raw pixels
   // Temporal dimension is 1 for single images
   let h_patches = new_height / patch_size;
   let w_patches = new_width / patch_size;

   // Create pixel_values tensor
   // Shape: (num_patches, C * temporal_patch * patch_size * patch_size)
   // Patches are ordered in spatial_merge x spatial_merge blocks (matching HF processor)
   let patch_dim = 3 * temporal_patch * patch_size * patch_size;
   let num_patches = h_patches * w_patches;
   let merged_h = h_patches / spatial_merge;
   let merged_w = w_patches / spatial_merge;

   let mut patches = vec![0f32; num_patches * patch_dim];

   // Patch ordering matches Python's Qwen2VLImageProcessorFast:
   // Iterate over merged blocks, then within each block iterate over merge positions.
   // Inner layout: C, T, patch_h, patch_w (channel-first, then temporal, then spatial).
   let mut patch_idx = 0;
   for mh in 0..merged_h
   {
      for mw in 0..merged_w
      {
         for sh in 0..spatial_merge
         {
            for sw in 0..spatial_merge
            {
               let ph = mh * spatial_merge + sh;
               let pw = mw * spatial_merge + sw;
               let mut patch_offset = 0;

               for c in 0..3
               {
                  for _t in 0..temporal_patch
                  {
                     for py in 0..patch_size
                     {
                        for px in 0..patch_size
                        {
                           let y = ph * patch_size + py;
                           let x = pw * patch_size + px;
                           let src_idx = c * new_height * new_width + y * new_width + x;
                           patches[patch_idx * patch_dim + patch_offset] = normalized[src_idx];
                           patch_offset += 1;
                        }
                     }
                  }
               }
               patch_idx += 1;
            }
         }
      }
   }

   let pixel_values = Tensor::from_vec(patches, (num_patches, patch_dim), device)?.to_dtype(dtype)?;

   // Grid THW: (temporal, height_patches, width_patches)
   // For single image, temporal = 1
   let grid_thw = Tensor::new(&[[1u32, h_patches as u32, w_patches as u32]], device)?;

   // merged_h, merged_w already computed above for patch ordering

   Ok((pixel_values, grid_thw, merged_h, merged_w))
}

/// Build input tokens for Qwen3-VL
fn build_input_tokens(tokenizer: &Tokenizer, prompt: &str, num_image_tokens: usize, image_token_id: u32, vision_start_token_id: u32,
                      vision_end_token_id: u32, device: &Device)
                      -> Result<(Tensor, Vec<(usize, usize)>)>
{
   // Qwen3-VL chat format (matches Python processor output):
   // <|im_start|>user\n<|vision_start|><|image_pad|>...<|vision_end|>prompt<|im_end|>
   // <|im_start|>assistant\n

   let im_start = tokenizer.token_to_id("<|im_start|>").unwrap_or(151644);
   let im_end = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
   let newline = tokenizer.token_to_id("\n").unwrap_or(198);

   // User prompt prefix
   let user_prefix = "user\n";
   let user_encoding = tokenizer.encode(user_prefix, false)
                                .map_err(|e| anyhow!("Tokenization error: {}", e))?;

   // The actual prompt after image
   let prompt_encoding = tokenizer.encode(prompt, false)
                                  .map_err(|e| anyhow!("Tokenization error: {}", e))?;

   // Assistant prefix
   let assistant_text = "assistant\n";
   let assistant_encoding = tokenizer.encode(assistant_text, false)
                                     .map_err(|e| anyhow!("Tokenization error: {}", e))?;

   // Build token sequence
   let mut input_ids: Vec<u32> = Vec::new();

   // User message with image
   input_ids.push(im_start);
   input_ids.extend(user_encoding.get_ids());
   input_ids.push(vision_start_token_id);

   // Track image placeholder positions
   let image_start = input_ids.len();
   input_ids.extend(vec![image_token_id; num_image_tokens]);
   let image_end = input_ids.len();

   input_ids.push(vision_end_token_id);
   input_ids.extend(prompt_encoding.get_ids());
   input_ids.push(im_end);
   input_ids.push(newline);

   // Assistant start
   input_ids.push(im_start);
   input_ids.extend(assistant_encoding.get_ids());

   let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
   let continuous_img_pad = vec![(image_start, image_end)];

   Ok((tensor, continuous_img_pad))
}


impl Qwen3VL
{
   pub fn load(variant: Qwen3VLVariant, device_id: i32) -> Result<Self>
   //----------------------------------------------------------------------
   {
      let device: Device = match get_hugging_face_device(device_id)
      {
         Ok(d) => d,
         Err(e) => return Err(e),
      };

      // Check GPU VRAM — Qwen3-VL 2B needs ~6GB free (4GB weights + 2GB inference overhead).
      if let Device::Cuda(_) = &device
      {
         const MIN_VRAM_BYTES: usize = 6 * 1024 * 1024 * 1024; // 6 GB
         match candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
         {
            | Ok((free, total)) =>
            {
               let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
               let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);
               eprintln!("Qwen3-VL: GPU {} VRAM: {:.1} GB free / {:.1} GB total", device_id, free_gb, total_gb);
               if free < MIN_VRAM_BYTES
               {
                  let errmsg = format!(
                     "GPU {} has only {:.1} GB free VRAM ({:.1} GB total). \
                      Qwen3-VL 2B requires at least 6 GB free. Use a GPU with more VRAM or select CPU.",
                     device_id, free_gb, total_gb);
                  crate::status::set_status(&errmsg);
                  return Err(anyhow!(errmsg));
               }
            }
            | Err(e) => eprintln!("Qwen3-VL: Could not query GPU memory: {:?}", e),
         }
      }

      // F16 halves VRAM usage (~4GB vs ~8GB for 2B model), critical for 8GB GPUs like RTX 2070.
      // CPU keeps F32 since system RAM is plentiful and F16 ops are slower on CPU.
      let dtype = if device_id < 0 { DType::F32 } else { DType::F16 };
      let model_id = variant.model_id();
      eprintln!("Qwen3-VL: Loading {} on {:?} with dtype {:?}", model_id, device, dtype);
      crate::status::set_status(&format!("Loading Qwen3-VL: {} on {:?}", model_id, device));

      let api = Api::new()?;
      let repo = api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, REVISION.to_string()));

      // Load config
      crate::status::set_status("Loading Qwen3-VL: downloading config...");
      let config_file = match repo.get("config.json")
      {
         Ok(file) => file,
         Err(e) =>
         {
            let errmsg = format!("Failed to get config.json from HuggingFace repo: {}", e);
            eprintln!("Qwen3-VL: {}", errmsg);
            return Err(anyhow!(errmsg))
         }
      };
      crate::status::set_status("Loading Qwen3-VL: reading config...");
      let config_data = match std::fs::read_to_string(&config_file)
      {
         Ok(data) => data,
         Err(e) =>
         {
            let errmsg = format!("Failed to read config.json from HuggingFace repo: {}", e);
            eprintln!("Qwen3-VL: {}", errmsg);
            return Err(anyhow!(errmsg))
         }
      };
      let mut config: Config = match serde_json::from_str(&config_data)
      {
         Ok(cfg) => cfg,
         Err(e) =>
         {
            let errmsg = format!("Failed to parse data\n{}\n in config.json from HuggingFace repo: {}",config_data, e);
            eprintln!("Qwen3-VL: {}", errmsg);
            return Err(anyhow!(errmsg))
         }
      };

      // Override max_position_embeddings to limit KvCache pre-allocation.
      // The default (262144) causes the KV cache to allocate ~30GB of VRAM across all layers.
      // For OCR, sequences are short (prompt + image tokens + generated text < 2048).
      const MAX_SEQ_FOR_OCR: usize = 2048;
      if config.text_config.max_position_embeddings > MAX_SEQ_FOR_OCR
      {
         eprintln!("Qwen3-VL: Capping max_position_embeddings from {} to {} for OCR use",
                   config.text_config.max_position_embeddings, MAX_SEQ_FOR_OCR);
         config.text_config.max_position_embeddings = MAX_SEQ_FOR_OCR;
      }

      // Load tokenizer
      crate::status::set_status("Loading Qwen3-VL: downloading tokenizer...");
      let tokenizer_file = repo.get("tokenizer.json")?;
      let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

      // Load model weights - Qwen3-VL uses sharded safetensors
      crate::status::set_status("Loading Qwen3-VL: downloading model weights...");
      let model_files: Vec<std::path::PathBuf> = if variant == Qwen3VLVariant::TwoB { vec![repo.get("model.safetensors")?] }
      else
      {
         (1..=3).filter_map(|i|
         {
            let filename = format!("model-{:05}-of-00002.safetensors", i);
            repo.get(&filename).ok()
         }).collect()
      };

      if model_files.is_empty()
      {
         let errmsg = format!("Qwen3-VL: Could not create model file list for variant {:?}", variant);
         eprintln!("{}", errmsg);
         return Err(anyhow!(errmsg))
      }

      crate::status::set_status(&format!("Loading Qwen3-VL: building model from {} shards...", model_files.len()));
      let model_files_ref: Vec<&std::path::PathBuf> = model_files.iter().collect();
      // let model_file_names = model_files.iter().map(|p| p.display()).join(", ");
      // (|p| p.file_name().unwrap().to_string_lossy().to_string())
      let model_file_names: String = model_files.iter().map(|p| p.display().to_string()).reduce(|acc, s| format!("{acc}, {s}")).unwrap_or_default();
      let vb = unsafe
      {
         match VarBuilder::from_mmaped_safetensors(&model_files_ref, dtype, &device)
         {
            Ok(vb) => vb,
            Err(e) =>
            {
               let errmsg = format!("Failed to load model weights from {}: {}",&model_file_names, e);
               eprintln!("Qwen3-VL: {}", errmsg);
               return Err(anyhow!(errmsg))
            }
         }
      };

      let model = match Qwen3VLModel::new(&config, vb)
      {
         Ok(m) => m,
         Err(e) =>
         {
            let errmsg = format!("Failed to create Qwen3-VL model from config and weights: {}", e);
            eprintln!("Qwen3-VL: {}", errmsg);
            return Err(anyhow!(errmsg))
         }
      };

      let eos_token_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);

      // Log VRAM after model load
      if let Device::Cuda(_) = &device
      {
         if let Ok((free, total)) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
         {
            let used_gb = (total - free) as f64 / (1024.0 * 1024.0 * 1024.0);
            let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
            eprintln!("Qwen3-VL: After model load — VRAM used: {:.1} GB, free: {:.1} GB", used_gb, free_gb);
         }
      }

      crate::status::set_status("Qwen3-VL: model loaded, running inference...");

      Ok(Self
         { model,
           tokenizer,
           config,
           device,
           dtype,
           eos_token_id,
           max_generation_length: MAX_GENERATION_LENGTH
         })
   }

   /// Run OCR on image bytes
   pub fn recognize(&self, image_bytes: &[u8], prompt: &str) -> Result<String>
   //-----------------------------------------------------------------------
   {
      // Preprocess image
      let (pixel_values, grid_thw, merged_h, merged_w) =
         match preprocess_image(image_bytes, &self.config, &self.device, self.dtype)
         {
            Ok(result) => result,
            Err(e) =>
            {
               let errmsg = format!("Image preprocessing failed: {}", e);
               eprintln!("Qwen3-VL: {}", errmsg);
               return Err(anyhow!(errmsg))
            }
         };

      let num_image_tokens = merged_h * merged_w;
      eprintln!("Qwen3-VL: image tokens: {} (merged {}x{}), pixel_values: {:?}",
                num_image_tokens, merged_h, merged_w, pixel_values.shape());

      // Build input tokens
      let (input_ids, continuous_img_pad) = build_input_tokens(&self.tokenizer, prompt, num_image_tokens, self.config.image_token_id,
                                                               self.config.vision_start_token_id, self.config.vision_end_token_id, &self.device)?;

      let seq_len = input_ids.dim(1)?;

      // Forward pass with image
      let logits = match self.model.forward(&input_ids, Some(pixel_values), None,  Some(grid_thw), None,  vec![seq_len], vec![continuous_img_pad], vec![vec![]],  &[0])
      {
         Ok(logits) => logits,
         Err(e) =>
         {
            let errmsg = format!("Model forward pass failed: {}", e);
            eprintln!("Qwen3-VL: {}", errmsg);
            return Err(anyhow!(errmsg))
         }
      };

      // Generate tokens
      let mut generated_tokens = Vec::new();
      let mut current_logits = logits;
      let mut seqlen_offset = seq_len;

      for _ in 0..self.max_generation_length
      {
         // Get last token logits — handle both 2D (batch, vocab) and 3D (batch, seq_len, vocab) shapes
         let last_logits = match current_logits.dims().len()
         {
            | 3 => current_logits.i((.., current_logits.dim(1)? - 1, ..))?,
            | 2 => current_logits.i((current_logits.dim(0)? - 1, ..))?,
            | n => return Err(anyhow!("Unexpected logits rank: {}, shape: {:?}", n, current_logits.shape())),
         };

         // Apply repetition penalty to discourage generating the same tokens
         let last_logits = if !generated_tokens.is_empty()
         {
            let mut logits_vec = last_logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for &tok in &generated_tokens
            {
               let idx = tok as usize;
               if idx < logits_vec.len()
               {
                  if logits_vec[idx] > 0.0
                  {
                     logits_vec[idx] /= REPETITION_PENALTY as f32;
                  }
                  else
                  {
                     logits_vec[idx] *= REPETITION_PENALTY as f32;
                  }
               }
            }
            Tensor::from_vec(logits_vec, last_logits.shape(), &self.device)?
         }
         else
         {
            last_logits
         };

         let next_token = last_logits.argmax(D::Minus1)?
                                     .to_dtype(DType::U32)?
                                     .to_vec0::<u32>()?;

         if next_token == self.eos_token_id
         {
            break;
         }

         generated_tokens.push(next_token);

         // Prepare next input
         let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

         // Forward without image (using KV cache)
         current_logits = self.model
                              .forward(&next_input, None, None, None, None, vec![1], vec![vec![]], vec![vec![]], &[seqlen_offset])?;

         seqlen_offset += 1;
      }

      // Decode tokens
      let output_text = self.tokenizer
                            .decode(&generated_tokens, true)
                            .map_err(|e| anyhow!("Decoding error: {}", e))?;

      Ok(output_text.trim().to_string())
   }
}

/// Global model cache
use std::sync::Mutex;

use crate::llm_util::get_hugging_face_device;
static MODEL_CACHE: Mutex<Option<(Qwen3VLVariant, i32, Qwen3VL)>> = Mutex::new(None);

/// Drop the cached model, freeing GPU memory.
pub fn clear_cache()
{
   if let Ok(mut cache) = MODEL_CACHE.lock()
   {
      *cache = None;
   }
}

/// Run Qwen3-VL inference on an image with caching
/// device_id: -1 for CPU, 0+ for GPU device index
pub fn qwen3_vl_recognize(model_id: &str, image_bytes: &[u8], device_id: i32) -> Result<String>
{
   let variant = Qwen3VLVariant::from_model_id(model_id).ok_or_else(|| anyhow!("Unknown Qwen3-VL model: {}", model_id))?;

   let mut cache = MODEL_CACHE.lock()
                              .map_err(|e| anyhow!("Failed to acquire model lock: {}", e))?;

   // Check if we need to load/reload the model (different variant or device)
   let need_load = match &*cache
   {
      | Some((cached_variant, cached_device, _)) => *cached_variant != variant || *cached_device != device_id,
      | None => true,
   };

   if need_load
   {
      crate::status::set_status(&format!("Loading Qwen3-VL: variant {:?} on device {}", variant, device_id));
      let model = Qwen3VL::load(variant, device_id)?;
      *cache = Some((variant, device_id, model));
   }

   // Run inference
   if let Some((_, _, model)) = cache.as_ref()
   {
      // Use formula recognition prompt
      let prompt = "Extract the mathematical formula or text from this image. Output in LaTeX format.";
      model.recognize(image_bytes, prompt)
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
   fn test_variant_from_id()
   {
      assert_eq!(Qwen3VLVariant::from_model_id("Qwen/Qwen3-VL-2B-Instruct"), Some(Qwen3VLVariant::TwoB));
      assert_eq!(Qwen3VLVariant::from_model_id("Qwen/Qwen3-VL-4B-Instruct"), Some(Qwen3VLVariant::FourB));
      assert_eq!(Qwen3VLVariant::from_model_id("unknown"), None);
   }

   /// Determine which device IDs to test based on OS and GPU availability.
   /// Always includes -1 (CPU). Adds GPU ordinals that Candle can open.
   fn test_device_ids() -> Vec<i32>
   {
      let mut ids = vec![-1i32]; // CPU always available

      for ordinal in 0..8u32
      {
         let available = if cfg!(target_os = "macos")
         {
            Device::new_metal(ordinal as usize).is_ok()
         }
         else
         {
            Device::new_cuda(ordinal as usize).is_ok()
         };

         if available
         {
            ids.push(ordinal as i32);
         }
         else
         {
            break;
         }
      }

      ids
   }

   /// Test that Qwen3VL::load succeeds on every available device.
   /// Downloads model weights from HuggingFace on first run.
   /// CPU load is required to pass. GPU loads may fail due to driver
   /// issues (e.g. PTX compilation errors on mixed-architecture setups)
   /// and are reported as warnings rather than hard failures.
   #[test]
   #[ignore] // requires network access and significant download
   fn test_load_on_available_devices()
   {
      let device_ids = test_device_ids();
      println!("Testing Qwen3VL::load on device_ids: {:?}", device_ids);

      let mut gpu_failures: Vec<String> = Vec::new();

      for &device_id in &device_ids
      {
         let label = if device_id < 0 { "CPU".to_string() } else { format!("GPU {}", device_id) };
         println!("Loading Qwen3VL (2B) on {}...", label);

         let result = Qwen3VL::load(Qwen3VLVariant::TwoB, device_id);

         if device_id < 0
         {
            // CPU load must succeed
            assert!(result.is_ok(), "Qwen3VL::load failed on CPU: {:?}", result.err());
            let model = result.unwrap();
            assert!(format!("{:?}", model.device).contains("Cpu"),
                    "Expected Cpu device, got {:?}", model.device);
         }
         else
         {
            match result
            {
               | Ok(model) =>
               {
                  let expected = if cfg!(target_os = "macos") { "Metal" } else { "Cuda" };
                  let device_debug = format!("{:?}", model.device);
                  assert!(device_debug.contains(expected),
                          "Expected device containing '{}' for device_id={}, got {:?}",
                          expected, device_id, model.device);
                  println!("{}: OK", label);
               }
               | Err(e) =>
               {
                  let msg = format!("{}: {} (non-fatal)", label, e);
                  eprintln!("WARNING: {}", msg);
                  gpu_failures.push(msg);
               }
            }
         }
      }

      if !gpu_failures.is_empty()
      {
         eprintln!("GPU load warnings ({}/{} GPUs failed):", gpu_failures.len(), device_ids.len() - 1);
         for f in &gpu_failures
         {
            eprintln!("  - {}", f);
         }
      }
   }

   #[test]
   #[ignore]
   fn test_load_gpu1_only()
   {
      println!("Loading Qwen3VL (2B) on GPU 1 only...");
      let result = Qwen3VL::load(Qwen3VLVariant::TwoB, 1);
      match result
      {
         | Ok(model) =>
         {
            let device_debug = format!("{:?}", model.device);
            println!("GPU 1: OK (device={device_debug})");
            assert!(device_debug.contains("Cuda"), "Expected Cuda device, got {device_debug}");
         }
         | Err(e) => panic!("GPU 1 load failed: {e}"),
      }
   }

   #[test]
   #[ignore]
   fn test_run_gpu1_only()
   {
      println!("Running Qwen3VL (2B) inference on GPU 1...");

      // qwen3_vl_recognize manages the model cache internally — only one copy on GPU.
      let img_bytes = match std::fs::read("test-data/eqn.png")
      {
         Ok(b) => b,
         Err(e) => panic!("Failed to read test image: {}", e),
      };
      match qwen3_vl_recognize(MODEL_ID_2B, &img_bytes, 1)
      {
         | Ok(text) =>
         {
            println!("Recognized text: {}", text);
         }
         | Err(e) =>
         {
            let msg = format!("{e}");
            if msg.contains("VRAM") || msg.contains("OUT_OF_MEMORY")
            {
               println!("GPU 1 has insufficient VRAM for 2B model (expected on 8GB cards): {e}");
            }
            else
            {
               panic!("Recognition failed: {e}");
            }
         }
      }
   }

   #[test]
   #[ignore]
   fn test_run_gpu0()
   {
      println!("Running Qwen3VL (2B) inference on GPU 0...");

      let img_bytes = match std::fs::read("test-data/eqn.png")
      {
         Ok(b) => b,
         Err(e) => panic!("Failed to read test image: {}", e),
      };
      match qwen3_vl_recognize(MODEL_ID_2B, &img_bytes, 0)
      {
         | Ok(text) =>
         {
            println!("Recognized text: {}", text);
         }
         | Err(e) => panic!("Recognition on GPU 0 failed: {e}"),
      }
   }

   /// Test using Python-preprocessed inputs to isolate preprocessing vs model issues
   #[test]
   #[ignore]
   fn test_run_python_inputs()
   {
      use candle_core::Tensor;

      println!("Loading Python-preprocessed inputs...");

      // Load numpy arrays saved by Python
      let pv_bytes = std::fs::read("/tmp/qwen3vl_pixel_values.npy").expect("pixel_values.npy");
      let ids_bytes = std::fs::read("/tmp/qwen3vl_input_ids.npy").expect("input_ids.npy");
      let thw_bytes = std::fs::read("/tmp/qwen3vl_grid_thw.npy").expect("grid_thw.npy");

      // Parse numpy .npy format: magic(6) + version(2) + header_len(2 or 4) + header + data
      fn npy_data_offset(data: &[u8]) -> usize
      {
         assert!(data.len() >= 10 && &data[..6] == b"\x93NUMPY", "Not a valid .npy file");
         let major = data[6];
         let header_len = if major == 1
         {
            u16::from_le_bytes([data[8], data[9]]) as usize
         }
         else
         {
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
         };
         let preamble = if major == 1 { 10 } else { 12 };
         preamble + header_len
      }

      fn parse_npy_f32(data: &[u8]) -> Vec<f32>
      {
         let offset = npy_data_offset(data);
         data[offset..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
      }

      fn parse_npy_i64(data: &[u8]) -> Vec<i64>
      {
         let offset = npy_data_offset(data);
         data[offset..].chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect()
      }

      fn parse_npy_i32(data: &[u8]) -> Vec<i32>
      {
         let offset = npy_data_offset(data);
         data[offset..].chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
      }

      let pixel_values_flat = parse_npy_f32(&pv_bytes);
      let input_ids_raw = parse_npy_i64(&ids_bytes);
      let grid_thw_raw = parse_npy_i64(&thw_bytes);

      println!("pixel_values: {} floats", pixel_values_flat.len());
      println!("input_ids: {} tokens", input_ids_raw.len());
      println!("grid_thw: {:?}", grid_thw_raw);

      // Load model on GPU 0
      let model = Qwen3VL::load(Qwen3VLVariant::TwoB, 0).expect("Model load");
      let device = &model.device;
      let dtype = model.dtype;

      // Create tensors from Python data
      let n_patches = pixel_values_flat.len() / 1536;
      let pixel_values = Tensor::from_vec(pixel_values_flat, (n_patches, 1536), device)
         .unwrap().to_dtype(dtype).unwrap();
      let input_ids: Vec<u32> = input_ids_raw.iter().map(|&x| x as u32).collect();
      let input_ids_tensor = Tensor::new(input_ids.as_slice(), device).unwrap().unsqueeze(0).unwrap();
      assert!(grid_thw_raw.len() >= 3, "Expected at least 3 values in grid_thw, got {}", grid_thw_raw.len());
      let grid_thw_tensor = Tensor::new(
         &[[grid_thw_raw[0] as u32, grid_thw_raw[1] as u32, grid_thw_raw[2] as u32]], device
      ).unwrap();

      println!("pixel_values: {:?}", pixel_values.shape());
      println!("input_ids: {:?}", input_ids_tensor.shape());
      println!("grid_thw: {:?}", grid_thw_tensor);

      // Find image_pad token positions
      let image_pad_id = 151655u32;
      let mut img_start = 0;
      let mut img_end = 0;
      for (i, &tok) in input_ids.iter().enumerate()
      {
         if tok == image_pad_id && img_start == 0 { img_start = i; }
         if tok == image_pad_id { img_end = i + 1; }
      }
      println!("Image pad: [{}, {}), {} tokens", img_start, img_end, img_end - img_start);

      let seq_len = input_ids.len();
      let continuous_img_pad = vec![(img_start, img_end)];

      // Forward pass
      let logits = model.model.forward(
         &input_ids_tensor, Some(pixel_values), None, Some(grid_thw_tensor), None,
         vec![seq_len], vec![continuous_img_pad], vec![vec![]], &[0]
      ).expect("Forward pass");

      println!("Logits shape: {:?}", logits.shape());

      // Generate tokens
      let mut generated_tokens = Vec::new();
      let mut current_logits = logits;
      let mut seqlen_offset = seq_len;
      let eos_token_id = model.eos_token_id;

      for _ in 0..128
      {
         let last_logits = match current_logits.dims().len()
         {
            | 3 => current_logits.i((.., current_logits.dim(1).unwrap() - 1, ..)).unwrap(),
            | 2 => current_logits.i((current_logits.dim(0).unwrap() - 1, ..)).unwrap(),
            | _ => panic!("Unexpected logits shape"),
         };

         let next_token = last_logits.to_dtype(DType::F32).unwrap()
            .argmax(D::Minus1).unwrap()
            .to_dtype(DType::U32).unwrap()
            .to_vec0::<u32>().unwrap();

         if next_token == eos_token_id { break; }
         generated_tokens.push(next_token);

         if generated_tokens.len() <= 20
         {
            let partial = model.tokenizer.decode(&generated_tokens, true).unwrap_or_default();
            eprintln!("  token[{}] = {} → {:?}", generated_tokens.len() - 1, next_token, partial);
         }

         let next_input = Tensor::new(&[next_token], device).unwrap().unsqueeze(0).unwrap();
         current_logits = model.model.forward(
            &next_input, None, None, None, None,
            vec![1], vec![vec![]], vec![vec![]], &[seqlen_offset]
         ).expect("Generation forward");
         seqlen_offset += 1;
      }

      let output = model.tokenizer.decode(&generated_tokens, true).unwrap_or_default();
      println!("\nWith Python inputs: {}", output);
   }
}
