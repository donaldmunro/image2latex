use std::collections::HashMap;

use anyhow::{Result, anyhow};
use candle_core::Device;
use base64::{Engine as _, engine::general_purpose};
use image::ImageFormat;
use serde::{Deserialize, Serialize};
use ureq::Agent;

pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

use crate::llm::{LLM};
use crate::{paddleocr_vl, qwen3_vl::{self}};
use crate::settings::Settings;

/// OpenAI-compatible chat completion response (works with Ollama /v1/chat/completions too)
#[derive(Debug, Deserialize)]
struct ChatCompletionChoice
{
   message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage
{
   content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse
{
   choices: Vec<ChatCompletionChoice>,
}

#[allow(clippy::vec_init_then_push)]
#[allow(non_snake_case)]
pub fn get_local_models() -> Vec<LLM>
//-----------------------------------
{
   let mut llms = Vec::new();

   // PaddleOCR-VL - local VLM inference via Candle
   llms.push(LLM::new_hugging_face(paddleocr_vl::MODEL_ID, 0));

   // Qwen3-VL models - local VLM inference via Candle
   llms.push(LLM::new_hugging_face(qwen3_vl::MODEL_ID_2B, 1));
   llms.push(LLM::new_hugging_face(qwen3_vl::MODEL_ID_4B, 2));

   llms
}

pub async fn load_llms(ollama_url: &str, settings: &Settings, sort_order_start: u32) -> Vec<LLM>
//-----------------------------------------
{
   let mut llms = get_local_models();

   let ollama_models = if !ollama_url.is_empty() && check_ollama_running(ollama_url).await
   {
      get_ollama_models(ollama_url).await.unwrap_or_default()
   }
   else
   {
      HashMap::new()
   };
   let mut i = 0;

   let ollama_chat_url = format!("{}/v1/chat/completions", ollama_url);
   let mut push_ollama = |llms: &mut Vec<LLM>, id: &str, name: &str, is_available: bool| {
      let rank = if is_available
      {
         i += 1;
         if name.to_lowercase().contains("cloud")
         {
            sort_order_start + 200 + i
         }
         else
         {
            sort_order_start + 100 + i
         }
      } else if name.to_lowercase().contains("cloud")
      {
         i += 1;
         sort_order_start + 300 + i
      }
      else
      {
         i += 1;
         if name.to_uppercase().contains("N/A")
         {
            sort_order_start + 1000 + i
         }
         else
         {
            sort_order_start + 2000 + i
         }
      };

      llms.push(LLM::new_ollama(id, name, &ollama_chat_url, is_available, rank));
      //llms.push(LLMInfo::new(id.to_string(), name.to_string(), LLMType::Ollama { url: ollama_chat_url.to_string(), },LlmCategory::Ollama, is_available, rank));
   };

   push_ollama(&mut llms, "glm-ocr:latest", "Ollama GLM-OCR 2B Latest (Local)", ollama_models.contains_key("glm-ocr:latest"));
   push_ollama(&mut llms, "glm-ocr:bf16", "Ollama GLM-OCR 2B 16 bit (Local)", ollama_models.contains_key("glm-ocr:bf16"));
   push_ollama(&mut llms, "glm-ocr:q8_0", "Ollama GLM-OCR 2B 8 bit (Local)", ollama_models.contains_key("glm-ocr:q8_0"));
   push_ollama(&mut llms, "deepseek-ocr:latest", "Ollama DeepSeek OCR Latest (Local)", ollama_models.contains_key("deepseek-ocr:latest"));
   push_ollama(&mut llms, "deepseek-ocr:3b", "Ollama DeepSeek OCR 3B (Local)", ollama_models.contains_key("deepseek-ocr:3b"));
   push_ollama(&mut llms, "qwen3-vl:latest", "Ollama Qwen3-vl Latest (Local)", ollama_models.contains_key("qwen3-vl:latest"));
   push_ollama(&mut llms, "qwen3-vl:2b", "Ollama Qwen3-vl 2B (Local)", ollama_models.contains_key("qwen3-vl:2b"));
   push_ollama(&mut llms, "qwen3-vl:4b", "Ollama Qwen3-vl 4B (Local)", ollama_models.contains_key("qwen3-vl:4b"));
   push_ollama(&mut llms, "qwen3-vl:8b", "Ollama Qwen3-vl 8B (Local)", ollama_models.contains_key("qwen3-vl:8b"));
   push_ollama(&mut llms, "qwen3-vl:235b-cloud", "Ollama Qwen3-vl 235B (Cloud)", ollama_models.contains_key("qwen3-vl:235b-cloud"));

   // if let Ok(key) = std::env::var("OPENAI_API_KEY")
   {
      let mut key = match std::env::var("OPENAI_API_KEY")
      {
         | Ok(k) => k,
         | Err(_) =>
         {
            String::new()
         }
      };
      if key.trim().is_empty()
      {
         key = match settings.get_encrypted_openapi_key()
         {
            | Ok(k) => k,
            | Err(_) => String::new(),
         };
      }
      llms.push(LLM::new_openai("gpt-4o-mini", "GPT-4o Mini (Web)", "https://api.openai.com/v1/chat/completions", &key, sort_order_start + 1000));
      llms.push(LLM::new_openai("gpt-4o", "GPT-4o (Web)", "https://api.openai.com/v1/chat/completions", &key, sort_order_start + 1001));
   }
   llms
}

pub async fn load_available_models() -> Vec<LLM>
//---------------------------------------------------------------------
{
   let mut settings = Settings::new();
   settings = settings.get_settings_or_default();
   let ollama_url = if settings.ollama_url.trim().is_empty() { DEFAULT_OLLAMA_URL } else { &settings.ollama_url };
   let mut llms = load_llms(ollama_url, &settings, 100).await;
   llms.sort_by_key(|llm| llm.info.sort_rank);
   llms
}

pub async fn check_ollama_running(url: &str) -> bool
//----------------------------------------------------------------------
{
   match reqwest::get(url).await
   {
      | Ok(response) =>
      {
         if let Ok(text) = response.text().await
         {
            text.contains("Ollama is running")
         }
         else
         {
            false
         }
      }
      | Err(_) => false,
   }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaModelListEntry
{
   pub name:    String,
   pub model:   Option<String>,
   pub size:    Option<u64>,
   pub details: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OllamaTagsResponse
{
   models: Vec<OllamaModelListEntry>,
}

pub async fn get_ollama_models(url: &str) -> Result<HashMap<String, OllamaModelListEntry>, Box<dyn std::error::Error>>
//-----------------------------------------------------------------------
{
   let mut ollama_models = HashMap::new();
   let models_url = format!("{}/api/tags", url);

   let response = reqwest::get(models_url).await?;
   let tags: OllamaTagsResponse = response.json().await?;

   for model in tags.models
   {
      ollama_models.insert(model.name.clone(), model);
   }

   Ok(ollama_models)
}

/// Unified OpenAI-compatible image-to-LaTeX function.
/// Works with both Ollama (/v1/chat/completions) and OpenAI-compatible APIs.
pub fn openai_chat_img_to_latex(url: &str, api_key: &str, model: &str, prompt: &str, medium_prompt: &str, simple_prompt: &str,
                                image: &[u8], image_format: &ImageFormat)
                                -> Result<String, Box<dyn std::error::Error + Send + Sync>>
//--------------------------------------------------------------------------------
{
   let prompts = [prompt.to_string(), medium_prompt.to_string(), simple_prompt.to_string()];
   fn trim_non_ascii(s: &str) -> String { s.chars().filter(|&c| c.is_ascii()).collect() }

   let mime = match image_format
   {
      | ImageFormat::Png => "image/png",
      | ImageFormat::Jpeg => "image/jpeg",
      | _ => "image/png",
   };
   let image_data_url = format!("data:{};base64,{}", mime, general_purpose::STANDARD.encode(image));

   let mut latex_text = String::new();
   for (i, prompt) in prompts.iter().enumerate()
   {
      let request = serde_json::json!({
         "model": model,
         "messages": [{
            "role": "user",
            "content": [
               { "type": "text", "text": prompt },
               { "type": "image_url", "image_url": { "url": &image_data_url } }
            ]
         }],
         "stream": false
      });

      let request_text = match serde_json::to_string(&request)
      {
         | Ok(s) => s,
         | Err(e) =>
         {
            let msg = format!("ERROR: Serializing request to JSON failed: {e}");
            return Err(msg.into());
         }
      };

      let config = Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(150))).https_only(false).build();
      let agent = Agent::new_with_config(config);
      let mut req = agent.post(url).header("Content-Type", "application/json;charset=UTF-8");
      if !api_key.is_empty()
      {
         req = req.header("Authorization", &format!("Bearer {}", api_key));
      }

      let mut response = match req.send(&request_text)
      {
         | Ok(r) => r,
         | Err(e) =>
         {
            let msg = format!("ERROR: Submitting request to {}: {}\n{}", url, e, &request_text);
            eprintln!("{}", &msg);
            return Err(msg.into());
         }
      };

      if !response.status().is_success()
      {
         if i == prompts.len() - 1
         {
            return Err(format!("API {} returned error: {}", url, response.status()).into());
         }
         continue;
      }

      let response_text = match response.body_mut().read_to_string()
      {
         | Ok(s) => s,
         | Err(e) =>
         {
            let msg = format!("ERROR: Reading response body failed: {e}");
            return Err(msg.into());
         }
      };

      let chat_response: ChatCompletionResponse = match serde_json::from_str(&response_text)
      {
         | Ok(response) => response,
         | Err(e) =>
         {
            eprintln!("API {} returned JSON deserialize error: {}", url, e);
            if i == prompts.len() - 1
            {
               return Err(e.to_string().into());
            }
            continue;
         }
      };

      latex_text = chat_response.choices
                                .first()
                                .map(|c| c.message.content.trim().to_string())
                                .unwrap_or_default();

      if !trim_non_ascii(latex_text.trim()).is_empty()
      {
         break;
      }
   }
   if latex_text.trim().is_empty()
   {
      return Err("API returned empty response".to_string().into());
   }
   Ok(latex_text)
}

/// Run PaddleOCR-VL inference on an image using local Candle model
/// device_id: -1 for CPU, 0+ for GPU device index
pub fn paddleocr_vl_img_to_text(image_bytes: &[u8], device_id: i32) -> Result<String, Box<dyn std::error::Error + Send + Sync>>
//--------------------------------------------------------------------------------
{
   // Use Formula task for math recognition, which is the primary use case
   match paddleocr_vl::paddleocr_vl_recognize(image_bytes, device_id, paddleocr_vl::PaddleOcrTask::Formula)
   {
      | Ok(text) => Ok(text),
      | Err(e) => Err(format!("PaddleOCR-VL error: {}", e).into()),
   }
}

/// Run Qwen3-VL inference on an image using local Candle model
/// device_id: -1 for CPU, 0+ for GPU device index
pub fn qwen3_vl_img_to_text(model_id: &str, image_bytes: &[u8], device_id: i32) -> Result<String, Box<dyn std::error::Error + Send + Sync>>
//--------------------------------------------------------------------------------
{
   match qwen3_vl::qwen3_vl_recognize(model_id, image_bytes, device_id)
   {
      | Ok(text) => Ok(text),
      | Err(e) => Err(format!("Qwen3-VL error: {}", e).into()),
   }
}

/// Clear all cached local models, freeing GPU/CPU memory.
pub fn clear_model_caches()
{
   paddleocr_vl::clear_cache();
   qwen3_vl::clear_cache();
}

pub fn extract_math(text: &str) -> String
//-----------------------------------------
{
   let re = regex::Regex::new(r"(?s)(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{(?:equation|align\*?|gather\*?|cases\*?)\}.*?\\end\{(?:(?:equation|align\*?|gather\*?|cases\*?))\})").unwrap();

   let matches: Vec<&str> = re.find_iter(text).map(|m| m.as_str()).collect();

   if matches.is_empty() { text.to_string() } else { matches.join("\n\n") }
}

pub fn get_hugging_face_device(device_id: i32) -> Result<Device, anyhow::Error>
//-----------------------------------------
{
   if device_id < 0
   {
      Ok(Device::Cpu)
   }
   else if cfg!(target_os = "macos")
   {
      match Device::metal_if_available(device_id as usize)
      {
         Ok(d) => Ok(d),
         Err(e) =>
         {
            let errmsg = format!("Metal GPU device {} not available. This may be due to running on non-Metal hardware, missing drivers, or incompatible GPU architecture. [{:?}]",
                device_id, e);
            eprintln!("Qwen3-VL {}:", &errmsg);
            crate::status::set_status(&errmsg);
            return Err(anyhow!("Error selecting Metal GPU {}: ({:?})", device_id, e));
            // Device::Cpu
         }
      }
   }
   else
   {
      match Device::cuda_if_available(device_id as usize)
      {
         Ok(d) => Ok(d),
         Err(e) =>
         {
            let errmsg = format!("CUDA device {} not available. This may be due to missing drivers, incompatible GPU architecture, or running on a CPU-only machine. [{:?}]",
                device_id, e);
            eprintln!("Qwen3-VL {}", &errmsg);
            crate::status::set_status(&errmsg);
            return Err(anyhow!("Error selecting GPU {}: ({:?})", device_id, e));
            //Device::Cpu
         }
      }
   }
}

#[cfg(test)]
mod tests
{
   use super::*;

   #[test]
   fn test_extract_math_simple()
   {
      let input = "Here is some math: $E=mc^2$ and more: \\(a^2 + b^2 = c^2\\)";
      let expected = "$E=mc^2$\n\n\\(a^2 + b^2 = c^2\\)";
      assert_eq!(extract_math(input), expected);
   }

   #[test]
   fn test_extract_math_multiline()
   {
      let input = "Block math: $$x = 1$$ and environment:\n\\begin{equation}\ny = 2\n\\end{equation}";
      let expected = "$$x = 1$$\n\n\\begin{equation}\ny = 2\n\\end{equation}";
      assert_eq!(extract_math(input), expected);
   }

   #[test]
   fn test_extract_math_none()
   {
      let input = "No math here, just plain text.";
      assert_eq!(extract_math(input), input);
   }
}
