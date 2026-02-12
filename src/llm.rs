use image::ImageFormat;

use crate::{MEDIUM_PROMPT, PROMPT, SIMPLE_PROMPT, qwen3_vl::{self, Qwen3VLVariant}, settings::Settings};
use crate::llm_util::{paddleocr_vl_img_to_text, qwen3_vl_img_to_text, openai_chat_img_to_latex, gemini_img_to_latex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmCategory
{
   /// Local inference via Candle (device selected at runtime)
   HuggingFace,
   /// Ollama API
   Ollama,
   /// OpenAI-compatible API
   KeyRequired
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HuggingFaceModels
{
   PaddleOCR,
   Qwen3 { variant: Qwen3VLVariant },
}

impl HuggingFaceModels
{
   pub fn from_model_id(model_id: &str) -> Option<Self>
   {
      if model_id.to_lowercase().contains("paddle")
      {
         Some(HuggingFaceModels::PaddleOCR)
      }
      else if model_id.to_lowercase().contains("qwen3")
      {
         if model_id.to_uppercase().contains("-2B")
         {
            Some( HuggingFaceModels::Qwen3 { variant: Qwen3VLVariant::TwoB } )
         }
         else if model_id.to_uppercase().contains("-4B")
         {
            Some( HuggingFaceModels::Qwen3 { variant: Qwen3VLVariant::FourB } )
         }
         else
         {
            None
         }
      }
      else
      {
         None
      }
   }
}

impl std::fmt::Display for HuggingFaceModels
{
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
   {
      match self
      {
         | HuggingFaceModels::PaddleOCR  => write!(f, "PaddleOCR-VL"),
         | HuggingFaceModels::Qwen3 { variant } => write!(f, "Qwen3-VL {:?}", variant),
      }
   }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LLMType
{
   HuggingFace { model: HuggingFaceModels },
   Ollama { url: String },
   OpenAI { url: String, api_key: String },
   Gemini { url: String, api_key: String },
   OllamaNonFree { url: String, api_key: String },
}

#[allow(dead_code)]
impl LLMType
{
   pub fn new_hugging_face(model_id: &str) -> Option<Self>
   {
      HuggingFaceModels::from_model_id(model_id).map(|model| LLMType::HuggingFace { model })
   }

   pub fn new_ollama(url: &str) -> Self
   {
      LLMType::Ollama { url: url.to_string() }
   }

   pub fn new_openai(url: &str, api_key: &str) -> Self
   {
      LLMType::OpenAI { url: url.to_string(), api_key: api_key.to_string() }
   }

   pub fn new_gemini(url: &str, api_key: &str) -> Self
   {
      LLMType::Gemini { url: url.to_string(), api_key: api_key.to_string() }
   }
}

impl std::fmt::Display for LLMType
{
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
   {
      match self
      {
         | LLMType::HuggingFace { model } => write!(f, "{:?}", model),
         | LLMType::Ollama { url } => write!(f, "Ollama ({})", url),
         | LLMType::OpenAI { url, .. } => write!(f, "OpenAI ({})", url),
         | LLMType::Gemini { url, .. } => write!(f, "Gemini ({})", url),
         | LLMType::OllamaNonFree { url, .. } => write!(f, "Ollama Cloud ({})", url),
      }
   }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LLMInfo
{
   pub id:           String,
   pub name:         String,
   pub category:     LlmCategory,
   pub is_available: bool,
   pub sort_rank:    u32,
}

impl LLMInfo
{
   pub fn new(id: String, name: String, category: LlmCategory, is_available: bool, sort_rank: u32) -> Self
//-----------------------------------------------------------------
   {
      Self { id, name, category, is_available, sort_rank }
   }
}

impl std::fmt::Display for LLMInfo
{
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
   {
      if self.is_available { write!(f, "{}", self.name) } else { write!(f, "{} (N/A)", self.name) }
   }
}

pub trait OCR: Send + Sync
{
   async fn img_to_latex(&self, image_bytes: &[u8], device_id: i32) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LLM
{
   pub info: LLMInfo,
   pub typ: LLMType
}

impl std::fmt::Display for LLM
{
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
   {
      write!(f, "{}", self.info)
   }
}

impl LLM
{
   pub fn new_hugging_face(model_id: &str, sort_rank: u32) -> Self
   {
      let typ = match HuggingFaceModels::from_model_id(model_id)
      {
         | Some(m) => LLMType::HuggingFace { model: m },
         | None => panic!("Unknown HuggingFace model ID: {}", model_id),
      };
      Self
      {
         info: LLMInfo::new(model_id.to_string(), model_id.to_string(), LlmCategory::HuggingFace, true, sort_rank),
         typ
      }
   }

   pub fn new_ollama(id: &str, name: &str, url: &str, is_available: bool, sort_rank: u32) -> Self
   {
      Self
      {
         info: LLMInfo::new(id.to_string(), name.to_string(), LlmCategory::Ollama, is_available, sort_rank),
         typ: LLMType::Ollama { url: url.to_string() },
      }
   }

   pub fn new_chatgpt(id: &str, name: &str, url: &str, api_key: &str, sort_rank: u32) -> Self
   {
      let has_key = ! api_key.to_string().trim().is_empty() && api_key.trim() != Settings::default_chatgpt_key();
      let fullname = if ! has_key { format!("{} (No Key)", name) } else { name.to_string() };
      Self
      {
         info: LLMInfo::new(id.to_string(), fullname, LlmCategory::KeyRequired, has_key, sort_rank),
         typ: LLMType::OpenAI { url: url.to_string(), api_key: api_key.to_string() },
       }
   }

   pub fn new_gemini(id: &str, name: &str, url: &str, api_key: &str, sort_rank: u32) -> Self
   {
      let has_key = !api_key.to_string().trim().is_empty() && api_key.trim() != Settings::default_gemini_key();
      let fullname = if !has_key { format!("{} (No Key)", name) } else { name.to_string() };
      Self
      {
         info: LLMInfo::new(id.to_string(), fullname, LlmCategory::KeyRequired, has_key, sort_rank),
         typ: LLMType::Gemini { url: url.to_string(), api_key: api_key.to_string() },
      }
   }

   pub fn new_ollama_web(id: &str, name: &str, url: &str, api_key: &str, sort_rank: u32) -> Self
   {
      let has_key = !api_key.to_string().trim().is_empty() && api_key.trim() != Settings::default_ollama_key();
      let fullname = if !has_key { format!("{} (No Key)", name) } else { name.to_string() };
      Self
      {
         info: LLMInfo::new(id.to_string(), fullname, LlmCategory::KeyRequired, has_key, sort_rank),
         typ: LLMType::OllamaNonFree { url: url.to_string(), api_key: api_key.to_string() },
      }
   }
   
}

impl OCR for LLM
//==============
{
   async fn img_to_latex(&self, image_bytes: &[u8], device_id: i32) -> Result<String, Box<dyn std::error::Error + Send + Sync>>
   {
      match &self.typ
      {
         | LLMType::HuggingFace { model } =>
         {
            match model
            {
               | HuggingFaceModels::PaddleOCR => paddleocr_vl_img_to_text(image_bytes, device_id),
               | HuggingFaceModels::Qwen3 { variant } =>
               {
                  let model_id = match variant
                  {
                     | Qwen3VLVariant::TwoB => qwen3_vl::MODEL_ID_2B,
                     | Qwen3VLVariant::FourB => qwen3_vl::MODEL_ID_4B,
                  };
                  qwen3_vl_img_to_text(model_id, image_bytes, device_id)
               }
            }
         }
         | LLMType::Ollama { url } =>
         {
            openai_chat_img_to_latex(url, "", "", &self.info.id, PROMPT, MEDIUM_PROMPT, SIMPLE_PROMPT, image_bytes, &ImageFormat::Png)
         }
         | LLMType::OpenAI { url, api_key } =>
         {
            if api_key.trim().is_empty() || api_key.trim() == Settings::default_chatgpt_key()
            {
               crate::status::set_status("Set a valid OpenAPI API key first");
               return Err("Set a valid OpenAPI API key first".into());
            }
            else
            {
               let key = match Settings::decrypt("OpenAPI", api_key)
               {
                  | Ok(k) => k,
                  | Err(e) =>
                  {
                     crate::status::set_status(&format!("Failed to decrypt OpenAPI key: {}", e));
                     return Err(format!("Failed to decrypt OpenAPI key: {}", e).into());
                  }
               };
               openai_chat_img_to_latex(url, "", &key, &self.info.id, PROMPT, MEDIUM_PROMPT, SIMPLE_PROMPT, image_bytes, &ImageFormat::Png)
            }
         }
         | LLMType::Gemini { url, api_key } =>
         {
            if api_key.trim().is_empty() || api_key.trim() == Settings::default_gemini_key()
            {
               crate::status::set_status("Set a valid Gemini API key first");
               return Err("Set a valid Gemini API key first".into());
            }
            else
            {
               let key = match Settings::decrypt("Gemini", api_key)
               {
                  | Ok(k) => k,
                  | Err(e) =>
                  {
                     crate::status::set_status(&format!("Failed to decrypt Gemini key: {}", e));
                     return Err(format!("Failed to decrypt Gemini key: {}", e).into());
                  }
               };
               gemini_img_to_latex(url, &key, PROMPT, MEDIUM_PROMPT, SIMPLE_PROMPT, image_bytes, &ImageFormat::Png)
            }
         }
         | LLMType::OllamaNonFree { url, api_key } =>
         {
            if api_key.trim().is_empty() || api_key.trim() == Settings::default_ollama_key()
            {
               crate::status::set_status("Set a valid Ollama API key first");
               return Err("Set a valid Ollama API key first".into());
            }
            else
            {
               let key = match Settings::decrypt("Ollama", api_key)
               {
                  | Ok(k) => k,
                  | Err(e) =>
                  {
                     crate::status::set_status(&format!("Failed to decrypt Ollama key: {}", e));
                     return Err(format!("Failed to decrypt Ollama key: {}", e).into());
                  }
               };
               openai_chat_img_to_latex(url, "", &key, &self.info.id, PROMPT, MEDIUM_PROMPT, SIMPLE_PROMPT, image_bytes, &ImageFormat::Png)
            }
         }
      }
   }
}
