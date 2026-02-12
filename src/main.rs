use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use iced::{Alignment, Element, Length, Task,
           futures::StreamExt,
           task::Handle as AbortHandle,
           widget::{button, checkbox, column, container, image as image_widget, mouse_area, pick_list, progress_bar, row, scrollable, text,
                    text_editor, text_input}};
use candle_core::Device;
use arboard::Clipboard;
use image::{self as image_crate, ImageEncoder, codecs::png::PngEncoder};
use image_crate::{ImageFormat, RgbaImage};
use sysinfo::System;

mod settings;
mod crypt;
mod llm;
mod llm_util;
mod paddleocr_vl;
mod qwen3_vl;
mod status;

use crate::settings::Settings;
use crate::llm::{LLM, LLMType, LlmCategory, OCR};
use crate::llm_util::{load_available_models, clear_model_caches, extract_math};

//Less sophisticated models appear to not understand what Markdown or Latex is and just return an empty string.
//So we try the more sophisticated prompt first and if it returns an empty string we try the simple prompt.
const SIMPLE_PROMPT: &str = r#"Return the text in this image."#;
const MEDIUM_PROMPT: &str = r#"Return the text in this image in Markdown Latex."#;
const PROMPT: &str = r#"Return the text in this image in Markdown Latex with math mode format.
Any Latex math should be delimited by \( and \).
Output only the converted content without any preamble, description, analysis or summary.
"#;

/// Static GPU list, initialized once on first access.
static GPU_LIST: LazyLock<Vec<String>> = LazyLock::new(enumerate_gpus);

pub fn main() -> iced::Result
{
   let settings = Settings::new();
   let settings = settings.get_settings_or_default();
   let (win_w, win_h) = match (settings.last_screen_width, settings.last_screen_height)
   {
      | (Some(w), Some(h)) if w >= 200 && h >= 200 => (w as f32, h as f32),
      | _ => (1024.0, 1024.0),
   };
   iced::application("Image2LaTeX", App::update, App::view).subscription(App::subscription)
                                                           .theme(|app| if app.dark_theme { iced::Theme::Dark } else { iced::Theme::Light })
                                                           .window_size((win_w, win_h))
                                                           .run()
}

/// Try to find an executable by name, checking PATH first, then known locations.
fn find_executable(name: &str, known_paths: &[&str]) -> Option<String>
{
   use std::process::Command;

   // Check if it's in PATH using `which` (Unix) or `where` (Windows)
   let which_cmd = if cfg!(target_os = "windows") { "where" } else { "which" };
   if let Ok(output) = Command::new(which_cmd).arg(name).output()
   {
      if output.status.success()
      {
         let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
         if !path.is_empty()
         {
            return Some(path);
         }
      }
   }

   // Check known locations
   for path in known_paths
   {
      if std::path::Path::new(path).exists()
      {
         return Some(path.to_string());
      }
   }

   // On Windows, search DriverStore for nvidia-smi
   if cfg!(target_os = "windows") && name == "nvidia-smi"
   {
      let driver_store = r"C:\Windows\System32\DriverStore\FileRepository";
      if let Ok(entries) = std::fs::read_dir(driver_store)
      {
         for entry in entries.flatten()
         {
            if entry.file_name().to_string_lossy().starts_with("nv")
            {
               let candidate = entry.path().join("nvidia-smi.exe");
               if candidate.exists()
               {
                  return Some(candidate.to_string_lossy().to_string());
               }
            }
         }
      }
   }

   None
}

/// Get GPU names on macOS by running `system_profiler SPDisplaysDataType`.
/// Returns a list of GPU names in the format "M4 10 Cores".
fn mac_gpus() -> Vec<String>
{
   // Example output:
   //
   // Graphics/Displays:
   //
   //     Apple M4:
   //
   //       Chipset Model: Apple M4
   //       Type: GPU
   //       Bus: Built-In
   //       Total Number of Cores: 10
   //       ...
   //
   let Some(exe) = find_executable("system_profiler", &["/usr/sbin/system_profiler"]) else {
      return Vec::new();
   };

   let Ok(output) = std::process::Command::new(&exe).arg("SPDisplaysDataType").output() else {
      return Vec::new();
   };

   if !output.status.success()
   {
      return Vec::new();
   }

   let stdout = String::from_utf8_lossy(&output.stdout);
   let mut names = Vec::new();
   let mut current_chipset: Option<String> = None;
   let mut current_cores: Option<String> = None;

   for line in stdout.lines()
   {
      let trimmed = line.trim();
      if let Some(val) = trimmed.strip_prefix("Chipset Model:")
      {
         // Emit the previous GPU if we had one
         if let Some(chipset) = current_chipset.take()
         {
            let name = match current_cores.take()
            {
               | Some(cores) => format!("{} {} Cores", chipset, cores),
               | None => chipset,
            };
            names.push(name);
         }
         current_chipset = Some(val.trim().to_string());
         current_cores = None;
      }
      else if let Some(val) = trimmed.strip_prefix("Total Number of Cores:")
      {
         current_cores = Some(val.trim().to_string());
      }
   }

   // Emit the last GPU
   if let Some(chipset) = current_chipset.take()
   {
      let name = match current_cores.take()
      {
         | Some(cores) => format!("{} {} Cores", chipset, cores),
         | None => chipset,
      };
      names.push(name);
   }

   names
}

/// Determine NVIDIA architecture from the GPU name.
fn gpu_architecture(name: &str) -> &'static str
{
   let upper = name.to_uppercase();
   // Blackwell: RTX 50xx
   if upper.contains("RTX 50") { return "Blackwell"; }
   // Ada Lovelace: RTX 40xx
   if upper.contains("RTX 40") || upper.contains("L40") || upper.contains("L4 ") { return "Ada"; }
   // Ampere: RTX 30xx, A100, A10, A30, A40, A6000
   if upper.contains("RTX 30") || upper.contains("RTX A") || upper.contains(" A100") || upper.contains(" A10") || upper.contains(" A30") || upper.contains(" A40") { return "Ampere"; }
   // Turing: RTX 20xx, GTX 16xx
   if upper.contains("RTX 20") || upper.contains("GTX 16") { return "Turing"; }
   // Pascal: GTX 10xx
   if upper.contains("GTX 10") || upper.contains("GP10") || upper.contains("P100") || upper.contains("P40") { return "Pascal"; }
   // Hopper
   if upper.contains("H100") || upper.contains("H200") { return "Hopper"; }
   ""
}

/// Get GPU names, memory, and architecture on Linux/Windows by running `nvidia-smi`.
/// Returns a list of GPU description strings (without GPU ID).
///
/// Parses the nvidia-smi table output. The GPU data section follows the `|===...===|`
/// separator. Each GPU has 3 data rows:
///   Row 0: GPU index + name + persistence-mode | bus-id + disp | ECC
///   Row 1: fan + temp + perf + power | memory-usage | gpu-util + compute
///   Row 2: (empty/MIG row)
fn cuda_gpus() -> Vec<String>
{
   let known_paths: &[&str] = if cfg!(target_os = "windows")
   {
      &[r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe"]
   }
   else
   {
      &["/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"]
   };

   let Some(exe) = find_executable("nvidia-smi", known_paths) else {
      return Vec::new();
   };

   let Ok(output) = std::process::Command::new(&exe).output() else {
      return Vec::new();
   };

   if !output.status.success()
   {
      return Vec::new();
   }

   let stdout = String::from_utf8_lossy(&output.stdout);
   let lines: Vec<&str> = stdout.lines().collect();

   // Find the |===...===| separator that marks the start of GPU data
   let Some(eq_idx) = lines.iter().position(|l| { let t = l.trim(); t.starts_with("|=") && t.contains("===") }) else {
      return Vec::new();
   };

   // Collect GPU data rows (|...|) after the separator, stopping at a blank line
   let mut gpu_data: Vec<&str> = Vec::new();
   for line in &lines[eq_idx + 1..]
   {
      let t = line.trim();
      if t.is_empty() { break; }
      if t.starts_with('|') && !t.starts_with("|=")
      {
         gpu_data.push(t);
      }
      // +--- separator lines between GPUs are skipped
   }

   let mut names = Vec::new();

   // Each GPU occupies 3 data rows
   for chunk in gpu_data.chunks(3)
   {
      if chunk.is_empty() { continue; }

      // Row 0: "|   0  NVIDIA GeForce RTX 4070 Ti     Off |   00000000:01:00.0  On |   N/A |"
      let cells0: Vec<&str> = chunk[0].split('|').collect();
      let mut gpu_name = String::new();
      if cells0.len() > 1
      {
         let first_cell = cells0[1].trim();
         // Format: "0  NVIDIA GeForce RTX 4070 Ti     Off"
         // Split on whitespace, skip the index, strip trailing "Off"/"On" (persistence mode)
         let parts: Vec<&str> = first_cell.split_whitespace().collect();
         if parts.len() >= 2
         {
            let end = if parts.last().map(|s| *s == "Off" || *s == "On").unwrap_or(false)
            {
               parts.len() - 1
            }
            else
            {
               parts.len()
            };
            gpu_name = parts[1..end].join(" ");
         }
      }

      // Row 1: "|  0%   45C    P8   15W / 285W |    1157MiB / 12282MiB |  14%  Default |"
      let mut memory_info = String::new();
      if chunk.len() > 1
      {
         let cells1: Vec<&str> = chunk[1].split('|').collect();
         if cells1.len() > 2
         {
            // Second cell: "    1157MiB / 12282MiB"
            let mem_cell = cells1[2].trim();
            if let Some(slash_pos) = mem_cell.find('/')
            {
               let total = mem_cell[slash_pos + 1..].trim().replace("MiB", "");
               let total = total.trim();
               if let Ok(mib) = total.parse::<f64>()
               {
                  let gb = mib / 1024.0;
                  if gb >= 1.0
                  {
                     memory_info = format!("{:.1}GB", gb);
                  }
                  else
                  {
                     memory_info = format!("{}MiB", total);
                  }
               }
            }
         }
      }

      if !gpu_name.is_empty()
      {
         let arch = gpu_architecture(&gpu_name);
         let mut desc = gpu_name;
         if !memory_info.is_empty()
         {
            desc = format!("{} {}", desc, memory_info);
         }
         if !arch.is_empty()
         {
            desc = format!("{} ({})", desc, arch);
         }
         names.push(desc);
      }
   }

   names
}

/// Enumerate available GPU devices using Candle and system tools.
/// Returns a list of strings in the format "{device_id}: {name}" for GPUs,
/// plus a "CPU: ..." entry.
fn enumerate_gpus() -> Vec<String>
{
   let mut list: Vec<String> = Vec::new();
   let mut ids: Vec<usize> = Vec::new();

   // Probe for available GPU devices via Candle
   if cfg!(target_os = "macos")
   {
      // On macOS, Metal typically only exposes device 0 (unified GPU),
      // even on multi-GPU systems. Candle's Metal backend panics on invalid
      // ordinals instead of returning an error, so we only probe device 0.
      if let Ok(_) = Device::new_metal(0)
      {
         ids.push(0);
         }
      }
      else
   {
      // CUDA can have multiple devices, probe up to 64
      for id in 0..64
      {
         match Device::new_cuda(id)
         {
            | Ok(_) => ids.push(id),
            | Err(e) =>
            {
               if ! e.to_string().to_lowercase().contains("invalid device ordinal")
               {
                  eprintln!("Error {} probing CUDA device {}: {:?}",e.to_string(), id, e);
               }
               break;
            }
         }
      }
   }

   // Get human-readable GPU names from system tools
   let names = if cfg!(target_os = "macos") { mac_gpus() } else { cuda_gpus() };

   // Build the list entries
   for (i, id) in ids.iter().enumerate()
   {
      let name = if i < names.len()
      {
         names[i].clone()
      }
      else if cfg!(target_os = "macos")
      {
         format!("Metal GPU {}", id)
      }
      else
      {
         format!("CUDA GPU {}", id)
      };
      list.push(format!("{}: {}", id, name));
   }

   // Always add CPU as an option
   let sys = System::new_all();
   let ram = (sys.total_memory() as f64 / 1_000_000_000.0).round() as u64;
   let cpu = sys.cpus()
                .iter()
                .next()
                .map_or("CPU".to_string(), |c| format!("CPU: {} {}GB RAM", c.brand(), ram));
   list.push(cpu);
   list
}

/// Parse device ID from selected GPU string.
/// Returns -1 for CPU, or the GPU device index (0, 1, etc.).
pub fn parse_device_id(selected_gpu: &str) -> i32
{
   if selected_gpu.to_lowercase().starts_with("cpu")
   {
      return -1;
   }
   // Format: "0: NVIDIA GeForce RTX 3080" — parse number before the colon
   if let Some(colon_pos) = selected_gpu.find(':')
   {
      if let Ok(device_id) = selected_gpu[..colon_pos].trim().parse::<i32>()
      {
         return device_id;
      }
   }
   -1
}

struct App
{
   monitor_clipboard:   bool,
   selected_model:      Option<LLM>,
   gpu_list:            Vec<String>,
   selected_gpu:        String,
   latex_content:       text_editor::Content,
   image_data:          Option<image_widget::Handle>,
   raw_image_bytes:     Option<Vec<u8>>,
   last_image_hash:     Option<u64>,
   auto_copy:           bool,
   auto_convert:        bool,
   models:              Vec<LLM>,
   initialized:         bool,
   converting:          bool,
   conversion_progress: f32,
   abort_handle:        Option<AbortHandle>,
   is_error:            bool,
   status_text:         String,
   dark_theme:          bool,
   show_settings:       bool,
   settings_ollama_url: String,
   settings_openai_key: String,
   settings_gemini_key: String,
   settings_ollama_key: String,
   show_openai_key:     bool,
   show_gemini_key:     bool,
   show_ollama_key:     bool,
   settings_status:     String,
}

impl Default for App
{
   fn default() -> Self
   {
      let gpu_list = GPU_LIST.clone();
      let selected_gpu = gpu_list.first()
                                 .cloned()
                                 .unwrap_or_else(|| "CPU".to_string());
      let models = Vec::new();
      let selected_model = models.first().cloned();
      let dark_theme = Settings::new().get_settings_or_default().dark_theme;
      Self { monitor_clipboard: false,
             selected_model,
             selected_gpu,
             gpu_list,
             latex_content: text_editor::Content::default(),
             image_data: None,
             raw_image_bytes: None,
             last_image_hash: None,
             auto_copy: false,
             auto_convert: false,
             models,
             initialized: false,
             converting: false,
             conversion_progress: 0.0,
             abort_handle: None,
             is_error: false,
             status_text: String::new(),
             dark_theme,
             show_settings: false,
             settings_ollama_url: String::new(),
             settings_openai_key: String::new(),
             settings_gemini_key: String::new(),
             settings_ollama_key: String::new(),
             show_openai_key: false,
             show_gemini_key: false,
             show_ollama_key: false,
             settings_status: String::new() }
   }
}

#[derive(Debug, Clone)]
enum Message
{
   ToggleMonitor(bool),
   PasteFromClipboard,
   ModelSelected(LLM),
   GpuSelected(String),
   LatexEdited(text_editor::Action),
   ToggleAutoCopy(bool),
   ToggleAutoConvert(bool),
   Convert,
   CancelConvert,
   ConversionResult(Result<String, String>),
   CopyLatex,
   ImageLoaded(Option<(image_widget::Handle, Vec<u8>, u64)>),
   ModelsLoaded(Vec<LLM>),
   RefreshModels,
   Tick,
   WindowResized(iced::Size),
   OpenSettings,
   CloseSettings,
   SettingsOllamaUrlChanged(String),
   SettingsOpenAIKeyChanged(String),
   SettingsGeminiKeyChanged(String),
   SettingsOllamaKeyChanged(String),
   ToggleShowOpenAIKey,
   ToggleShowGeminiKey,
   ToggleShowOllamaKey,
   ToggleDarkTheme(bool),
   SaveSettings,
}

impl App
{
   fn update(&mut self, message: Message) -> Task<Message>
   {
      let msg: String;
      match message
      {
         | Message::ToggleMonitor(value) =>
         {
            self.monitor_clipboard = value;
            Task::none()
         }
         | Message::PasteFromClipboard =>
         {
            Task::perform(async {
                             let mut clipboard = Clipboard::new().ok()?;
                             let image = clipboard.get_image().ok()?; //RGBA

                             let mut hasher = std::collections::hash_map::DefaultHasher::new();
                             image.bytes.hash(&mut hasher);
                             let hash = hasher.finish();

                             let mut png_bytes = Vec::new();
                             let mut cursor = std::io::Cursor::new(&mut png_bytes);
                             let encoder = PngEncoder::new(&mut cursor);
                             match encoder.write_image(image.bytes.as_ref(), image.width as u32, image.height as u32, image::ExtendedColorType::Rgba8)
                             {
                                | Ok(_) =>
                                {}
                                | Err(e) =>
                                {
                                   eprintln!("Failed to encode image: {}", e);
                                   return None;
                                }
                             }

                             //   let mut hasher = std::collections::hash_map::DefaultHasher::new();
                             //   png_bytes.hash(&mut hasher);
                             //   let hash2 = hasher.finish();
                             //   println!("Png Image hash: {}", hash2);

                             let handle = image_widget::Handle::from_rgba(image.width as u32, image.height as u32, image.bytes.into_owned());
                             Some((handle, png_bytes, hash))
                          },
                          Message::ImageLoaded)
         }
         | Message::ImageLoaded(data) =>
         {
            if let Some((handle, raw_bytes, hash)) = data
            {
               self.image_data = Some(handle);
               self.raw_image_bytes = Some(raw_bytes);
               self.last_image_hash = Some(hash);

               if self.auto_convert
               {
                  return Task::done(Message::Convert);
               }
            }
            else
            {
               self.image_data = None;
               self.raw_image_bytes = None;
               self.last_image_hash = None;
            }
            Task::none()
         }
         | Message::ModelSelected(model) =>
         {            
            if !model.info.is_available
            {
               msg = format!("{} API key is not set. Configure it in Settings or set the corresponding environment variable.", model.info.name);
               let reason = match model.info.category
               {
                  | LlmCategory::Ollama => "Ollama is not running or this model is not installed. Start Ollama and/or pull/run the model, then click refresh.",
                  | LlmCategory::KeyRequired => &msg,
                  | LlmCategory::HuggingFace => "This model is not available. It may need to be downloaded first.",
               };
               self.is_error = true;
               self.status_text = format!("{} is unavailable: {}", model.info.name, reason);
               return Task::none();
            }
            // Free GPU memory from the previously cached model
            clear_model_caches();
            self.is_error = false;
            self.status_text.clear();
            self.selected_model = Some(model);
            Task::none()
         }
         | Message::GpuSelected(gpu) =>
         {
            self.selected_gpu = gpu;
            Task::none()
         }
         | Message::LatexEdited(action) =>
         {
            self.is_error = false;
            self.latex_content.perform(action);
            Task::none()
         }
         | Message::ToggleAutoCopy(value) =>
         {
            self.auto_copy = value;
            Task::none()
         }
         | Message::ToggleAutoConvert(value) =>
         {
            self.auto_convert = value;
            // Cancel conversion if auto-convert is toggled off while converting
            if !value && self.converting
            {
               return Task::done(Message::CancelConvert);
            }
            Task::none()
         }
         | Message::Convert =>
         {
            if self.converting
            {
               return Task::none();
            }
            if let Some(model) = &self.selected_model
            {
               if !model.info.is_available
               {
                  return Task::none();
               }
            }
            if let (Some(model), Some(image_bytes)) = (&self.selected_model, &self.raw_image_bytes)
            {
               self.converting = true;
               self.conversion_progress = 0.0;
               self.is_error = false;
               self.latex_content = text_editor::Content::new();
               let image_bytes = image_bytes.clone();
               let device_id = parse_device_id(&self.selected_gpu);
               let model = model.clone();
               let task =
               {
                  Task::perform( async move
                  {
                     model.img_to_latex(&image_bytes, device_id).await.map_err(|e| e.to_string())
                  },
                  Message::ConversionResult )
               };
               let (abortable_task, handle) = task.abortable();
               self.abort_handle = Some(handle);
               abortable_task
            }
            else
            {
               Task::none()
            }
         }
         | Message::CancelConvert =>
         {
            if let Some(handle) = self.abort_handle.take()
            {
               handle.abort();
            }
            self.converting = false;
            self.conversion_progress = 0.0;
            status::clear_status();
            self.status_text.clear();
            Task::none()
         }
         | Message::ConversionResult(result) =>
         {
            self.converting = false;
            self.abort_handle = None;
            status::clear_status();
            self.status_text.clear();
            match result
            {
               | Ok(content) =>
               {
                  let extracted = extract_math(&content);
                  self.latex_content = text_editor::Content::with_text(&extracted);
                  if let Some(model) = &self.selected_model
                  {
                     let mut settings = Settings::new().get_settings_or_default();
                     settings.last_model = Some(model.info.id.clone());
                     let _ = settings.write_settings();
                  }
                  if self.auto_copy
                  {
                     return Task::done(Message::CopyLatex);
                  }
               }
               | Err(err) =>
               {
                  self.is_error = true;
                  let errmsg = format!("Conversion error: {}", err);
                  self.latex_content = text_editor::Content::with_text(&errmsg);
               }
            }
            Task::none()
         }
         | Message::CopyLatex =>
         {
            let text = self.latex_content.text();
            let auto_copy = self.auto_copy;
            Task::perform(async move {
                             if let Ok(mut clipboard) = Clipboard::new()
                             {
                                let _ = clipboard.set_text(text);
                             }
                          },
                          move |_| Message::ToggleAutoCopy(auto_copy))
         }
         | Message::ModelsLoaded(models) =>
         {
            self.models = models;
            if self.selected_model.is_none() && !self.models.is_empty()
            {
               let settings = Settings::new().get_settings_or_default();
               self.selected_model = if let Some(last_id) = &settings.last_model
               {
                  self.models.iter().find(|m| m.info.id == *last_id).cloned()
               }
               else
               {
                  None
               };
               if self.selected_model.is_none()
               {
                  self.selected_model = self.models.first().cloned();
               }
            }
            self.initialized = true;
            Task::none()
         }
         | Message::RefreshModels => Task::perform(async { load_available_models().await }, Message::ModelsLoaded),
         | Message::Tick =>
         {
            self.conversion_progress = (self.conversion_progress + 2.0) % 100.0;
            let s = status::get_status();
            if s != self.status_text
            {
               self.status_text = s;
            }
            Task::none()
         }
         | Message::WindowResized(size) =>
         {
            let w = size.width as u32;
            let h = size.height as u32;
            if w >= 200 && h >= 200
            {
               let mut settings = Settings::new().get_settings_or_default();
               settings.last_screen_width = Some(w);
               settings.last_screen_height = Some(h);
               let _ = settings.write_settings();
            }
            Task::none()
         }
         | Message::OpenSettings =>
         {
            self.show_settings = true;
            let settings = Settings::new().get_settings_or_default();
            self.settings_ollama_url = settings.ollama_url;
            self.settings_openai_key.clear();
            self.settings_gemini_key.clear();
            self.settings_ollama_key.clear();
            self.show_openai_key = false;
            self.show_gemini_key = false;
            self.show_ollama_key = false;
            self.settings_status.clear();
            Task::none()
         }
         | Message::CloseSettings =>
         {
            self.show_settings = false;
            self.settings_ollama_url.clear();
            self.settings_openai_key.clear();
            self.settings_gemini_key.clear();
            self.settings_ollama_key.clear();
            self.settings_status.clear();
            Task::none()
         }
         | Message::SettingsOllamaUrlChanged(value) =>
         {
            self.settings_ollama_url = value;
            Task::none()
         }
         | Message::SettingsOpenAIKeyChanged(value) =>
         {
            self.settings_openai_key = value;
            Task::none()
         }
         | Message::SettingsGeminiKeyChanged(value) =>
         {
            self.settings_gemini_key = value;
            Task::none()
         }
         | Message::ToggleShowOpenAIKey =>
         {
            self.show_openai_key = !self.show_openai_key;
            Task::none()
         }
         | Message::ToggleShowGeminiKey =>
         {
            self.show_gemini_key = !self.show_gemini_key;
            Task::none()
         }
         | Message::SettingsOllamaKeyChanged(value) =>
         {
            self.settings_ollama_key = value;
            Task::none()
         }
         | Message::ToggleShowOllamaKey =>
         {
            self.show_ollama_key = !self.show_ollama_key;
            Task::none()
         }
         | Message::ToggleDarkTheme(value) =>
         {
            self.dark_theme = value;
            let mut settings = Settings::new().get_settings_or_default();
            settings.dark_theme = value;
            let _ = settings.write_settings();
            Task::none()
         }
         | Message::SaveSettings =>
         {
            let mut settings = Settings::new().get_settings_or_default();
            settings.ollama_url = if self.settings_ollama_url.trim().is_empty()
            {
               llm_util::DEFAULT_OLLAMA_URL.to_string()
            }
            else
            {
               self.settings_ollama_url.trim().to_string()
            };
            let openai_key = self.settings_openai_key.trim();
            let gemini_key = self.settings_gemini_key.trim();

            if !openai_key.is_empty()
            {
               match Settings::encrypt_value(openai_key)
               {
                  | Ok(encrypted) => settings.set_chatgpt_key(encrypted),
                  | Err(e) =>
                  {
                     self.settings_status = format!("Error encrypting OpenAI key: {}", e);
                     return Task::none();
                  }
               }
            }
            if !gemini_key.is_empty()
            {
               match Settings::encrypt_value(gemini_key)
               {
                  | Ok(encrypted) => settings.set_gemini_key(encrypted),
                  | Err(e) =>
                  {
                     self.settings_status = format!("Error encrypting Gemini key: {}", e);
                     return Task::none();
                  }
               }
            }
            let ollama_key = self.settings_ollama_key.trim();
            if !ollama_key.is_empty()
            {
               match Settings::encrypt_value(ollama_key)
               {
                  | Ok(encrypted) => settings.set_ollama_key(encrypted),
                  | Err(e) =>
                  {
                     self.settings_status = format!("Error encrypting Ollama key: {}", e);
                     return Task::none();
                  }
               }
            }
            match settings.write_settings()
            {
               | Ok(_) =>
               {
                  self.settings_status = "Settings saved successfully".to_string();

                  // Update is_available and stored api_key on models whose key was just entered
                  let entered_openai = !self.settings_openai_key.trim().is_empty();
                  let entered_gemini = !self.settings_gemini_key.trim().is_empty();
                  let entered_ollama = !self.settings_ollama_key.trim().is_empty();

                  for llm in &mut self.models
                  {
                     let new_key = match &llm.typ
                     {
                        | LLMType::OpenAI { .. } if entered_openai => settings.get_encrypted_chatgpt_key().ok(),
                        | LLMType::Gemini { .. } if entered_gemini => settings.get_encrypted_gemini_key().ok(),
                        | LLMType::OllamaNonFree { .. } if entered_ollama => settings.get_encrypted_ollama_key().ok(),
                        | _ => None,
                     };
                     if let Some(key) = new_key
                     {
                        match &mut llm.typ
                        {
                           | LLMType::OpenAI { api_key, .. }
                           | LLMType::Gemini { api_key, .. }
                           | LLMType::OllamaNonFree { api_key, .. } => *api_key = key,
                           | _ => {}
                        }
                        llm.info.is_available = true;
                        llm.info.name = llm.info.name.trim_end_matches(" (No Key)").to_string();
                     }
                  }

                  self.settings_openai_key.clear();
                  self.settings_gemini_key.clear();
                  self.settings_ollama_key.clear();
               }
               | Err(e) =>
               {
                  self.settings_status = format!("Error saving settings: {}", e);
               }
            }
            Task::none()
         }
      }
   }

   fn subscription(&self) -> iced::Subscription<Message>
   {
      let models_subscription = if !self.initialized
      {
         iced::Subscription::run(|| {
            iced::futures::stream::once(async { Message::ModelsLoaded(load_available_models().await) })
         })
      }
      else
      {
         iced::Subscription::none()
      };

      let clipboard_subscription = if self.monitor_clipboard
      {
         iced::Subscription::run(|| {
            iced::futures::stream::unfold(None, |state: Option<u64>| async move {
               tokio::time::sleep(std::time::Duration::from_millis(500)).await;

               let mut clipboard = match Clipboard::new()
               {
                  | Ok(c) => c,
                  | Err(_) => return Some((None, state)),
               };

               if let Ok(image) = clipboard.get_image()
               {
                  use std::hash::{Hash, Hasher};
                  let mut hasher = std::collections::hash_map::DefaultHasher::new();
                  image.bytes.hash(&mut hasher);
                  let hash = hasher.finish();

                  if Some(hash) != state
                  {
                     let mut png_bytes = Vec::new();
                     let img_buf = match RgbaImage::from_raw(image.width as u32, image.height as u32, image.bytes.to_vec())
                     {
                        | Some(buf) => buf,
                        | None => return Some((None, state)),
                     };
                     if img_buf.write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
                               .is_err()
                     {
                        return Some((None, state));
                     }

                     let handle = image_widget::Handle::from_rgba(image.width as u32, image.height as u32, image.bytes.into_owned());
                     return Some((Some(Message::ImageLoaded(Some((handle, png_bytes, hash)))), Some(hash)));
                  }
               }

               Some((None, state))
            }).filter_map(|msg| async move { msg })
         })
      }
      else
      {
         iced::Subscription::none()
      };

      let tick_subscription = if self.converting
      {
         iced::time::every(std::time::Duration::from_millis(50)).map(|_| Message::Tick)
      }
      else
      {
         iced::Subscription::none()
      };

      let resize_subscription = iced::event::listen_with(|event, _status, _window| {
         match event
         {
            | iced::Event::Window(iced::window::Event::Resized(size)) => Some(Message::WindowResized(size)),
            | _ => None,
         }
      });

      iced::Subscription::batch(vec![models_subscription, clipboard_subscription, tick_subscription, resize_subscription])
   }

   fn view(&self) -> Element<'_, Message>
   {
      let monitor_checkbox = if self.converting
      {
         checkbox("Monitor Clipboard", self.monitor_clipboard)
      }
      else
      {
         checkbox("Monitor Clipboard", self.monitor_clipboard).on_toggle(Message::ToggleMonitor)
      };

      let paste_button = button("Paste Image").on_press_maybe((!self.converting && !self.monitor_clipboard).then_some(Message::PasteFromClipboard));

      let settings_button = button("Settings").on_press_maybe((!self.converting).then_some(Message::OpenSettings));

      let header = row![monitor_checkbox, paste_button, iced::widget::Space::with_width(Length::Fill), settings_button]
                        .spacing(20)
                        .align_y(Alignment::Center);

      let image_display = if let Some(handle) = &self.image_data
      {
         container(scrollable(image_widget(handle.clone()).width(Length::Shrink)
                                                          .height(Length::Shrink)).width(Length::Fill)
                                                                                  .height(Length::Fill))
      }
      else
      {
         container(text("No image in clipboard").size(24)).width(Length::Fill)
                                                          .height(Length::Fill)
                                                          .center_x(Length::Fill)
                                                          .center_y(Length::Fill)
      };

      let convert_button = if self.converting
      {
         button("Cancel").on_press(Message::CancelConvert)
      }
      else
      {
         let model_available = self.selected_model.as_ref().is_some_and(|m| m.info.is_available);
         button("Convert").on_press_maybe((self.image_data.is_some() && model_available).then_some(Message::Convert))
      };

      // Auto Convert checkbox stays enabled during conversion so user can cancel by unchecking
      let auto_convert_checkbox = checkbox("Auto Convert", self.auto_convert).on_toggle(Message::ToggleAutoConvert);

      let refresh_button = button("Refresh").on_press_maybe((!self.converting).then_some(Message::RefreshModels));

      if self.selected_model.is_none()
      {
         eprintln!("No model selected");
      }
      // Show GPU selector for all local Candle-based models
      let is_local_model = matches!(self.selected_model.as_ref().map(|m| &m.info.category), Some(crate::llm::LlmCategory::HuggingFace));

      let mut gpu_selector = row![text("GPU/CPU: ")];
      if is_local_model
      {
         gpu_selector =
            gpu_selector.push(pick_list(self.gpu_list.as_slice(), Some(&self.selected_gpu), Message::GpuSelected).placeholder("Select GPU/CPU..."));
      }

      let mut model_selector = row![text("Model:")];
      model_selector =
         model_selector.push(pick_list(self.models.as_slice(), self.selected_model.as_ref(), Message::ModelSelected).placeholder("Select a model..."));

      let model_selector = model_selector.push(convert_button)
                                         .push(auto_convert_checkbox)
                                         .push(refresh_button)
                                         .spacing(20)
                                         .align_y(Alignment::Center);

      let loading_indicator = if self.converting
      {
         container(progress_bar(0.0..=100.0, self.conversion_progress).height(4)).width(Length::Fill)
      }
      else
      {
         container(iced::widget::Space::with_height(4)).width(Length::Fill)
      };

      let is_error = self.is_error;
      let error_style = move |theme: &iced::Theme, status: text_editor::Status| {
         let mut style = text_editor::default(theme, status);
         if is_error
         {
            style.value = iced::Color::from_rgb(1.0, 0.3, 0.3);
         }
         style
      };

      let latex_editor = if self.converting
      {
         text_editor(&self.latex_content).style(error_style)
                                         .height(Length::Fill)
      }
      else
      {
         text_editor(&self.latex_content).on_action(Message::LatexEdited)
                                         .style(error_style)
                                         .height(Length::Fill)
      };

      let latex_input = container(latex_editor).padding(10)
                                               .style(container::bordered_box);

      let auto_copy_checkbox = if self.converting
      {
         checkbox("Auto-Copy", self.auto_copy)
      }
      else
      {
         checkbox("Auto-Copy", self.auto_copy).on_toggle(Message::ToggleAutoCopy)
      };

      let copy_button = button("Copy LaTeX").on_press_maybe((!self.converting).then_some(Message::CopyLatex));

      let footer = row![auto_copy_checkbox, copy_button].spacing(20)
                                                        .align_y(Alignment::Center);

      let mut content = column![header,
                            container(image_display).height(Length::FillPortion(2))
                                                    .padding(10)
                                                    .style(container::bordered_box),
                            loading_indicator];
      if is_local_model
      {
         content = content.push(gpu_selector);
      }
      content = content.push(model_selector)
                       .push(container(latex_input).height(Length::FillPortion(1)))
                       .push(footer);

      if !self.status_text.is_empty()
      {
         let status_color = if self.is_error
         {
            iced::Color::from_rgb(1.0, 0.3, 0.3)
         }
         else
         {
            iced::Color::from_rgb(0.4, 0.7, 1.0)
         };
         let status_bar = container(text(&self.status_text).size(12)
                                                           .color(status_color))
                                    .padding([2, 8])
                                    .width(Length::Fill);
         content = content.push(status_bar);
      }

      content = content.padding(20).spacing(15);

      let main_view = mouse_area(content)
         .interaction(if self.converting { iced::mouse::Interaction::Pointer } else { iced::mouse::Interaction::Idle });

      if self.show_settings
      {
         let settings = Settings::new().get_settings_or_default();

         let ollama_input = text_input(llm_util::DEFAULT_OLLAMA_URL, &self.settings_ollama_url)
            .on_input(Message::SettingsOllamaUrlChanged)
            .width(Length::Fill);

         let openai_placeholder = if settings.has_chatgpt_key() { "Key is set (enter new to replace)" } else { "Enter OpenAI API key" };
         let gemini_placeholder = if settings.has_gemini_key() { "Key is set (enter new to replace)" } else { "Enter Gemini API key" };

         let openai_input = text_input(openai_placeholder, &self.settings_openai_key)
            .on_input(Message::SettingsOpenAIKeyChanged)
            .secure(!self.show_openai_key)
            .width(Length::Fill);
         let show_openai_label = if self.show_openai_key { "Hide" } else { "Show" };
         let openai_row = row![openai_input, button(show_openai_label).on_press(Message::ToggleShowOpenAIKey)]
            .spacing(10)
            .align_y(Alignment::Center);

         let gemini_input = text_input(gemini_placeholder, &self.settings_gemini_key)
            .on_input(Message::SettingsGeminiKeyChanged)
            .secure(!self.show_gemini_key)
            .width(Length::Fill);
         let show_gemini_label = if self.show_gemini_key { "Hide" } else { "Show" };
         let gemini_row = row![gemini_input, button(show_gemini_label).on_press(Message::ToggleShowGeminiKey)]
            .spacing(10)
            .align_y(Alignment::Center);

         let ollama_key_placeholder = if settings.has_ollama_key() { "Key is set (enter new to replace)" } else { "Enter Ollama API key" };
         let ollama_key_input = text_input(ollama_key_placeholder, &self.settings_ollama_key)
            .on_input(Message::SettingsOllamaKeyChanged)
            .secure(!self.show_ollama_key)
            .width(Length::Fill);
         let show_ollama_label = if self.show_ollama_key { "Hide" } else { "Show" };
         let ollama_key_row = row![ollama_key_input, button(show_ollama_label).on_press(Message::ToggleShowOllamaKey)]
            .spacing(10)
            .align_y(Alignment::Center);

         let theme_checkbox = checkbox("Dark Theme", self.dark_theme).on_toggle(Message::ToggleDarkTheme);

         let button_row = row![iced::widget::Space::with_width(Length::Fill),
                               button("Cancel").on_press(Message::CloseSettings),
                               button("Save").on_press(Message::SaveSettings)]
            .spacing(10);

         let mut dialog_column = column![text("Settings").size(20),
                                         text("Ollama URL").size(14),
                                         ollama_input,
                                         text("OpenAI API Key").size(14),
                                         openai_row,
                                         text("Gemini API Key").size(14),
                                         gemini_row,
                                         text("Ollama Paid Web API Key (https://ollama.com/api)").size(14),
                                         ollama_key_row,
                                         theme_checkbox,
                                         button_row]
            .spacing(10)
            .padding(25)
            .width(500);

         if !self.settings_status.is_empty()
         {
            let status_color = if self.settings_status.starts_with("Error")
            {
               iced::Color::from_rgb(1.0, 0.3, 0.3)
            }
            else
            {
               iced::Color::from_rgb(0.3, 1.0, 0.3)
            };
            dialog_column = dialog_column.push(text(&self.settings_status).size(12).color(status_color));
         }

         let dialog_box = container(dialog_column)
            .style(|_: &iced::Theme| container::Style {
               background: Some(iced::Background::Color(iced::Color::from_rgb(0.15, 0.15, 0.2))),
               border: iced::Border { color: iced::Color::from_rgb(0.4, 0.4, 0.5), width: 1.0, radius: 8.0.into() },
               ..Default::default()
            });

         let overlay = container(dialog_box)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_: &iced::Theme| container::Style {
               background: Some(iced::Background::Color(iced::Color::from_rgba(0.0, 0.0, 0.0, 0.5))),
               ..Default::default()
            });

         iced::widget::stack![main_view, iced::widget::opaque(overlay)].into()
      }
      else
      {
         main_view.into()
      }
   }
}
