//#![feature(os_str_display)]
use std::{env, fs::File, io::Write, path::PathBuf};

use crate::crypt;
use crate::crypt::generate_key;

const PROGRAM: &str = "image2latex";

use crate::llm_util::DEFAULT_OLLAMA_URL;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Settings
{
   #[serde(default = "Settings::default_ollama_url")]
   pub ollama_url:                 String,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub last_model:                           Option<String>,

   pub last_screen_width:                    Option<u32>,
   pub last_screen_height:                   Option<u32>,
   #[serde(default = "Settings::default_dark_theme")]
   pub dark_theme:                           bool,
   // #[serde(skip_serializing_if = "Option::is_none")]
   #[serde(default = "Settings::default_openapi_key")]
   openapi_encrypted_key:           String,

   #[serde(default = "Settings::default_openapi_key")]
   gemini_encrypted_key:            String,
}

impl Default for Settings
{
   fn default() -> Self
//------------------
   {
      Self
      {
         ollama_url: Settings::default_ollama_url(),
         last_model: None,
         last_screen_width: None,
         last_screen_height: None,
         dark_theme: Settings::default_dark_theme(),
         openapi_encrypted_key: Settings::default_openapi_key(),
         gemini_encrypted_key: Settings::default_gemini_key(),
      }
   }
}

impl Settings
//===========
{
   pub fn new() -> Self { Settings::default() }

   fn default_ollama_url() -> String { DEFAULT_OLLAMA_URL.to_string() }

   pub fn default_openapi_key() -> String { "Paste your OpenAPI key here".to_string() }

   pub fn default_gemini_key() -> String { "Paste your Gemini API key here".to_string() }

   fn default_dark_theme() -> bool { true }

   pub fn get_encrypted_openapi_key(&self) -> Result<String, String>
   {
      let value = self.openapi_encrypted_key.trim();
      match value
      {
         "Paste your OpenAPI key here"=> Err("OpenAI API Key is not set".to_string()),
         "" => Err("OpenAI API Key is empty".to_string()),
         _ => Ok(value.to_string()),
      }
   }

   pub fn get_encrypted_gemini_key(&self) -> Result<String, String>
   {
      let value = self.gemini_encrypted_key.trim();
      match value
      {
         "Paste your Gemini API key here" => Err("Gemini API Key is not set".to_string()),
         "" => Err("Gemini API Key is empty".to_string()),
         _ => Ok(value.to_string()),
      }
   }

   pub fn has_openapi_key(&self) -> bool
   {
      let v = self.openapi_encrypted_key.trim();
      !v.is_empty() && v != Self::default_openapi_key()
   }

   pub fn has_gemini_key(&self) -> bool
   {
      let v = self.gemini_encrypted_key.trim();
      !v.is_empty() && v != Self::default_gemini_key()
   }

   pub fn set_openapi_key(&mut self, encrypted_hex: String)
   {
      self.openapi_encrypted_key = encrypted_hex;
   }

   pub fn set_gemini_key(&mut self, encrypted_hex: String)
   {
      self.gemini_encrypted_key = encrypted_hex;
   }

   pub fn encrypt_value(plaintext: &str) -> Result<String, String>
   {
      let key = Settings::get_encryption_key()?;
      match crypt::encrypt(plaintext, &key)
      {
         | Ok(encrypted) => Ok(hex::encode(encrypted)),
         | Err(e) => Err(format!("Encryption failed: {:?}", e)),
      }
   }

   pub fn get_settings(&self) -> Result<Settings, String>
//-------------------------------------------
   {
      let settings_path = match Settings::get_settings_path()
      {
         | Ok(p) => p,
         | Err(_e) => match Settings::write_default_settings()
         {
            | Ok(pp) => pp,
            | Err(e) =>
            {
               let errmsg = format!("Error on get settings: {}", e);
               return Err(errmsg);
            }
         },
      };

      if !settings_path.exists()
      {
         match Settings::write_default_settings()
         {
            | Ok(_) => (),
            | Err(e) =>
            {
               eprintln!("Error creating default settings: {}", e);
               // PathBuf::new()
            }
         };
      }
      Ok(self.read_settings())
   }

   pub fn get_settings_or_default(&self) -> Settings
   //-------------------------------------------
   {
      match self.get_settings()
      {
         | Ok(s) => s,
         | Err(_) => Settings::default(),
      }
   }

   pub fn write_settings(&self) -> Result<PathBuf, std::io::Error>
//-----------------------------------------------------------------------
   {
      let settings_path = match Settings::get_settings_path()
      {
         | Ok(p) => p,
         | Err(_) => match Settings::write_default_settings()
         {
            | Ok(pp) => pp,
            | Err(e) =>
            {
               // println!("Error on get settings: {}", e);
               return Err(e);
            }
         },
      };
      let mut file = File::create(&settings_path)?;
      for retry in 0..3
      {
         match file.try_lock()
         {
            | Ok(_) => break,
            | Err(e) =>
            {
               if retry == 2
               {
                  let errmsg = format!("Failed to lock settings file {}: {}", settings_path.display(), e);
                  // println!("{errmsg}");
                  return Err(std::io::Error::other(errmsg));
               }
               std::thread::sleep(std::time::Duration::from_millis(500));
            }
         }
      }
      let json = serde_json::to_string_pretty(&self)?;
      file.write_all(json.as_bytes())?;
      // println!("Wrote settings {} to {}", json, settings_path.display());
      Ok(settings_path)
   }

   fn get_encryption_key() -> Result<String, String>
   //-----------------------------------------------
   {
      // Read encryption key from hidden file encryption-key with read permissions only for current user
      let encryption_file_path = match Settings::get_config_path()
      {
         | Ok(mut p) =>
         {
            p.push("encryption-key");
            p
         }
         | Err(e) =>
         {
            let errmsg = format!("Failed to get config path for encryption key: {}", e);
            eprintln!("{errmsg}");
            return Err(errmsg);
         }
      };
      let hex_key: String;
      if !encryption_file_path.exists()
      {
         let new_key = generate_key();
         hex_key = hex::encode(new_key);
         match std::fs::write(&encryption_file_path, &hex_key)
         {
            | Ok(_) => (),
            | Err(e) =>
            {
               let errmsg = format!("Failed to write encryption key to file {}: {}", encryption_file_path.display(), e);
               eprintln!("{errmsg}");
               return Err(errmsg);
            }
         };
         // Restrict file permissions to owner read/write only
         #[cfg(unix)]
         {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&encryption_file_path, perms);
         }
         #[cfg(windows)]
         {
            // Use icacls to remove inherited permissions and grant only the current user full control
            if let Ok(username) = env::var("USERNAME")
            {
               let path_str = encryption_file_path.display().to_string();
               let _ = std::process::Command::new("icacls")
                  .args([&path_str, "/inheritance:r", "/grant:r", &format!("{}:(R,W)", username)])
                  .output();
            }
         }
      }
      else
      {
         hex_key = match std::fs::read_to_string(&encryption_file_path)
         {
            | Ok(key) => key.trim().to_string(),
            | Err(e) =>
            {
               let errmsg = format!("Failed to read encryption key from file {}: {}", encryption_file_path.display(), e);
               eprintln!("{errmsg}");
               return Err(errmsg);
            }
         };
      }
      Ok(hex_key)
   }

   pub fn decrypt(name: &str, encrypted: &str) -> Result<String, String>
   //-------------------------------------------------------
   {
      if encrypted.trim().is_empty()
      {
         return Err(format!("Empty key for {}", name));
      }
      let encrypted_bytes = match hex::decode(encrypted)
      {
         Ok(bytes) => bytes,
         Err(e) =>
         {
            let errmsg = format!("Failed to hex decode encrypted password: {}", e);
            return Err(errmsg);
         }
      };
      if encrypted_bytes.is_empty()
      {
         return Err(format!("Empty decrypted key for {}", name));
      }
      {
         let key = match Settings::get_encryption_key()
         {
            |  Ok(k) => k,
               Err(e) => { return Err(format!("Encryption key is missing [{}]", e)); }
         };
         match crypt::decrypt(&encrypted_bytes, &key)
         {
            |  Ok(decrypted) =>
               {
                  Ok(decrypted)
               }
               Err(e) =>
               {
                  let errmsg = format!("Failed to decrypt key for {}: {}",name, e);
                  eprintln!("{errmsg}");
                  Err(errmsg)
               }
         }
      }
   }

   /// Get OS specific path to the config directory for the program
   pub fn get_config_path() -> Result<PathBuf, std::io::Error>
   //-----------------------------------------------------------------------------------------
   {
      match dirs::config_dir() // cargo add dirs
      {
         | Some(p) =>
         {
            let pp = p.join(PROGRAM);
            if !pp.exists()
            {
               match std::fs::create_dir_all(pp.as_path())
               {
                  | Ok(_) => (),
                  | Err(e) =>
                  {
                     return Err(std::io::Error::other(format!("Failed to create config directory {}: {}",
                                                            pp.display(), e)));
                  }
               }
            }
            Ok(pp)
         }
         | None =>
         {
            let mut config_path = Settings::get_home_dir();

            if env::consts::OS == "windows"
            {
               let mut pp = config_path.clone();
               pp.push("AppData/Local");
               if pp.is_dir()
               {
                  config_path.push("AppData/Local");
               }
               else
               {
                  pp.pop();
                  pp.pop();
                  pp.push("Local Settings/");
                  if pp.is_dir()
                  {
                     config_path.push("Local Settings/");
                  }
                  else
                  {
                     config_path.push("Application Data/Local Settings/");
                  }
               }
            }
            else if env::consts::OS == "macos"
            {
               config_path.push(Settings::get_home_dir());
               config_path.push(".config/");
               if ! config_path.is_dir()
               {
                  config_path.pop();
                  config_path.push("Library/");
                  config_path.push("Application Support/");
                  if ! config_path.is_dir()
                  {
                     config_path.pop();
                     config_path.pop();
                  }
               }
            }
            else
            {
               config_path.push(".config/");
            }
            config_path.push(PROGRAM);
            if config_path.exists() && !config_path.is_dir()
            {
               return Err(std::io::Error::other(format!("Config path {} exists and is not a directory",
                                                      config_path.display())));
            }
            if !config_path.exists()
            {
               std::fs::create_dir_all(config_path.as_path())?;
            }
            Ok(config_path)
         }
      }
   }

   /// Get the path to the settings file for the program.
   pub fn get_settings_path() -> Result<PathBuf, std::io::Error>
   //-------------------------------------------------------------------
   {
      let mut config_path = match Settings::get_config_path()
      {
         | Ok(p) => p,
         | Err(e) =>
         {
            eprintln!("Error getting settings path: {}", e);
            return Err(e);
         }
      };
      config_path.push("settings.json");
      Ok(config_path)
   }

   pub fn write_default_settings() -> Result<PathBuf, std::io::Error>
//-----------------------------------------------------------------------
   {
      let settings = Settings::default();
      let mut config_file = Settings::get_config_path()?;
      config_file.push("settings.json");
      let mut file = File::create(&config_file)?;
      let json = serde_json::to_string_pretty(&settings)?;
      file.write_all(json.as_bytes())?;
      // let file = File::create(&config_file)?;
      // let mut writer = BufWriter::new(file);
      // serde_json::to_writer(&mut writer, &settings)?;
      Ok(config_file)
   }

   fn read_settings(&self) -> Settings
//-----------------------------------------------------------------
   {
      let mut config_file = match Settings::get_config_path()
      {
         | Ok(p) => p,
         | Err(e) =>
         {
            eprintln!("Error getting settings path: {}", e);
            return Settings::default();
         }
      };
      config_file.push("settings.json");
      if !config_file.exists()
      {
         return Settings::default();
      }
      let file = match File::open(&config_file)
      {
         | Ok(f) => f,
         | Err(e) =>
         {
            eprintln!("Error opening settings file: {}", e);
            return Settings::default();
         }
      };
      let settings: Settings = match serde_json::from_reader(file)
      {
         | Ok(s) => s,
         | Err(e) =>
         {
            eprintln!("Error reading settings: {}", e);
            Settings::default()
         }
      };
      settings.clone()
   }

   fn get_home_fallbacks() -> PathBuf
//--------------------------------
   {
      if cfg!(target_os = "linux")
      {
         return PathBuf::from("~/");
      }
      else if cfg!(target_os = "windows")
      {
         return PathBuf::from("C:/Users/Public");
      }
      return PathBuf::from("~/");
   }

   pub fn get_home_dir() -> PathBuf
//-------------------------------
   {
      match dirs::home_dir()
      {
         | Some(h) => h,
         | None => Settings::get_home_fallbacks(),
      }
   }

   #[allow(dead_code)]
   pub fn get_home_dir_string() -> String
//-------------------------------
   {
      match dirs::home_dir()
      {
         | Some(h) => h.display().to_string(),
         | None =>
         {
            let pp = Settings::get_home_fallbacks();
            pp.display().to_string()
         }
      }
   }
}

unsafe impl Sync for Settings {}
