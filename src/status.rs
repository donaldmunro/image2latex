use std::sync::{LazyLock, RwLock};

static STATUS: LazyLock<RwLock<String>> = LazyLock::new(|| RwLock::new(String::new()));

pub fn set_status(msg: &str)
{
   if let Ok(mut s) = STATUS.write()
   {
      *s = msg.to_string();
   }
}

pub fn get_status() -> String
{
   STATUS.read().map(|s| s.clone()).unwrap_or_default()
}

pub fn clear_status()
{
   set_status("");
}
