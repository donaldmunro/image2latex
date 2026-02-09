use std::error::Error;

use aes_gcm::{ // cargo add aes-gcm
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Nonce
};

type EncryptedData = Vec<u8>;

// const KEY: &str = "f40efce4dbefc325d25779aaf18340e10aef9b053f61901d8f3b4ce72ba81c2f";

pub fn generate_key() -> String
//--------------------------------
{
   let key = Aes256Gcm::generate_key(&mut OsRng);
   hex::encode(key)
}

pub fn encrypt(password: &str, key: &str) -> Result<EncryptedData, aes_gcm::Error>
//-----------------------------------------------------------------------------------------------
{
   // let key = Aes256Gcm::generate_key(&mut OsRng);
   let key_bytes = hex::decode(key).expect("Invalid hex key");
   let cipher = Aes256Gcm::new_from_slice(&key_bytes).expect("Invalid key length");
   let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

   let mut ciphertext = cipher.encrypt(&nonce, password.as_bytes())?;
   let mut result = Vec::with_capacity(nonce.len() + ciphertext.len());
   result.extend_from_slice(&nonce);
   result.append(&mut ciphertext);

   Ok(result)
}

pub fn decrypt(data: &[u8], key: &str) -> Result<String, Box<dyn Error>>
//---------------------------------------------------------------------------------------
{
   // let key = Aes256Gcm::generate_key(&mut OsRng);
   let key_bytes = hex::decode(key).expect("Invalid hex key");
   const NONCE_LEN: usize = 12; // GCM nonce size

   if data.len() < NONCE_LEN
   {
      return Err("Encrypted data too short".into());
   }

   let cipher = Aes256Gcm::new_from_slice(&key_bytes).expect("Invalid key length");
   let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);
   let nonce = Nonce::clone_from_slice(nonce_bytes);

   let plaintext = cipher.decrypt(&nonce, ciphertext).map_err(|e| format!("Decryption failed: {:?}", e))?;
   Ok(String::from_utf8(plaintext)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip()
    {
       let key = generate_key();
       let password = "my_secret_password";
        let encrypted = encrypt(password, &key).expect("Encryption failed");
        let decrypted = decrypt(&encrypted, &key).expect("Decryption failed");
        assert_eq!(password, decrypted);
    }

    #[test]
    fn test_decrypt_invalid_data() {
        // Create data that is long enough (nonce + ciphertext) but invalid
        // 12 bytes nonce + some ciphertext
        let mut data = vec![0u8; 20];
        // Fill with some random values to ensure it's not a valid tag/ciphertext
        for i in 0..data.len() {
            data[i] = (i % 255) as u8;
        }
        let key = generate_key();
        let result = decrypt(&data, &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_short_data() {
        let data = vec![0u8; 5]; // Too short for nonce (12 bytes)
         let key = generate_key();
        let result = decrypt(&data, &key);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Encrypted data too short");
    }

    #[test]
    fn test_encrypt_produces_different_outputs() {
        let password = "password";
         let key = generate_key();
        let enc1 = encrypt(password, &key).expect("Encryption failed");
        let enc2 = encrypt(password, &key).expect("Encryption failed");
        // Because of the random nonce, outputs should be different even for same input
        assert_ne!(enc1, enc2);
    }
}
