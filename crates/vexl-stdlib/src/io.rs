use std::fs;
use std::io::{self};
use std::path::Path;
use anyhow::{Result, Context};

/// Read entire file to string
pub fn read_file(path: &str) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("Failed to read file: {}", path))
}

/// Write string to file
pub fn write_file(path: &str, content: &str) -> Result<()> {
    fs::write(path, content).with_context(|| format!("Failed to write to file: {}", path))
}

/// Read line from stdin
pub fn read_line() -> Result<String> {
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).context("Failed to read line from stdin")?;
    Ok(buffer.trim_end().to_string())
}

/// Check if file exists
pub fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}
