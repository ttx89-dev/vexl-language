//! Universal Vector Serialization for VEXL
//!
//! This crate implements the "everything is a vector" serialization paradigm,
//! providing multiple formats (.xd, .xt, .xg, .xa, .xs) for different use cases.

use std::collections::HashMap;
use vexl_core::vector::{Vector, Dim1, Dim2, Dim3, DimN};

/// Errors that can occur during serialization/deserialization
#[derive(Debug, Clone, PartialEq)]
pub enum SerializeError {
    /// Invalid format or corrupted data
    InvalidFormat(String),
    /// Unsupported version
    UnsupportedVersion(u16),
    /// Type mismatch during deserialization
    TypeMismatch(String),
    /// I/O error
    IoError(String),
    /// Compression/decompression error
    CompressionError(String),
    /// Encryption/decryption error
    EncryptionError(String),
    /// Checksum validation failed
    ChecksumMismatch,
    /// Feature not implemented
    NotImplemented(String),
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializeError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            SerializeError::UnsupportedVersion(version) => write!(f, "Unsupported version: {}", version),
            SerializeError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            SerializeError::IoError(msg) => write!(f, "IO error: {}", msg),
            SerializeError::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            SerializeError::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
            SerializeError::ChecksumMismatch => write!(f, "Checksum mismatch"),
            SerializeError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
        }
    }
}

pub type Result<T> = std::result::Result<T, SerializeError>;

/// Core trait for vector serialization
pub trait VectorSerialize {
    /// Serialize to binary .xd format
    fn to_xd(&self) -> Result<Vec<u8>>;

    /// Deserialize from binary .xd format
    fn from_xd(data: &[u8]) -> Result<Self>
    where
        Self: Sized;

    /// Serialize to text .xt format
    fn to_xt(&self) -> Result<String>;

    /// Deserialize from text .xt format
    fn from_xt(text: &str) -> Result<Self>
    where
        Self: Sized;

    /// Serialize to graphics .xg format
    fn to_xg(&self) -> Result<String>;

    /// Deserialize from graphics .xg format
    fn from_xg(xml: &str) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for archive operations
pub trait VectorArchive {
    /// Create a multi-vector archive
    fn create_archive(vectors: HashMap<String, &dyn VectorSerialize>) -> Result<Vec<u8>>;

    /// Read a multi-vector archive
    fn read_archive(data: &[u8]) -> Result<HashMap<String, Box<dyn VectorSerialize>>>;
}

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Stream ID
    pub stream_id: u32,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Enable encryption
    pub encrypted: bool,
    /// Buffer size
    pub buffer_size: usize,
}

/// Handle for stream operations
#[derive(Debug)]
pub struct StreamHandle {
    pub stream_id: u32,
    pub sequence: u64,
}

/// Trait for streaming operations
pub trait VectorStream {
    /// Start a new stream
    fn start_stream(&mut self, config: StreamConfig) -> Result<StreamHandle>;

    /// Send a vector over the stream
    fn send_vector(&mut self, handle: &StreamHandle, vector: &dyn VectorSerialize) -> Result<()>;

    /// Receive a vector from the stream
    fn receive_vector(&mut self, handle: &StreamHandle) -> Result<Box<dyn VectorSerialize>>;
}

// Re-export submodules
pub mod binary;
pub mod text;
pub mod graphics;
pub mod archive;
pub mod stream;

/// Utility functions for type conversion
pub mod utils {
    use vexl_core::vector::ElementType;
    use crate::Result;

    /// Convert VEXL ElementType to type tag
    pub fn element_type_to_tag(element_type: ElementType) -> u8 {
        match element_type {
            ElementType::I64 => 0,
            ElementType::F64 => 1,
            ElementType::Bool => 2,
            ElementType::String => 3,
            // ElementType::Nested => 4, // Future extension
        }
    }

    /// Convert type tag to VEXL ElementType
    pub fn tag_to_element_type(tag: u8) -> Result<ElementType> {
        match tag {
            0 => Ok(ElementType::I64),
            1 => Ok(ElementType::F64),
            2 => Ok(ElementType::Bool),
            3 => Ok(ElementType::String),
            4 => Err(crate::SerializeError::NotImplemented("Nested types not yet supported".to_string())),
            _ => Err(crate::SerializeError::InvalidFormat(format!("Unknown type tag: {}", tag))),
        }
    }

    /// Convert storage mode to tag
    pub fn storage_mode_to_tag(storage_mode: vexl_core::vector::VectorStorage) -> u8 {
        match storage_mode {
            vexl_core::vector::VectorStorage::Dense => 0,
            vexl_core::vector::VectorStorage::Coo => 1,
            vexl_core::vector::VectorStorage::Csr => 2,
            vexl_core::vector::VectorStorage::Generator => 3,
            vexl_core::vector::VectorStorage::Memoized => 4,
        }
    }

    /// Convert tag to storage mode
    pub fn tag_to_storage_mode(tag: u8) -> Result<vexl_core::vector::VectorStorage> {
        match tag {
            0 => Ok(vexl_core::vector::VectorStorage::Dense),
            1 => Ok(vexl_core::vector::VectorStorage::Coo),
            2 => Ok(vexl_core::vector::VectorStorage::Csr),
            3 => Ok(vexl_core::vector::VectorStorage::Generator),
            4 => Ok(vexl_core::vector::VectorStorage::Memoized),
            _ => Err(crate::SerializeError::InvalidFormat(format!("Unknown storage mode tag: {}", tag))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_core::vector::{VectorStorage, ElementType};

    #[test]
    fn test_element_type_conversion() {
        assert_eq!(utils::element_type_to_tag(ElementType::I64), 0);
        assert_eq!(utils::element_type_to_tag(ElementType::F64), 1);
        assert_eq!(utils::element_type_to_tag(ElementType::Bool), 2);
        assert_eq!(utils::element_type_to_tag(ElementType::String), 3);

        assert_eq!(utils::tag_to_element_type(0).unwrap(), ElementType::I64);
        assert_eq!(utils::tag_to_element_type(1).unwrap(), ElementType::F64);
        assert_eq!(utils::tag_to_element_type(2).unwrap(), ElementType::Bool);
        assert_eq!(utils::tag_to_element_type(3).unwrap(), ElementType::String);
    }

    #[test]
    fn test_storage_mode_conversion() {
        assert_eq!(utils::storage_mode_to_tag(VectorStorage::Dense), 0);
        assert_eq!(utils::storage_mode_to_tag(VectorStorage::Coo), 1);
        assert_eq!(utils::storage_mode_to_tag(VectorStorage::Csr), 2);
        assert_eq!(utils::storage_mode_to_tag(VectorStorage::Generator), 3);
        assert_eq!(utils::storage_mode_to_tag(VectorStorage::Memoized), 4);

        assert_eq!(utils::tag_to_storage_mode(0).unwrap(), VectorStorage::Dense);
        assert_eq!(utils::tag_to_storage_mode(1).unwrap(), VectorStorage::Coo);
        assert_eq!(utils::tag_to_storage_mode(2).unwrap(), VectorStorage::Csr);
        assert_eq!(utils::tag_to_storage_mode(3).unwrap(), VectorStorage::Generator);
        assert_eq!(utils::tag_to_storage_mode(4).unwrap(), VectorStorage::Memoized);
    }
}
