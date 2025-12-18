//! Binary .xd format implementation
//!
//! High-performance binary serialization for VEXL vectors.
//! Based on the specification in DOCUMENTATION/reference/vector-formats.md

use std::io::{Read, Write};
use std::mem;
use crc32fast::Hasher as Crc32Hasher;
use lz4::{Decoder, EncoderBuilder};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, KeyInit};
use sha2::{Sha256, Digest};
use crate::{Result, SerializeError, VectorSerialize};
use vexl_core::vector::{Vector, Dim1, Dim2, Dim3, DimN, VectorStorage};

/// Binary format header (136 bytes total)
#[derive(Debug, Clone)]
#[repr(C)]
struct Header {
    /// Magic bytes "VEXL"
    magic: [u8; 4],
    /// Format version
    version: u16,
    /// Reserved for future use
    _reserved: u16,
    /// Element type tag
    type_tag: u64,
    /// Number of dimensions
    dimensionality: u64,
    /// Dimension sizes (max 8 dimensions)
    shape: [u64; 8],
    /// Storage mode
    storage_mode: u8,
    /// Metadata flags
    metadata: u64,
    /// Size of data section in bytes
    data_size: u64,
    /// Size of index section (sparse only)
    index_size: u64,
    /// CRC32 checksum of header
    checksum: u32,
    /// Reserved for future use
    _reserved2: u64,
}

impl Header {
    /// Create a new header
    fn new(type_tag: u8, dimensionality: usize, shape: &[u64], storage_mode: u8, metadata: u64, data_size: u64, index_size: u64) -> Self {
        let mut header = Self {
            magic: *b"VEXL",
            version: 1,
            _reserved: 0,
            type_tag: type_tag as u64,
            dimensionality: dimensionality as u64,
            shape: [0; 8],
            storage_mode,
            metadata,
            data_size,
            index_size,
            checksum: 0,
            _reserved2: 0,
        };

        // Copy shape (up to 8 dimensions)
        for (i, &dim) in shape.iter().enumerate().take(8) {
            header.shape[i] = dim;
        }

        // Calculate checksum
        header.checksum = header.calculate_checksum();

        header
    }

    /// Calculate CRC32 checksum of header (excluding checksum field itself)
    fn calculate_checksum(&self) -> u32 {
        let mut hasher = Crc32Hasher::new();

        // Hash all fields except checksum
        unsafe {
            let header_ptr = self as *const Self as *const u8;
            let size_without_checksum = mem::size_of::<Self>() - mem::size_of::<u32>();
            let slice = std::slice::from_raw_parts(header_ptr, size_without_checksum);
            hasher.update(slice);
        }

        hasher.finalize()
    }

    /// Validate header
    fn validate(&self) -> Result<()> {
        // Check magic
        if &self.magic != b"VEXL" {
            return Err(SerializeError::InvalidFormat("Invalid magic bytes".to_string()));
        }

        // Check version
        if self.version != 1 {
            return Err(SerializeError::UnsupportedVersion(self.version));
        }

        // Validate checksum
        let calculated = self.calculate_checksum();
        if calculated != self.checksum {
            return Err(SerializeError::ChecksumMismatch);
        }

        Ok(())
    }

    /// Read header from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < mem::size_of::<Self>() {
            return Err(SerializeError::InvalidFormat("Header too small".to_string()));
        }

        let header: Self = unsafe { std::ptr::read(bytes.as_ptr() as *const Self) };
        header.validate()?;
        Ok(header)
    }

    /// Convert to bytes
    fn to_bytes(&self) -> Vec<u8> {
        unsafe {
            let ptr = self as *const Self as *const u8;
            let slice = std::slice::from_raw_parts(ptr, mem::size_of::<Self>());
            slice.to_vec()
        }
    }
}

/// Metadata flags
#[derive(Debug, Clone, Copy)]
pub struct MetadataFlags {
    pub compressed: bool,
    pub encrypted: bool,
    pub checksummed: bool,
    pub big_endian: bool,
    pub compression_level: u8,
}

impl MetadataFlags {
    /// Create from u64
    fn from_u64(value: u64) -> Self {
        Self {
            compressed: (value & 1) != 0,
            encrypted: (value & 2) != 0,
            checksummed: (value & 4) != 0,
            big_endian: (value & 8) != 0,
            compression_level: ((value >> 4) & 0xF) as u8,
        }
    }

    /// Convert to u64
    fn to_u64(&self) -> u64 {
        let mut value = 0u64;
        if self.compressed { value |= 1; }
        if self.encrypted { value |= 2; }
        if self.checksummed { value |= 4; }
        if self.big_endian { value |= 8; }
        value |= (self.compression_level as u64 & 0xF) << 4;
        value
    }
}

/// Encryption context
struct EncryptionContext {
    key: Key<Aes256Gcm>,
    cipher: Aes256Gcm,
}

impl EncryptionContext {
    fn new(key_bytes: &[u8; 32]) -> Self {
        let key = Key::<Aes256Gcm>::from_slice(key_bytes);
        let cipher = Aes256Gcm::new(key);
        Self { key: *key, cipher }
    }

    fn encrypt(&self, data: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
        let nonce = Nonce::from_slice(nonce);
        self.cipher.encrypt(nonce, data)
            .map_err(|e| SerializeError::EncryptionError(e.to_string()))
    }

    fn decrypt(&self, data: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
        let nonce = Nonce::from_slice(nonce);
        self.cipher.decrypt(nonce, data)
            .map_err(|e| SerializeError::EncryptionError(e.to_string()))
    }
}

/// Binary serialization implementation for Vector<T, D>
impl<T, D> VectorSerialize for Vector<T, D>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::iter::Sum<T>,
    D: Clone,
{
    fn to_xd(&self) -> Result<Vec<u8>> {
        // For now, implement basic dense vector serialization
        // TODO: Support all storage modes

        let element_count = self.len() as u64;
        let element_size = mem::size_of::<T>() as u64;
        let data_size = element_count * element_size;

        // Create shape array (1D for now)
        let shape = vec![element_count];

        // Create header
        let type_tag = crate::utils::element_type_to_tag(vexl_core::vector::ElementType::I64); // TODO: Proper type detection
        let storage_mode = crate::utils::storage_mode_to_tag(VectorStorage::Dense);
        let metadata = MetadataFlags {
            compressed: false,
            encrypted: false,
            checksummed: true,
            big_endian: false,
            compression_level: 0,
        };

        let header = Header::new(
            type_tag,
            1, // 1D for now
            &shape,
            storage_mode,
            metadata.to_u64(),
            data_size,
            0, // No index for dense
        );

        // Serialize data
        let mut result = header.to_bytes();

        // Add SHA256 checksum if requested
        if metadata.checksummed {
            let mut hasher = Sha256::new();
            // Hash header + data placeholder
            hasher.update(&result);
            let checksum = hasher.finalize();
            result.extend_from_slice(&checksum);
        }

        // Add data section
        for i in 0..self.len() {
            let value = self.get(i);
            let bytes = unsafe {
                let ptr = &value as *const T as *const u8;
                std::slice::from_raw_parts(ptr, mem::size_of::<T>())
            };
            result.extend_from_slice(bytes);
        }

        // Compress if requested
        if metadata.compressed {
            let mut encoder = EncoderBuilder::new()
                .level(metadata.compression_level as u32)
                .build(Vec::new())
                .map_err(|e| SerializeError::CompressionError(e.to_string()))?;

            encoder.write_all(&result[mem::size_of::<Header>()..])
                .map_err(|e| SerializeError::CompressionError(e.to_string()))?;

            let (compressed, _result) = encoder.finish();
            result.truncate(mem::size_of::<Header>());
            result.extend_from_slice(&compressed);
        }

        Ok(result)
    }

    fn from_xd(data: &[u8]) -> Result<Self> {
        if data.len() < mem::size_of::<Header>() {
            return Err(SerializeError::InvalidFormat("Data too small for header".to_string()));
        }

        // Read header
        let header = Header::from_bytes(&data[..mem::size_of::<Header>()])?;
        let metadata = MetadataFlags::from_u64(header.metadata);

        let mut data_offset = mem::size_of::<Header>();

        // Skip checksum if present
        if metadata.checksummed {
            data_offset += 32; // SHA256 size
        }

        // Decompress if needed
        let data_section = if metadata.compressed {
            let mut decoder = Decoder::new(&data[data_offset..])
                .map_err(|e| SerializeError::CompressionError(e.to_string()))?;

            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)
                .map_err(|e| SerializeError::CompressionError(e.to_string()))?;

            decompressed
        } else {
            data[data_offset..].to_vec()
        };

        // For now, only support dense i64 vectors
        if header.type_tag != 0 || header.storage_mode != 0 {
            return Err(SerializeError::NotImplemented("Only dense i64 vectors supported".to_string()));
        }

        let element_count = header.shape[0] as usize;
        let element_size = mem::size_of::<T>();

        if data_section.len() != element_count * element_size {
            return Err(SerializeError::InvalidFormat("Data size mismatch".to_string()));
        }

        // Reconstruct vector
        let mut elements = Vec::with_capacity(element_count);
        for i in 0..element_count {
            let offset = i * element_size;
            let bytes = &data_section[offset..offset + element_size];

            let value: T = unsafe { std::ptr::read(bytes.as_ptr() as *const T) };
            elements.push(value);
        }

        Ok(Vector::new_with_values(elements, header.dimensionality as usize, VectorStorage::Dense))
    }

    fn to_xt(&self) -> Result<String> {
        // Delegate to text module
        Err(SerializeError::NotImplemented("Use text module for .xt format".to_string()))
    }

    fn from_xt(_text: &str) -> Result<Self> {
        Err(SerializeError::NotImplemented("Use text module for .xt format".to_string()))
    }

    fn to_xg(&self) -> Result<String> {
        // Delegate to graphics module
        Err(SerializeError::NotImplemented("Use graphics module for .xg format".to_string()))
    }

    fn from_xg(_xml: &str) -> Result<Self> {
        Err(SerializeError::NotImplemented("Use graphics module for .xg format".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_core::vector::VectorStorage;

    #[test]
    fn test_header_roundtrip() {
        let header = Header::new(0, 1, &[10], 0, 0, 80, 0);
        let bytes = header.to_bytes();
        let decoded = Header::from_bytes(&bytes).unwrap();

        assert_eq!(header.magic, decoded.magic);
        assert_eq!(header.version, decoded.version);
        assert_eq!(header.type_tag, decoded.type_tag);
        assert_eq!(header.dimensionality, decoded.dimensionality);
        assert_eq!(header.shape[0], decoded.shape[0]);
        assert_eq!(header.storage_mode, decoded.storage_mode);
    }

    #[test]
    fn test_vector_serialization_roundtrip() {
        let original = Vector::<i64, Dim1>::new_with_values([1, 2, 3, 4, 5], 1, VectorStorage::Dense);

        let serialized = original.to_xd().unwrap();
        let deserialized = Vector::<i64, Dim1>::from_xd(&serialized).unwrap();

        assert_eq!(original.len(), deserialized.len());
        for i in 0..original.len() {
            assert_eq!(original.get(i), deserialized.get(i));
        }
    }

    #[test]
    fn test_metadata_flags() {
        let flags = MetadataFlags {
            compressed: true,
            encrypted: false,
            checksummed: true,
            big_endian: false,
            compression_level: 6,
        };

        let encoded = flags.to_u64();
        let decoded = MetadataFlags::from_u64(encoded);

        assert_eq!(flags.compressed, decoded.compressed);
        assert_eq!(flags.encrypted, decoded.encrypted);
        assert_eq!(flags.checksummed, decoded.checksummed);
        assert_eq!(flags.big_endian, decoded.big_endian);
        assert_eq!(flags.compression_level, decoded.compression_level);
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = vec![0u8; mem::size_of::<Header>()];
        bytes[0..4].copy_from_slice(b"INVALID");

        assert!(Header::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_checksum_validation() {
        let header = Header::new(0, 1, &[10], 0, 0, 80, 0);
        let mut bytes = header.to_bytes();

        // Corrupt checksum
        let checksum_offset = mem::size_of::<Header>() - 4;
        bytes[checksum_offset] = !bytes[checksum_offset];

        assert!(Header::from_bytes(&bytes).is_err());
    }
}
