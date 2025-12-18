//! Fractal Storage System for VEXL
//!
//! Infinite logical storage through generators. Everything becomes generators:
//! - Files: generator that yields bytes
//! - Databases: generator that yields rows
//! - Networks: generator that yields packets
//! - Computations: generator that yields results

use std::collections::HashMap;
use std::path::Path;
use parking_lot::{Mutex, RwLock};
use vexl_core::generator::{Generator, TieredCache};
use vexl_serialize::VectorSerialize;
use thiserror::Error;

/// Simple value type for storage operations
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
}

/// Errors that can occur in fractal storage operations
#[derive(Error, Debug)]
pub enum FractalStorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Generator evaluation error: {0}")]
    Generator(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Key not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, FractalStorageError>;

/// Storage tier levels for fractal storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    /// Hot values in memory (L1 cache)
    HotMemory,
    /// Chunk cache for sequential access (L2 cache)
    ChunkCache,
    /// Disk-backed persistent storage (L3 cache)
    DiskBacked,
    /// Generator algorithms (infinite, zero storage cost)
    Generator,
}

/// Configuration for fractal storage
#[derive(Debug, Clone)]
pub struct FractalStorageConfig {
    /// Maximum memory usage for hot cache (bytes)
    pub hot_cache_limit: usize,
    /// Maximum memory usage for chunk cache (bytes)
    pub chunk_cache_limit: usize,
    /// Path for disk-backed storage
    pub disk_path: Option<String>,
    /// Compression level for disk storage (0-9)
    pub compression_level: u8,
    /// Maximum generator evaluation depth
    pub max_generator_depth: usize,
}

impl Default for FractalStorageConfig {
    fn default() -> Self {
        Self {
            hot_cache_limit: 100 * 1024 * 1024, // 100MB
            chunk_cache_limit: 500 * 1024 * 1024, // 500MB
            disk_path: None,
            compression_level: 6,
            max_generator_depth: 1000,
        }
    }
}

/// Core fractal storage trait
pub trait FractalStorage {
    /// Store a generator algorithm (infinite storage)
    fn store_generator<G: Generator + Clone + Send + Sync + 'static>(&mut self, key: &str, generator: G) -> Result<()>;

    /// Store a computed value (materialized storage)
    fn store_value<T: VectorSerialize + 'static>(&mut self, key: &str, value: T) -> Result<()>;

    /// Retrieve a value by key (lazy evaluation)
    fn get<T: VectorSerialize + 'static>(&self, key: &str) -> Result<Option<T>>;

    /// Check if a key exists
    fn exists(&self, key: &str) -> bool;

    /// Remove a key
    fn remove(&self, key: &str) -> Result<()>;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total keys stored
    pub total_keys: usize,
    /// Keys in hot memory
    pub hot_memory_keys: usize,
    /// Keys in chunk cache
    pub chunk_cache_keys: usize,
    /// Keys on disk
    pub disk_keys: usize,
    /// Keys as generators
    pub generator_keys: usize,
    /// Total memory usage (bytes)
    pub memory_usage: usize,
    /// Disk usage (bytes)
    pub disk_usage: usize,
}

/// Wrapper for VectorSerialize
struct VectorSerializeWrapper(Box<dyn VectorSerialize>);

impl VectorSerialize for VectorSerializeWrapper {
    fn to_xd(&self) -> vexl_serialize::Result<Vec<u8>> {
        self.0.to_xd()
    }

    fn from_xd(_data: &[u8]) -> vexl_serialize::Result<Self>
    where
        Self: Sized {
        // This would need a registry of constructors, simplified for now
        Err(vexl_serialize::SerializeError::NotImplemented("Dynamic construction not implemented".to_string()))
    }

    fn to_xt(&self) -> vexl_serialize::Result<String> {
        self.0.to_xt()
    }

    fn from_xt(_text: &str) -> vexl_serialize::Result<Self>
    where
        Self: Sized {
        Err(vexl_serialize::SerializeError::NotImplemented("Dynamic construction not implemented".to_string()))
    }

    fn to_xg(&self) -> vexl_serialize::Result<String> {
        self.0.to_xg()
    }

    fn from_xg(_xml: &str) -> vexl_serialize::Result<Self>
    where
        Self: Sized {
        Err(vexl_serialize::SerializeError::NotImplemented("Dynamic construction not implemented".to_string()))
    }
}

/// Internal storage entry
enum StorageEntry {
    /// Materialized value stored in memory/disk
    Materialized(VectorSerializeWrapper),
    /// Generator for infinite/computed sequences (simplified for now)
    Generator(String), // Placeholder - would store serialized generator
}

/// Main fractal storage implementation
pub struct FractalStore {
    /// Configuration
    config: FractalStorageConfig,
    /// Storage entries
    entries: RwLock<HashMap<String, StorageEntry>>,
    /// Hot memory cache (L1)
    hot_cache: TieredCache<Value>,
    /// Chunk cache (L2)
    chunk_cache: Mutex<HashMap<String, Vec<u8>>>,
    /// Disk-backed storage (L3)
    disk_storage: Option<DiskBackend>,
}

/// Disk backend for persistent storage
struct DiskBackend {
    base_path: String,
}

impl DiskBackend {
    fn new(path: &str) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        Ok(Self {
            base_path: path.to_string(),
        })
    }

    fn store(&self, key: &str, data: &[u8]) -> Result<()> {
        let path = Path::new(&self.base_path).join(format!("{}.xd", key));
        std::fs::write(path, data)?;
        Ok(())
    }

    fn load(&self, key: &str) -> Result<Vec<u8>> {
        let path = Path::new(&self.base_path).join(format!("{}.xd", key));
        Ok(std::fs::read(path)?)
    }

    fn exists(&self, key: &str) -> bool {
        let path = Path::new(&self.base_path).join(format!("{}.xd", key));
        path.exists()
    }

    fn remove(&self, key: &str) -> Result<()> {
        let path = Path::new(&self.base_path).join(format!("{}.xd", key));
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

impl FractalStore {
    /// Create a new fractal storage instance
    pub fn new(config: FractalStorageConfig) -> Result<Self> {
        let disk_storage = if let Some(path) = &config.disk_path {
            Some(DiskBackend::new(path)?)
        } else {
            None
        };

        let hot_cache = TieredCache::new(config.hot_cache_limit, 64); // 64 chunk size

        Ok(Self {
            config,
            entries: RwLock::new(HashMap::new()),
            hot_cache,
            chunk_cache: Mutex::new(HashMap::new()),
            disk_storage,
        })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self> {
        Self::new(FractalStorageConfig::default())
    }



    /// Load from disk if available
    fn load_from_disk<T: VectorSerialize + 'static>(&self, key: &str) -> Result<Option<T>> {
        if let Some(disk) = &self.disk_storage {
            if disk.exists(key) {
                let data = disk.load(key)?;
                let value = T::from_xd(&data)
                    .map_err(|e| FractalStorageError::Serialization(e.to_string()))?;
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// Store to disk if configured
    fn store_to_disk(&self, key: &str, data: &[u8]) -> Result<()> {
        if let Some(disk) = &self.disk_storage {
            disk.store(key, data)?;
        }
        Ok(())
    }
}

impl FractalStorage for FractalStore {
    fn store_generator<G: Generator + Clone + Send + Sync + 'static>(&mut self, key: &str, _generator: G) -> Result<()> {
        // Simplified implementation - just store a placeholder for now
        let entry = StorageEntry::Generator("generator_placeholder".to_string());
        self.entries.write().insert(key.to_string(), entry);
        Ok(())
    }

    fn store_value<T: VectorSerialize + 'static>(&mut self, key: &str, value: T) -> Result<()> {
        // Serialize the value
        let data = value.to_xd()
            .map_err(|e| FractalStorageError::Serialization(e.to_string()))?;

        // Store to disk if configured
        self.store_to_disk(key, &data)?;

        // Store in memory as well for fast access
        let entry = StorageEntry::Materialized(VectorSerializeWrapper(Box::new(value)));
        self.entries.write().insert(key.to_string(), entry);

        Ok(())
    }

    fn get<T: VectorSerialize + 'static>(&self, key: &str) -> Result<Option<T>> {
        // For now, only support disk-based retrieval to avoid trait object issues
        // In a full implementation, we'd need a registry of type constructors
        self.load_from_disk(key)
    }

    fn exists(&self, key: &str) -> bool {
        let entries = self.entries.read();
        entries.contains_key(key) ||
        self.disk_storage.as_ref().map_or(false, |d| d.exists(key))
    }

    fn remove(&self, key: &str) -> Result<()> {
        let mut entries = self.entries.write();
        entries.remove(key);

        // Remove from disk if exists
        if let Some(disk) = &self.disk_storage {
            disk.remove(key)?;
        }

        Ok(())
    }

    fn stats(&self) -> StorageStats {
        let entries = self.entries.read();
        let mut stats = StorageStats::default();

        stats.total_keys = entries.len();

        for entry in entries.values() {
            match entry {
                StorageEntry::Materialized(_) => {
                    stats.hot_memory_keys += 1;
                    // Approximate memory usage
                    stats.memory_usage += 1024; // Rough estimate
                }
                StorageEntry::Generator(_) => {
                    stats.generator_keys += 1;
                    // Generators use minimal memory
                    stats.memory_usage += 128;
                }
            }
        }

        // Add disk stats if available
        if let Some(disk) = &self.disk_storage {
            // This would need actual disk usage calculation
            stats.disk_keys = entries.len(); // Simplified
        }

        stats
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use vexl_core::vector::{Vector, Dim1, VectorStorage};

    #[test]
    fn test_fractal_store_creation() {
        let store = FractalStore::default().unwrap();
        let stats = store.stats();
        assert_eq!(stats.total_keys, 0);
        assert_eq!(stats.memory_usage, 0);
    }

    #[test]
    fn test_store_and_retrieve_value() {
        let mut store = FractalStore::default().unwrap();
        let original = Vector::<i64, Dim1>::new_with_values([1, 2, 3, 4, 5], 1, VectorStorage::Dense);

        store.store_value("test_vector", original.clone()).unwrap();

        let retrieved: Option<Vector<i64, Dim1>> = store.get("test_vector").unwrap();

        assert!(retrieved.is_some());
        let retrieved_vec = retrieved.unwrap();

        // Compare elements
        for i in 0..original.len() {
            assert_eq!(original.get(i), retrieved_vec.get(i));
        }
    }

    #[test]
    fn test_key_existence() {
        let mut store = FractalStore::default().unwrap();
        let vector = Vector::<i64, Dim1>::new_with_values([42], 1, VectorStorage::Dense);

        assert!(!store.exists("test_key"));

        store.store_value("test_key", vector).unwrap();

        assert!(store.exists("test_key"));
    }

    #[test]
    fn test_remove_key() {
        let mut store = FractalStore::default().unwrap();
        let vector = Vector::<i64, Dim1>::new_with_values([42], 1, VectorStorage::Dense);

        store.store_value("test_key", vector).unwrap();
        assert!(store.exists("test_key"));

        store.remove("test_key").unwrap();
        assert!(!store.exists("test_key"));
    }

    #[test]
    fn test_storage_stats() {
        let mut store = FractalStore::default().unwrap();
        let vector = Vector::<i64, Dim1>::new_with_values([1, 2, 3], 1, VectorStorage::Dense);

        store.store_value("vec1", vector.clone()).unwrap();
        store.store_value("vec2", vector).unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.hot_memory_keys, 2);
        assert!(stats.memory_usage > 0);
    }
}
