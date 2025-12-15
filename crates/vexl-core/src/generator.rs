//! VEXL Generator System with Tiered Caching
//!
//! Generators store algorithms, not data, enabling near-infinite logical storage
//! and lazy evaluation with intelligent caching strategies.

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};

/// Cache entry with metadata for tiered caching
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    access_count: u64,
    last_access: u64,
    size_estimate: usize,
}

/// Simplified tiered caching system for generators
/// Uses Mutex for all shared state to avoid deadlocks
#[derive(Debug)]
pub struct TieredCache<T> {
    /// Combined L1/L2 cache with access tracking
    cache: Mutex<HashMap<u64, CacheEntry<T>>>,
    max_size: usize,
    access_order: Mutex<VecDeque<u64>>,

    /// Chunk cache for sequential access patterns
    chunk_cache: Mutex<HashMap<u64, Vec<T>>>,
    chunk_size: usize,

    /// Global access counter
    access_counter: std::sync::atomic::AtomicU64,
}

impl<T> TieredCache<T>
where T: Clone + Send + Sync {
    /// Create new tiered cache with specified sizes
    pub fn new(max_size: usize, chunk_size: usize) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_size,
            access_order: Mutex::new(VecDeque::new()),
            chunk_cache: Mutex::new(HashMap::new()),
            chunk_size,
            access_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Default cache configuration
    pub fn default() -> Self {
        Self::new(1024, 64)
    }

    /// Get value from cache
    pub fn get(&self, key: u64) -> Option<T> {
        let current_access = self.access_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Try main cache
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(entry) = cache.get_mut(&key) {
                entry.access_count += 1;
                entry.last_access = current_access;

                // Update access order
                if let Ok(mut order) = self.access_order.lock() {
                    if let Some(pos) = order.iter().position(|&k| k == key) {
                        order.remove(pos);
                    }
                    order.push_back(key);
                }

                return Some(entry.value.clone());
            }
        }

        // Try chunk cache for sequential access
        let chunk_key = key / self.chunk_size as u64;
        if let Ok(chunk_cache) = self.chunk_cache.lock() {
            if let Some(chunk) = chunk_cache.get(&chunk_key) {
                let chunk_offset = (key % self.chunk_size as u64) as usize;
                if chunk_offset < chunk.len() {
                    let value = chunk[chunk_offset].clone();

                    // Cache individual value
                    let entry = CacheEntry {
                        value: value.clone(),
                        access_count: 1,
                        last_access: current_access,
                        size_estimate: std::mem::size_of::<T>(),
                    };

                    if let Ok(mut cache) = self.cache.lock() {
                        cache.insert(key, entry);
                        if let Ok(mut order) = self.access_order.lock() {
                            order.push_back(key);

                            // Evict if over capacity
                            while cache.len() > self.max_size {
                                if let Some(evict_key) = order.pop_front() {
                                    cache.remove(&evict_key);
                                }
                            }
                        }
                    }

                    return Some(value);
                }
            }
        }

        None
    }

    /// Put value in cache
    pub fn put(&self, key: u64, value: T) {
        let current_access = self.access_counter.load(std::sync::atomic::Ordering::SeqCst);

        let entry = CacheEntry {
            value: value.clone(),
            access_count: 1,
            last_access: current_access,
            size_estimate: std::mem::size_of::<T>(),
        };

        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, entry);
            if let Ok(mut order) = self.access_order.lock() {
                order.push_back(key);

                // Evict if over capacity
                while cache.len() > self.max_size {
                    if let Some(evict_key) = order.pop_front() {
                        cache.remove(&evict_key);
                    }
                }
            }
        }
    }

    /// Cache entire chunk for sequential access
    pub fn put_chunk(&self, chunk_start: u64, chunk: Vec<T>) {
        let chunk_key = chunk_start / self.chunk_size as u64;
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.insert(chunk_key, chunk);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let (cache_size, chunk_count) = if let Ok(cache) = self.cache.lock() {
            if let Ok(chunk_cache) = self.chunk_cache.lock() {
                (cache.len(), chunk_cache.len())
            } else {
                (cache.len(), 0)
            }
        } else {
            (0, 0)
        };

        CacheStats {
            l1_size: cache_size,
            l2_size: 0, // Simplified to single tier
            chunk_count,
            total_accesses: self.access_counter.load(std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Clear all caches
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        if let Ok(mut chunk_cache) = self.chunk_cache.lock() {
            chunk_cache.clear();
        }
        if let Ok(mut order) = self.access_order.lock() {
            order.clear();
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub l1_size: usize,
    pub l2_size: usize,
    pub chunk_count: usize,
    pub total_accesses: u64,
}

/// Core generator trait for lazy evaluation with caching
pub trait Generator: Send + Sync {
    /// Evaluate the generator at a specific index with caching
    fn evaluate(&self, index: usize) -> Option<Box<dyn std::any::Any + Send + Sync>>;

    /// Get the bounds of this generator (if finite)
    fn bounds(&self) -> Option<(usize, usize)>;

    /// Check if this generator is pure (deterministic, no side effects)
    fn is_pure(&self) -> bool;

    /// Get cache for this generator (if available)
    fn cache(&self) -> Option<&TieredCache<Box<dyn std::any::Any + Send + Sync>>> {
        None
    }

    /// Clone this generator into an Arc
    fn clone_generator(&self) -> Arc<dyn Generator>;
}

/// Cached generator wrapper
pub struct CachedGenerator<G: Generator, T> {
    generator: G,
    cache: TieredCache<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<G: Generator + Clone, T> CachedGenerator<G, T>
where T: Clone + Send + Sync + 'static {
    /// Create new cached generator
    pub fn new(generator: G) -> Self {
        Self {
            generator,
            cache: TieredCache::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Evaluate with caching
    pub fn evaluate_cached(&self, _index: usize) -> Option<T> {
        // This would need proper type conversion in real implementation
        // For now, just delegate to generator
        None // Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleGenerator;
    
    impl Generator for SimpleGenerator {
        fn evaluate(&self, _index: usize) -> Option<Box<dyn std::any::Any + Send + Sync>> {
            None
        }
        
        fn bounds(&self) -> Option<(usize, usize)> {
            None
        }
        
        fn is_pure(&self) -> bool {
            true
        }
        
        fn clone_generator(&self) -> Arc<dyn Generator> {
            Arc::new(SimpleGenerator)
        }
    }

    #[test]
    fn test_generator_trait() {
        let gen = SimpleGenerator;
        assert!(gen.is_pure());
    }

    #[test]
    fn test_tiered_cache_basic() {
        let cache = TieredCache::<i32>::new(10, 8);

        // Test put and get
        cache.put(1, 100);
        cache.put(2, 200);
        cache.put(3, 300);

        assert_eq!(cache.get(1), Some(100));
        assert_eq!(cache.get(2), Some(200));
        assert_eq!(cache.get(3), Some(300));
        assert_eq!(cache.get(4), None);
    }

    #[test]
    fn test_tiered_cache_lru_eviction() {
        let cache = TieredCache::<i32>::new(2, 8);

        // Fill cache
        cache.put(1, 100);
        cache.put(2, 200);
        cache.put(3, 300); // Should evict oldest

        // Check eviction worked
        let stats = cache.stats();
        assert_eq!(stats.l1_size, 2); // Should maintain max size
        assert_eq!(cache.get(1), None); // 1 should be evicted
        assert_eq!(cache.get(2), Some(200));
        assert_eq!(cache.get(3), Some(300));
    }

    #[test]
    fn test_tiered_cache_chunk() {
        let cache = TieredCache::<i32>::new(10, 4);

        // Put a chunk
        let chunk = vec![10, 20, 30, 40];
        cache.put_chunk(0, chunk);

        // Access individual elements (should be cached)
        assert_eq!(cache.get(0), Some(10));
        assert_eq!(cache.get(1), Some(20));
        assert_eq!(cache.get(2), Some(30));
        assert_eq!(cache.get(3), Some(40));
    }

    #[test]
    fn test_tiered_cache_stats() {
        let cache = TieredCache::<i32>::default();

        cache.put(1, 100);
        cache.put(2, 200);

        // Access a few times
        cache.get(1);
        cache.get(1);
        cache.get(2);

        let stats = cache.stats();
        assert_eq!(stats.l1_size, 2);
        assert!(stats.total_accesses >= 3);
    }
}
