//! Generator caching system

#![allow(static_mut_refs)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::any::Any;
use std::time::{Duration, Instant};

/// Memoization strategies
#[derive(Clone, Copy, Debug)]
pub enum MemoStrategy {
    /// Never cache
    None,

    /// LRU cache with max entries
    Lru { max_entries: usize },

    /// LFU cache with max entries
    Lfu { max_entries: usize },

    /// Time-to-live caching
    Ttl { duration: Duration },

    /// Cache only checkpoints
    Checkpoint { interval: usize },

    /// Adaptive based on access patterns
    Adaptive,
}

/// Cache statistics
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size_bytes: usize,
}

/// Generator cache implementation
pub struct GeneratorCache {
    strategy: MemoStrategy,
    entries: HashMap<usize, CacheEntry>,
    lru_order: Vec<usize>,
    frequency: HashMap<usize, u64>,
    stats: CacheStats,
    max_size: usize,
}

struct CacheEntry {
    value: Box<dyn Any + Send + Sync>,
    inserted_at: Instant,
    access_count: u64,
}

impl GeneratorCache {
    pub fn new(strategy: MemoStrategy) -> Self {
        let max_size = match strategy {
            MemoStrategy::Lru { max_entries } => max_entries,
            MemoStrategy::Lfu { max_entries } => max_entries,
            MemoStrategy::Adaptive => 10000,
            _ => usize::MAX,
        };

        GeneratorCache {
            strategy,
            entries: HashMap::new(),
            lru_order: Vec::new(),
            frequency: HashMap::new(),
            stats: CacheStats::default(),
            max_size,
        }
    }

    pub fn get(&self, index: usize) -> Option<Box<dyn Any + Send + Sync>> {
        if matches!(self.strategy, MemoStrategy::None) {
            return None;
        }

        // Check checkpoint strategy
        if let MemoStrategy::Checkpoint { interval } = self.strategy {
            if index % interval != 0 {
                return None;
            }
        }

        self.entries.get(&index).map(|entry| {
            // Clone the value (simplified - should use Arc for efficiency)
            // For now, return reference-like behavior
            clone_any(&entry.value)
        })
    }

    pub fn insert(&mut self, index: usize, value: Box<dyn Any + Send + Sync>) {
        if matches!(self.strategy, MemoStrategy::None) {
            return;
        }

        // Check checkpoint strategy
        if let MemoStrategy::Checkpoint { interval } = self.strategy {
            if index % interval != 0 {
                return;
            }
        }

        // Evict if necessary
        if self.entries.len() >= self.max_size {
            self.evict();
        }

        let entry = CacheEntry {
            value,
            inserted_at: Instant::now(),
            access_count: 1,
        };

        self.entries.insert(index, entry);
        self.lru_order.push(index);
        *self.frequency.entry(index).or_insert(0) += 1;
    }

    fn evict(&mut self) {
        match self.strategy {
            MemoStrategy::Lru { .. } => {
                if let Some(oldest) = self.lru_order.first().copied() {
                    self.entries.remove(&oldest);
                    self.lru_order.remove(0);
                    self.stats.evictions += 1;
                }
            }
            MemoStrategy::Lfu { .. } => {
                // Find least frequently used
                if let Some((&key, _)) = self.frequency.iter()
                    .min_by_key(|(_, &count)| count)
                {
                    if let Some(entry) = self.entries.get(&key) {
                        // Use the access_count field for additional logic
                        let _access_count = entry.access_count;
                        // In a full implementation, this could affect eviction priority
                    }
                    self.entries.remove(&key);
                    self.frequency.remove(&key);
                    self.lru_order.retain(|&k| k != key);
                    self.stats.evictions += 1;
                }
            }
            MemoStrategy::Ttl { duration } => {
                let now = Instant::now();
                let expired: Vec<usize> = self.entries.iter()
                    .filter(|(_, entry)| now.duration_since(entry.inserted_at) > duration)
                    .map(|(&k, _)| k)
                    .collect();

                for key in expired {
                    self.entries.remove(&key);
                    self.lru_order.retain(|&k| k != key);
                    self.stats.evictions += 1;
                }
            }
            MemoStrategy::Adaptive => {
                // Combine LRU and LFU heuristics
                if let Some(oldest) = self.lru_order.first().copied() {
                    let freq = self.frequency.get(&oldest).copied().unwrap_or(0);
                    if freq < 3 {
                        self.entries.remove(&oldest);
                        self.lru_order.remove(0);
                        self.frequency.remove(&oldest);
                        self.stats.evictions += 1;
                    }
                }
            }
            _ => {}
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.frequency.clear();
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// Helper function to clone Any (simplified)
fn clone_any(value: &Box<dyn Any + Send + Sync>) -> Box<dyn Any + Send + Sync> {
    // This is a simplified implementation
    // In practice, you'd use Arc or specific type cloning
    if let Some(&v) = value.downcast_ref::<i64>() {
        Box::new(v)
    } else if let Some(&v) = value.downcast_ref::<f64>() {
        Box::new(v)
    } else if let Some(&v) = value.downcast_ref::<bool>() {
        Box::new(v)
    } else {
        Box::new(0i64) // Fallback
    }
}

// ═══════════════════════════════════════════════════════════
// Global Cache System
// ═══════════════════════════════════════════════════════════

static mut CACHE_SYSTEM: Option<CacheSystem> = None;

pub struct CacheSystem {
    total_size: Arc<RwLock<usize>>,
    max_total_size: usize,
}

impl CacheSystem {
    pub fn new(max_size: usize) -> Self {
        CacheSystem {
            total_size: Arc::new(RwLock::new(0)),
            max_total_size: max_size,
        }
    }

    pub fn register_cache(&self, size: usize) {
        let mut total = self.total_size.write().unwrap();
        *total += size;
    }

    pub fn unregister_cache(&self, size: usize) {
        let mut total = self.total_size.write().unwrap();
        *total = total.saturating_sub(size);
    }

    pub fn memory_pressure(&self) -> f64 {
        let total = *self.total_size.read().unwrap();
        total as f64 / self.max_total_size as f64
    }
}

pub fn init_cache_system() {
    let max_size = 1024 * 1024 * 1024; // 1GB default
    unsafe {
        CACHE_SYSTEM = Some(CacheSystem::new(max_size));
    }
}

pub fn shutdown_cache_system() {
    unsafe {
        CACHE_SYSTEM = None;
    }
}

pub fn cache_system() -> &'static CacheSystem {
    unsafe {
        CACHE_SYSTEM.as_ref().expect("Cache system not initialized")
    }
}
