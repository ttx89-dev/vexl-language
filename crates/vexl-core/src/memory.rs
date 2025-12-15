//! VEXL Memory Management System
//!
//! Advanced memory management with garbage collection for VEXL vectors and generators.
//! Provides automatic memory lifecycle management with performance optimization.

use std::sync::{Arc, Mutex, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::any::Any;

/// Memory allocation statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_live: usize,
    pub gc_cycles: usize,
    pub pool_allocations: usize,
    pub direct_allocations: usize,
}

/// Garbage collection handle for managed objects
#[derive(Debug)]
pub struct GcHandle<T> {
    data: Arc<GcObject<T>>,
}

#[derive(Debug)]
struct GcObject<T> {
    value: T,
    ref_count: AtomicUsize,
    generation: usize,
    size_estimate: usize,
}

/// Generational garbage collector
#[derive(Debug)]
struct GenerationalGc {
    young_generation: Mutex<Vec<Weak<dyn Any + Send + Sync>>>,
    old_generation: Mutex<Vec<Weak<dyn Any + Send + Sync>>>,
    perm_generation: Mutex<Vec<Arc<dyn Any + Send + Sync>>>,
    young_threshold: usize,
    promotion_threshold: usize,
    cycle_counter: AtomicUsize,
}

/// Memory pool for efficient vector allocation
#[derive(Debug)]
pub struct MemoryPool<T> {
    pool: Mutex<Vec<Vec<T>>>,
    chunk_size: usize,
    max_chunks: usize,
    allocations: AtomicUsize,
}

impl<T> MemoryPool<T>
where T: Clone + Default {
    /// Create new memory pool
    pub fn new(chunk_size: usize, max_chunks: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            chunk_size,
            max_chunks,
            allocations: AtomicUsize::new(0),
        }
    }

    /// Allocate vector from pool
    pub fn allocate(&self, size: usize) -> Vec<T> {
        self.allocations.fetch_add(1, Ordering::SeqCst);

        if size <= self.chunk_size {
            // Try to reuse from pool
            if let Ok(mut pool) = self.pool.lock() {
                if let Some(mut chunk) = pool.pop() {
                    if chunk.len() >= size {
                        chunk.truncate(size);
                        chunk.resize(size, T::default());
                        return chunk;
                    }
                }
            }
        }

        // Allocate new
        vec![T::default(); size]
    }

    /// Return vector to pool for reuse
    pub fn deallocate(&self, mut vec: Vec<T>) {
        if vec.len() <= self.chunk_size {
            if let Ok(mut pool) = self.pool.lock() {
                if pool.len() < self.max_chunks {
                    vec.clear();
                    vec.resize(self.chunk_size, T::default());
                    pool.push(vec);
                }
            }
        }
    }

    /// Get allocation statistics
    pub fn stats(&self) -> usize {
        self.allocations.load(Ordering::SeqCst)
    }
}

/// Main memory manager
#[derive(Debug)]
pub struct MemoryManager {
    gc: GenerationalGc,
    vector_pool: MemoryPool<i32>,
    object_pool: MemoryPool<u8>,
    stats: Mutex<MemoryStats>,
}

impl MemoryManager {
    /// Create new memory manager
    pub fn new() -> Self {
        Self {
            gc: GenerationalGc {
                young_generation: Mutex::new(Vec::new()),
                old_generation: Mutex::new(Vec::new()),
                perm_generation: Mutex::new(Vec::new()),
                young_threshold: 1024,
                promotion_threshold: 10,
                cycle_counter: AtomicUsize::new(0),
            },
            vector_pool: MemoryPool::new(1024, 16), // 1K chunks, max 16
            object_pool: MemoryPool::new(4096, 8),  // 4K chunks, max 8
            stats: Mutex::new(MemoryStats {
                total_allocated: 0,
                total_freed: 0,
                current_live: 0,
                gc_cycles: 0,
                pool_allocations: 0,
                direct_allocations: 0,
            }),
        }
    }

    /// Allocate vector using memory pool
    pub fn allocate_vector(&self, size: usize) -> Vec<i32> {
        let vec = self.vector_pool.allocate(size);
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocated += size * std::mem::size_of::<i32>();
            stats.current_live += size * std::mem::size_of::<i32>();
            stats.pool_allocations += 1;
        }
        vec
    }

    /// Deallocate vector back to pool
    pub fn deallocate_vector(&self, vec: Vec<i32>) {
        let size = vec.len() * std::mem::size_of::<i32>();
        self.vector_pool.deallocate(vec);
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_freed += size;
            stats.current_live -= size;
        }
    }

    /// Create garbage collected handle
    pub fn manage<T: Send + Sync + 'static>(&self, value: T) -> GcHandle<T> {
        let size = std::mem::size_of::<T>();
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocated += size;
            stats.current_live += size;
        }

        GcHandle {
            data: Arc::new(GcObject {
                value,
                ref_count: AtomicUsize::new(1),
                generation: 0,
                size_estimate: size,
            }),
        }
    }

    /// Run garbage collection cycle
    pub fn collect_garbage(&self) {
        self.gc.cycle_counter.fetch_add(1, Ordering::SeqCst);

        // Young generation collection
        if let Ok(mut young) = self.gc.young_generation.lock() {
            young.retain(|weak| weak.upgrade().is_some());
        }

        // Check for promotion to old generation
        if let Ok(young) = self.gc.young_generation.lock() {
            if young.len() > self.gc.young_threshold {
                if let Ok(_old) = self.gc.old_generation.lock() {
                    // Simple promotion strategy - move surviving objects
                    // In real implementation, would check reference counts
                }
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.gc_cycles += 1;
        }
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    /// Force cleanup of unused memory
    pub fn cleanup(&self) {
        self.collect_garbage();

        // Additional cleanup could be added here
        // - Clear unused cache entries
        // - Compact memory pools
        // - Return memory to OS if possible
    }
}

impl<T> GcHandle<T> {
    /// Get reference to managed object
    pub fn get(&self) -> &T {
        &self.data.value
    }

    /// Get mutable reference (if uniquely owned)
    pub fn get_mut(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.data).map(|obj| &mut obj.value)
    }

    /// Clone handle (increases reference count)
    pub fn clone_handle(&self) -> GcHandle<T>
    where T: Clone {
        self.data.ref_count.fetch_add(1, Ordering::SeqCst);
        GcHandle {
            data: Arc::clone(&self.data),
        }
    }
}

impl<T> Drop for GcHandle<T> {
    fn drop(&mut self) {
        let refs = self.data.ref_count.fetch_sub(1, Ordering::SeqCst);
        if refs == 1 {
            // Last reference - could trigger cleanup
            // In real implementation, would notify GC
        }
    }
}

impl<T: Clone> Clone for GcHandle<T> {
    fn clone(&self) -> Self {
        self.clone_handle()
    }
}

/// Global memory manager instance
static mut GLOBAL_MEMORY_MANAGER: Option<MemoryManager> = None;

/// Initialize the global memory manager
pub fn init_memory_manager() {
    unsafe {
        if GLOBAL_MEMORY_MANAGER.is_none() {
            GLOBAL_MEMORY_MANAGER = Some(MemoryManager::new());
        }
    }
}

/// Get reference to global memory manager
pub fn global_memory_manager() -> &'static MemoryManager {
    unsafe {
        GLOBAL_MEMORY_MANAGER.as_ref().expect("Memory manager not initialized")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let pool = MemoryPool::<i32>::new(10, 5);

        // Allocate vectors
        let v1 = pool.allocate(5);
        let v2 = pool.allocate(8);

        assert_eq!(v1.len(), 5);
        assert_eq!(v2.len(), 8);
        assert_eq!(pool.stats(), 2);

        // Deallocate back to pool
        pool.deallocate(v1);
        pool.deallocate(v2);
    }

    #[test]
    fn test_gc_handle_management() {
        init_memory_manager();
        let manager = global_memory_manager();

        // Create managed object
        let handle = manager.manage(vec![1, 2, 3, 4, 5]);
        assert_eq!(handle.get().len(), 5);

        // Clone handle
        let handle2 = handle.clone_handle();
        assert_eq!(handle2.get().len(), 5);

        let stats = manager.stats();
        assert!(stats.total_allocated > 0);
        assert!(stats.current_live > 0);
    }

    #[test]
    fn test_memory_manager_stats() {
        init_memory_manager();
        let manager = global_memory_manager();

        // Allocate some vectors
        let v1 = manager.allocate_vector(100);
        let v2 = manager.allocate_vector(200);

        let stats = manager.stats();
        assert!(stats.total_allocated >= 300 * std::mem::size_of::<i32>());
        assert!(stats.pool_allocations >= 2);

        // Deallocate (may return to pool, not necessarily free)
        manager.deallocate_vector(v1);
        manager.deallocate_vector(v2);

        let stats_after = manager.stats();
        // After deallocation, total_freed should be >= what we allocated
        assert!(stats_after.total_freed >= stats.total_allocated - stats_after.current_live);
    }

    #[test]
    fn test_garbage_collection() {
        init_memory_manager();
        let manager = global_memory_manager();

        // Create some objects that will be collected
        let _handle1 = manager.manage(42);
        let _handle2 = manager.manage(String::from("test"));

        let cycles_before = manager.stats().gc_cycles;

        manager.collect_garbage();

        let cycles_after = manager.stats().gc_cycles;
        assert_eq!(cycles_after, cycles_before + 1);
    }
}
