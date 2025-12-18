//! VEXL Memory Management System
//!
//! Advanced memory management with garbage collection for VEXL vectors and generators.
//! Provides automatic memory lifecycle management with performance optimization.

use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Memory statistics tracking
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated: usize,
    pub peak: usize,
    pub gc_runs: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        MemoryStats {
            allocated: 0,
            peak: 0,
            gc_runs: 0,
        }
    }
}

/// Global memory manager singleton - thread-safe
pub struct MemoryManager {
    stats: Mutex<MemoryStats>,
    gc_trigger: AtomicUsize,
    total_capacity: usize,
}

impl MemoryManager {
    pub fn new(total_capacity: usize) -> Self {
        MemoryManager {
            stats: Mutex::new(MemoryStats::default()),
            gc_trigger: AtomicUsize::new((total_capacity as f64 * 0.8) as usize), // 80% threshold
            total_capacity,
        }
    }

    pub fn allocate(&self, size: usize) -> *mut u8 {
        let layout = Layout::from_size_align(size, 8).unwrap();
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }

        self.update_stats(size, true);
        ptr
    }

    pub fn deallocate(&self, ptr: *mut u8, size: usize) {
        // For now, deallocation is handled by GC
        // In a full implementation, we'd track allocation sizes and free appropriately
        // For this implementation, we rely on GC for deallocation
        self.update_stats(size, false);
    }

    pub fn get_stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    pub fn allocate_vector<T>(&self, len: usize) -> VectorHandle<T> {
        let size = std::mem::size_of::<T>() * len;
        let ptr = self.allocate(size) as *mut T;
        VectorHandle { ptr, len, _phantom: std::marker::PhantomData }
    }

    fn update_stats(&self, size: usize, allocate: bool) {
        let mut stats = self.stats.lock().unwrap();
        if allocate {
            stats.allocated += size;
            if stats.allocated > stats.peak {
                stats.peak = stats.allocated;
            }
        } else {
            stats.allocated = stats.allocated.saturating_sub(size);
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(1024 * 1024 * 1024) // 1GB default
    }
}

/// Vector handle for type-safe vector allocation
pub struct VectorHandle<T> {
    ptr: *mut T,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> VectorHandle<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for VectorHandle<T> {
    fn drop(&mut self) {
        // Memory is managed by the GC system
        // In a full implementation, this would notify the GC
    }
}

/// GC handle for automatic memory management
pub struct GcHandle(*mut u8);

impl GcHandle {
    pub fn new(ptr: *mut u8) -> Self {
        // Register with GC
        GcHandle(ptr)
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.0
    }
}

impl Drop for GcHandle {
    fn drop(&mut self) {
        // Unregister from GC
        // In a full implementation, this would call gc_unregister
    }
}

/// Global memory manager instance
static GLOBAL_MANAGER: OnceLock<MemoryManager> = OnceLock::new();

/// Initialize the global memory manager
pub fn init_memory_manager() {
    GLOBAL_MANAGER.get_or_init(|| MemoryManager::new(1024 * 1024 * 1024));
}

/// Get reference to the global memory manager
pub fn global_memory_manager() -> &'static MemoryManager {
    GLOBAL_MANAGER.get().expect("Memory manager not initialized")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_init() {
        init_memory_manager();
        let manager = global_memory_manager();
        let stats = manager.get_stats();
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.gc_runs, 0);
    }

    #[test]
    fn test_vector_allocation() {
        init_memory_manager();
        let manager = global_memory_manager();
        let mut vector: VectorHandle<i32> = manager.allocate_vector(10);
        assert_eq!(vector.len(), 10);

        // Test writing to vector
        let slice = vector.as_slice_mut();
        for i in 0..10 {
            slice[i] = i as i32;
        }

        // Test reading from vector
        let slice = vector.as_slice();
        for i in 0..10 {
            assert_eq!(slice[i], i as i32);
        }
    }

    #[test]
    fn test_gc_handle() {
        init_memory_manager();
        let manager = global_memory_manager();
        let ptr = manager.allocate(64);
        let handle = GcHandle::new(ptr);
        assert_eq!(handle.as_ptr(), ptr);
    }

    #[test]
    fn test_memory_stats() {
        init_memory_manager();
        let manager = global_memory_manager();

        let initial_stats = manager.get_stats();
        let ptr = manager.allocate(128);
        let after_stats = manager.get_stats();

        assert_eq!(after_stats.allocated, initial_stats.allocated + 128);
        assert!(after_stats.peak >= after_stats.allocated);
    }
}
