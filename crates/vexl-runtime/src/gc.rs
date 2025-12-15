//! Garbage collection for VEXL runtime

#![allow(static_mut_refs)]

use std::collections::HashSet;
use std::sync::RwLock;

/// GC root registry
static mut GC_REGISTRY: Option<GcRegistry> = None;

/// Garbage collector state
pub struct GcRegistry {
    /// All allocated objects
    allocations: RwLock<HashSet<usize>>,

    /// Allocation sizes
    sizes: RwLock<std::collections::HashMap<usize, usize>>,

    /// Total allocated bytes
    total_bytes: std::sync::atomic::AtomicUsize,

    /// GC threshold
    threshold: usize,
}

impl GcRegistry {
    pub fn new(threshold: usize) -> Self {
        GcRegistry {
            allocations: RwLock::new(HashSet::new()),
            sizes: RwLock::new(std::collections::HashMap::new()),
            total_bytes: std::sync::atomic::AtomicUsize::new(0),
            threshold,
        }
    }

    pub fn register(&self, ptr: *mut u8, size: usize) {
        let addr = ptr as usize;
        self.allocations.write().unwrap().insert(addr);
        self.sizes.write().unwrap().insert(addr, size);
        self.total_bytes.fetch_add(size, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn unregister(&self, ptr: *mut u8) {
        let addr = ptr as usize;
        self.allocations.write().unwrap().remove(&addr);

        if let Some(size) = self.sizes.write().unwrap().remove(&addr) {
            self.total_bytes.fetch_sub(size, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn should_collect(&self) -> bool {
        self.total_bytes() > self.threshold
    }

    pub fn collect(&self) {
        // Simple collection: release unreferenced allocations
        // In a real implementation, this would trace roots

        // For now, just log
        let total = self.total_bytes();
        let count = self.allocations.read().unwrap().len();
        eprintln!("GC: {} allocations, {} bytes", count, total);
    }
}

/// Initialize garbage collector
pub fn init_gc() {
    let threshold = 256 * 1024 * 1024; // 256MB default
    unsafe {
        GC_REGISTRY = Some(GcRegistry::new(threshold));
    }
}

/// Shutdown garbage collector
pub fn shutdown_gc() {
    unsafe {
        GC_REGISTRY = None;
    }
}

/// Register allocation with GC
pub fn gc_register(ptr: *mut u8, size: usize) {
    unsafe {
        if let Some(ref gc) = GC_REGISTRY {
            gc.register(ptr, size);
        }
    }
}

/// Unregister allocation from GC
pub fn gc_unregister(ptr: *mut u8) {
    unsafe {
        if let Some(ref gc) = GC_REGISTRY {
            gc.unregister(ptr);
        }
    }
}

/// Trigger garbage collection
pub fn gc_collect() {
    unsafe {
        if let Some(ref gc) = GC_REGISTRY {
            gc.collect();
        }
    }
}

/// Check if GC should run
pub fn gc_should_collect() -> bool {
    unsafe {
        GC_REGISTRY.as_ref()
            .map(|gc| gc.should_collect())
            .unwrap_or(false)
    }
}
