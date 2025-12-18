//! Garbage collection for VEXL runtime

use std::collections::HashSet;
use std::sync::RwLock;

/// GC root registry
static GC_REGISTRY: std::sync::OnceLock<GcRegistry> = std::sync::OnceLock::new();

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
    GC_REGISTRY.get_or_init(|| GcRegistry::new(threshold));
}

/// Shutdown garbage collector
pub fn shutdown_gc() {
    // Note: OnceLock doesn't allow dropping, so this is a no-op for now
    // In a full implementation, we'd need a different approach for shutdown
}

/// Get reference to GC registry
fn gc_registry() -> &'static GcRegistry {
    GC_REGISTRY.get().expect("GC not initialized")
}

/// Register allocation with GC
pub fn gc_register(ptr: *mut u8, size: usize) {
    gc_registry().register(ptr, size);
}

/// Unregister allocation from GC
pub fn gc_unregister(ptr: *mut u8) {
    gc_registry().unregister(ptr);
}

/// Trigger garbage collection
pub fn gc_collect() {
    gc_registry().collect();
}

/// Check if GC should run
pub fn gc_should_collect() -> bool {
    gc_registry().should_collect()
}
