use vexl_gpu::{init_best_backend, GpuBackend, GpuBuffer, ComputeKernel, GpuArg};
use std::sync::Arc;
use parking_lot::Mutex;

pub struct VPU {
    backend: Box<dyn GpuBackend>,
}

impl VPU {
    pub fn new() -> Self {
        log::info!("Initializing VPU...");
        let backend = init_best_backend();
        log::info!("VPU initialized with backend: {}", backend.name());
        
        Self {
            backend,
        }
    }
    
    pub fn name(&self) -> &str {
        self.backend.name()
    }
    
    // Wrapper methods for runtime to use
    pub fn allocate(&self, size: usize) -> anyhow::Result<GpuBuffer> {
        self.backend.allocate(size)
    }
    
    pub fn execute(&self, kernel_name: &str, source: &str, args: &[GpuArg]) -> anyhow::Result<()> {
        let kernel = ComputeKernel {
            name: kernel_name.to_string(),
            source: source.to_string(),
            entry_point: "main".to_string(),
        };
        self.backend.execute(&kernel, args)
    }
}

// Global VPU instance (lazy initialized could be better but for now simple)
lazy_static::lazy_static! {
    pub static ref GLOBAL_VPU: Arc<Mutex<VPU>> = Arc::new(Mutex::new(VPU::new()));
}
