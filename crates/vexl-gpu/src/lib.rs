pub mod backend;
pub mod cuda;
pub mod opencl;
pub mod vulkan;
pub mod vector_ops;
pub mod hardware_validation;
pub mod performance_benchmarks;
pub mod safety_monitor;

pub use backend::{GpuBackend, GpuContext, ComputeKernel, GpuBuffer, GpuArg};
pub use vector_ops::GpuVectorOps;
pub use hardware_validation::run_hardware_validation;
pub use performance_benchmarks::run_performance_benchmarks;
pub use safety_monitor::run_safety_validation;

/// Initialize the best available GPU backend
pub fn init_best_backend() -> Box<dyn GpuBackend> {
    // Priority: CUDA > Vulkan > OpenCL

    #[cfg(feature = "cuda")]
    {
        log::info!("Attempting to initialize CUDA backend");
        if let Ok(backend) = cuda::CudaBackend::try_new() {
            log::info!("Successfully initialized CUDA backend");
            return Box::new(backend);
        }
        log::warn!("CUDA backend initialization failed, trying Vulkan");
    }

    #[cfg(feature = "vulkan")]
    {
        log::info!("Attempting to initialize Vulkan backend");
        if let Ok(backend) = vulkan::VulkanBackend::try_new() {
            log::info!("Successfully initialized Vulkan backend");
            return Box::new(backend);
        }
        log::warn!("Vulkan backend initialization failed, trying OpenCL");
    }

    #[cfg(feature = "opencl")]
    {
        log::info!("Attempting to initialize OpenCL backend");
        if let Ok(backend) = opencl::OpenCLBackend::try_new() {
            log::info!("Successfully initialized OpenCL backend");
            return Box::new(backend);
        }
        log::warn!("OpenCL backend initialization failed");
    }

    log::warn!("No GPU backend available, falling back to CPU simulation");
    Box::new(backend::CpuFallbackBackend::new())
}
