use anyhow::Result;

pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &str;
    fn allocate(&self, size: usize) -> Result<GpuBuffer>;
    fn execute(&self, kernel: &ComputeKernel, args: &[GpuArg]) -> Result<()>;
    fn read_back(&self, buffer: &GpuBuffer, size: usize) -> Result<Vec<u8>>;
}

pub struct GpuContext {
    pub backend: Box<dyn GpuBackend>,
}

pub struct ComputeKernel {
    pub name: String,
    pub source: String,
    pub entry_point: String,
}

pub struct GpuBuffer {
    pub id: u64,
    pub size: usize,
    pub ptr: *mut (), // Opaque pointer to backend-specific handle
}

impl Clone for GpuBuffer {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            size: self.size,
            ptr: self.ptr,
        }
    }
}

pub enum GpuArg {
    Buffer(GpuBuffer),
    ScalarI64(i64),
    ScalarF64(f64),
}

// === CPU Fallback Backend ===

pub struct CpuFallbackBackend;

impl CpuFallbackBackend {
    pub fn new() -> Self { Self }
}

impl GpuBackend for CpuFallbackBackend {
    fn name(&self) -> &str { "CpuFallback" }
    
    fn allocate(&self, size: usize) -> Result<GpuBuffer> {
        // Validate allocation size
        if size == 0 {
            return Err(anyhow::anyhow!("Cannot allocate buffer of size 0"));
        }
        if size > 1 * 1024 * 1024 * 1024 { // 1GB limit for CPU fallback
            return Err(anyhow::anyhow!("Allocation size {} exceeds CPU fallback limit", size));
        }

        // Just allocate on heap
        let layout = std::alloc::Layout::from_size_align(size, 8)?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow::anyhow!("CPU allocation failed"));
        }
        Ok(GpuBuffer {
            id: 0,
            size,
            ptr: ptr as *mut (),
        })
    }
    
    fn execute(&self, _kernel: &ComputeKernel, _args: &[GpuArg]) -> Result<()> {
        // In a real fallback, this would interpret the kernel or run a CPU equivalent
        log::warn!("Executing kernel on CPU fallback (no-op)");
        Ok(())
    }
    
    fn read_back(&self, buffer: &GpuBuffer, size: usize) -> Result<Vec<u8>> {
        let mut vec = Vec::with_capacity(size);
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.ptr as *const u8, vec.as_mut_ptr(), size);
            vec.set_len(size);
        }
        Ok(vec)
    }
}
