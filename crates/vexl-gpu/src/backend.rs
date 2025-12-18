use anyhow::Result;

pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &str;
    fn allocate(&self, size: usize) -> Result<GpuBuffer>;
    fn write_buffer(&self, buffer: &GpuBuffer, data: &[u8]) -> Result<()>;
    fn read_buffer(&self, buffer: &GpuBuffer, data: &mut [u8]) -> Result<()>;
    fn execute(&self, kernel: &ComputeKernel, args: &[GpuArg]) -> Result<()>;
    fn execute_kernel(&self, name: &str, source: &str, args: &[GpuArg], work_size: u32) -> Result<()> {
        let kernel = ComputeKernel {
            name: name.to_string(),
            source: source.to_string(),
            entry_point: "main".to_string(),
        };
        self.execute(&kernel, args)
    }
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

    fn write_buffer(&self, buffer: &GpuBuffer, data: &[u8]) -> Result<()> {
        if data.len() > buffer.size {
            return Err(anyhow::anyhow!("Data size {} exceeds buffer size {}", data.len(), buffer.size));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr as *mut u8, data.len());
        }
        Ok(())
    }

    fn read_buffer(&self, buffer: &GpuBuffer, data: &mut [u8]) -> Result<()> {
        if data.len() > buffer.size {
            return Err(anyhow::anyhow!("Requested size {} exceeds buffer size {}", data.len(), buffer.size));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.ptr as *const u8, data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    fn execute(&self, kernel: &ComputeKernel, args: &[GpuArg]) -> Result<()> {
        // Simple CPU kernel execution for vector operations
        if kernel.name == "vector_add" && args.len() == 3 {
            if let (GpuArg::Buffer(buf_a), GpuArg::Buffer(buf_b), GpuArg::Buffer(buf_c)) = (&args[0], &args[1], &args[2]) {
                // Read input data
                let data_a = self.read_back(buf_a, buf_a.size)?;
                let data_b = self.read_back(buf_b, buf_b.size)?;

                // Convert to f32 slices
                let slice_a: &[f32] = bytemuck::cast_slice(&data_a);
                let slice_b: &[f32] = bytemuck::cast_slice(&data_b);

                // Perform vector addition
                let result: Vec<f32> = slice_a.iter().zip(slice_b.iter()).map(|(a, b)| a + b).collect();

                // Write result back
                let result_bytes: &[u8] = bytemuck::cast_slice(&result);
                self.write_buffer(buf_c, result_bytes)?;

                return Ok(());
            }
        }

        // For unsupported kernels, log warning
        log::warn!("Executing kernel '{}' on CPU fallback (limited support)", kernel.name);
        Ok(())
    }

    fn read_back(&self, buffer: &GpuBuffer, size: usize) -> Result<Vec<u8>> {
        if size > buffer.size {
            return Err(anyhow::anyhow!("Requested size {} exceeds buffer size {}", size, buffer.size));
        }
        let mut vec = Vec::with_capacity(size);
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.ptr as *const u8, vec.as_mut_ptr(), size);
            vec.set_len(size);
        }
        Ok(vec)
    }
}
