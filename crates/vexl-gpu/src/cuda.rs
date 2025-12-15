#[cfg(feature = "cuda")]
use crate::backend::{GpuBackend, ComputeKernel, GpuArg, GpuBuffer};
#[cfg(feature = "cuda")]
use anyhow::{Result, anyhow};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    compiled_kernels: HashMap<String, cudarc::driver::CudaFunction>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn try_new() -> Result<Self> {
        unsafe {
            let device = CudaDevice::new(0)?;
            let stream = device.fork_default_stream()?;

            Ok(Self {
                device,
                stream,
                compiled_kernels: HashMap::new(),
            })
        }
    }

    fn compile_cuda_kernel(&mut self, kernel: &ComputeKernel) -> Result<()> {
        if self.compiled_kernels.contains_key(&kernel.name) {
            return Ok(()); // Already compiled
        }

        // Convert GLSL to CUDA kernel
        let cuda_source = Self::glsl_to_cuda(&kernel.source)?;

        // Compile with NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(cuda_source)?;

        // Load the PTX and get the function
        self.device.load_ptx(ptx, &[kernel.entry_point.as_str()])?;
        let function = self.device.get_func(kernel.entry_point.as_str())?;

        self.compiled_kernels.insert(kernel.name.clone(), function);

        Ok(())
    }

    fn glsl_to_cuda(glsl_source: &str) -> Result<String> {
        // Basic GLSL to CUDA conversion
        // This is a simplified converter - real implementation would be more comprehensive
        let mut cuda_code = String::from("#include <cuda_runtime.h>\n\n");

        // Convert GLSL main function to CUDA global function
        let main_start = glsl_source.find("void main()").ok_or_else(|| anyhow!("No main function found"))?;
        let main_body = &glsl_source[main_start..];

        cuda_code.push_str("extern \"C\" __global__ void main_kernel(");
        cuda_code.push_str("float* inputA, float* inputB, float* output");
        cuda_code.push_str(") {\n");

        // Convert GLSL indexing to CUDA
        let converted_body = main_body
            .replace("gl_GlobalInvocationID.x", "blockIdx.x * blockDim.x + threadIdx.x")
            .replace("inputA.data[", "inputA[")
            .replace("inputB.data[", "inputB[")
            .replace("output.data[", "output[");

        cuda_code.push_str(&converted_body[12..]); // Skip "void main()"
        cuda_code.push_str("\n}\n");

        Ok(cuda_code)
    }
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    fn name(&self) -> &str { "CUDA" }

    fn allocate(&self, size: usize) -> Result<GpuBuffer> {
        unsafe {
            let device_ptr = self.device.alloc(size)?;

            Ok(GpuBuffer {
                id: device_ptr.as_raw() as u64,
                size,
                ptr: device_ptr.as_raw() as *mut (),
            })
        }
    }

    fn execute(&self, kernel: &ComputeKernel, args: &[GpuArg]) -> Result<()> {
        // For CUDA, we need mutable access to compile kernels
        let mut_self = unsafe { &mut *(self as *const Self as *mut Self) };
        mut_self.compile_cuda_kernel(kernel)?;

        let cuda_function = mut_self.compiled_kernels.get(&kernel.name)
            .ok_or_else(|| anyhow!("Kernel not compiled"))?;

        // Prepare kernel arguments
        let mut kernel_args = Vec::new();
        for arg in args {
            match arg {
                GpuArg::Buffer(buffer) => {
                    let device_ptr = unsafe { DevicePtr::from_raw(buffer.id as *mut std::ffi::c_void) };
                    kernel_args.push(device_ptr);
                },
                GpuArg::ScalarF64(val) => {
                    kernel_args.push(*val as f32); // Convert to f32 for CUDA
                },
                _ => return Err(anyhow!("Unsupported argument type")),
            }
        }

        // Launch configuration
        let num_elements = args.iter().find_map(|arg| {
            if let GpuArg::Buffer(buffer) = arg {
                Some(buffer.size / std::mem::size_of::<f32>())
            } else {
                None
            }
        }).unwrap_or(1024);

        let block_size = 256;
        let grid_size = (num_elements + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            cuda_function.launch_on_stream(&mut_self.stream, config, &kernel_args)?;
        }

        Ok(())
    }

    fn read_back(&self, buffer: &GpuBuffer, size: usize) -> Result<Vec<u8>> {
        unsafe {
            let device_ptr = DevicePtr::from_raw(buffer.id as *mut std::ffi::c_void);
            let mut host_data = vec![0u8; size];

            self.device.dtoh_sync_copy_into(&device_ptr.slice(0..size), &mut host_data)?;

            Ok(host_data)
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaBackend {
    fn drop(&mut self) {
        // CUDA resources are automatically cleaned up by cudarc
    }
}

#[cfg(not(feature = "cuda"))]
use crate::backend::{GpuBackend, ComputeKernel, GpuArg, GpuBuffer};
#[cfg(not(feature = "cuda"))]
use anyhow::{Result, anyhow};

#[cfg(not(feature = "cuda"))]
pub struct CudaBackend;

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    pub fn try_new() -> Result<Self> {
        Err(anyhow!("CUDA feature not enabled"))
    }
}

#[cfg(not(feature = "cuda"))]
impl GpuBackend for CudaBackend {
    fn name(&self) -> &str { "CUDA" }

    fn allocate(&self, _size: usize) -> Result<GpuBuffer> {
        unimplemented!("CUDA feature not enabled")
    }

    fn execute(&self, _kernel: &ComputeKernel, _args: &[GpuArg]) -> Result<()> {
        unimplemented!("CUDA feature not enabled")
    }

    fn read_back(&self, _buffer: &GpuBuffer, _size: usize) -> Result<Vec<u8>> {
        unimplemented!("CUDA feature not enabled")
    }
}
