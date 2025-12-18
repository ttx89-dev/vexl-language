#[cfg(feature = "opencl")]
use crate::backend::{GpuBackend, ComputeKernel, GpuArg, GpuBuffer};
#[cfg(feature = "opencl")]
use anyhow::{Result, anyhow};
#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::Buffer,
    platform::get_platforms,
    program::Program,
};
#[cfg(feature = "opencl")]
use std::ptr;

#[cfg(feature = "opencl")]
pub struct OpenCLBackend {
    context: Context,
    queue: CommandQueue,
    device: Device,
    program: Program,
}

#[cfg(feature = "opencl")]
impl OpenCLBackend {
    pub fn try_new() -> Result<Self> {
        // Get OpenCL platform
        let platforms = get_platforms()?;
        if platforms.is_empty() {
            return Err(anyhow!("No OpenCL platforms found"));
        }
        let platform = &platforms[0];

        // Get GPU devices
        let devices = get_all_devices(platform, CL_DEVICE_TYPE_GPU)?;
        if devices.is_empty() {
            return Err(anyhow!("No OpenCL GPU devices found"));
        }
        let device = devices[0].clone();

        // Create context
        let context = Context::from_device(&device)?;

        // Create command queue
        let queue = CommandQueue::create_default(&context, opencl3::CL_QUEUE_PROFILING_ENABLE)?;

        // Create program (empty initially, will be built when kernels are loaded)
        let program = Program::create_from_source(&context, "")?;

        Ok(Self {
            context,
            queue,
            device,
            program,
        })
    }

    fn build_kernel(&mut self, kernel: &ComputeKernel) -> Result<()> {
        // Build program with the kernel source
        self.program = Program::create_from_source(&self.context, &kernel.source)?;
        self.program.build(&[&self.device], "")?;

        Ok(())
    }

    fn glsl_to_opencl(glsl_source: &str) -> Result<String> {
        // Convert GLSL compute shader to OpenCL kernel
        let mut cl_code = String::new();

        // Convert GLSL main to OpenCL kernel
        let main_start = glsl_source.find("void main()").ok_or_else(|| anyhow!("No main function found"))?;
        let main_body = &glsl_source[main_start..];

        cl_code.push_str("__kernel void main_kernel(");
        cl_code.push_str("__global float* inputA, __global float* inputB, __global float* output");
        cl_code.push_str(") {\n");

        // Convert GLSL built-ins to OpenCL
        let converted_body = main_body
            .replace("gl_GlobalInvocationID.x", "get_global_id(0)")
            .replace("inputA.data[", "inputA[")
            .replace("inputB.data[", "inputB[")
            .replace("output.data[", "output[");

        cl_code.push_str(&converted_body[12..]); // Skip "void main()"
        cl_code.push_str("\n}\n");

        Ok(cl_code)
    }
}

#[cfg(feature = "opencl")]
impl GpuBackend for OpenCLBackend {
    fn write_buffer(&self, _buffer: &GpuBuffer, _data: &[u8]) -> Result<()> {
        Err(anyhow::anyhow!("OpenCL backend not fully implemented"))
    }

    fn read_buffer(&self, _buffer: &GpuBuffer, _data: &mut [u8]) -> Result<()> {
        Err(anyhow::anyhow!("OpenCL backend not fully implemented"))
    }

    fn name(&self) -> &str { "OpenCL" }

    fn allocate(&self, size: usize) -> Result<GpuBuffer> {
        let buffer = Buffer::<u8>::create(&self.context, opencl3::CL_MEM_READ_WRITE, size, ptr::null_mut())?;

        Ok(GpuBuffer {
            id: buffer.get_cl_mem() as u64,
            size,
            ptr: buffer.get_cl_mem() as *mut (),
        })
    }

    fn execute(&self, kernel: &ComputeKernel, args: &[GpuArg]) -> Result<()> {
        // For OpenCL, we need mutable access to build kernels
        let mut_self = unsafe { &mut *(self as *const Self as *mut Self) };
        mut_self.build_kernel(kernel)?;

        // Create kernel
        let cl_kernel = Kernel::create(&mut_self.program, &kernel.entry_point)?;

        // Set kernel arguments
        let mut arg_index = 0;
        for arg in args {
            match arg {
                GpuArg::Buffer(buffer) => {
                    let cl_buffer = unsafe { Buffer::<u8>::from_cl_mem(buffer.id as opencl3::cl_mem) };
                    cl_kernel.set_arg(arg_index, &cl_buffer)?;
                },
                GpuArg::ScalarF64(val) => {
                    let scalar_val = *val as f32;
                    cl_kernel.set_arg(arg_index, &scalar_val)?;
                },
                _ => return Err(anyhow!("Unsupported argument type")),
            }
            arg_index += 1;
        }

        // Calculate work size
        let num_elements = args.iter().find_map(|arg| {
            if let GpuArg::Buffer(buffer) = arg {
                Some(buffer.size / std::mem::size_of::<f32>())
            } else {
                None
            }
        }).unwrap_or(1024);

        let work_size = num_elements as usize;
        let local_work_size = 256.min(work_size);

        // Enqueue kernel
        unsafe {
            cl_kernel.enqueue_nd_range(
                &mut_self.queue,
                &[work_size],
                &[local_work_size],
                &[],
            )?;
        }

        // Wait for completion
        mut_self.queue.finish()?;

        Ok(())
    }

    fn read_back(&self, buffer: &GpuBuffer, size: usize) -> Result<Vec<u8>> {
        let cl_buffer = unsafe { Buffer::<u8>::from_cl_mem(buffer.id as opencl3::cl_mem) };
        let mut host_data = vec![0u8; size];

        unsafe {
            self.queue.enqueue_read_buffer(
                &cl_buffer,
                opencl3::CL_BLOCKING,
                0,
                &mut host_data,
                &[],
            )?;
        }

        self.queue.finish()?;
        Ok(host_data)
    }
}

#[cfg(feature = "opencl")]
impl Drop for OpenCLBackend {
    fn drop(&mut self) {
        // OpenCL resources are automatically cleaned up
    }
}

#[cfg(not(feature = "opencl"))]
use crate::backend::{GpuBackend, ComputeKernel, GpuArg, GpuBuffer};
#[cfg(not(feature = "opencl"))]
use anyhow::{Result, anyhow};

#[cfg(not(feature = "opencl"))]
pub struct OpenCLBackend;

#[cfg(not(feature = "opencl"))]
impl OpenCLBackend {
    pub fn try_new() -> Result<Self> {
        Err(anyhow!("OpenCL feature not enabled"))
    }
}

#[cfg(not(feature = "opencl"))]
impl GpuBackend for OpenCLBackend {
    fn write_buffer(&self, _buffer: &GpuBuffer, _data: &[u8]) -> Result<()> {
        unimplemented!("OpenCL feature not enabled")
    }

    fn read_buffer(&self, _buffer: &GpuBuffer, _data: &mut [u8]) -> Result<()> {
        unimplemented!("OpenCL feature not enabled")
    }

    fn name(&self) -> &str { "OpenCL" }

    fn allocate(&self, _size: usize) -> Result<GpuBuffer> {
        unimplemented!("OpenCL feature not enabled")
    }

    fn execute(&self, _kernel: &ComputeKernel, _args: &[GpuArg]) -> Result<()> {
        unimplemented!("OpenCL feature not enabled")
    }

    fn read_back(&self, _buffer: &GpuBuffer, _size: usize) -> Result<Vec<u8>> {
        unimplemented!("OpenCL feature not enabled")
    }
}
