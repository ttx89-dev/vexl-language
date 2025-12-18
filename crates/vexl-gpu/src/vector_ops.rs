//! GPU-Accelerated Vector Operations for VEXL
//!
//! High-performance vector operations using GPU compute shaders.
//! Automatically dispatches to GPU when beneficial, with CPU fallback.
//!
//! ## Future Enhancements (Phase 6.1)
//!
//! ### Advanced GPU Features
//! - SPIR-V code generation pipeline
//! - Enhanced Vulkan compute shaders
//! - Multi-GPU support
//! - GPU memory management optimization
//! - Kernel fusion for complex operations
//! - Dynamic kernel compilation
//!
//! ### Performance Optimizations
//! - Memory coalescing strategies
//! - Occupancy optimization
//! - Asynchronous execution
//! - GPU-CPU hybrid scheduling
//! - Profile-guided kernel selection

use crate::backend::GpuBackend;
use anyhow::Result;

/// GPU-accelerated vector operations
pub struct GpuVectorOps {
    backend: Box<dyn GpuBackend>,
}

impl GpuVectorOps {
    /// Create new GPU vector operations with automatic backend selection
    pub fn new() -> Result<Self> {
        let backend = crate::init_best_backend();
        Ok(Self { backend })
    }

    /// Create GPU vector operations with specific backend
    pub fn with_backend(backend: Box<dyn GpuBackend>) -> Self {
        Self { backend }
    }

    /// Vector addition: c[i] = a[i] + b[i]
    pub fn add_vectors(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(anyhow::anyhow!("Vector length mismatch: {} vs {}", a.len(), b.len()));
        }

        // Allocate GPU buffers
        let buffer_a = self.backend.allocate(a.len() * std::mem::size_of::<f32>())?;
        let buffer_b = self.backend.allocate(b.len() * std::mem::size_of::<f32>())?;
        let buffer_c = self.backend.allocate(a.len() * std::mem::size_of::<f32>())?;

        // Upload input data
        self.backend.write_buffer(&buffer_a, bytemuck::cast_slice(a))?;
        self.backend.write_buffer(&buffer_b, bytemuck::cast_slice(b))?;

        // Create compute kernel for vector addition
        let kernel_source = r#"
            #version 450
            layout(local_size_x = 256) in;

            layout(binding = 0) buffer InputA { float data[]; } inputA;
            layout(binding = 1) buffer InputB { float data[]; } inputB;
            layout(binding = 2) buffer Output { float data[]; } output;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx < inputA.data.length()) {
                    output.data[idx] = inputA.data[idx] + inputB.data[idx];
                }
            }
        "#;

        let args = vec![
            crate::GpuArg::Buffer(buffer_a.clone()),
            crate::GpuArg::Buffer(buffer_b.clone()),
            crate::GpuArg::Buffer(buffer_c.clone()),
        ];

        // Execute kernel
        self.backend.execute_kernel("vector_add", kernel_source, &args, a.len() as u32)?;

        // Read back result
        let mut result_bytes = vec![0u8; a.len() * std::mem::size_of::<f32>()];
        self.backend.read_buffer(&buffer_c, &mut result_bytes)?;
        let result = bytemuck::cast_slice::<u8, f32>(&result_bytes).to_vec();

        Ok(result)
    }

    /// Vector multiplication: c[i] = a[i] * b[i]
    pub fn mul_vectors(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(anyhow::anyhow!("Vector length mismatch: {} vs {}", a.len(), b.len()));
        }

        // For now, use CPU fallback implementation
        let mut result = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            result.push(a[i] * b[i]);
        }
        Ok(result)
    }

    /// Vector dot product: sum(a[i] * b[i])
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow::anyhow!("Vector length mismatch: {} vs {}", a.len(), b.len()));
        }

        // For now, use CPU fallback implementation
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        Ok(sum)
    }

    /// Map function over vector: c[i] = f(a[i])
    pub fn map_vector<F>(&self, input: &[f32], f: F) -> Result<Vec<f32>>
    where
        F: Fn(f32) -> f32,
    {
        // For now, use CPU fallback implementation
        let mut result = Vec::with_capacity(input.len());
        for &x in input {
            result.push(f(x));
        }
        Ok(result)
    }

    /// Scalar multiplication: c[i] = a[i] * scalar
    pub fn scale_vector(&self, input: &[f32], scalar: f32) -> Result<Vec<f32>> {
        self.map_vector(input, |x| x * scalar)
    }

    /// Vector reduction (sum): sum of all elements
    pub fn sum_vector(&self, input: &[f32]) -> Result<f32> {
        let mut sum = 0.0f32;
        for &x in input {
            sum += x;
        }
        Ok(sum)
    }

    /// Get backend name for debugging
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_addition() {
        let ops = GpuVectorOps::new().unwrap();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = ops.add_vectors(&a, &b).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_multiplication() {
        let ops = GpuVectorOps::new().unwrap();
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![3.0, 4.0, 5.0];
        let result = ops.mul_vectors(&a, &b).unwrap();
        assert_eq!(result, vec![6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_dot_product() {
        let ops = GpuVectorOps::new().unwrap();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = ops.dot_product(&a, &b).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_map_vector() {
        let ops = GpuVectorOps::new().unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let result = ops.map_vector(&input, |x| x * 2.0).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_vector() {
        let ops = GpuVectorOps::new().unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let result = ops.scale_vector(&input, 3.0).unwrap();
        assert_eq!(result, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_sum_vector() {
        let ops = GpuVectorOps::new().unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = ops.sum_vector(&input).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_length_mismatch_error() {
        let ops = GpuVectorOps::new().unwrap();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(ops.add_vectors(&a, &b).is_err());
        assert!(ops.mul_vectors(&a, &b).is_err());
        assert!(ops.dot_product(&a, &b).is_err());
    }
}
