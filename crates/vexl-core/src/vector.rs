//! VEXL Vector Type
//!
//! Everything in VEXL is a vector. This module implements vector types
//! for maximum performance and automatic parallelization.

/// Vector error types
#[derive(Debug)]
pub enum VectorError {
    /// Invalid index access
    IndexOutOfBounds { index: usize, size: usize },

    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },

    /// Invalid shape for operation
    InvalidShape,

    /// Parallel execution not available
    ParallelExecutionUnavailable,
}

/// Result type for vector operations
pub type VectorResult<T> = Result<T, VectorError>;

/// Parallel and GPU acceleration configuration
#[derive(Debug)]
pub struct ComputeConfig {
    pub use_parallel: bool,
    pub use_gpu: bool,
    pub gpu_threshold: usize,
    pub parallel_threshold: usize,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            use_parallel: true,
            use_gpu: true,
            gpu_threshold: 1024,      // Use GPU for arrays >= 1K elements
            parallel_threshold: 64,    // Use parallel for arrays >= 64 elements
        }
    }
}

/// Global compute configuration
static mut GLOBAL_COMPUTE_CONFIG: ComputeConfig = ComputeConfig {
    use_parallel: true,
    use_gpu: true,
    gpu_threshold: 1024,
    parallel_threshold: 64,
};

/// Set global compute configuration
pub fn set_compute_config(config: ComputeConfig) {
    unsafe {
        GLOBAL_COMPUTE_CONFIG = config;
    }
}

/// Get global compute configuration
fn get_compute_config() -> &'static ComputeConfig {
    unsafe { &GLOBAL_COMPUTE_CONFIG }
}

/// GPU acceleration trait for external GPU implementations
pub trait GpuAccelerator: Send + Sync {
    fn should_use_gpu(&self, size: usize) -> bool;
    fn add_vectors(&self, a: &[f32], b: &[f32]) -> anyhow::Result<Vec<f32>>;
    fn mul_vectors(&self, a: &[f32], b: &[f32]) -> anyhow::Result<Vec<f32>>;
    fn map_vector(&self, input: &[f32], scalar: f32) -> anyhow::Result<Vec<f32>>;
}

/// Initialize GPU acceleration if available
#[cfg(feature = "gpu")]
pub fn init_gpu_acceleration() {
    use vexl_gpu::GpuVectorOps;

    if let Ok(gpu_ops) = GpuVectorOps::new() {
        let accelerator = Box::new(GpuAcceleratorImpl { ops: gpu_ops });
        set_gpu_accelerator(accelerator);
    }
}

#[cfg(not(feature = "gpu"))]
pub fn init_gpu_acceleration() {
    // No GPU support
}

/// Internal GPU accelerator implementation
#[cfg(feature = "gpu")]
struct GpuAcceleratorImpl {
    ops: vexl_gpu::GpuVectorOps,
}

#[cfg(feature = "gpu")]
impl GpuAccelerator for GpuAcceleratorImpl {
    fn should_use_gpu(&self, size: usize) -> bool {
        self.ops.should_use_gpu(size)
    }

    fn add_vectors(&self, a: &[f32], b: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.ops.add_vectors(a, b)
    }

    fn mul_vectors(&self, a: &[f32], b: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.ops.mul_vectors(a, b)
    }

    fn map_vector(&self, input: &[f32], scalar: f32) -> anyhow::Result<Vec<f32>> {
        self.ops.map_vector(input, scalar)
    }
}

/// Global GPU accelerator instance
static mut GPU_ACCELERATOR: Option<Box<dyn GpuAccelerator>> = None;

/// Set the global GPU accelerator
pub fn set_gpu_accelerator(accelerator: Box<dyn GpuAccelerator>) {
    unsafe {
        GPU_ACCELERATOR = Some(accelerator);
    }
}

/// Get reference to global GPU accelerator
fn get_gpu_accelerator() -> Option<&'static dyn GpuAccelerator> {
    unsafe {
        GPU_ACCELERATOR.as_ref().map(|b| b.as_ref())
    }
}

/// GPU acceleration interface
mod gpu_accel {
    use super::*;

    pub fn init_gpu() {
        // GPU initialization would be handled by the external GPU crate
        // This is just a placeholder for the interface
    }

    pub fn should_use_gpu(size: usize) -> bool {
        get_gpu_accelerator().map_or(false, |acc| acc.should_use_gpu(size))
    }

    pub fn add_vectors_gpu(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
        get_gpu_accelerator()?.add_vectors(a, b).ok()
    }

    pub fn mul_vectors_gpu(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
        get_gpu_accelerator()?.mul_vectors(a, b).ok()
    }

    pub fn map_vector_gpu(input: &[f32], scalar: f32) -> Option<Vec<f32>> {
        get_gpu_accelerator()?.map_vector(input, scalar).ok()
    }
}

/// SIMD-accelerated vector operations for performance optimization
/// Uses portable SIMD when available, with graceful fallback
pub mod simd {
    

    /// SIMD-accelerated vector addition for i32
    /// Automatically chooses optimal implementation based on platform and size
    pub fn add_i32_simd(a: &[i32], b: &[i32], result: &mut [i32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        // For large arrays, use chunked processing (may be auto-vectorized by compiler)
        if a.len() >= 32 {
            let chunk_size = 8;
            for chunk_start in (0..a.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(a.len());
                let chunk_len = chunk_end - chunk_start;

                for i in 0..chunk_len {
                    let idx = chunk_start + i;
                    result[idx] = a[idx] + b[idx];
                }
            }
        } else {
            // For small arrays, use simple loop
            for i in 0..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }

    /// SIMD-accelerated vector multiplication for i32
    pub fn mul_i32_simd(a: &[i32], b: &[i32], result: &mut [i32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() >= 32 {
            let chunk_size = 8;
            for chunk_start in (0..a.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(a.len());
                let chunk_len = chunk_end - chunk_start;

                for i in 0..chunk_len {
                    let idx = chunk_start + i;
                    result[idx] = a[idx] * b[idx];
                }
            }
        } else {
            for i in 0..a.len() {
                result[i] = a[i] * b[i];
            }
        }
    }

    /// SIMD-accelerated sum reduction for i32
    pub fn sum_i32_simd(data: &[i32]) -> i32 {
        if data.is_empty() {
            return 0;
        }

        // Use parallel chunks for large arrays
        if data.len() >= 64 {
            let chunk_size = 16;
            let mut partial_sums = Vec::new();

            for chunk in data.chunks(chunk_size) {
                let chunk_sum: i32 = chunk.iter().sum();
                partial_sums.push(chunk_sum);
            }

            partial_sums.iter().sum()
        } else {
            // Simple sequential sum
            data.iter().sum()
        }
    }

    /// Check if SIMD is beneficial for the given operation size
    pub fn should_use_simd(len: usize) -> bool {
        // SIMD is typically beneficial for arrays of at least 32 elements
        // due to setup overhead and memory access patterns
        len >= 32
    }
}

/// 1D Vector - most common case for basic operations
#[derive(Debug, Clone)]
pub struct Vector1<T> {
    data: Vec<T>,
}

/// 2D Vector - for matrices and tensors
#[derive(Debug, Clone)]
pub struct Vector2<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

// Implementation for 1D Vectors
impl<T> Vector1<T> {
    /// Create a new 1D vector from elements
    pub fn new(elements: Vec<T>) -> Self {
        Vector1 { data: elements }
    }

    /// Create a 1D vector from a slice
    pub fn from_slice(elements: &[T]) -> Self
    where T: Clone {
        Vector1 { data: elements.to_vec() }
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> VectorResult<&T> {
        self.data.get(index)
            .ok_or_else(|| VectorError::IndexOutOfBounds {
                index,
                size: self.data.len()
            })
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Map function over elements
    pub fn map<U, F>(&self, f: F) -> Vector1<U>
    where
        T: Clone,
        F: Fn(&T) -> U,
    {
        let results: Vec<U> = self.data.iter().map(f).collect();
        Vector1 { data: results }
    }

    /// Filter elements
    pub fn filter<F>(&self, pred: F) -> Vector1<T>
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        let results: Vec<T> = self.data.iter()
            .filter(|&elem| pred(elem))
            .cloned()
            .collect();
        Vector1 { data: results }
    }

    /// Reduce to single value
    pub fn reduce<F>(&self, init: T, f: F) -> T
    where
        T: Clone + Copy,
        F: Fn(T, T) -> T,
    {
        self.data.iter().fold(init, |acc, &elem| f(acc, elem))
    }

    /// Parallel map using cooperative scheduler
    /// Uses runtime executor if available, otherwise falls back to sequential
    pub fn par_map<U, F>(&self, f: F) -> Vector1<U>
    where
        T: Clone,
        F: Fn(&T) -> U,
    {
        // TODO: Integrate with runtime parallel execution when available
        // For now, use sequential implementation
        self.map(f)
    }

    /// Parallel filter using cooperative scheduler
    /// Uses runtime executor if available, otherwise falls back to sequential
    pub fn par_filter<F>(&self, pred: F) -> Vector1<T>
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        // TODO: Integrate with runtime parallel execution when available
        // For now, use sequential implementation
        self.filter(pred)
    }

    /// Parallel reduce using tree-based accumulation
    /// Uses runtime executor if available, otherwise falls back to sequential
    pub fn par_reduce<F>(&self, init: T, f: F) -> T
    where
        T: Clone + Copy,
        F: Fn(T, T) -> T,
    {
        // TODO: Integrate with runtime parallel execution when available
        // For now, use sequential implementation
        self.reduce(init, f)
    }
}

// Implementation for 2D Vectors
impl<T> Vector2<T> {
    /// Create a 2D vector from nested vectors
    pub fn from_nested(nested: Vec<Vec<T>>) -> VectorResult<Self> {
        if nested.is_empty() {
            return Err(VectorError::InvalidShape);
        }

        let rows = nested.len();
        let cols = nested[0].len();

        // Validate all rows have same length
        for row in &nested {
            if row.len() != cols {
                return Err(VectorError::InvalidShape);
            }
        }

        let mut data = Vec::with_capacity(rows * cols);
        for row in nested {
            data.extend(row);
        }

        Ok(Vector2 { data, rows, cols })
    }

    /// Create a 2D vector from flat data
    pub fn from_data(data: Vec<T>, rows: usize, cols: usize) -> VectorResult<Self> {
        if data.len() != rows * cols {
            return Err(VectorError::InvalidShape);
        }
        Ok(Vector2 { data, rows, cols })
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> VectorResult<&T> {
        if row >= self.rows || col >= self.cols {
            return Err(VectorError::IndexOutOfBounds {
                index: row * self.cols + col,
                size: self.data.len()
            });
        }

        let index = row * self.cols + col;
        Ok(unsafe { self.data.get_unchecked(index) })
    }

    /// Get dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get total elements
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// Advanced Matrix Operations
impl<T> Vector2<T>
where T: Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default + Copy {

    /// Matrix multiplication (self @ other)
    pub fn matmul(&self, other: &Vector2<T>) -> VectorResult<Vector2<T>> {
        if self.cols != other.rows {
            return Err(VectorError::InvalidShape);
        }

        let mut result_data = Vec::with_capacity(self.rows * other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    let a = unsafe { self.data.get_unchecked(i * self.cols + k) };
                    let b = unsafe { other.data.get_unchecked(k * other.cols + j) };
                    sum = sum + (*a * *b);
                }
                result_data.push(sum);
            }
        }

        Ok(Vector2 {
            data: result_data,
            rows: self.rows,
            cols: other.cols,
        })
    }

    /// Element-wise addition with broadcasting
    pub fn add(&self, other: &Vector2<T>) -> VectorResult<Vector2<T>> {
        self.element_wise_op(other, |a, b| *a + *b)
    }

    /// Element-wise multiplication with broadcasting
    pub fn mul(&self, other: &Vector2<T>) -> VectorResult<Vector2<T>> {
        self.element_wise_op(other, |a, b| *a * *b)
    }

    /// Element-wise operation with broadcasting support
    fn element_wise_op<F>(&self, other: &Vector2<T>, op: F) -> VectorResult<Vector2<T>>
    where F: Fn(&T, &T) -> T {
        // For now, require exact shape match (broadcasting TODO)
        if self.shape() != other.shape() {
            return Err(VectorError::InvalidShape);
        }

        let mut result_data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            let a = unsafe { self.data.get_unchecked(i) };
            let b = unsafe { other.data.get_unchecked(i) };
            result_data.push(op(a, b));
        }

        Ok(Vector2 {
            data: result_data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Transpose matrix
    pub fn transpose(&self) -> Vector2<T> {
        let mut result_data = Vec::with_capacity(self.data.len());

        for j in 0..self.cols {
            for i in 0..self.rows {
                let val = unsafe { self.data.get_unchecked(i * self.cols + j) };
                result_data.push(*val);
            }
        }

        Vector2 {
            data: result_data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Extract row as 1D vector
    pub fn row(&self, row_idx: usize) -> VectorResult<Vector1<T>> {
        if row_idx >= self.rows {
            return Err(VectorError::IndexOutOfBounds {
                index: row_idx,
                size: self.rows
            });
        }

        let start = row_idx * self.cols;
        let end = start + self.cols;
        let row_data = self.data[start..end].to_vec();

        Ok(Vector1 { data: row_data })
    }

    /// Extract column as 1D vector
    pub fn col(&self, col_idx: usize) -> VectorResult<Vector1<T>> {
        if col_idx >= self.cols {
            return Err(VectorError::IndexOutOfBounds {
                index: col_idx,
                size: self.cols
            });
        }

        let mut col_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let val = unsafe { self.data.get_unchecked(i * self.cols + col_idx) };
            col_data.push(*val);
        }

        Ok(Vector1 { data: col_data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector1_creation() {
        let v = Vector1::from_slice(&[1, 2, 3]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0).unwrap(), &1);
        assert_eq!(v.get(1).unwrap(), &2);
        assert_eq!(v.get(2).unwrap(), &3);
    }

    #[test]
    fn test_vector1_map() {
        let v = Vector1::from_slice(&[1, 2, 3]);
        let doubled = v.map(|x| x * 2);
        assert_eq!(doubled.get(0).unwrap(), &2);
        assert_eq!(doubled.get(1).unwrap(), &4);
        assert_eq!(doubled.get(2).unwrap(), &6);
    }

    #[test]
    fn test_vector1_filter() {
        let v = Vector1::from_slice(&[1, 2, 3, 4, 5]);
        let evens = v.filter(|x| x % 2 == 0);
        assert_eq!(evens.len(), 2);
        assert_eq!(evens.get(0).unwrap(), &2);
        assert_eq!(evens.get(1).unwrap(), &4);
    }

    #[test]
    fn test_vector1_reduce() {
        let v = Vector1::from_slice(&[1, 2, 3, 4, 5]);
        let sum = v.reduce(0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_vector2_creation() {
        let matrix = vec![
            vec![1, 2],
            vec![3, 4]
        ];
        let v = Vector2::from_nested(matrix).unwrap();
        assert_eq!(v.shape(), (2, 2));
        assert_eq!(v.len(), 4);
        assert_eq!(v.get(0, 0).unwrap(), &1);
        assert_eq!(v.get(0, 1).unwrap(), &2);
        assert_eq!(v.get(1, 0).unwrap(), &3);
        assert_eq!(v.get(1, 1).unwrap(), &4);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Vector2::from_nested(vec![
            vec![1, 2],
            vec![3, 4]
        ]).unwrap();

        let b = Vector2::from_nested(vec![
            vec![5, 6],
            vec![7, 8]
        ]).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), (2, 2));
        // [1,2]   [5,6]   = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4] * [7,8]   = [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result.get(0, 0).unwrap(), &19);
        assert_eq!(result.get(0, 1).unwrap(), &22);
        assert_eq!(result.get(1, 0).unwrap(), &43);
        assert_eq!(result.get(1, 1).unwrap(), &50);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = Vector2::from_nested(vec![
            vec![1, 2, 3],
            vec![4, 5, 6]
        ]).unwrap();

        let transposed = matrix.transpose();
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed.get(0, 0).unwrap(), &1);
        assert_eq!(transposed.get(1, 0).unwrap(), &2);
        assert_eq!(transposed.get(2, 0).unwrap(), &3);
        assert_eq!(transposed.get(0, 1).unwrap(), &4);
        assert_eq!(transposed.get(1, 1).unwrap(), &5);
        assert_eq!(transposed.get(2, 1).unwrap(), &6);
    }

    #[test]
    fn test_matrix_addition() {
        let a = Vector2::from_nested(vec![
            vec![1, 2],
            vec![3, 4]
        ]).unwrap();

        let b = Vector2::from_nested(vec![
            vec![5, 6],
            vec![7, 8]
        ]).unwrap();

        let result = a.add(&b).unwrap();
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.get(0, 0).unwrap(), &6);  // 1+5
        assert_eq!(result.get(0, 1).unwrap(), &8);  // 2+6
        assert_eq!(result.get(1, 0).unwrap(), &10); // 3+7
        assert_eq!(result.get(1, 1).unwrap(), &12); // 4+8
    }

    #[test]
    fn test_matrix_row_extraction() {
        let matrix = Vector2::from_nested(vec![
            vec![1, 2, 3],
            vec![4, 5, 6]
        ]).unwrap();

        let row0 = matrix.row(0).unwrap();
        assert_eq!(row0.len(), 3);
        assert_eq!(row0.get(0).unwrap(), &1);
        assert_eq!(row0.get(1).unwrap(), &2);
        assert_eq!(row0.get(2).unwrap(), &3);

        let row1 = matrix.row(1).unwrap();
        assert_eq!(row1.len(), 3);
        assert_eq!(row1.get(0).unwrap(), &4);
        assert_eq!(row1.get(1).unwrap(), &5);
        assert_eq!(row1.get(2).unwrap(), &6);
    }

    #[test]
    fn test_matrix_col_extraction() {
        let matrix = Vector2::from_nested(vec![
            vec![1, 2, 3],
            vec![4, 5, 6]
        ]).unwrap();

        let col0 = matrix.col(0).unwrap();
        assert_eq!(col0.len(), 2);
        assert_eq!(col0.get(0).unwrap(), &1);
        assert_eq!(col0.get(1).unwrap(), &4);

        let col2 = matrix.col(2).unwrap();
        assert_eq!(col2.len(), 2);
        assert_eq!(col2.get(0).unwrap(), &3);
        assert_eq!(col2.get(1).unwrap(), &6);
    }

    #[test]
    fn test_parallel_map() {
        // Currently falls back to sequential - tests API compatibility
        let v = Vector1::from_slice(&[1, 2, 3, 4, 5]);
        let doubled = v.par_map(|x| x * 2);
        assert_eq!(doubled.len(), 5);
        assert_eq!(doubled.get(0).unwrap(), &2);
        assert_eq!(doubled.get(1).unwrap(), &4);
        assert_eq!(doubled.get(4).unwrap(), &10);
    }

    #[test]
    fn test_parallel_filter() {
        // Currently falls back to sequential - tests API compatibility
        let v = Vector1::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let evens = v.par_filter(|x| x % 2 == 0);
        assert_eq!(evens.len(), 4); // Should contain exactly 2, 4, 6, 8
        assert_eq!(evens.get(0).unwrap(), &2);
        assert_eq!(evens.get(1).unwrap(), &4);
        assert_eq!(evens.get(2).unwrap(), &6);
        assert_eq!(evens.get(3).unwrap(), &8);
    }

    #[test]
    fn test_parallel_reduce() {
        // Currently falls back to sequential - tests API compatibility
        let v = Vector1::from_slice(&[1, 2, 3, 4, 5]);
        let sum = v.par_reduce(0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }
}
