//! Vector runtime operations for VEXL
//!
//! This module provides the core vector operations that are called
//! from LLVM-generated code via FFI.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Vector header (64 bytes) - matches specification in vexl-core
#[repr(C)]
#[derive(Debug)]
pub struct VectorHeader {
    pub type_tag: u64,
    pub dimensionality: u64,
    pub total_size: u64,
    pub shape_ptr: *const u64,
    pub storage_mode: u64, // 0 = Dense
    pub data_ptr: *mut u8,
    pub stride_ptr: *const u64,
    pub metadata: u64,
}

/// Runtime vector representation
#[derive(Debug)]
pub struct Vector {
    header: NonNull<VectorHeader>,
}

impl Vector {
    /// Create a Vector from a raw pointer (consumes the pointer)
    pub unsafe fn from_raw(ptr: *mut Vector) -> Vector {
        *Box::from_raw(ptr)
    }

    /// Create a new vector from raw parts (called from LLVM)
    pub unsafe fn from_raw_parts(
        type_tag: u64,
        dimensionality: u64,
        total_size: u64,
        data_ptr: *mut u8,
    ) -> *mut Vector {
        // Allocate header
        let header_layout = Layout::new::<VectorHeader>();
        let header_ptr = alloc(header_layout) as *mut VectorHeader;

        // Initialize header
        (*header_ptr) = VectorHeader {
            type_tag,
            dimensionality,
            total_size,
            shape_ptr: std::ptr::null(), // TODO: Implement shape
            storage_mode: 0, // Dense
            data_ptr,
            stride_ptr: std::ptr::null(), // TODO: Implement stride
            metadata: 1, // Reference count
        };

        let vec = Vector {
            header: NonNull::new_unchecked(header_ptr),
        };

        // Return raw pointer for FFI
        Box::into_raw(Box::new(vec))
    }

    /// Get element at index (i64 elements)
    pub unsafe fn get_i64(&self, index: u64) -> i64 {
        let header = self.header.as_ref();
        if index >= header.total_size {
            return 0; // Error case
        }

        let elem_ptr = header.data_ptr.add((index * 8) as usize) as *const i64;
        *elem_ptr
    }

    /// Set element at index (i64 elements)
    pub unsafe fn set_i64(&mut self, index: u64, value: i64) {
        let header = self.header.as_ref();
        if index >= header.total_size {
            return; // Error case
        }

        let elem_ptr = header.data_ptr.add((index * 8) as usize) as *mut i64;
        *elem_ptr = value;
    }

    /// Get element at index (f64 elements)
    pub unsafe fn get_f64(&self, index: u64) -> f64 {
        let header = self.header.as_ref();
        if index >= header.total_size {
            return 0.0; // Error case
        }

        let elem_ptr = header.data_ptr.add((index * 8) as usize) as *const f64;
        *elem_ptr
    }

    /// Set element at index (f64 elements)
    pub unsafe fn set_f64(&mut self, index: u64, value: f64) {
        let header = self.header.as_ref();
        if index >= header.total_size {
            return; // Error case
        }

        let elem_ptr = header.data_ptr.add((index * 8) as usize) as *mut f64;
        *elem_ptr = value;
    }

    /// Get element at index (bool elements)
    pub unsafe fn get_bool(&self, index: u64) -> bool {
        let header = self.header.as_ref();
        if index >= header.total_size {
            return false; // Error case
        }

        let elem_ptr = header.data_ptr.add(index as usize) as *const bool;
        *elem_ptr
    }

    /// Set element at index (bool elements)
    pub unsafe fn set_bool(&mut self, index: u64, value: bool) {
        let header = self.header.as_ref();
        if index >= header.total_size {
            return; // Error case
        }

        let elem_ptr = header.data_ptr.add(index as usize) as *mut bool;
        *elem_ptr = value;
    }

    /// Get vector length
    pub fn len(&self) -> u64 {
        unsafe { self.header.as_ref().total_size }
    }

    /// Parallel map operation
    pub fn parallel_map<F>(&self, f: F) -> *mut Vector
    where
        F: Fn(i64) -> i64 + Send + Sync + 'static,
    {
        let len = self.len() as usize;
        let result = vexl_vec_alloc_i64(len as u64);

        // Simple sequential implementation for now
        // TODO: Use actual parallel execution
        for i in 0..len {
            let val = unsafe { self.get_i64(i as u64) };
            let mapped = f(val);
            vexl_vec_set_i64(result, i as u64, mapped);
        }

        result
    }

    /// Filter operation
    pub fn filter<F>(&self, pred: F) -> *mut Vector
    where
        F: Fn(&i64) -> bool,
    {
        let len = self.len() as usize;
        let mut filtered = Vec::new();

        for i in 0..len {
            let val = unsafe { self.get_i64(i as u64) };
            if pred(&val) {
                filtered.push(val);
            }
        }

        let filtered_len = filtered.len() as u64;
        let result = vexl_vec_alloc_i64(filtered_len);

        for (i, &val) in filtered.iter().enumerate() {
            vexl_vec_set_i64(result, i as u64, val);
        }

        result
    }

    /// Reduce operation
    pub fn reduce<F>(&self, init: i64, f: F) -> i64
    where
        F: Fn(i64, i64) -> i64,
    {
        let len = self.len() as usize;
        let mut acc = init;

        for i in 0..len {
            let val = unsafe { self.get_i64(i as u64) };
            acc = f(acc, val);
        }

        acc
    }

    /// Parallel reduce operation
    pub fn parallel_reduce<F>(&self, init: i64, f: F) -> i64
    where
        F: Fn(i64, i64) -> i64 + Send + Sync + 'static,
    {
        // Simple sequential implementation for now
        // TODO: Use actual parallel execution
        self.reduce(init, f)
    }

    /// Sequential map operation (fallback)
    pub fn map<F>(&self, f: F) -> *mut Vector
    where
        F: Fn(i64) -> i64 + Send + Sync + 'static,
    {
        self.parallel_map(f)
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        unsafe {
            let header = self.header.as_ptr();
            let metadata = (*header).metadata;

            if metadata <= 1 {
                // Last reference - deallocate
                if !(*header).data_ptr.is_null() {
                    // TODO: Proper deallocation based on type
                    // For now, assume i64 elements
                    let data_layout = Layout::from_size_align(
                        ((*header).total_size * 8) as usize,
                        8
                    ).unwrap();
                    dealloc((*header).data_ptr, data_layout);
                }

                let header_layout = Layout::new::<VectorHeader>();
                dealloc(header as *mut u8, header_layout);
            } else {
                (*header).metadata = metadata - 1;
            }
        }
    }
}

/// FFI Functions called from LLVM IR

/// Allocate a new vector with i64 elements
#[no_mangle]
pub extern "C" fn vexl_vec_alloc_i64(count: u64) -> *mut Vector {
    if count == 0 {
        return std::ptr::null_mut();
    }

    // Allocate data array
    let data_layout = Layout::from_size_align((count * 8) as usize, 8).unwrap();
    let data_ptr = unsafe { alloc(data_layout) };

    // Initialize to zeros
    unsafe {
        std::ptr::write_bytes(data_ptr, 0, (count * 8) as usize);
    }

    // Type tag: 0 = i64
    unsafe { Vector::from_raw_parts(0, 1, count, data_ptr) }
}

/// Get i64 element from vector
#[no_mangle]
pub extern "C" fn vexl_vec_get_i64(vec: *mut Vector, index: u64) -> i64 {
    if vec.is_null() {
        return 0;
    }

    let vector = unsafe { &*vec };
    unsafe { vector.get_i64(index) }
}

/// Set i64 element in vector
#[no_mangle]
pub extern "C" fn vexl_vec_set_i64(vec: *mut Vector, index: u64, value: i64) {
    if vec.is_null() {
        return;
    }

    let vector = unsafe { &mut *vec };
    unsafe { vector.set_i64(index, value) };
}

/// Get vector length
#[no_mangle]
pub extern "C" fn vexl_vec_len(vec: *mut Vector) -> u64 {
    if vec.is_null() {
        return 0;
    }

    let vector = unsafe { &*vec };
    vector.len()
}

/// Free vector
#[no_mangle]
pub extern "C" fn vexl_vec_free(vec: *mut Vector) {
    if !vec.is_null() {
        unsafe {
            drop(Box::from_raw(vec));
        }
    }
}

/// Create vector from i64 array (called from LLVM)
#[no_mangle]
pub extern "C" fn vexl_vec_from_i64_array(data: *const i64, count: u64) -> *mut Vector {
    if data.is_null() || count == 0 {
        return std::ptr::null_mut();
    }

    // Allocate data
    let data_layout = Layout::from_size_align((count * 8) as usize, 8).unwrap();
    let data_ptr = unsafe { alloc(data_layout) };

    // Copy data
    unsafe {
        std::ptr::copy_nonoverlapping(
            data as *const u8,
            data_ptr,
            (count * 8) as usize
        );
    }

    unsafe { Vector::from_raw_parts(0, 1, count, data_ptr) }
}

/// Vector addition (element-wise)
#[no_mangle]
pub extern "C" fn vexl_vec_add_i64(a: *mut Vector, b: *mut Vector) -> *mut Vector {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    let vec_a = unsafe { &*a };
    let vec_b = unsafe { &*b };

    let len_a = vec_a.len();
    let len_b = vec_b.len();

    if len_a != len_b {
        return std::ptr::null_mut(); // Error: mismatched lengths
    }

    let result = vexl_vec_alloc_i64(len_a);
    if result.is_null() {
        return std::ptr::null_mut();
    }

    for i in 0..len_a {
        let val_a = vexl_vec_get_i64(a, i);
        let val_b = vexl_vec_get_i64(b, i);
        vexl_vec_set_i64(result, i, val_a + val_b);
    }

    result
}

/// Vector scalar multiplication
#[no_mangle]
pub extern "C" fn vexl_vec_mul_scalar_i64(vec: *mut Vector, scalar: i64) -> *mut Vector {
    if vec.is_null() {
        return std::ptr::null_mut();
    }

    let vector = unsafe { &*vec };
    let len = vector.len();

    let result = vexl_vec_alloc_i64(len);
    if result.is_null() {
        return std::ptr::null_mut();
    }

    for i in 0..len {
        let val = vexl_vec_get_i64(vec, i);
        vexl_vec_set_i64(result, i, val * scalar);
    }

    result
}

/// Allocate a new vector with f64 elements
#[no_mangle]
pub extern "C" fn vexl_vec_alloc_f64(count: u64) -> *mut Vector {
    if count == 0 {
        return std::ptr::null_mut();
    }

    // Allocate data array
    let data_layout = Layout::from_size_align((count * 8) as usize, 8).unwrap();
    let data_ptr = unsafe { alloc(data_layout) };

    // Initialize to zeros
    unsafe {
        std::ptr::write_bytes(data_ptr, 0, (count * 8) as usize);
    }

    // Type tag: 1 = f64
    unsafe { Vector::from_raw_parts(1, 1, count, data_ptr) }
}

/// Get f64 element from vector
#[no_mangle]
pub extern "C" fn vexl_vec_get_f64(vec: *mut Vector, index: u64) -> f64 {
    if vec.is_null() {
        return 0.0;
    }

    let vector = unsafe { &*vec };
    unsafe { vector.get_f64(index) }
}

/// Set f64 element in vector
#[no_mangle]
pub extern "C" fn vexl_vec_set_f64(vec: *mut Vector, index: u64, value: f64) {
    if vec.is_null() {
        return;
    }

    let vector = unsafe { &mut *vec };
    unsafe { vector.set_f64(index, value) };
}

/// Allocate a new vector with bool elements
#[no_mangle]
pub extern "C" fn vexl_vec_alloc_bool(count: u64) -> *mut Vector {
    if count == 0 {
        return std::ptr::null_mut();
    }

    // Allocate data array
    let data_layout = Layout::from_size_align(count as usize, 1).unwrap();
    let data_ptr = unsafe { alloc(data_layout) };

    // Initialize to false
    unsafe {
        std::ptr::write_bytes(data_ptr, 0, count as usize);
    }

    // Type tag: 2 = bool
    unsafe { Vector::from_raw_parts(2, 1, count, data_ptr) }
}

/// Get bool element from vector
#[no_mangle]
pub extern "C" fn vexl_vec_get_bool(vec: *mut Vector, index: u64) -> bool {
    if vec.is_null() {
        return false;
    }

    let vector = unsafe { &*vec };
    unsafe { vector.get_bool(index) }
}

/// Set bool element in vector
#[no_mangle]
pub extern "C" fn vexl_vec_set_bool(vec: *mut Vector, index: u64, value: bool) {
    if vec.is_null() {
        return;
    }

    let vector = unsafe { &mut *vec };
    unsafe { vector.set_bool(index, value) };
}

/// Vector dot product (i64)
#[no_mangle]
pub extern "C" fn vexl_vec_dot_i64(a: *mut Vector, b: *mut Vector) -> i64 {
    if a.is_null() || b.is_null() {
        return 0;
    }

    let vec_a = unsafe { &*a };
    let vec_b = unsafe { &*b };

    let len_a = vec_a.len();
    let len_b = vec_b.len();

    if len_a != len_b {
        return 0; // Error: mismatched lengths
    }

    let mut result = 0i64;
    for i in 0..len_a {
        let val_a = vexl_vec_get_i64(a, i);
        let val_b = vexl_vec_get_i64(b, i);
        result += val_a * val_b;
    }

    result
}

/// Vector dot product (f64)
#[no_mangle]
pub extern "C" fn vexl_vec_dot_f64(a: *mut Vector, b: *mut Vector) -> f64 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }

    let vec_a = unsafe { &*a };
    let vec_b = unsafe { &*b };

    let len_a = vec_a.len();
    let len_b = vec_b.len();

    if len_a != len_b {
        return 0.0; // Error: mismatched lengths
    }

    let mut result = 0.0f64;
    for i in 0..len_a {
        let val_a = vexl_vec_get_f64(a, i);
        let val_b = vexl_vec_get_f64(b, i);
        result += val_a * val_b;
    }

    result
}

/// Matrix multiplication (simplified 2D case)
/// Assumes vectors are stored in row-major order
#[no_mangle]
pub extern "C" fn vexl_mat_mul_i64(a: *mut Vector, b: *mut Vector, m: u64, n: u64, p: u64) -> *mut Vector {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    // Result matrix is m x p
    let result = vexl_vec_alloc_i64(m * p);
    if result.is_null() {
        return std::ptr::null_mut();
    }

    // C[i][j] = sum_k A[i][k] * B[k][j]
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0i64;
            for k in 0..n {
                let a_val = vexl_vec_get_i64(a, i * n + k);
                let b_val = vexl_vec_get_i64(b, k * p + j);
                sum += a_val * b_val;
            }
            vexl_vec_set_i64(result, i * p + j, sum);
        }
    }

    result
}

/// Matrix transpose (2D matrix)
#[no_mangle]
pub extern "C" fn vexl_mat_transpose_i64(vec: *mut Vector, rows: u64, cols: u64) -> *mut Vector {
    if vec.is_null() {
        return std::ptr::null_mut();
    }

    // Result is cols x rows
    let result = vexl_vec_alloc_i64(cols * rows);
    if result.is_null() {
        return std::ptr::null_mut();
    }

    for i in 0..rows {
        for j in 0..cols {
            let val = vexl_vec_get_i64(vec, i * cols + j);
            vexl_vec_set_i64(result, j * rows + i, val);
        }
    }

    result
}

/// Vector element-wise multiplication
#[no_mangle]
pub extern "C" fn vexl_vec_mul_i64(a: *mut Vector, b: *mut Vector) -> *mut Vector {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    let vec_a = unsafe { &*a };
    let vec_b = unsafe { &*b };

    let len_a = vec_a.len();
    let len_b = vec_b.len();

    if len_a != len_b {
        return std::ptr::null_mut(); // Error: mismatched lengths
    }

    let result = vexl_vec_alloc_i64(len_a);
    if result.is_null() {
        return std::ptr::null_mut();
    }

    for i in 0..len_a {
        let val_a = vexl_vec_get_i64(a, i);
        let val_b = vexl_vec_get_i64(b, i);
        vexl_vec_set_i64(result, i, val_a * val_b);
    }

    result
}

/// Vector element-wise subtraction
#[no_mangle]
pub extern "C" fn vexl_vec_sub_i64(a: *mut Vector, b: *mut Vector) -> *mut Vector {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    let vec_a = unsafe { &*a };
    let vec_b = unsafe { &*b };

    let len_a = vec_a.len();
    let len_b = vec_b.len();

    if len_a != len_b {
        return std::ptr::null_mut(); // Error: mismatched lengths
    }

    let result = vexl_vec_alloc_i64(len_a);
    if result.is_null() {
        return std::ptr::null_mut();
    }

    for i in 0..len_a {
        let val_a = vexl_vec_get_i64(a, i);
        let val_b = vexl_vec_get_i64(b, i);
        vexl_vec_set_i64(result, i, val_a - val_b);
    }

    result
}
