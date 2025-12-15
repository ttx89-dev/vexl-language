//! VEXL Runtime - Vector operations, generators, and cooperative scheduler

pub mod vector;
pub mod generator;
pub mod scheduler;
pub mod cache;
pub mod gc;
pub mod ffi; // Export FFI module

// Parallel execution functions for vexl-core integration
mod parallel_ops;

pub use vector::{Vector, VectorHeader};
pub use scheduler::CooperativeScheduler;
pub use parallel_ops::{ParallelOps};

/// Initialize the VEXL runtime
/// Must be called before any VEXL operations
#[no_mangle]
pub extern "C" fn vexl_runtime_init() {
    // Initialize thread pool for parallel operations
    scheduler::init_thread_pool();

    // Initialize parallel operations for vexl-core integration
    parallel_ops::init_vexl_core_integration();

    // Initialize GPU acceleration if available
    #[cfg(feature = "gpu")]
    vexl_core::gpu_accel::init_gpu();

    // Initialize garbage collector
    gc::init_gc();

    // Initialize generator cache system
    cache::init_cache_system();
}

/// Shutdown the VEXL runtime
#[no_mangle]
pub extern "C" fn vexl_runtime_shutdown() {
    // Cleanup resources
    scheduler::shutdown_thread_pool();
    gc::shutdown_gc();
    cache::shutdown_cache_system();
}

// ═══════════════════════════════════════════════════════════
// PARALLEL VECTOR OPERATIONS
// ═══════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn vexl_vec_map_parallel(
    vec_ptr: *mut crate::vector::Vector,
    fn_ptr: *const (),
    _num_threads: u64
) -> *mut crate::vector::Vector {
    let vec = unsafe { &*vec_ptr };

    type MapFn = extern "C" fn(i64) -> i64;
    let map_fn: MapFn = unsafe { std::mem::transmute(fn_ptr) };

    let result = vec.parallel_map(move |x: i64| map_fn(x));
    result
}

#[no_mangle]
pub extern "C" fn vexl_vec_filter(
    vec_ptr: *mut crate::vector::Vector,
    pred_ptr: *const ()
) -> *mut crate::vector::Vector {
    let vec = unsafe { &*vec_ptr };

    type PredFn = extern "C" fn(i64) -> bool;
    let pred_fn: PredFn = unsafe { std::mem::transmute(pred_ptr) };

    let result = vec.filter(move |x: &i64| pred_fn(*x));
    result
}

#[no_mangle]
pub extern "C" fn vexl_vec_reduce_parallel(
    vec_ptr: *mut crate::vector::Vector,
    init_ptr: *const u8,
    fn_ptr: *const (),
    _num_threads: u64
) -> *mut u8 {
    let vec = unsafe { &*vec_ptr };
    let init: i64 = unsafe { *(init_ptr as *const i64) };

    type ReduceFn = extern "C" fn(i64, i64) -> i64;
    let reduce_fn: ReduceFn = unsafe { std::mem::transmute(fn_ptr) };

    let result = vec.parallel_reduce(init, move |a, b| reduce_fn(a, b));
    // Allocate memory for the result
    let layout = std::alloc::Layout::new::<i64>();
    let ptr = unsafe { std::alloc::alloc(layout) as *mut i64 };
    unsafe { *ptr = result; }
    ptr as *mut u8
}

// Non-parallel versions for fallback
#[no_mangle]
pub extern "C" fn vexl_vec_map_sequential(
    vec_ptr: *mut crate::vector::Vector,
    fn_ptr: *const ()
) -> *mut crate::vector::Vector {
    let vec = unsafe { &*vec_ptr };

    type MapFn = extern "C" fn(i64) -> i64;
    let map_fn: MapFn = unsafe { std::mem::transmute(fn_ptr) };

    let result = vec.map(move |x: i64| map_fn(x));
    result
}

#[no_mangle]
pub extern "C" fn vexl_vec_reduce_sequential(
    vec_ptr: *mut crate::vector::Vector,
    init_ptr: *const u8,
    fn_ptr: *const ()
) -> *mut u8 {
    let vec = unsafe { &*vec_ptr };
    let init: i64 = unsafe { *(init_ptr as *const i64) };

    type ReduceFn = extern "C" fn(i64, i64) -> i64;
    let reduce_fn: ReduceFn = unsafe { std::mem::transmute(fn_ptr) };

    let result = vec.reduce(init, move |a, b| reduce_fn(a, b));
    // Allocate memory for the result
    let layout = std::alloc::Layout::new::<i64>();
    let ptr = unsafe { std::alloc::alloc(layout) as *mut i64 };
    unsafe { *ptr = result; }
    ptr as *mut u8
}

/// Standard Library Functions
/// These are exposed as runtime functions for VEXL programs

#[no_mangle]
pub extern "C" fn vexl_vec_sum(vec_ptr: *mut crate::vector::Vector) -> i64 {
    let vec = unsafe { &*vec_ptr };
    let len = vec.len() as usize;

    let mut sum = 0i64;
    for i in 0..len {
        sum += unsafe { vec.get_i64(i as u64) };
    }
    sum
}

#[no_mangle]
pub extern "C" fn vexl_vec_product(vec_ptr: *mut crate::vector::Vector) -> i64 {
    let vec = unsafe { &*vec_ptr };
    let len = vec.len() as usize;

    let mut product = 1i64;
    for i in 0..len {
        product *= unsafe { vec.get_i64(i as u64) };
    }
    product
}

#[no_mangle]
pub extern "C" fn vexl_vec_max(vec_ptr: *mut crate::vector::Vector) -> i64 {
    let vec = unsafe { &*vec_ptr };
    let len = vec.len() as usize;

    if len == 0 {
        return 0;
    }

    let mut max_val = unsafe { vec.get_i64(0) };
    for i in 1..len {
        let val = unsafe { vec.get_i64(i as u64) };
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

#[no_mangle]
pub extern "C" fn vexl_vec_min(vec_ptr: *mut crate::vector::Vector) -> i64 {
    let vec = unsafe { &*vec_ptr };
    let len = vec.len() as usize;

    if len == 0 {
        return 0;
    }

    let mut min_val = unsafe { vec.get_i64(0) };
    for i in 1..len {
        let val = unsafe { vec.get_i64(i as u64) };
        if val < min_val {
            min_val = val;
        }
    }
    min_val
}

#[no_mangle]
pub extern "C" fn vexl_math_sin(x: f64) -> f64 {
    x.sin()
}

#[no_mangle]
pub extern "C" fn vexl_math_cos(x: f64) -> f64 {
    x.cos()
}

#[no_mangle]
pub extern "C" fn vexl_math_sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[no_mangle]
pub extern "C" fn vexl_math_pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

// ═══════════════════════════════════════════════════════════
// STRING OPERATIONS
// ═══════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn vexl_string_concat(
    s1_ptr: *const u8,
    s1_len: u64,
    s2_ptr: *const u8,
    s2_len: u64
) -> *mut u8 {
    let s1 = unsafe { std::slice::from_raw_parts(s1_ptr, s1_len as usize) };
    let s2 = unsafe { std::slice::from_raw_parts(s2_ptr, s2_len as usize) };

    let result = format!("{}{}",
        std::str::from_utf8(s1).unwrap_or(""),
        std::str::from_utf8(s2).unwrap_or("")
    );

    // Return as C string (null-terminated)
    let c_str = std::ffi::CString::new(result).unwrap();
    c_str.into_raw() as *mut u8
}

#[no_mangle]
pub extern "C" fn vexl_string_len(s_ptr: *const u8) -> u64 {
    let c_str = unsafe { std::ffi::CStr::from_ptr(s_ptr as *const i8) };
    c_str.to_bytes().len() as u64
}

// IO operations are now defined in the ffi.rs module
pub mod vpu;
