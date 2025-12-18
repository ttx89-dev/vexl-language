//! VEXL Runtime - Vector operations, generators, and cooperative scheduler

pub mod vector;
pub mod generator;
pub mod scheduler;
pub mod cache;
pub mod gc;
pub mod ffi; // Export FFI module
pub mod ffi_bridge; // Type-safe FFI bridge
pub mod interpreter;
pub mod context;

// Import FFI functions for bridge registration
use crate::ffi::*;

// Parallel execution functions for vexl-core integration
mod parallel_ops;

pub use vector::{Vector, VectorHeader};
pub use scheduler::CooperativeScheduler;
pub use parallel_ops::{ParallelOps};

// JIT execution support
use std::cell::RefCell;
thread_local! {
    static JIT_RUNTIME: RefCell<Option<JitRuntime>> = RefCell::new(None);
}

/// JIT runtime for on-demand expression evaluation
pub struct JitRuntime {
    context: inkwell::context::Context,
    jit_engine: vexl_codegen::JitEngine<'static>,
}

impl JitRuntime {
    /// Initialize the JIT runtime
    pub fn new() -> Result<Self, String> {
        let context = inkwell::context::Context::create();

        // Create JIT engine - we need to transmute the lifetime since we store it in TLS
        let jit_engine = unsafe {
            std::mem::transmute(vexl_codegen::JitEngine::new(&context)?)
        };

        Ok(Self {
            context,
            jit_engine,
        })
    }

    /// Evaluate a VIR module and return the result
    pub fn eval_module(&mut self, vir_module: &vexl_ir::VirModule) -> Result<i64, String> {
        self.jit_engine.compile_and_execute(vir_module)
    }
}

/// Initialize JIT runtime for expression evaluation
pub fn init_jit_runtime() -> Result<(), String> {
    JIT_RUNTIME.with(|runtime| {
        if runtime.borrow().is_none() {
            *runtime.borrow_mut() = Some(JitRuntime::new()?);
        }
        Ok(())
    })
}

/// Evaluate an expression using JIT compilation
pub fn eval_expression(vir_module: &vexl_ir::VirModule) -> Result<i64, String> {
    JIT_RUNTIME.with(|runtime| {
        if let Some(ref mut jit_runtime) = *runtime.borrow_mut() {
            jit_runtime.eval_module(vir_module)
        } else {
            Err("JIT runtime not initialized. Call init_jit_runtime() first.".to_string())
        }
    })
}

/// Global FFI bridge instance for type-safe function calls
static mut FFI_BRIDGE: Option<ffi_bridge::FfiBridge> = None;

/// Get or initialize the FFI bridge
fn get_ffi_bridge() -> &'static mut ffi_bridge::FfiBridge {
    unsafe {
        if FFI_BRIDGE.is_none() {
            FFI_BRIDGE = Some(ffi_bridge::FfiBridge::new());
            let bridge = FFI_BRIDGE.as_mut().unwrap();

            // Register runtime functions with proper signatures using raw pointers
            // For now, we'll register them in the symbol resolver instead of the bridge
            // The bridge will be used for type-safe calls once the infrastructure is complete

            // Vector operations - registered in symbol resolver
            // Math functions - registered in symbol resolver
            // String operations - registered in symbol resolver
            // IO operations - registered in symbol resolver
        }
        FFI_BRIDGE.as_mut().unwrap()
    }
}

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
    {
        // GPU acceleration temporarily disabled
        // TODO: Implement GPU acceleration module
    }

    // Initialize garbage collector
    gc::init_gc();

    // Initialize generator cache system
    cache::init_cache_system();

    // Initialize FFI bridge with runtime functions
    let _ = get_ffi_bridge();
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
    let bridge = get_ffi_bridge();

    // Use FFI bridge for type-safe function calls
    // For now, still using direct transmute but this will be replaced
    // with proper callback registration system
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
