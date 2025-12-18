//! Parallel Runtime for VEXL Core
//!
//! This module provides functions for parallel execution
//! that can be implemented by external crates like vexl-runtime.

use std::sync::OnceLock;

/// Trait for parallel runtime implementations
pub trait ParallelRuntime: Send + Sync {
    /// Execute a parallel map operation
    fn execute_map_i32(&self, data: &[i32], f: fn(&i32) -> i32) -> Vec<i32>;

    /// Execute a parallel filter operation
    fn execute_filter_i32(&self, data: &[i32], pred: fn(&&i32) -> bool) -> Vec<i32>;

    /// Execute a parallel reduce operation
    fn execute_reduce_i32(&self, data: &[i32], init: i32, f: fn(i32, i32) -> i32) -> i32;
}

// Global parallel runtime instance
static PARALLEL_RUNTIME: OnceLock<&'static dyn ParallelRuntime> = OnceLock::new();

/// Set the parallel runtime for VEXL operations
pub fn set_parallel_runtime(runtime: &'static dyn ParallelRuntime) {
    PARALLEL_RUNTIME.set(runtime).ok(); // Ignore if already set
}

/// Get the current parallel runtime
pub fn get_parallel_runtime() -> Option<&'static dyn ParallelRuntime> {
    PARALLEL_RUNTIME.get().copied()
}

/// Execute parallel map operation using the global runtime
pub fn execute_parallel_map(data: &[i32], f: fn(&i32) -> i32) -> Vec<i32> {
    if let Some(runtime) = get_parallel_runtime() {
        runtime.execute_map_i32(data, f)
    } else {
        // Fallback to sequential execution
        data.iter().map(f).collect()
    }
}

/// Execute parallel filter operation using the global runtime
pub fn execute_parallel_filter(data: &[i32], pred: fn(&&i32) -> bool) -> Vec<i32> {
    if let Some(runtime) = get_parallel_runtime() {
        runtime.execute_filter_i32(data, pred)
    } else {
        // Fallback to sequential execution
        data.iter().filter(pred).cloned().collect()
    }
}

/// Execute parallel reduce operation using the global runtime
pub fn execute_parallel_reduce(data: &[i32], init: i32, f: fn(i32, i32) -> i32) -> i32 {
    if let Some(runtime) = get_parallel_runtime() {
        runtime.execute_reduce_i32(data, init, f)
    } else {
        // Fallback to sequential execution
        data.iter().fold(init, |acc, &x| f(acc, x))
    }
}
