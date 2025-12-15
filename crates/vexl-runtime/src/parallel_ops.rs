//! Parallel Operations for VEXL Core Integration
//!
//! This module provides parallel execution functions that can be called
//! from vexl-core through function pointers, avoiding circular dependencies.

use std::sync::{Arc, Mutex};
use crate::scheduler::global_scheduler;

/// Parallel operations executor
pub struct ParallelOps;

impl ParallelOps {
    /// Execute map operation in parallel using cooperative scheduler
    pub fn map_parallel_i32(
        input: &[i32],
        map_fn: fn(&i32) -> i32,
    ) -> Vec<i32> {
        if input.is_empty() {
            return Vec::new();
        }

        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (input.len() / num_cpus.max(1)).max(1);
        let chunks: Vec<Vec<i32>> = input.chunks(chunk_size).map(|c| c.to_vec()).collect();

        let results = Arc::new(Mutex::new(vec![None; chunks.len()]));
        let results_clone = Arc::clone(&results);

        // Submit parallel tasks
        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let map_fn = map_fn;
            let results = Arc::clone(&results);

            global_scheduler().submit(move || {
                let chunk_result: Vec<i32> = chunk.iter().map(|x| map_fn(x)).collect();
                let mut results = results.lock().unwrap();
                results[chunk_idx] = Some(chunk_result);
            });
        }

        // Wait for completion and collect results
        loop {
            let results = results_clone.lock().unwrap();
            if results.iter().all(|r| r.is_some()) {
                let mut final_result = Vec::with_capacity(input.len());
                for chunk_result in results.iter() {
                    if let Some(chunk) = chunk_result {
                        final_result.extend_from_slice(chunk);
                    }
                }
                return final_result;
            }
            std::thread::yield_now();
        }
    }

    /// Execute filter operation in parallel
    pub fn filter_parallel_i32(
        input: &[i32],
        pred_fn: fn(&&i32) -> bool,
    ) -> Vec<i32> {
        if input.is_empty() {
            return Vec::new();
        }

        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (input.len() / num_cpus.max(1)).max(1);
        let chunks: Vec<Vec<i32>> = input.chunks(chunk_size).map(|c| c.to_vec()).collect();

        let results = Arc::new(Mutex::new(vec![None; chunks.len()]));
        let results_clone = Arc::clone(&results);

        // Submit parallel tasks
        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let pred_fn = pred_fn;
            let results = Arc::clone(&results);

            global_scheduler().submit(move || {
                let chunk_result: Vec<i32> = chunk.iter()
                    .filter(pred_fn)
                    .cloned()
                    .collect();
                let mut results = results.lock().unwrap();
                results[chunk_idx] = Some(chunk_result);
            });
        }

        // Wait for completion and collect results
        loop {
            let results = results_clone.lock().unwrap();
            if results.iter().all(|r| r.is_some()) {
                let mut final_result = Vec::new();
                for chunk_result in results.iter() {
                    if let Some(chunk) = chunk_result {
                        final_result.extend_from_slice(chunk);
                    }
                }
                return final_result;
            }
            std::thread::yield_now();
        }
    }

    /// Execute reduce operation in parallel
    pub fn reduce_parallel_i32(
        input: &[i32],
        init: i32,
        reduce_fn: fn(i32, i32) -> i32,
    ) -> i32 {
        if input.is_empty() {
            return init;
        }

        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (input.len() / num_cpus.max(1)).max(1);
        let chunks: Vec<Vec<i32>> = input.chunks(chunk_size).map(|c| c.to_vec()).collect();

        let partial_results = Arc::new(Mutex::new(vec![None; chunks.len()]));
        let partial_results_clone = Arc::clone(&partial_results);

        // Submit parallel partial reductions
        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let reduce_fn = reduce_fn;
            let partial_results = Arc::clone(&partial_results);

            global_scheduler().submit(move || {
                let partial = chunk.iter().fold(init, |acc, &x| reduce_fn(acc, x));
                let mut results = partial_results.lock().unwrap();
                results[chunk_idx] = Some(partial);
            });
        }

        // Wait for completion and combine partial results
        loop {
            let partial_results = partial_results_clone.lock().unwrap();
            if partial_results.iter().all(|r| r.is_some()) {
                let final_result = partial_results.iter()
                    .map(|r| r.unwrap())
                    .fold(init, reduce_fn);
                return final_result;
            }
            std::thread::yield_now();
        }
    }
}

/// Concrete implementation of ParallelRuntime for vexl-core integration
pub struct VexlParallelRuntime;

impl VexlParallelRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl vexl_core::ParallelRuntime for VexlParallelRuntime {
    fn execute_map_i32(&self, data: &[i32], f: fn(&i32) -> i32) -> Vec<i32> {
        ParallelOps::map_parallel_i32(data, f)
    }

    fn execute_filter_i32(&self, data: &[i32], pred: fn(&&i32) -> bool) -> Vec<i32> {
        ParallelOps::filter_parallel_i32(data, pred)
    }

    fn execute_reduce_i32(&self, data: &[i32], init: i32, f: fn(i32, i32) -> i32) -> i32 {
        ParallelOps::reduce_parallel_i32(data, init, f)
    }
}

/// Initialize vexl-core parallel runtime integration
pub fn init_vexl_core_integration() {
    let runtime = Box::new(VexlParallelRuntime::new());
    vexl_core::set_parallel_runtime(Box::leak(runtime));
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_core::ParallelRuntime;

    #[test]
    fn test_parallel_map_i32() {
        // Initialize scheduler for testing
        crate::scheduler::init_thread_pool();

        let input = vec![1, 2, 3, 4, 5];
        let result = ParallelOps::map_parallel_i32(&input, |x| x * 2);

        assert_eq!(result, vec![2, 4, 6, 8, 10]);

        crate::scheduler::shutdown_thread_pool();
    }

    #[test]
    fn test_parallel_filter_i32() {
        crate::scheduler::init_thread_pool();

        let input = vec![1, 2, 3, 4, 5, 6];
        let result = ParallelOps::filter_parallel_i32(&input, |&&x| x % 2 == 0);

        assert_eq!(result, vec![2, 4, 6]);

        crate::scheduler::shutdown_thread_pool();
    }

    #[test]
    fn test_parallel_reduce_i32() {
        crate::scheduler::init_thread_pool();

        let input = vec![1, 2, 3, 4, 5];
        let result = ParallelOps::reduce_parallel_i32(&input, 0, |a, b| a + b);

        assert_eq!(result, 15);

        crate::scheduler::shutdown_thread_pool();
    }

    #[test]
    fn test_vexl_parallel_runtime() {
        crate::scheduler::init_thread_pool();

        let runtime = VexlParallelRuntime::new();

        // Test map
        let data = vec![1, 2, 3, 4, 5];
        let result = runtime.execute_map_i32(&data, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);

        // Test filter
        let result = runtime.execute_filter_i32(&data, |&&x| x % 2 == 0);
        assert_eq!(result, vec![2, 4]);

        // Test reduce
        let result = runtime.execute_reduce_i32(&data, 0, |a, b| a + b);
        assert_eq!(result, 15);

        crate::scheduler::shutdown_thread_pool();
    }
}
