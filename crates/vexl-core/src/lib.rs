//! VEXL Core - Universal Vector Type System
//!
//! This crate implements the foundational types for VEXL:
//! - `Vector1<T>` - 1D vector type
//! - `Vector2<T>` - 2D vector type (matrices)
//! - Parallel execution framework
//! - Generator trait and implementations
//! - Memory management and GC system
//! - Effect type system

pub mod vector;
pub mod generator;
pub mod memory;
pub mod effect;
pub mod parallel;

// Note: Vector types not yet implemented - placeholder for future implementation
// pub use vector::{
//     Vector1, Vector2, VectorError, VectorResult,
//     ComputeConfig, set_compute_config
// };
pub use generator::{
    Generator, TieredCache, CacheStats, CachedGenerator
};
pub use memory::{
    MemoryManager, GcHandle, MemoryStats, init_memory_manager
};
pub use effect::Effect;
pub use parallel::{
    ParallelRuntime, set_parallel_runtime, execute_parallel_map,
    execute_parallel_filter, execute_parallel_reduce
};
