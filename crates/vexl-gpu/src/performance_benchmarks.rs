//! Performance Benchmarking Suite for VEXL GPU
//!
//! This module provides comprehensive performance benchmarking across different
//! workloads, backends, and optimization levels.

use std::time::{Duration, Instant};
use crate::*;

/// Run performance benchmarks
pub fn run_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Running Performance Benchmark Suite");
    println!("═══════════════════════════════════════");

    let backends = get_available_backends()?;
    println!("🎯 Testing backends: {:?}", backends.iter().map(|b| b.name()).collect::<Vec<_>>());

    // Run vector operation benchmarks
    benchmark_vector_operations(&backends)?;

    // Run memory benchmarks
    benchmark_memory_operations(&backends)?;

    println!("🎉 Performance benchmarking completed successfully!");
    Ok(())
}

/// Get available GPU backends for benchmarking
fn get_available_backends() -> Result<Vec<Box<dyn GpuBackend>>, Box<dyn std::error::Error>> {
    let mut backends = Vec::new();

    // Try Vulkan backend
    #[cfg(feature = "vulkan")]
    {
        if let Ok(backend) = crate::vulkan::VulkanBackend::try_new() {
            backends.push(Box::new(backend) as Box<dyn GpuBackend>);
        }
    }

    // Try CUDA backend
    #[cfg(feature = "cuda")]
    {
        if let Ok(backend) = crate::cuda::CudaBackend::try_new() {
            backends.push(Box::new(backend) as Box<dyn GpuBackend>);
        }
    }

    // Try OpenCL backend
    #[cfg(feature = "opencl")]
    {
        if let Ok(backend) = crate::opencl::OpenCLBackend::try_new() {
            backends.push(Box::new(backend) as Box<dyn GpuBackend>);
        }
    }

    // Always include CPU fallback for comparison
    backends.push(Box::new(crate::backend::CpuFallbackBackend::new()));

    Ok(backends)
}

/// Benchmark vector operations across different backends
fn benchmark_vector_operations(backends: &[Box<dyn GpuBackend>]) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Benchmarking Vector Operations");

    let test_sizes = [1000, 10000, 100000];
    let iterations = 10;

    for backend in backends {
        println!("   Backend: {}", backend.name());

        for &size in &test_sizes {
            let gpu_ops = GpuVectorOps::new()?;

            // Benchmark vector addition
            let mut total_time = Duration::default();
            for _ in 0..iterations {
                let a = vec![1.0f32; size];
                let b = vec![2.0f32; size];

                let start = Instant::now();
                let _ = gpu_ops.add_vectors(&a, &b);
                total_time += start.elapsed();
            }

            let avg_time = total_time / iterations as u32;
            let throughput = size as f64 / avg_time.as_secs_f64();

            println!("      Size {:>6}: {:>8.2}ms avg, {:>10.0} ops/sec",
                size, avg_time.as_millis(), throughput);
        }
    }

    Ok(())
}

/// Benchmark memory operations
fn benchmark_memory_operations(backends: &[Box<dyn GpuBackend>]) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Benchmarking Memory Operations");

    let test_sizes = [1024, 65536, 1048576]; // 1KB, 64KB, 1MB
    let iterations = 100;

    for backend in backends {
        println!("   Backend: {}", backend.name());

        for &size in &test_sizes {
            let mut total_time = Duration::default();

            for _ in 0..iterations {
                let start = Instant::now();
                let buffer = backend.allocate(size)?;
                let _data = backend.read_back(&buffer, size.min(1024))?;
                total_time += start.elapsed();
            }

            let avg_time = total_time / iterations as u32;
            println!("      Size {:>6}: {:>8.2}ms avg", size, avg_time.as_millis());
        }
    }

    Ok(())
}
