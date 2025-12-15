//! Performance Benchmarking Suite for VEXL GPU
//!
//! This module provides comprehensive performance benchmarking across different
//! workloads, backends, and optimization levels with detailed logging and metrics.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use vexl_gpu::*;

/// Performance benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub backend_name: String,
    pub operation: String,
    pub workload_size: usize,
    pub iterations: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub throughput: f64, // operations/second
    pub memory_bandwidth: f64, // bytes/second
    pub efficiency_metrics: HashMap<String, f64>,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub min_workload_size: usize,
    pub max_workload_size: usize,
    pub iterations_per_workload: usize,
    pub warmup_iterations: usize,
    pub backends_to_test: Vec<String>,
    pub safety_limits: SafetyLimits,
}

#[derive(Debug, Clone)]
pub struct SafetyLimits {
    pub max_execution_time: Duration,
    pub max_memory_usage: usize,
    pub temperature_threshold: f32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            min_workload_size: 1000,
            max_workload_size: 1000000,
            iterations_per_workload: 100,
            warmup_iterations: 10,
            backends_to_test: vec!["all".to_string()],
            safety_limits: SafetyLimits {
                max_execution_time: Duration::from_secs(300), // 5 minutes
                max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
                temperature_threshold: 85.0,
            },
        }
    }
}

/// Comprehensive performance benchmarking suite
pub struct PerformanceBenchmarkSuite {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl PerformanceBenchmarkSuite {
    /// Create new benchmark suite with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            results: Vec::new(),
        }
    }

    /// Create benchmark suite with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run comprehensive performance benchmarks
    pub fn run_full_benchmarks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting VEXL GPU Performance Benchmark Suite");
        println!("═══════════════════════════════════════════════════════════════");

        let backends = self.get_available_backends()?;
        println!("🎯 Testing Backends: {:?}", backends.iter().map(|b| b.name()).collect::<Vec<_>>());

        // Run all benchmark categories
        self.benchmark_vector_operations(&backends)?;
        self.benchmark_memory_operations(&backends)?;
        self.benchmark_concurrent_workloads(&backends)?;
        self.benchmark_scalability(&backends)?;

        self.print_benchmark_summary();
        Ok(())
    }

    /// Get available GPU backends for testing
    fn get_available_backends(&self) -> Result<Vec<Box<dyn GpuBackend>>, Box<dyn std::error::Error>> {
        let mut backends = Vec::new();

        if self.config.backends_to_test.contains(&"all".to_string()) {
            #[cfg(feature = "cuda")]
            {
                if let Ok(backend) = cuda::CudaBackend::try_new() {
                    backends.push(Box::new(backend) as Box<dyn GpuBackend>);
                }
            }

            #[cfg(feature = "vulkan")]
            {
                if let Ok(backend) = vulkan::VulkanBackend::try_new() {
                    backends.push(Box::new(backend) as Box<dyn GpuBackend>);
                }
            }

            #[cfg(feature = "opencl")]
            {
                if let Ok(backend) = opencl::OpenCLBackend::try_new() {
                    backends.push(Box::new(backend) as Box<dyn GpuBackend>);
                }
            }
        } else {
            // Specific backends requested
            for backend_name in &self.config.backends_to_test {
                match backend_name.as_str() {
                    #[cfg(feature = "cuda")]
                    "cuda" => {
                        if let Ok(backend) = cuda::CudaBackend::try_new() {
                            backends.push(Box::new(backend) as Box<dyn GpuBackend>);
                        }
                    }
                    #[cfg(feature = "vulkan")]
                    "vulkan" => {
                        if let Ok(backend) = vulkan::VulkanBackend::try_new() {
                            backends.push(Box::new(backend) as Box<dyn GpuBackend>);
                        }
                    }
                    #[cfg(feature = "opencl")]
                    "opencl" => {
                        if let Ok(backend) = opencl::OpenCLBackend::try_new() {
                            backends.push(Box::new(backend) as Box<dyn GpuBackend>);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Always include CPU fallback for comparison
        backends.push(Box::new(backend::CpuFallbackBackend::new()));

        Ok(backends)
    }

    /// Benchmark vector operations across different backends and sizes
    fn benchmark_vector_operations(&mut self, backends: &[Box<dyn GpuBackend>]) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔢 Benchmarking Vector Operations");

        let operations = vec![
            ("vector_addition", "Vector Addition"),
            ("vector_multiplication", "Vector Multiplication"),
            ("vector_map", "Vector Map/Transform"),
        ];

        let workload_sizes = self.generate_workload_sizes();

        for backend in backends {
            for (op_key, op_name) in &operations {
                for &size in &workload_sizes {
                    let result = self.benchmark_single_operation(
                        backend.as_ref(),
                        op_key,
                        op_name,
                        size,
                    )?;

                    if let Some(result) = result {
                        self.results.push(result);
                    }
                }
            }
        }

        Ok(())
    }

    /// Benchmark single vector operation
    fn benchmark_single_operation(
        &self,
        backend: &dyn GpuBackend,
        op_key: &str,
        op_name: &str,
        size: usize,
    ) -> Result<Option<BenchmarkResult>, Box<dyn std::error::Error>> {
        let mut gpu_ops = GpuVectorOps::new(Box::new(backend.clone()));

        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            let a = vec![1.0f32; size];
            let b = vec![2.0f32; size];
            let _ = gpu_ops.add_vectors(&a, &b);
        }

        // Benchmark phase
        let mut total_time = Duration::default();
        let iterations = self.config.iterations_per_workload;

        for _ in 0..iterations {
            let a = vec![1.0f32; size];
            let b = vec![2.0f32; size];

            let start = Instant::now();
            let result = match op_key {
                "vector_addition" => gpu_ops.add_vectors(&a, &b),
                "vector_multiplication" => gpu_ops.mul_vectors(&a, &b),
                "vector_map" => gpu_ops.map_vector(&a, |x| x * 2.0),
                _ => return Ok(None),
            };
            let elapsed = start.elapsed();

            // Check for errors
            if result.is_err() {
                return Ok(None); // Skip failed operations
            }

            total_time += elapsed;

            // Safety check: execution time
            if elapsed > self.config.safety_limits.max_execution_time {
                println!("⚠️ Safety violation: Operation took too long ({:?})", elapsed);
                return Ok(None);
            }
        }

        let avg_time = total_time / iterations as u32;
        let throughput = size as f64 / avg_time.as_secs_f64();
        let memory_bandwidth = (size * std::mem::size_of::<f32>() * 3) as f64 / avg_time.as_secs_f64(); // read 2, write 1

        let mut efficiency_metrics = HashMap::new();
        efficiency_metrics.insert("throughput_ops_sec".to_string(), throughput);
        efficiency_metrics.insert("memory_bandwidth_mbps".to_string(), memory_bandwidth / (1024.0 * 1024.0));

        Ok(Some(BenchmarkResult {
            backend_name: backend.name().to_string(),
            operation: op_name.to_string(),
            workload_size: size,
            iterations,
            total_time,
            avg_time,
            throughput,
            memory_bandwidth,
            efficiency_metrics,
        }))
    }

    /// Benchmark memory operations
    fn benchmark_memory_operations(&mut self, backends: &[Box<dyn GpuBackend>]) -> Result<(), Box<dyn std::error::Error>> {
        println!("💾 Benchmarking Memory Operations");

        let memory_sizes = vec![
            1024,        // 1KB
            1024 * 64,   // 64KB
            1024 * 1024, // 1MB
            1024 * 1024 * 64, // 64MB
        ];

        for backend in backends {
            for &size in &memory_sizes {
                let result = self.benchmark_memory_operation(backend.as_ref(), size)?;
                if let Some(result) = result {
                    self.results.push(result);
                }
            }
        }

        Ok(())
    }

    /// Benchmark single memory operation
    fn benchmark_memory_operation(
        &self,
        backend: &dyn GpuBackend,
        size: usize,
    ) -> Result<Option<BenchmarkResult>, Box<dyn std::error::Error>> {
        let mut total_time = Duration::default();
        let iterations = self.config.iterations_per_workload / 10; // Fewer iterations for memory ops

        for _ in 0..iterations {
            let start = Instant::now();
            let buffer = backend.allocate(size)?;
            // Simulate some work with the buffer
            let _data = backend.read_back(&buffer, size.min(1024))?;
            let elapsed = start.elapsed();

            total_time += elapsed;
        }

        let avg_time = total_time / iterations as u32;
        let throughput = size as f64 / avg_time.as_secs_f64();
        let memory_bandwidth = size as f64 / avg_time.as_secs_f64();

        let mut efficiency_metrics = HashMap::new();
        efficiency_metrics.insert("allocation_throughput".to_string(), throughput);
        efficiency_metrics.insert("memory_bandwidth_mbps".to_string(), memory_bandwidth / (1024.0 * 1024.0));

        Ok(Some(BenchmarkResult {
            backend_name: backend.name().to_string(),
            operation: "Memory Allocation".to_string(),
            workload_size: size,
            iterations,
            total_time,
            avg_time,
            throughput,
            memory_bandwidth,
            efficiency_metrics,
        }))
    }

    /// Benchmark concurrent workloads
    fn benchmark_concurrent_workloads(&mut self, backends: &[Box<dyn GpuBackend>]) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔄 Benchmarking Concurrent Workloads");

        let concurrent_tasks = 4;
        let workload_size = 100000;

        for backend in backends {
            let result = self.benchmark_concurrent_operations(backend.as_ref(), concurrent_tasks, workload_size)?;
            if let Some(result) = result {
                self.results.push(result);
            }
        }

        Ok(())
    }

    /// Benchmark concurrent operations
    fn benchmark_concurrent_operations(
        &self,
        backend: &dyn GpuBackend,
        num_tasks: usize,
        workload_size: usize,
    ) -> Result<Option<BenchmarkResult>, Box<dyn std::error::Error>> {
        let start = Instant::now();

        let mut handles = Vec::new();

        for _ in 0..num_tasks {
            let backend_clone = Box::new(backend.clone());
            let handle = std::thread::spawn(move || {
                let mut gpu_ops = GpuVectorOps::new(backend_clone);
                let a = vec![1.0f32; workload_size];
                let b = vec![2.0f32; workload_size];

                let task_start = Instant::now();
                let _ = gpu_ops.add_vectors(&a, &b);
                task_start.elapsed()
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut total_task_time = Duration::default();
        for handle in handles {
            match handle.join() {
                Ok(duration) => total_task_time += duration,
                Err(_) => return Ok(None), // Skip if thread failed
            }
        }

        let total_time = start.elapsed();
        let avg_task_time = total_task_time / num_tasks as u32;

        let mut efficiency_metrics = HashMap::new();
        efficiency_metrics.insert("total_wall_time_ms".to_string(), total_time.as_millis() as f64);
        efficiency_metrics.insert("avg_task_time_ms".to_string(), avg_task_time.as_millis() as f64);
        efficiency_metrics.insert("parallel_efficiency".to_string(), total_task_time.as_secs_f64() / total_time.as_secs_f64());

        Ok(Some(BenchmarkResult {
            backend_name: backend.name().to_string(),
            operation: "Concurrent Operations".to_string(),
            workload_size: workload_size * num_tasks,
            iterations: num_tasks,
            total_time,
            avg_time: avg_task_time,
            throughput: (workload_size * num_tasks) as f64 / total_time.as_secs_f64(),
            memory_bandwidth: 0.0,
            efficiency_metrics,
        }))
    }

    /// Benchmark scalability across different workload sizes
    fn benchmark_scalability(&mut self, backends: &[Box<dyn GpuBackend>]) -> Result<(), Box<dyn std::error::Error>> {
        println!("📈 Benchmarking Scalability");

        let scalability_sizes = vec![
            1000, 10000, 100000, 1000000, 10000000
        ];

        for backend in backends {
            for &size in &scalability_sizes {
                // Skip very large sizes for CPU backend to avoid excessive test time
                if backend.name() == "CPU" && size > 1000000 {
                    continue;
                }

                let result = self.benchmark_scalability_operation(backend.as_ref(), size)?;
                if let Some(result) = result {
                    self.results.push(result);
                }
            }
        }

        Ok(())
    }

    /// Benchmark single scalability operation
    fn benchmark_scalability_operation(
        &self,
        backend: &dyn GpuBackend,
        size: usize,
    ) -> Result<Option<BenchmarkResult>, Box<dyn std::error::Error>> {
        let mut gpu_ops = GpuVectorOps::new(Box::new(backend.clone()));

        // Use fewer iterations for larger workloads
        let iterations = if size > 1000000 { 5 } else { 10 };

        let mut total_time = Duration::default();

        for _ in 0..iterations {
            let a = (0..size).map(|x| x as f32).collect::<Vec<_>>();
            let b = (0..size).map(|x| (x * 2) as f32).collect::<Vec<_>>();

            let start = Instant::now();
            let _ = gpu_ops.add_vectors(&a, &b);
            total_time += start.elapsed();
        }

        let avg_time = total_time / iterations as u32;
        let throughput = size as f64 / avg_time.as_secs_f64();

        let mut efficiency_metrics = HashMap::new();
        efficiency_metrics.insert("operations_per_second".to_string(), throughput);
        efficiency_metrics.insert("efficiency_score".to_string(),
            if size > 100000 { throughput / (size as f64).log10() } else { throughput });

        Ok(Some(BenchmarkResult {
            backend_name: backend.name().to_string(),
            operation: "Scalability Test".to_string(),
            workload_size: size,
            iterations,
            total_time,
            avg_time,
            throughput,
            memory_bandwidth: (size * std::mem::size_of::<f32>() * 3) as f64 / avg_time.as_secs_f64(),
            efficiency_metrics,
        }))
    }

    /// Generate workload sizes for benchmarking
    fn generate_workload_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut size = self.config.min_workload_size;

        while size <= self.config.max_workload_size {
            sizes.push(size);
            size *= 10; // Geometric progression
        }

        sizes
    }

    /// Print comprehensive benchmark summary
    fn print_benchmark_summary(&self) {
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("📊 VEXL GPU Performance Benchmark Suite - RESULTS");
        println!("═══════════════════════════════════════════════════════════════");

        if self.results.is_empty() {
            println!("❌ No benchmark results available");
            return;
        }

        // Group results by operation
        let mut operation_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();

        for result in &self.results {
            operation_groups.entry(result.operation.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        // Print results for each operation
        for (operation, results) in operation_groups {
            println!("\n🎯 Operation: {}", operation);

            // Group by backend
            let mut backend_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();

            for result in results {
                backend_groups.entry(result.backend_name.clone())
                    .or_insert_with(Vec::new)
                    .push(result);
            }

            for (backend, backend_results) in backend_groups {
                println!("   🔧 Backend: {}", backend);

                // Sort by workload size
                let mut sorted_results = backend_results.clone();
                sorted_results.sort_by_key(|r| r.workload_size);

                for result in sorted_results {
                    println!("      📏 Size: {:>8} | Time: {:>8.2}ms | Throughput: {:>10.0} ops/sec | Bandwidth: {:>8.2} MB/s",
                        result.workload_size,
                        result.avg_time.as_millis(),
                        result.throughput,
                        result.memory_bandwidth / (1024.0 * 1024.0)
                    );

                    // Print additional efficiency metrics
                    for (metric, value) in &result.efficiency_metrics {
                        if !metric.contains("throughput") && !metric.contains("bandwidth") {
                            println!("         {}: {:.2}", metric, value);
                        }
                    }
                }
            }
        }

        // Performance analysis
        println!("\n📈 Performance Analysis:");

        // Find best performing backend for each operation
        for (operation, results) in &operation_groups {
            if let Some(best_result) = results.iter()
                .filter(|r| r.workload_size >= 100000) // Only consider meaningful workloads
                .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap()) {

                println!("   🏆 {}: {} achieves {:.0} ops/sec at size {}",
                    operation,
                    best_result.backend_name,
                    best_result.throughput,
                    best_result.workload_size
                );
            }
        }

        // Scalability analysis
        if let Some(scalability_results) = operation_groups.get("Scalability Test") {
            println!("\n📈 Scalability Analysis:");
            for backend in ["CUDA", "Vulkan", "OpenCL", "CPU"] {
                let backend_results: Vec<_> = scalability_results.iter()
                    .filter(|r| r.backend_name == backend)
                    .collect();

                if backend_results.len() >= 2 {
                    let small = backend_results.iter().min_by_key(|r| r.workload_size).unwrap();
                    let large = backend_results.iter().max_by_key(|r| r.workload_size).unwrap();

                    let speedup = large.throughput / small.throughput;
                    println!("   📊 {}: {:.2}x speedup from {} to {} elements",
                        backend, speedup, small.workload_size, large.workload_size);
                }
            }
        }

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("🏁 Performance Benchmarking Complete - {} Results Recorded", self.results.len());
        println!("═══════════════════════════════════════════════════════════════");
    }
}

/// Run performance benchmarks with default configuration
pub fn run_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    let mut suite = PerformanceBenchmarkSuite::new();
    suite.run_full_benchmarks()
}

/// Run performance benchmarks with custom configuration
pub fn run_performance_benchmarks_with_config(config: BenchmarkConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut suite = PerformanceBenchmarkSuite::with_config(config);
    suite.run_full_benchmarks()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = PerformanceBenchmarkSuite::new();
        assert_eq!(suite.results.len(), 0);
        assert!(suite.config.max_workload_size > suite.config.min_workload_size);
    }

    #[test]
    fn test_benchmark_config_validation() {
        let config = BenchmarkConfig::default();
        assert!(config.iterations_per_workload > 0);
        assert!(config.warmup_iterations > 0);
        assert!(config.safety_limits.max_execution_time > Duration::default());
    }

    #[test]
    fn test_workload_size_generation() {
        let suite = PerformanceBenchmarkSuite::new();
        let sizes = suite.generate_workload_sizes();
        assert!(!sizes.is_empty());
        assert!(sizes.windows(2).all(|w| w[0] < w[1])); // Should be monotonically increasing
    }
}
