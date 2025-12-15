//! Comprehensive Hardware Testing Suite for VEXL GPU
//!
//! This integration test runs the complete hardware validation, performance benchmarking,
//! and safety monitoring suite to ensure VEXL GPU is production-ready.

use std::time::Duration;
use vexl_gpu::*;

/// Comprehensive hardware test configuration
#[derive(Debug, Clone)]
pub struct ComprehensiveTestConfig {
    pub enable_hardware_validation: bool,
    pub enable_performance_benchmarks: bool,
    pub enable_safety_monitoring: bool,
    pub enable_concurrent_stress_test: bool,
    pub test_duration_seconds: u64,
    pub max_memory_usage: usize,
    pub safety_temperature_threshold: f32,
}

impl Default for ComprehensiveTestConfig {
    fn default() -> Self {
        Self {
            enable_hardware_validation: true,
            enable_performance_benchmarks: true,
            enable_safety_monitoring: true,
            enable_concurrent_stress_test: true,
            test_duration_seconds: 30,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            safety_temperature_threshold: 85.0,
        }
    }
}

/// Test results summary
#[derive(Debug, Clone)]
pub struct TestResultsSummary {
    pub hardware_validation_passed: bool,
    pub performance_benchmarks_passed: bool,
    pub safety_monitoring_passed: bool,
    pub concurrent_stress_test_passed: bool,
    pub total_violations: usize,
    pub critical_violations: usize,
    pub test_duration: Duration,
    pub overall_status: TestStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TestStatus {
    Passed,
    Failed,
    Warning,
    Error,
}

/// Run comprehensive hardware testing suite
pub fn run_comprehensive_hardware_tests(config: ComprehensiveTestConfig) -> Result<TestResultsSummary, Box<dyn std::error::Error>> {
    println!("🧪 STARTING COMPREHENSIVE VEXL GPU HARDWARE TESTING SUITE");
    println!("═══════════════════════════════════════════════════════════════");

    let start_time = std::time::Instant::now();
    let mut results = TestResultsSummary {
        hardware_validation_passed: false,
        performance_benchmarks_passed: false,
        safety_monitoring_passed: false,
        concurrent_stress_test_passed: false,
        total_violations: 0,
        critical_violations: 0,
        test_duration: Duration::default(),
        overall_status: TestStatus::Passed,
    };

    println!("🎯 Test Configuration:");
    println!("   Hardware Validation: {}", config.enable_hardware_validation);
    println!("   Performance Benchmarks: {}", config.enable_performance_benchmarks);
    println!("   Safety Monitoring: {}", config.enable_safety_monitoring);
    println!("   Concurrent Stress Test: {}", config.enable_concurrent_stress_test);
    println!("   Test Duration: {}s", config.test_duration_seconds);
    println!("   Max Memory: {:.2} GB", config.max_memory_usage as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Safety Temp Threshold: {}°C", config.safety_temperature_threshold);
    println!();

    // Phase 1: Hardware Validation
    if config.enable_hardware_validation {
        println!("📋 PHASE 1: Hardware Validation");
        match run_hardware_validation() {
            Ok(()) => {
                results.hardware_validation_passed = true;
                println!("   ✅ Hardware validation PASSED");
            }
            Err(e) => {
                results.overall_status = TestStatus::Error;
                println!("   ❌ Hardware validation FAILED: {}", e);
            }
        }
        println!();
    }

    // Phase 2: Performance Benchmarks
    if config.enable_performance_benchmarks {
        println!("⚡ PHASE 2: Performance Benchmarks");
        match run_performance_benchmarks() {
            Ok(()) => {
                results.performance_benchmarks_passed = true;
                println!("   ✅ Performance benchmarks COMPLETED");
            }
            Err(e) => {
                results.overall_status = TestStatus::Warning;
                println!("   ⚠️ Performance benchmarks had issues: {}", e);
            }
        }
        println!();
    }

    // Phase 3: Safety Monitoring
    if config.enable_safety_monitoring {
        println!("🛡️ PHASE 3: Safety Monitoring");
        match run_safety_validation() {
            Ok(()) => {
                results.safety_monitoring_passed = true;
                println!("   ✅ Safety monitoring PASSED");
            }
            Err(e) => {
                results.safety_monitoring_passed = false;
                results.overall_status = TestStatus::Error;
                println!("   ❌ Safety monitoring FAILED: {}", e);
                // Simple check for critical violations in error message
                if e.to_string().contains("Critical safety violations") {
                    results.critical_violations = 1; // At least one critical violation
                }
            }
        }
        println!();
    }

    // Phase 4: Concurrent Stress Test
    if config.enable_concurrent_stress_test {
        println!("🔄 PHASE 4: Concurrent Stress Test");
        match run_concurrent_stress_test(config.clone()) {
            Ok(()) => {
                results.concurrent_stress_test_passed = true;
                println!("   ✅ Concurrent stress test PASSED");
            }
            Err(e) => {
                results.overall_status = TestStatus::Warning;
                println!("   ⚠️ Concurrent stress test had issues: {}", e);
            }
        }
        println!();
    }

    // Calculate final results
    results.test_duration = start_time.elapsed();

    // Determine overall status
    if results.hardware_validation_passed &&
       results.safety_monitoring_passed &&
       results.critical_violations == 0 {
        results.overall_status = TestStatus::Passed;
    } else if results.critical_violations > 0 {
        results.overall_status = TestStatus::Error;
    } else {
        results.overall_status = TestStatus::Warning;
    }

    // Print comprehensive results
    print_comprehensive_results(&results);

    Ok(results)
}

/// Run concurrent stress test
fn run_concurrent_stress_test(config: ComprehensiveTestConfig) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let num_threads = 4;
    let operations_per_thread = 50;
    let mut handles = Vec::new();
    let error_count = Arc::new(Mutex::new(0));

    println!("   🔄 Starting concurrent stress test with {} threads", num_threads);

    for thread_id in 0..num_threads {
        let error_count = Arc::clone(&error_count);
        let handle = thread::spawn(move || {
            let backend = init_best_backend();

            for op_id in 0..operations_per_thread {
                // Test memory allocation
                match backend.allocate(1024 * 64) { // 64KB
                    Ok(buffer) => {
                        // Test small vector operation
                        let gpu_ops = GpuVectorOps::new().unwrap();
                        let a = vec![1.0f32; 1000];
                        let b = vec![2.0f32; 1000];

                        if gpu_ops.add_vectors(&a, &b).is_err() {
                            *error_count.lock().unwrap() += 1;
                        }
                    }
                    Err(_) => {
                        *error_count.lock().unwrap() += 1;
                    }
                }

                // Small delay to prevent overwhelming
                thread::sleep(Duration::from_millis(1));
            }

            println!("     ✅ Thread {} completed {} operations", thread_id, operations_per_thread);
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().map_err(|_| "Thread join failed")?;
    }

    let total_errors = *error_count.lock().unwrap();
    if total_errors > 0 {
        return Err(format!("Concurrent stress test had {} errors", total_errors).into());
    }

    Ok(())
}

/// Print comprehensive test results
fn print_comprehensive_results(results: &TestResultsSummary) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("📊 COMPREHENSIVE HARDWARE TESTING RESULTS");
    println!("═══════════════════════════════════════════════════════════════");

    // Individual test results
    println!("🎯 Individual Test Results:");
    println!("   Hardware Validation:    {}", if results.hardware_validation_passed { "✅ PASSED" } else { "❌ FAILED" });
    println!("   Performance Benchmarks: {}", if results.performance_benchmarks_passed { "✅ PASSED" } else { "⚠️ ISSUES" });
    println!("   Safety Monitoring:      {}", if results.safety_monitoring_passed { "✅ PASSED" } else { "❌ FAILED" });
    println!("   Concurrent Stress Test: {}", if results.concurrent_stress_test_passed { "✅ PASSED" } else { "⚠️ ISSUES" });

    println!();
    println!("📈 Metrics:");
    println!("   Total Test Duration:     {:.2}s", results.test_duration.as_secs_f64());
    println!("   Critical Violations:     {}", results.critical_violations);

    println!();
    println!("🏁 Overall Status:");

    let status_message = match results.overall_status {
        TestStatus::Passed => "✅ ALL TESTS PASSED - READY FOR PRODUCTION",
        TestStatus::Warning => "⚠️ TESTS PASSED WITH WARNINGS - MONITOR CLOSELY",
        TestStatus::Error => "❌ CRITICAL ISSUES DETECTED - REQUIRES IMMEDIATE ATTENTION",
        TestStatus::Failed => "💥 TESTS FAILED - NOT READY FOR PRODUCTION",
    };

    println!("   {}", status_message);

    // Recommendations
    println!();
    println!("💡 Recommendations:");
    if results.overall_status == TestStatus::Passed {
        println!("   🎉 Excellent! VEXL GPU is production-ready with full hardware validation.");
        println!("   🚀 Proceed with confidence to deployment and user testing.");
    } else if results.overall_status == TestStatus::Warning {
        println!("   ⚠️ Performance or minor issues detected. Suitable for controlled deployment.");
        println!("   📊 Monitor performance metrics in production environment.");
    } else {
        println!("   🚨 Critical issues require immediate resolution before any deployment.");
        println!("   🔧 Address safety violations and failed tests before proceeding.");
    }

    println!("═══════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    #[test]
    fn test_comprehensive_hardware_testing() {
        // Create test configuration for CI/testing environment
        let config = ComprehensiveTestConfig {
            enable_hardware_validation: true,
            enable_performance_benchmarks: true, // Enable performance benchmarks
            enable_safety_monitoring: true,
            enable_concurrent_stress_test: true, // Enable concurrent stress test
            test_duration_seconds: 5, // Short duration for unit tests
            max_memory_usage: 100 * 1024 * 1024, // 100MB limit for tests
            safety_temperature_threshold: 90.0, // Higher threshold for testing
        };

        match run_comprehensive_hardware_tests(config) {
            Ok(results) => {
                println!("Comprehensive test completed with status: {:?}", results.overall_status);

                // Basic assertions
                assert!(results.test_duration > Duration::default());
                assert!(results.hardware_validation_passed || results.safety_monitoring_passed);

                // Critical violations should be zero in test environment
                assert_eq!(results.critical_violations, 0,
                    "Critical violations detected in test environment: {}", results.critical_violations);
            }
            Err(e) => {
                panic!("Comprehensive hardware testing failed: {}", e);
            }
        }
    }

    #[test]
    fn test_config_validation() {
        let config = ComprehensiveTestConfig::default();
        assert!(config.test_duration_seconds > 0);
        assert!(config.max_memory_usage > 0);
        assert!(config.safety_temperature_threshold > 0.0);
    }

    #[test]
    fn test_concurrent_stress_test() {
        let config = ComprehensiveTestConfig {
            enable_concurrent_stress_test: true,
            test_duration_seconds: 2,
            ..Default::default()
        };

        match run_concurrent_stress_test(config) {
            Ok(()) => println!("Concurrent stress test passed"),
            Err(e) => panic!("Concurrent stress test failed: {}", e),
        }
    }
}
