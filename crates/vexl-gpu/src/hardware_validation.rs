//! Hardware Validation Suite for VEXL GPU
//!
//! This module provides comprehensive hardware validation testing
//! to ensure GPU functionality works correctly across different scenarios.

use std::time::{Duration, Instant};
use crate::*;

/// Run hardware validation tests
pub fn run_hardware_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Running Hardware Validation Suite");
    println!("═══════════════════════════════════════");

    let mut passed_tests = 0;
    let mut total_tests = 0;

    // Test 1: Backend Initialization
    total_tests += 1;
    match test_backend_initialization() {
        Ok(_) => {
            passed_tests += 1;
            println!("✅ Backend initialization: PASSED");
        }
        Err(e) => {
            println!("❌ Backend initialization: FAILED - {}", e);
        }
    }

    // Test 2: Memory Management
    total_tests += 1;
    match test_memory_management() {
        Ok(_) => {
            passed_tests += 1;
            println!("✅ Memory management: PASSED");
        }
        Err(e) => {
            println!("❌ Memory management: FAILED - {}", e);
        }
    }

    // Test 3: Vector Operations
    total_tests += 1;
    match test_vector_operations() {
        Ok(_) => {
            passed_tests += 1;
            println!("✅ Vector operations: PASSED");
        }
        Err(e) => {
            println!("❌ Vector operations: FAILED - {}", e);
        }
    }

    // Test 4: Error Handling
    total_tests += 1;
    match test_error_handling() {
        Ok(_) => {
            passed_tests += 1;
            println!("✅ Error handling: PASSED");
        }
        Err(e) => {
            println!("❌ Error handling: FAILED - {}", e);
        }
    }

    println!();
    println!("📊 Hardware Validation Results:");
    println!("   Passed: {}/{}", passed_tests, total_tests);
    println!("   Success Rate: {:.1}%", (passed_tests as f64 / total_tests as f64) * 100.0);

    if passed_tests == total_tests {
        println!("🎉 All hardware validation tests PASSED!");
        Ok(())
    } else {
        Err(format!("Hardware validation failed: {}/{} tests passed", passed_tests, total_tests).into())
    }
}

/// Test backend initialization
fn test_backend_initialization() -> Result<(), Box<dyn std::error::Error>> {
    let backend = init_best_backend();

    // Verify backend has a name
    if backend.name().is_empty() {
        return Err("Backend name is empty".into());
    }

    // Test basic buffer allocation
    let buffer = backend.allocate(1024)?;
    if buffer.size != 1024 {
        return Err(format!("Buffer size mismatch: expected 1024, got {}", buffer.size).into());
    }

    Ok(())
}

/// Test memory management capabilities
fn test_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    let backend = init_best_backend();

    // Test various buffer sizes
    let test_sizes = [64, 1024, 65536, 1048576]; // 64B, 1KB, 64KB, 1MB

    for &size in &test_sizes {
        let buffer = backend.allocate(size)?;
        if buffer.size != size {
            return Err(format!("Buffer allocation size mismatch: expected {}, got {}", size, buffer.size).into());
        }
    }

    Ok(())
}

/// Test vector operations
fn test_vector_operations() -> Result<(), Box<dyn std::error::Error>> {
    let gpu_ops = GpuVectorOps::new()?;

    // Test vector addition
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
    let result = gpu_ops.add_vectors(&a, &b)?;

    let expected = vec![6.0f32, 6.0, 6.0, 6.0, 6.0];
    if result != expected {
        return Err(format!("Vector addition result mismatch: expected {:?}, got {:?}", expected, result).into());
    }

    // Test vector map
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mapped = gpu_ops.map_vector(&input, |x| x * 3.0)?;
    let expected_mapped = vec![3.0f32, 6.0, 9.0, 12.0, 15.0];

    if mapped != expected_mapped {
        return Err(format!("Vector map result mismatch: expected {:?}, got {:?}", expected_mapped, mapped).into());
    }

    Ok(())
}

/// Test error handling
fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Use CPU fallback backend to avoid GPU hardware issues in tests
    let backend = crate::backend::CpuFallbackBackend::new();

    // Test invalid buffer allocation (should fail gracefully)
    match backend.allocate(0) {
        Ok(_) => return Err("Should have rejected zero-sized buffer allocation".into()),
        Err(_) => {} // Expected
    }

    // Test invalid buffer allocation (very large)
    match backend.allocate(usize::MAX) {
        Ok(_) => return Err("Should have rejected extremely large buffer allocation".into()),
        Err(_) => {} // Expected
    }

    Ok(())
}
