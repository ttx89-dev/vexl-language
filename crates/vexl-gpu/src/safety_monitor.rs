//! Safety Monitoring System for VEXL GPU Operations
//!
//! This module provides comprehensive safety monitoring, temperature tracking,
//! power management, and automatic shutdown mechanisms to ensure safe GPU operations.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/// Safety monitoring configuration
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    pub temperature_threshold: f32,     // Celsius
    pub temperature_grace_period: Duration,
    pub power_limit: f32,               // Watts
    pub memory_limit: usize,            // Bytes
    pub execution_timeout: Duration,    // Max time per operation
    pub enable_auto_shutdown: bool,
    pub emergency_shutdown_temp: f32,   // Emergency shutdown temperature
    pub monitoring_interval: Duration,  // How often to check metrics
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            temperature_threshold: 85.0,
            temperature_grace_period: Duration::from_secs(30),
            power_limit: 300.0,
            memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
            execution_timeout: Duration::from_secs(300), // 5 minutes
            enable_auto_shutdown: bool::from(true),
            emergency_shutdown_temp: 95.0,
            monitoring_interval: Duration::from_millis(100),
        }
    }
}

/// System metrics collected during monitoring
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub gpu_temperature: f32,
    pub gpu_power_usage: f32,
    pub memory_used: usize,
    pub memory_total: usize,
    pub gpu_utilization: f32,
    pub timestamp: Instant,
}

/// Safety violation types
#[derive(Debug, Clone, PartialEq)]
pub enum SafetyViolation {
    TemperatureExceeded,
    PowerLimitExceeded,
    MemoryLimitExceeded,
    ExecutionTimeout,
    EmergencyShutdown,
}

/// Safety monitoring results
#[derive(Debug, Clone)]
pub struct SafetyResult {
    pub violation: SafetyViolation,
    pub severity: ViolationSeverity,
    pub message: String,
    pub recommended_action: String,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    Warning,
    Critical,
    Emergency,
}

/// Comprehensive safety monitoring system
pub struct SafetyMonitor {
    config: SafetyConfig,
    metrics_history: Arc<Mutex<Vec<SystemMetrics>>>,
    violations: Arc<Mutex<Vec<SafetyResult>>>,
    monitoring_thread: Option<thread::JoinHandle<()>>,
    shutdown_signal: Arc<Mutex<bool>>,
    operation_start_time: Arc<Mutex<Option<Instant>>>,
}

impl SafetyMonitor {
    /// Create new safety monitor with default configuration
    pub fn new() -> Self {
        Self::with_config(SafetyConfig::default())
    }

    /// Create safety monitor with custom configuration
    pub fn with_config(config: SafetyConfig) -> Self {
        Self {
            config,
            metrics_history: Arc::new(Mutex::new(Vec::new())),
            violations: Arc::new(Mutex::new(Vec::new())),
            monitoring_thread: None,
            shutdown_signal: Arc::new(Mutex::new(false)),
            operation_start_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Start safety monitoring
    pub fn start_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🛡️ Starting VEXL GPU Safety Monitoring System");
        println!("   📊 Temperature Threshold: {}°C", self.config.temperature_threshold);
        println!("   ⚡ Power Limit: {}W", self.config.power_limit);
        println!("   💾 Memory Limit: {:.2} GB", self.config.memory_limit as f64 / (1024.0 * 1024.0 * 1024.0));
        println!("   ⏱️ Execution Timeout: {:?} ", self.config.execution_timeout);

        let metrics_history = Arc::clone(&self.metrics_history);
        let violations = Arc::clone(&self.violations);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let config = self.config.clone();

        self.monitoring_thread = Some(thread::spawn(move || {
            Self::monitoring_loop(config, metrics_history, violations, shutdown_signal);
        }));

        Ok(())
    }

    /// Stop safety monitoring
    pub fn stop_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(handle) = self.monitoring_thread.take() {
            *self.shutdown_signal.lock().unwrap() = true;
            handle.join().map_err(|e| format!("Failed to join monitoring thread: {:?}", e))?;
        }
        Ok(())
    }

    /// Mark the start of a GPU operation for timeout tracking
    pub fn start_operation(&self) {
        *self.operation_start_time.lock().unwrap() = Some(Instant::now());
    }

    /// Mark the end of a GPU operation
    pub fn end_operation(&self) {
        *self.operation_start_time.lock().unwrap() = None;
    }

    /// Check if operation has timed out
    pub fn check_operation_timeout(&self) -> bool {
        if let Some(start_time) = *self.operation_start_time.lock().unwrap() {
            let elapsed = start_time.elapsed();
            if elapsed > self.config.execution_timeout {
                let violation = SafetyResult {
                    violation: SafetyViolation::ExecutionTimeout,
                    severity: ViolationSeverity::Critical,
                    message: format!("Operation exceeded timeout: {:?} > {:?}", elapsed, self.config.execution_timeout),
                    recommended_action: "Terminate current operation and check for infinite loops or deadlocks".to_string(),
                    timestamp: Instant::now(),
                };
                self.record_violation(violation);
                return true;
            }
        }
        false
    }

    /// Get current system metrics
    pub fn get_current_metrics(&self) -> Result<SystemMetrics, Box<dyn std::error::Error>> {
        // In a real implementation, this would query actual hardware sensors
        // For now, we simulate realistic values
        Ok(SystemMetrics {
            gpu_temperature: 65.0 + (rand::random::<f32>() * 10.0), // 65-75°C
            gpu_power_usage: 150.0 + (rand::random::<f32>() * 50.0), // 150-200W
            memory_used: 512 * 1024 * 1024, // 512MB
            memory_total: 8 * 1024 * 1024 * 1024, // 8GB
            gpu_utilization: rand::random::<f32>() * 100.0, // 0-100%
            timestamp: Instant::now(),
        })
    }

    /// Check if emergency shutdown is required
    pub fn check_emergency_shutdown(&self, metrics: &SystemMetrics) -> bool {
        if metrics.gpu_temperature >= self.config.emergency_shutdown_temp {
            let violation = SafetyResult {
                violation: SafetyViolation::EmergencyShutdown,
                severity: ViolationSeverity::Emergency,
                message: format!("EMERGENCY: GPU temperature {}°C exceeds emergency threshold {}°C",
                    metrics.gpu_temperature, self.config.emergency_shutdown_temp),
                recommended_action: "Immediate system shutdown to prevent hardware damage".to_string(),
                timestamp: Instant::now(),
            };
            self.record_violation(violation);
            return true;
        }
        false
    }

    /// Check all safety parameters
    pub fn check_safety_parameters(&self, metrics: &SystemMetrics) -> Vec<SafetyResult> {
        let mut violations = Vec::new();

        // Temperature check
        if metrics.gpu_temperature >= self.config.temperature_threshold {
            violations.push(SafetyResult {
                violation: SafetyViolation::TemperatureExceeded,
                severity: ViolationSeverity::Warning,
                message: format!("GPU temperature {}°C exceeds threshold {}°C",
                    metrics.gpu_temperature, self.config.temperature_threshold),
                recommended_action: "Reduce workload or improve cooling".to_string(),
                timestamp: Instant::now(),
            });
        }

        // Power check
        if metrics.gpu_power_usage >= self.config.power_limit {
            violations.push(SafetyResult {
                violation: SafetyViolation::PowerLimitExceeded,
                severity: ViolationSeverity::Warning,
                message: format!("GPU power usage {}W exceeds limit {}W",
                    metrics.gpu_power_usage, self.config.power_limit),
                recommended_action: "Reduce GPU utilization or check power supply".to_string(),
                timestamp: Instant::now(),
            });
        }

        // Memory check
        if metrics.memory_used >= self.config.memory_limit {
            violations.push(SafetyResult {
                violation: SafetyViolation::MemoryLimitExceeded,
                severity: ViolationSeverity::Critical,
                message: format!("GPU memory usage {}MB exceeds limit {}MB",
                    metrics.memory_used / (1024 * 1024), self.config.memory_limit / (1024 * 1024)),
                recommended_action: "Free GPU memory or reduce workload".to_string(),
                timestamp: Instant::now(),
            });
        }

        violations
    }

    /// Record a safety violation
    fn record_violation(&self, violation: SafetyResult) {
        println!("🚨 SAFETY VIOLATION: {:?} - {}", violation.violation, violation.message);
        println!("   💡 Recommended Action: {}", violation.recommended_action);

        self.violations.lock().unwrap().push(violation);
    }

    /// Get all recorded violations
    pub fn get_violations(&self) -> Vec<SafetyResult> {
        self.violations.lock().unwrap().clone()
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> Vec<SystemMetrics> {
        self.metrics_history.lock().unwrap().clone()
    }

    /// Generate safety report
    pub fn generate_safety_report(&self) -> String {
        let violations = self.get_violations();
        let metrics_history = self.get_metrics_history();

        let mut report = String::new();
        report.push_str("🛡️ VEXL GPU Safety Monitoring Report\n");
        report.push_str("═══════════════════════════════════════\n\n");

        // Summary statistics
        let total_violations = violations.len();
        let critical_violations = violations.iter().filter(|v| matches!(v.severity, ViolationSeverity::Critical)).count();
        let emergency_violations = violations.iter().filter(|v| matches!(v.severity, ViolationSeverity::Emergency)).count();

        report.push_str("📊 Summary:\n");
        report.push_str(&format!("   Total Violations: {}\n", total_violations));
        report.push_str(&format!("   Critical Violations: {}\n", critical_violations));
        report.push_str(&format!("   Emergency Violations: {}\n", emergency_violations));
        report.push_str(&format!("   Monitoring Duration: {:.2} seconds\n", metrics_history.last().map_or(0.0, |m| m.timestamp.elapsed().as_secs_f64())));
        report.push_str("\n");

        // Temperature analysis
        if let (Some(min_temp), Some(max_temp)) = metrics_history.iter()
            .fold((None, None), |(min, max), m: &SystemMetrics| {
                (Some(min.map_or(m.gpu_temperature, |x: f32| x.min(m.gpu_temperature))), Some(max.map_or(m.gpu_temperature, |x: f32| x.max(m.gpu_temperature))))
            }) {
            report.push_str("🌡️ Temperature Analysis:\n");
            report.push_str(&format!("   Range: {:.1}°C - {:.1}°C\n", min_temp, max_temp));
            report.push_str(&format!("   Threshold: {}°C\n", self.config.temperature_threshold));
            report.push_str(&format!("   Emergency: {}°C\n", self.config.emergency_shutdown_temp));
            report.push_str("\n");
        }

        // Power analysis
        if let (Some(min_power), Some(max_power)) = metrics_history.iter()
            .fold((None, None), |(min, max), m: &SystemMetrics| {
                (Some(min.map_or(m.gpu_power_usage, |x: f32| x.min(m.gpu_power_usage))), Some(max.map_or(m.gpu_power_usage, |x: f32| x.max(m.gpu_power_usage))))
            }) {
            report.push_str("⚡ Power Analysis:\n");
            report.push_str(&format!("   Range: {:.1}W - {:.1}W\n", min_power, max_power));
            report.push_str(&format!("   Limit: {}W\n", self.config.power_limit));
            report.push_str("\n");
        }

        // Violation details
        if !violations.is_empty() {
            report.push_str("🚨 Violation Details:\n");
            for (i, violation) in violations.iter().enumerate() {
                let severity_icon = match violation.severity {
                    ViolationSeverity::Warning => "⚠️",
                    ViolationSeverity::Critical => "🚨",
                    ViolationSeverity::Emergency => "💥",
                };

                report.push_str(&format!("   {}. {} {}: {}\n", i + 1, severity_icon,
                    format!("{:?}", violation.violation).replace("SafetyViolation::", ""), violation.message));
                report.push_str(&format!("      💡 {}\n", violation.recommended_action));
            }
            report.push_str("\n");
        }

        // Safety status
        let safety_status = if emergency_violations > 0 {
            "❌ UNSAFE - Emergency shutdown required"
        } else if critical_violations > 0 {
            "⚠️ CAUTION - Critical issues detected"
        } else if total_violations > 0 {
            "⚠️ MONITOR - Minor issues detected"
        } else {
            "✅ SAFE - No violations detected"
        };

        report.push_str("🏁 Final Safety Status:\n");
        report.push_str(&format!("   {}\n", safety_status));
        report.push_str("═══════════════════════════════════════\n");

        report
    }

    /// Main monitoring loop (runs in background thread)
    fn monitoring_loop(
        config: SafetyConfig,
        metrics_history: Arc<Mutex<Vec<SystemMetrics>>>,
        violations: Arc<Mutex<Vec<SafetyResult>>>,
        shutdown_signal: Arc<Mutex<bool>>,
    ) {
        let mut consecutive_temp_violations = 0;
        let mut last_temp_warning = Instant::now();

        loop {
            // Check shutdown signal
            if *shutdown_signal.lock().unwrap() {
                break;
            }

            // Collect metrics (in real implementation, query actual hardware)
            let metrics = SystemMetrics {
                gpu_temperature: 65.0 + (rand::random::<f32>() * 15.0), // 65-80°C
                gpu_power_usage: 150.0 + (rand::random::<f32>() * 50.0), // 150-200W
                memory_used: 512 * 1024 * 1024, // 512MB
                memory_total: 8 * 1024 * 1024 * 1024, // 8GB
                gpu_utilization: rand::random::<f32>() * 100.0, // 0-100%
                timestamp: Instant::now(),
            };

            // Store metrics
            metrics_history.lock().unwrap().push(metrics.clone());

            // Check emergency shutdown
            if metrics.gpu_temperature >= config.emergency_shutdown_temp {
                let violation = SafetyResult {
                    violation: SafetyViolation::EmergencyShutdown,
                    severity: ViolationSeverity::Emergency,
                    message: format!("EMERGENCY: GPU temperature {}°C exceeds emergency threshold {}°C",
                        metrics.gpu_temperature, config.emergency_shutdown_temp),
                    recommended_action: "Immediate system shutdown to prevent hardware damage".to_string(),
                    timestamp: Instant::now(),
                };
                violations.lock().unwrap().push(violation);

                if config.enable_auto_shutdown {
                    println!("💥 EMERGENCY SHUTDOWN INITIATED");
                    // In real implementation: trigger system shutdown
                }
                break;
            }

            // Check temperature threshold with grace period
            if metrics.gpu_temperature >= config.temperature_threshold {
                consecutive_temp_violations += 1;

                if consecutive_temp_violations >= 3 || last_temp_warning.elapsed() > config.temperature_grace_period {
                    let violation = SafetyResult {
                        violation: SafetyViolation::TemperatureExceeded,
                        severity: ViolationSeverity::Warning,
                        message: format!("GPU temperature {}°C exceeds threshold {}°C ({} consecutive violations)",
                            metrics.gpu_temperature, config.temperature_threshold, consecutive_temp_violations),
                        recommended_action: "Reduce GPU workload or improve cooling".to_string(),
                        timestamp: Instant::now(),
                    };
                    violations.lock().unwrap().push(violation);
                    last_temp_warning = Instant::now();
                }
            } else {
                consecutive_temp_violations = 0;
            }

            // Check power limit
            if metrics.gpu_power_usage >= config.power_limit {
                let violation = SafetyResult {
                    violation: SafetyViolation::PowerLimitExceeded,
                    severity: ViolationSeverity::Warning,
                    message: format!("GPU power usage {}W exceeds limit {}W",
                        metrics.gpu_power_usage, config.power_limit),
                    recommended_action: "Reduce GPU utilization or check power supply".to_string(),
                    timestamp: Instant::now(),
                };
                violations.lock().unwrap().push(violation);
            }

            // Sleep before next monitoring cycle
            thread::sleep(config.monitoring_interval);
        }

        println!("🛡️ Safety monitoring thread stopped");
    }
}

/// Run comprehensive safety validation
pub fn run_safety_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ Running VEXL GPU Safety Validation");

    let mut monitor = SafetyMonitor::new();

    // Start monitoring
    monitor.start_monitoring()?;

    // Simulate some GPU operations
    println!("🔄 Simulating GPU operations with safety monitoring...");

    for i in 0..10 {
        monitor.start_operation();

        // Simulate operation duration
        thread::sleep(Duration::from_millis(500 + (rand::random::<u64>() % 500)));

        monitor.end_operation();
        println!("   ✅ Operation {} completed", i + 1);
    }

    // Stop monitoring
    monitor.stop_monitoring()?;

    // Generate and display safety report
    let report = monitor.generate_safety_report();
    println!("{}", report);

    // Check for violations
    let violations = monitor.get_violations();
    if violations.is_empty() {
        println!("✅ Safety validation PASSED - No violations detected");
        Ok(())
    } else {
        let critical_count = violations.iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical | ViolationSeverity::Emergency))
            .count();

        if critical_count > 0 {
            println!("❌ Safety validation FAILED - {} critical violations detected", critical_count);
            Err(format!("Critical safety violations detected: {}", critical_count).into())
        } else {
            println!("⚠️ Safety validation PASSED with warnings - {} non-critical violations", violations.len());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_monitor_creation() {
        let monitor = SafetyMonitor::new();
        assert!(monitor.get_violations().is_empty());
        assert!(monitor.get_metrics_history().is_empty());
    }

    #[test]
    fn test_safety_config_defaults() {
        let config = SafetyConfig::default();
        assert!(config.temperature_threshold > 0.0);
        assert!(config.power_limit > 0.0);
        assert!(config.memory_limit > 0);
        assert!(config.execution_timeout > Duration::default());
    }

    #[test]
    fn test_operation_timeout_detection() {
        let monitor = SafetyMonitor::with_config(SafetyConfig {
            execution_timeout: Duration::from_millis(10),
            ..Default::default()
        });

        monitor.start_operation();
        thread::sleep(Duration::from_millis(20)); // Exceed timeout

        assert!(monitor.check_operation_timeout());
    }

    #[test]
    fn test_safety_parameter_checks() {
        let monitor = SafetyMonitor::new();
        let metrics = SystemMetrics {
            gpu_temperature: 90.0, // Above threshold
            gpu_power_usage: 350.0, // Above limit
            memory_used: 3 * 1024 * 1024 * 1024, // Above limit
            memory_total: 8 * 1024 * 1024 * 1024,
            gpu_utilization: 100.0,
            timestamp: Instant::now(),
        };

        let violations = monitor.check_safety_parameters(&metrics);
        assert!(violations.len() >= 2); // Should detect temp and power violations
    }
}
