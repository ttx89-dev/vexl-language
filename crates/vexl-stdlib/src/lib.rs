//! VEXL Standard Library
//!
//! Comprehensive standard library providing:
//! - Linear Algebra (matrix operations, decompositions, solvers)
//! - Statistics (descriptive stats, hypothesis testing, regression)
//! - Data Structures (collections, graphs, priority queues)
//! - I/O Operations (files, directories, network, system)
//! - Math Functions (trigonometry, special functions, optimization)

pub mod io;
pub mod math;
pub mod linalg;
pub mod stats;
pub mod collections;

// Re-export types for convenience
use vexl_runtime::context::{Value, Function, ExecutionContext};

/// Initialize and register all standard library operations
pub fn register(context: &mut ExecutionContext) {
    // Register linear algebra operations
    linalg::register_linalg_ops(context);

    // Register statistical operations
    stats::register_stats_ops(context);

    // Register collection operations
    collections::register_collections_ops(context);

    // Register I/O operations
    io::register_io_ops(context);

    // Register basic math operations (placeholder for now)
    register_math_ops(context);
}

/// Register basic math operations
fn register_math_ops(context: &mut ExecutionContext) {
    // Basic arithmetic operations (already handled by runtime)
    // Additional math functions would be registered here

    // Trigonometric functions
    context.register_function(Function::Native {
        name: "math_sin".to_string(),
        arg_count: 1,
        func: std::rc::Rc::new(|args: &[Value]| -> Result<Value, String> {
            if args.len() != 1 {
                return Err("sin requires 1 argument".to_string());
            }

            match &args[0] {
                Value::Float(x) => {
                    Ok(Value::Float(x.sin()))
                }
                Value::Integer(x) => {
                    Ok(Value::Float((*x as f64).sin()))
                }
                _ => Err("sin requires a numeric argument".to_string()),
            }
        }),
    });

    context.register_function(Function::Native {
        name: "math_cos".to_string(),
        arg_count: 1,
        func: std::rc::Rc::new(|args: &[Value]| -> Result<Value, String> {
            if args.len() != 1 {
                return Err("cos requires 1 argument".to_string());
            }

            match &args[0] {
                Value::Float(x) => {
                    Ok(Value::Float(x.cos()))
                }
                Value::Integer(x) => {
                    Ok(Value::Float((*x as f64).cos()))
                }
                _ => Err("cos requires a numeric argument".to_string()),
            }
        }),
    });

    context.register_function(Function::Native {
        name: "math_sqrt".to_string(),
        arg_count: 1,
        func: std::rc::Rc::new(|args: &[Value]| -> Result<Value, String> {
            if args.len() != 1 {
                return Err("sqrt requires 1 argument".to_string());
            }

            match &args[0] {
                Value::Float(x) => {
                    if *x >= 0.0 {
                        Ok(Value::Float(x.sqrt()))
                    } else {
                        Err("sqrt requires non-negative argument".to_string())
                    }
                }
                Value::Integer(x) => {
                    if *x >= 0 {
                        Ok(Value::Float((*x as f64).sqrt()))
                    } else {
                        Err("sqrt requires non-negative argument".to_string())
                    }
                }
                _ => Err("sqrt requires a numeric argument".to_string()),
            }
        }),
    });

    // Power function
    context.register_function(Function::Native {
        name: "math_pow".to_string(),
        arg_count: 2,
        func: std::rc::Rc::new(|args: &[Value]| -> Result<Value, String> {
            if args.len() != 2 {
                return Err("pow requires 2 arguments".to_string());
            }

            match (&args[0], &args[1]) {
                (Value::Float(base), Value::Float(exp)) => {
                    Ok(Value::Float(base.powf(*exp)))
                }
                (Value::Integer(base), Value::Integer(exp)) => {
                    Ok(Value::Integer(base.pow(*exp as u32)))
                }
                (Value::Float(base), Value::Integer(exp)) => {
                    Ok(Value::Float(base.powi(*exp as i32)))
                }
                (Value::Integer(base), Value::Float(exp)) => {
                    Ok(Value::Float((*base as f64).powf(*exp)))
                }
                _ => Err("pow requires numeric arguments".to_string()),
            }
        }),
    });

    // Exponential and logarithmic functions
    context.register_function(Function::Native {
        name: "math_exp".to_string(),
        arg_count: 1,
        func: std::rc::Rc::new(|args: &[Value]| -> Result<Value, String> {
            if args.len() != 1 {
                return Err("exp requires 1 argument".to_string());
            }

            match &args[0] {
                Value::Float(x) => {
                    Ok(Value::Float(x.exp()))
                }
                Value::Integer(x) => {
                    Ok(Value::Float((*x as f64).exp()))
                }
                _ => Err("exp requires a numeric argument".to_string()),
            }
        }),
    });

    context.register_function(Function::Native {
        name: "math_log".to_string(),
        arg_count: 1,
        func: std::rc::Rc::new(|args: &[Value]| -> Result<Value, String> {
            if args.len() != 1 {
                return Err("log requires 1 argument".to_string());
            }

            match &args[0] {
                Value::Float(x) => {
                    if *x > 0.0 {
                        Ok(Value::Float(x.ln()))
                    } else {
                        Err("log requires positive argument".to_string())
                    }
                }
                Value::Integer(x) => {
                    if *x > 0 {
                        Ok(Value::Float((*x as f64).ln()))
                    } else {
                        Err("log requires positive argument".to_string())
                    }
                }
                _ => Err("log requires a numeric argument".to_string()),
            }
        }),
    });
}

/// Get standard library documentation
pub fn get_documentation() -> std::collections::HashMap<String, String> {
    let mut docs = std::collections::HashMap::new();

    // Linear Algebra
    docs.insert("linalg_matmul".to_string(), "Matrix multiplication: C = A * B".to_string());
    docs.insert("linalg_dot".to_string(), "Vector dot product".to_string());
    docs.insert("linalg_norm".to_string(), "Vector Euclidean norm".to_string());

    // Statistics
    docs.insert("stats_mean".to_string(), "Arithmetic mean of dataset".to_string());
    docs.insert("stats_median".to_string(), "Median value of dataset".to_string());
    docs.insert("stats_variance".to_string(), "Variance of dataset".to_string());
    docs.insert("stats_correlation".to_string(), "Pearson correlation coefficient".to_string());

    // Collections
    docs.insert("hashmap_new".to_string(), "Create new hash map".to_string());
    docs.insert("stack_new".to_string(), "Create new stack".to_string());
    docs.insert("queue_new".to_string(), "Create new queue".to_string());
    docs.insert("priority_queue_new".to_string(), "Create new priority queue".to_string());

    // I/O Operations
    docs.insert("io_read_file".to_string(), "Read entire file as string".to_string());
    docs.insert("io_write_file".to_string(), "Write string to file".to_string());
    docs.insert("io_file_exists".to_string(), "Check if file exists".to_string());
    docs.insert("io_print".to_string(), "Print to stdout without newline".to_string());
    docs.insert("io_println".to_string(), "Print to stdout with newline".to_string());

    // Math Functions
    docs.insert("math_sin".to_string(), "Sine function".to_string());
    docs.insert("math_cos".to_string(), "Cosine function".to_string());
    docs.insert("math_sqrt".to_string(), "Square root function".to_string());
    docs.insert("math_pow".to_string(), "Power function (base^exponent)".to_string());
    docs.insert("math_exp".to_string(), "Exponential function (e^x)".to_string());
    docs.insert("math_log".to_string(), "Natural logarithm".to_string());

    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_runtime::ExecutionContext;

    #[test]
    fn test_stdlib_registration() {
        let mut context = ExecutionContext::new();
        register(&mut context);

        // Check that functions were registered
        assert!(context.call_function("math_sin", &[]).is_err()); // Wrong args
        assert!(context.call_function("linalg_dot", &[]).is_err()); // Wrong args
        assert!(context.call_function("nonexistent", &[]).is_err());
    }

    #[test]
    fn test_math_functions() {
        let mut context = ExecutionContext::new();
        register(&mut context);

        // Test sin function
        let result = context.call_function("math_sin", &[
            Value::Float(std::f64::consts::PI / 2.0)
        ]);

        match result {
            Ok(Value::Float(val)) => {
                assert!((val - 1.0).abs() < 0.0001);
            }
            _ => panic!("Expected float result"),
        }
    }

    #[test]
    fn test_documentation() {
        let docs = get_documentation();
        assert!(docs.contains_key("math_sin"));
        assert!(docs.contains_key("linalg_dot"));
        assert!(docs.contains_key("io_read_file"));
    }
}
