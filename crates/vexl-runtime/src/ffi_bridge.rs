//! Type-Safe FFI Bridge for VEXL Runtime
//!
//! This module provides a type-safe Foreign Function Interface (FFI) bridge
//! that enables safe calling between JIT-compiled code and the VEXL runtime.
//! It eliminates unsafe transmute operations and provides proper type checking.

use std::collections::HashMap;
use std::ffi::c_void;

/// Function pointer types for different signatures
pub enum FunctionPointer {
    /// () -> i64
    NullaryInt(fn() -> i64),
    /// (i64) -> i64
    UnaryInt(fn(i64) -> i64),
    /// (i64, i64) -> i64
    BinaryInt(fn(i64, i64) -> i64),
    /// (i64, i64, i64) -> i64
    TernaryInt(fn(i64, i64, i64) -> i64),
    /// (i64, i64, i64, i64) -> i64
    QuaternaryInt(fn(i64, i64, i64, i64) -> i64),
    /// () -> *mut c_void
    NullaryPtr(fn() -> *mut c_void),
    /// (i64) -> *mut c_void
    UnaryPtr(fn(i64) -> *mut c_void),
    /// (*mut c_void) -> i64
    UnaryPtrInt(fn(*mut c_void) -> i64),
    /// (*mut c_void, i64) -> i64
    BinaryPtrInt(fn(*mut c_void, i64) -> i64),
    /// (*mut c_void, i64, i64) -> ()
    TernaryPtrVoid(fn(*mut c_void, i64, i64)),
    /// (*mut c_void, *mut c_void) -> *mut c_void
    BinaryPtrPtr(fn(*mut c_void, *mut c_void) -> *mut c_void),
    /// (f64) -> f64
    UnaryFloatFloat(fn(f64) -> f64),
    /// (f64, f64) -> f64
    BinaryFloatFloat(fn(f64, f64) -> f64),
    /// (*const u8) -> u64
    UnaryPtrU64(fn(*const u8) -> u64),
    /// (*const u8, u64, *const u8, u64) -> *mut u8
    QuaternaryPtrPtr(fn(*const u8, u64, *const u8, u64) -> *mut u8),
}

/// Type-safe FFI bridge for managing function pointers
pub struct FfiBridge {
    functions: HashMap<String, FunctionPointer>,
}

impl FfiBridge {
    /// Create a new empty FFI bridge
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register a function pointer with type safety
    pub fn register_function<F: FnPtr>(&mut self, name: &str, func: F) {
        let ptr = FunctionPointer::from_func(func);
        self.functions.insert(name.to_string(), ptr);
    }

    /// Call a registered function with type checking
    pub fn call_function(&self, name: &str, args: &[FfiValue]) -> Result<FfiValue, String> {
        let func_ptr = self.functions.get(name)
            .ok_or_else(|| format!("Function '{}' not registered in FFI bridge", name))?;

        match func_ptr {
            FunctionPointer::NullaryInt(f) => {
                if args.is_empty() {
                    Ok(FfiValue::Int(f()))
                } else {
                    Err(format!("Function '{}' expects 0 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::UnaryInt(f) => {
                if args.len() == 1 {
                    match &args[0] {
                        FfiValue::Int(arg) => Ok(FfiValue::Int(f(*arg))),
                        _ => Err(format!("Function '{}' expects integer argument", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 1 argument, got {}", name, args.len()))
                }
            }
            FunctionPointer::BinaryInt(f) => {
                if args.len() == 2 {
                    match (&args[0], &args[1]) {
                        (FfiValue::Int(a), FfiValue::Int(b)) => Ok(FfiValue::Int(f(*a, *b))),
                        _ => Err(format!("Function '{}' expects integer arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 2 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::TernaryInt(f) => {
                if args.len() == 3 {
                    match (&args[0], &args[1], &args[2]) {
                        (FfiValue::Int(a), FfiValue::Int(b), FfiValue::Int(c)) => Ok(FfiValue::Int(f(*a, *b, *c))),
                        _ => Err(format!("Function '{}' expects integer arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 3 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::QuaternaryInt(f) => {
                if args.len() == 4 {
                    match (&args[0], &args[1], &args[2], &args[3]) {
                        (FfiValue::Int(a), FfiValue::Int(b), FfiValue::Int(c), FfiValue::Int(d)) => {
                            Ok(FfiValue::Int(f(*a, *b, *c, *d)))
                        }
                        _ => Err(format!("Function '{}' expects integer arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 4 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::NullaryPtr(f) => {
                if args.is_empty() {
                    Ok(FfiValue::Ptr(f()))
                } else {
                    Err(format!("Function '{}' expects 0 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::UnaryPtr(f) => {
                if args.len() == 1 {
                    match &args[0] {
                        FfiValue::Int(arg) => Ok(FfiValue::Ptr(f(*arg))),
                        _ => Err(format!("Function '{}' expects integer argument", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 1 argument, got {}", name, args.len()))
                }
            }
            FunctionPointer::UnaryPtrInt(f) => {
                if args.len() == 1 {
                    match &args[0] {
                        FfiValue::Ptr(arg) => Ok(FfiValue::Int(f(*arg))),
                        _ => Err(format!("Function '{}' expects pointer argument", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 1 argument, got {}", name, args.len()))
                }
            }
            FunctionPointer::BinaryPtrInt(f) => {
                if args.len() == 2 {
                    match (&args[0], &args[1]) {
                        (FfiValue::Ptr(a), FfiValue::Int(b)) => Ok(FfiValue::Int(f(*a, *b))),
                        _ => Err(format!("Function '{}' expects (pointer, integer) arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 2 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::TernaryPtrVoid(f) => {
                if args.len() == 3 {
                    match (&args[0], &args[1], &args[2]) {
                        (FfiValue::Ptr(a), FfiValue::Int(b), FfiValue::Int(c)) => {
                            f(*a, *b, *c);
                            Ok(FfiValue::Void)
                        }
                        _ => Err(format!("Function '{}' expects (pointer, integer, integer) arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 3 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::BinaryPtrPtr(f) => {
                if args.len() == 2 {
                    match (&args[0], &args[1]) {
                        (FfiValue::Ptr(a), FfiValue::Ptr(b)) => Ok(FfiValue::Ptr(f(*a, *b))),
                        _ => Err(format!("Function '{}' expects pointer arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 2 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::UnaryFloatFloat(f) => {
                if args.len() == 1 {
                    match &args[0] {
                        FfiValue::Int(arg) => {
                            // Convert i64 to f64 for math functions
                            let float_val = *arg as f64;
                            Ok(FfiValue::Int(f(float_val) as i64))
                        }
                        _ => Err(format!("Function '{}' expects numeric argument", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 1 argument, got {}", name, args.len()))
                }
            }
            FunctionPointer::BinaryFloatFloat(f) => {
                if args.len() == 2 {
                    match (&args[0], &args[1]) {
                        (FfiValue::Int(a), FfiValue::Int(b)) => {
                            let result = f(*a as f64, *b as f64);
                            Ok(FfiValue::Int(result as i64))
                        }
                        _ => Err(format!("Function '{}' expects numeric arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 2 arguments, got {}", name, args.len()))
                }
            }
            FunctionPointer::UnaryPtrU64(f) => {
                if args.len() == 1 {
                    match &args[0] {
                        FfiValue::Ptr(arg) => Ok(FfiValue::Int(f(*arg as *const u8) as i64)),
                        _ => Err(format!("Function '{}' expects pointer argument", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 1 argument, got {}", name, args.len()))
                }
            }
            FunctionPointer::QuaternaryPtrPtr(f) => {
                if args.len() == 4 {
                    match (&args[0], &args[1], &args[2], &args[3]) {
                        (FfiValue::Ptr(a), FfiValue::Int(b), FfiValue::Ptr(c), FfiValue::Int(d)) => {
                            let result = f(*a as *const u8, *b as u64, *c as *const u8, *d as u64);
                            Ok(FfiValue::Ptr(result as *mut c_void))
                        }
                        _ => Err(format!("Function '{}' expects (pointer, int, pointer, int) arguments", name)),
                    }
                } else {
                    Err(format!("Function '{}' expects 4 arguments, got {}", name, args.len()))
                }
            }
        }
    }

    /// Check if a function is registered
    pub fn is_registered(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Get all registered function names
    pub fn get_function_names(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }

    /// Get raw function pointer (for advanced use cases)
    pub unsafe fn get_raw_pointer(&self, name: &str) -> Option<*mut c_void> {
        self.functions.get(name).map(|fp| fp.as_raw_ptr())
    }
}

impl FunctionPointer {
    /// Create FunctionPointer from a function
    pub fn from_func<F: FnPtr>(func: F) -> Self {
        func.into_function_pointer()
    }

    /// Get raw pointer for low-level operations
    pub fn as_raw_ptr(&self) -> *mut c_void {
        match self {
            FunctionPointer::NullaryInt(f) => *f as *mut c_void,
            FunctionPointer::UnaryInt(f) => *f as *mut c_void,
            FunctionPointer::BinaryInt(f) => *f as *mut c_void,
            FunctionPointer::TernaryInt(f) => *f as *mut c_void,
            FunctionPointer::QuaternaryInt(f) => *f as *mut c_void,
            FunctionPointer::NullaryPtr(f) => *f as *mut c_void,
            FunctionPointer::UnaryPtr(f) => *f as *mut c_void,
            FunctionPointer::UnaryPtrInt(f) => *f as *mut c_void,
            FunctionPointer::BinaryPtrInt(f) => *f as *mut c_void,
            FunctionPointer::TernaryPtrVoid(f) => *f as *mut c_void,
            FunctionPointer::BinaryPtrPtr(f) => *f as *mut c_void,
            FunctionPointer::UnaryFloatFloat(f) => *f as *mut c_void,
            FunctionPointer::BinaryFloatFloat(f) => *f as *mut c_void,
            FunctionPointer::UnaryPtrU64(f) => *f as *mut c_void,
            FunctionPointer::QuaternaryPtrPtr(f) => *f as *mut c_void,
        }
    }
}

/// Values that can be passed across the FFI boundary
#[derive(Debug, Clone)]
pub enum FfiValue {
    Int(i64),
    Ptr(*mut c_void),
    Void,
}

/// Trait for function pointers that can be converted to FunctionPointer
pub trait FnPtr {
    fn into_function_pointer(self) -> FunctionPointer;
}

impl FnPtr for fn() -> i64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::NullaryInt(self)
    }
}

impl FnPtr for fn(i64) -> i64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::UnaryInt(self)
    }
}

impl FnPtr for fn(i64, i64) -> i64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::BinaryInt(self)
    }
}

impl FnPtr for fn(i64, i64, i64) -> i64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::TernaryInt(self)
    }
}

impl FnPtr for fn(i64, i64, i64, i64) -> i64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::QuaternaryInt(self)
    }
}

impl FnPtr for fn() -> *mut c_void {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::NullaryPtr(self)
    }
}

impl FnPtr for fn(i64) -> *mut c_void {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::UnaryPtr(self)
    }
}

impl FnPtr for fn(*mut c_void) -> i64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::UnaryPtrInt(self)
    }
}

impl FnPtr for fn(*mut c_void, i64) -> i64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::BinaryPtrInt(self)
    }
}

impl FnPtr for fn(*mut c_void, i64, i64) {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::TernaryPtrVoid(self)
    }
}

impl FnPtr for fn(*mut c_void, *mut c_void) -> *mut c_void {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::BinaryPtrPtr(self)
    }
}



impl FnPtr for fn(f64) -> f64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::UnaryFloatFloat(self)
    }
}

impl FnPtr for fn(f64, f64) -> f64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::BinaryFloatFloat(self)
    }
}

impl FnPtr for fn(*const u8, u64, *const u8, u64) -> *mut u8 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::QuaternaryPtrPtr(self)
    }
}

impl FnPtr for fn(*const u8) -> u64 {
    fn into_function_pointer(self) -> FunctionPointer {
        FunctionPointer::UnaryPtrU64(self)
    }
}

/// Macro for automatic FFI function registration
#[macro_export]
macro_rules! ffi_function {
    ($bridge:expr, $name:expr, $func:expr) => {
        $bridge.register_function($name, $func);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test functions
    fn test_nullary() -> i64 { 42 }
    fn test_unary(x: i64) -> i64 { x * 2 }
    fn test_binary(x: i64, y: i64) -> i64 { x + y }
    fn test_ptr_func() -> *mut c_void { std::ptr::null_mut() }

    #[test]
    fn test_ffi_bridge_creation() {
        let bridge = FfiBridge::new();
        assert!(!bridge.is_registered("test"));
        assert!(bridge.get_function_names().is_empty());
    }

    #[test]
    fn test_register_nullary_function() {
        let mut bridge = FfiBridge::new();
        bridge.register_function("test_nullary", test_nullary as fn() -> i64);

        assert!(bridge.is_registered("test_nullary"));
        let result = bridge.call_function("test_nullary", &[]).unwrap();
        match result {
            FfiValue::Int(42) => {},
            _ => panic!("Expected Int(42)"),
        }
    }

    #[test]
    fn test_register_unary_function() {
        let mut bridge = FfiBridge::new();
        bridge.register_function("test_unary", test_unary as fn(i64) -> i64);

        let args = vec![FfiValue::Int(21)];
        let result = bridge.call_function("test_unary", &args).unwrap();
        match result {
            FfiValue::Int(42) => {},
            _ => panic!("Expected Int(42)"),
        }
    }

    #[test]
    fn test_register_binary_function() {
        let mut bridge = FfiBridge::new();
        bridge.register_function("test_binary", test_binary as fn(i64, i64) -> i64);

        let args = vec![FfiValue::Int(10), FfiValue::Int(32)];
        let result = bridge.call_function("test_binary", &args).unwrap();
        match result {
            FfiValue::Int(42) => {},
            _ => panic!("Expected Int(42)"),
        }
    }

    #[test]
    fn test_call_unregistered_function() {
        let bridge = FfiBridge::new();
        let result = bridge.call_function("nonexistent", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_call_with_wrong_args() {
        let mut bridge = FfiBridge::new();
        bridge.register_function("test_unary", test_unary as fn(i64) -> i64);

        // Wrong number of arguments
        let result = bridge.call_function("test_unary", &[]);
        assert!(result.is_err());

        // Wrong argument types
        let args = vec![FfiValue::Ptr(std::ptr::null_mut())];
        let result = bridge.call_function("test_unary", &args);
        assert!(result.is_err());
    }

    #[test]
    fn test_macro_registration() {
        let mut bridge = FfiBridge::new();
        ffi_function!(&mut bridge, "test_macro", test_nullary as fn() -> i64);

        assert!(bridge.is_registered("test_macro"));
        let result = bridge.call_function("test_macro", &[]).unwrap();
        match result {
            FfiValue::Int(42) => {},
            _ => panic!("Expected Int(42)"),
        }
    }
}
