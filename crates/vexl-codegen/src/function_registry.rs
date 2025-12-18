//! Type-Safe Function Registry for VEXL
//!
//! This module provides a centralized registry for managing function signatures
//! and calling conventions used throughout the VEXL compilation pipeline.

use std::collections::HashMap;
use vexl_ir::{VirType, FunctionSignature};

/// Calling convention for function calls
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallingConvention {
    /// C calling convention (extern "C")
    C,
    /// Fast calling convention (extern "fastcall")
    Fast,
    /// Cold calling convention (extern "cold")
    Cold,
}

/// Function descriptor containing all metadata about a function
#[derive(Debug, Clone)]
pub struct FunctionDescriptor {
    pub name: String,
    pub param_types: Vec<VirType>,
    pub return_type: VirType,
    pub calling_convention: CallingConvention,
    pub is_variadic: bool,
}

impl FunctionDescriptor {
    pub fn new(
        name: String,
        param_types: Vec<VirType>,
        return_type: VirType,
        calling_convention: CallingConvention,
        is_variadic: bool,
    ) -> Self {
        Self {
            name,
            param_types,
            return_type,
            calling_convention,
            is_variadic,
        }
    }

    /// Create a function descriptor from a VIR function signature
    pub fn from_signature(name: String, signature: &FunctionSignature) -> Self {
        Self::new(
            name,
            signature.param_types.clone(),
            signature.return_type.clone(),
            CallingConvention::C, // Default to C calling convention
            false, // Not variadic by default
        )
    }

    /// Get the function signature
    pub fn signature(&self) -> FunctionSignature {
        FunctionSignature::new(
            self.param_types.clone(),
            self.return_type.clone(),
        )
    }
}

/// Central registry for function signatures and calling conventions
#[derive(Debug)]
pub struct FunctionRegistry {
    functions: HashMap<String, FunctionDescriptor>,
}

impl FunctionRegistry {
    /// Create a new empty function registry
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register a runtime function with its signature
    pub fn register_runtime_function(
        &mut self,
        name: &str,
        param_types: Vec<VirType>,
        return_type: VirType,
        calling_convention: CallingConvention,
        is_variadic: bool,
    ) {
        let descriptor = FunctionDescriptor::new(
            name.to_string(),
            param_types,
            return_type,
            calling_convention,
            is_variadic,
        );
        self.functions.insert(name.to_string(), descriptor);
    }

    /// Register a user-defined function from VIR
    pub fn register_user_function(
        &mut self,
        name: String,
        signature: &FunctionSignature,
    ) {
        let descriptor = FunctionDescriptor::from_signature(name.clone(), signature);
        self.functions.insert(name, descriptor);
    }

    /// Look up a function descriptor by name
    pub fn get_descriptor(&self, name: &str) -> Option<&FunctionDescriptor> {
        self.functions.get(name)
    }

    /// Get function signature by name
    pub fn get_signature(&self, name: &str) -> Option<FunctionSignature> {
        self.get_descriptor(name).map(|desc| desc.signature())
    }

    /// Get calling convention by name
    pub fn get_calling_convention(&self, name: &str) -> Option<CallingConvention> {
        self.get_descriptor(name).map(|desc| desc.calling_convention)
    }

    /// Check if a function is registered
    pub fn is_registered(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Get all registered function names
    pub fn get_function_names(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }

    /// Initialize registry with common runtime functions
    pub fn initialize_with_runtime_functions(&mut self) {
        // Vector operations
        self.register_runtime_function(
            "vexl_vec_alloc_i64",
            vec![VirType::Int64],
            VirType::Pointer,
            CallingConvention::C,
            false,
        );

        self.register_runtime_function(
            "vexl_vec_len",
            vec![VirType::Pointer],
            VirType::Int64,
            CallingConvention::C,
            false,
        );

        self.register_runtime_function(
            "vexl_vec_get_i64",
            vec![VirType::Pointer, VirType::Int64],
            VirType::Int64,
            CallingConvention::C,
            false,
        );

        self.register_runtime_function(
            "vexl_vec_set_i64",
            vec![VirType::Pointer, VirType::Int64, VirType::Int64],
            VirType::Void,
            CallingConvention::C,
            false,
        );

        self.register_runtime_function(
            "vexl_vec_sum",
            vec![VirType::Pointer],
            VirType::Int64,
            CallingConvention::C,
            false,
        );

        // Print operations
        self.register_runtime_function(
            "vexl_print_int",
            vec![VirType::Int64],
            VirType::Void,
            CallingConvention::C,
            false,
        );

        // Parallel operations
        self.register_runtime_function(
            "vexl_vec_map_parallel",
            vec![VirType::Pointer, VirType::Pointer, VirType::Int64],
            VirType::Pointer,
            CallingConvention::C,
            false,
        );

        self.register_runtime_function(
            "vexl_vec_reduce_parallel",
            vec![VirType::Pointer, VirType::Pointer, VirType::Pointer, VirType::Int64],
            VirType::Pointer,
            CallingConvention::C,
            false,
        );

        // Exit function
        self.register_runtime_function(
            "vexl_exit",
            vec![VirType::Int64],
            VirType::Void,
            CallingConvention::C,
            false,
        );
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        registry.initialize_with_runtime_functions();
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_registry_creation() {
        let registry = FunctionRegistry::new();
        assert!(registry.functions.is_empty());
    }

    #[test]
    fn test_register_runtime_function() {
        let mut registry = FunctionRegistry::new();

        registry.register_runtime_function(
            "test_func",
            vec![VirType::Int64, VirType::Float64],
            VirType::Pointer,
            CallingConvention::C,
            false,
        );

        let descriptor = registry.get_descriptor("test_func").unwrap();
        assert_eq!(descriptor.name, "test_func");
        assert_eq!(descriptor.param_types.len(), 2);
        assert_eq!(descriptor.return_type, VirType::Pointer);
        assert_eq!(descriptor.calling_convention, CallingConvention::C);
        assert!(!descriptor.is_variadic);
    }

    #[test]
    fn test_register_user_function() {
        let mut registry = FunctionRegistry::new();

        let signature = FunctionSignature::new(
            vec![VirType::Int64],
            VirType::Int64,
        );

        registry.register_user_function("user_func".to_string(), &signature);

        let descriptor = registry.get_descriptor("user_func").unwrap();
        assert_eq!(descriptor.name, "user_func");
        assert_eq!(descriptor.param_types, vec![VirType::Int64]);
        assert_eq!(descriptor.return_type, VirType::Int64);
        assert_eq!(descriptor.calling_convention, CallingConvention::C);
    }

    #[test]
    fn test_get_signature() {
        let mut registry = FunctionRegistry::new();

        registry.register_runtime_function(
            "test_func",
            vec![VirType::Int64],
            VirType::Float64,
            CallingConvention::C,
            false,
        );

        let signature = registry.get_signature("test_func").unwrap();
        assert_eq!(signature.param_types, vec![VirType::Int64]);
        assert_eq!(signature.return_type, VirType::Float64);
    }

    #[test]
    fn test_initialize_with_runtime_functions() {
        let registry = FunctionRegistry::default();

        // Check that some runtime functions are registered
        assert!(registry.is_registered("vexl_vec_alloc_i64"));
        assert!(registry.is_registered("vexl_print_int"));
        assert!(registry.is_registered("vexl_exit"));

        let names = registry.get_function_names();
        assert!(!names.is_empty());
        assert!(names.contains(&"vexl_vec_sum".to_string()));
    }

    #[test]
    fn test_unknown_function() {
        let registry = FunctionRegistry::new();
        assert!(!registry.is_registered("unknown_func"));
        assert!(registry.get_descriptor("unknown_func").is_none());
        assert!(registry.get_signature("unknown_func").is_none());
    }
}
