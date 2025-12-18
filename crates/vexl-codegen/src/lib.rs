//! VEXL Code Generation - LLVM Backend

pub mod function_registry;
pub mod symbol_resolver;
pub mod llvm;
pub mod jit;

pub use function_registry::{FunctionRegistry, FunctionDescriptor, CallingConvention};
pub use symbol_resolver::{SymbolResolver, SymbolInfo};
pub use llvm::codegen_to_string;
pub use jit::JitEngine;
