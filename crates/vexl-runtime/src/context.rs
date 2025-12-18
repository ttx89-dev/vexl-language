//! Execution Context for VEXL Runtime
//!
//! Manages variable bindings, function registry, and execution state.
//! Supports both interpreter and JIT execution modes.

use crate::vector::Vector;
use crate::interpreter::{Interpreter, BytecodeProgram, Value as InterpreterValue};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

/// Simple garbage collector placeholder
struct Gc;

impl Gc {
    fn new() -> Self { Gc }
    fn collect(&self) {}
    fn stats(&self) -> (usize, usize) { (0, 0) }
}

/// Execution context for VEXL programs
pub struct ExecutionContext {
    /// Variable bindings (name -> value)
    variables: HashMap<String, Value>,
    /// Function registry (name -> function)
    functions: HashMap<String, Function>,
    /// Current execution mode
    mode: ExecutionMode,
    /// Garbage collector
    gc: Rc<RefCell<Gc>>,
    /// Interpreter instance (for interpreter mode)
    interpreter: Option<Interpreter>,
    /// JIT runtime (for JIT mode)
    jit_runtime: Option<crate::JitRuntime>,
}

/// Execution modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    /// Use bytecode interpreter (fast startup)
    Interpreter,
    /// Use JIT compilation (high performance)
    Jit,
    /// Auto-select based on program characteristics
    Auto,
}

/// VEXL runtime value types
#[derive(Debug, Clone)]
pub enum Value {
    /// 64-bit integer
    Integer(i64),
    /// 64-bit float
    Float(f64),
    /// String
    String(String),
    /// Vector (reference to runtime vector)
    Vector(VectorRef),
    /// Boolean
    Boolean(bool),
    /// Function reference
    Function(String),
    /// Unit type (no value)
    Unit,
}

/// Vector reference (handles ownership and borrowing)
#[derive(Debug, Clone)]
pub struct VectorRef {
    /// Pointer to the vector
    ptr: *mut Vector,
    /// Whether this context owns the vector (responsible for cleanup)
    owned: bool,
}

impl VectorRef {
    /// Create a new owned vector reference
    pub fn owned(ptr: *mut Vector) -> Self {
        Self { ptr, owned: true }
    }

    /// Create a new borrowed vector reference
    pub fn borrowed(ptr: *mut Vector) -> Self {
        Self { ptr, owned: false }
    }

    /// Get the underlying pointer
    pub fn ptr(&self) -> *mut Vector {
        self.ptr
    }

    /// Check if this reference owns the vector
    pub fn is_owned(&self) -> bool {
        self.owned
    }
}

impl Drop for VectorRef {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                // Free the vector if we own it
                crate::vector::vexl_vec_free(self.ptr);
            }
        }
    }
}

/// Function representation
#[derive(Clone)]
pub enum Function {
    /// Native Rust function
    Native {
        name: String,
        arg_count: usize,
        func: Rc<dyn Fn(&[Value]) -> Result<Value, String>>,
    },
    /// VEXL bytecode function
    Bytecode {
        name: String,
        parameters: Vec<String>,
        body: Rc<BytecodeProgram>,
    },
    /// JIT-compiled function
    Jit {
        name: String,
        parameters: Vec<String>,
        // JIT function pointer would be stored here
    },
}

impl ExecutionContext {
    /// Create a new execution context with default settings
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            mode: ExecutionMode::Auto,
            gc: Rc::new(RefCell::new(Gc::new())),
            interpreter: Some(Interpreter::new()),
            jit_runtime: None,
        }
    }

    /// Create execution context with specific mode
    pub fn with_mode(mode: ExecutionMode) -> Self {
        let mut ctx = Self::new();
        ctx.mode = mode;
        ctx
    }

    /// Set the execution mode
    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.mode = mode;
    }

    /// Initialize JIT runtime (if needed)
    pub fn init_jit(&mut self) -> Result<(), String> {
        if self.jit_runtime.is_none() {
            self.jit_runtime = Some(crate::JitRuntime::new()?);
        }
        Ok(())
    }

    /// Bind a variable
    pub fn bind_variable(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }

    /// Get a variable value
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// Register a native function
    pub fn register_function(&mut self, func: Function) {
        let name = match &func {
            Function::Native { name, .. } => name.clone(),
            Function::Bytecode { name, .. } => name.clone(),
            Function::Jit { name, .. } => name.clone(),
        };
        self.functions.insert(name, func);
    }

    /// Call a function
    pub fn call_function(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        if let Some(Function::Native { func, .. }) = self.functions.get(name) {
            return func(args);
        }

        if let Some(Function::Bytecode { body, .. }) = self.functions.get(name) {
            let body = body.clone(); // Clone to avoid borrowing issues
            return self.execute_bytecode(&body, args);
        }

        if let Some(Function::Jit { .. }) = self.functions.get(name) {
            return Err("JIT function execution not implemented yet".to_string());
        }

        Err(format!("Function '{}' not found", name))
    }

    /// Execute bytecode program
    fn execute_bytecode(&mut self, program: &BytecodeProgram, args: &[Value]) -> Result<Value, String> {
        if let Some(interpreter) = &mut self.interpreter {
            // Set up arguments as variables
            // This is a simplified implementation

            let result = interpreter.execute(program)?;

            match result {
                Some(interp_value) => self.convert_interpreter_value(interp_value),
                None => Ok(Value::Unit),
            }
        } else {
            Err("Interpreter not available".to_string())
        }
    }

    /// Convert interpreter value to context value
    fn convert_interpreter_value(&self, value: InterpreterValue) -> Result<Value, String> {
        match value {
            InterpreterValue::Integer(i) => Ok(Value::Integer(i)),
            InterpreterValue::Float(f) => Ok(Value::Float(f)),
            InterpreterValue::String(s) => Ok(Value::String(s)),
            InterpreterValue::Boolean(b) => Ok(Value::Boolean(b)),
            InterpreterValue::Vector(ptr) => Ok(Value::Vector(VectorRef::owned(ptr))),
        }
    }

    /// Evaluate an expression
    pub fn evaluate(&mut self, expr: &str) -> Result<Value, String> {
        match self.mode {
            ExecutionMode::Interpreter => {
                self.evaluate_interpreter(expr)
            }
            ExecutionMode::Jit => {
                self.evaluate_jit(expr)
            }
            ExecutionMode::Auto => {
                // Auto-select: use interpreter for simple expressions, JIT for complex ones
                if expr.len() < 100 {
                    self.evaluate_interpreter(expr)
                } else {
                    self.evaluate_jit(expr)
                }
            }
        }
    }

    /// Evaluate using interpreter
    fn evaluate_interpreter(&mut self, expr: &str) -> Result<Value, String> {
        // Parse and compile expression to bytecode
        // This would require integration with the syntax crate
        // For now, return a placeholder
        Err("Interpreter evaluation not fully implemented".to_string())
    }

    /// Evaluate using JIT
    fn evaluate_jit(&mut self, expr: &str) -> Result<Value, String> {
        // Compile expression to VIR, then to machine code
        // This would require full integration with syntax, types, ir, and codegen
        // For now, return a placeholder
        Err("JIT evaluation not fully implemented".to_string())
    }

    /// Create a vector from slice
    pub fn create_vector(&mut self, data: &[f64]) -> Result<Value, String> {
        unsafe {
            // Convert f64 slice to i64 (simplified - assumes integer data)
            let i64_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();

            // Allocate vector
            let vec_ptr = crate::vector::vexl_vec_alloc_i64(i64_data.len() as u64);

            // Populate vector
            for (i, &val) in i64_data.iter().enumerate() {
                crate::vector::vexl_vec_set_i64(vec_ptr, i as u64, val);
            }

            Ok(Value::Vector(VectorRef::owned(vec_ptr)))
        }
    }

    /// Get vector data as slice
    pub fn vector_as_slice(&self, vector: &Value) -> Result<&[f64], String> {
        match vector {
            Value::Vector(vec_ref) => {
                unsafe {
                    let vec = &*vec_ref.ptr();
                    let len = vec.len() as usize;
                    // This is a simplified conversion - in practice, would need proper type handling
                    Err("Vector slice access not implemented".to_string())
                }
            }
            _ => Err("Expected vector value".to_string()),
        }
    }

    /// Perform garbage collection
    pub fn collect_garbage(&mut self) {
        self.gc.borrow_mut().collect();
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        self.gc.borrow().stats()
    }

    /// Clear all variables and functions
    pub fn clear(&mut self) {
        self.variables.clear();
        self.functions.clear();
        self.collect_garbage();
    }
}

/// Scope management for nested contexts
pub struct Scope {
    /// Parent scope (if any)
    pub parent: Option<Rc<RefCell<Scope>>>,
    /// Local variables
    pub locals: HashMap<String, Value>,
}

impl Scope {
    /// Create root scope
    pub fn root() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            parent: None,
            locals: HashMap::new(),
        }))
    }

    /// Create child scope
    pub fn child(parent: Rc<RefCell<Scope>>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            parent: Some(parent),
            locals: HashMap::new(),
        }))
    }

    /// Set a local variable
    pub fn set_local(scope: &Rc<RefCell<Scope>>, name: String, value: Value) {
        scope.borrow_mut().locals.insert(name, value);
    }

    /// Get a variable (search local then parent scopes)
    pub fn get_variable(scope: &Rc<RefCell<Scope>>, name: &str) -> Option<Value> {
        let scope_ref = scope.borrow();

        // Check local scope first
        if let Some(value) = scope_ref.locals.get(name) {
            return Some(value.clone());
        }

        // Check parent scopes
        if let Some(ref parent) = scope_ref.parent {
            return Self::get_variable(parent, name);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_context_creation() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.variables.len(), 0);
        assert_eq!(ctx.functions.len(), 0);
    }

    #[test]
    fn test_variable_binding() {
        let mut ctx = ExecutionContext::new();

        ctx.bind_variable("x".to_string(), Value::Integer(42));
        ctx.bind_variable("y".to_string(), Value::Float(3.14));

        assert!(matches!(ctx.get_variable("x"), Some(Value::Integer(42))));
        assert!(matches!(ctx.get_variable("y"), Some(Value::Float(_))));
        assert!(ctx.get_variable("z").is_none());
    }

    #[test]
    fn test_scope_management() {
        let root_scope = Scope::root();

        Scope::set_local(&root_scope, "global".to_string(), Value::Integer(100));

        let child_scope = Scope::child(root_scope.clone());
        Scope::set_local(&child_scope, "local".to_string(), Value::Integer(200));

        // Test local variable access
        assert!(matches!(Scope::get_variable(&child_scope, "local"), Some(Value::Integer(200))));

        // Test parent variable access
        assert!(matches!(Scope::get_variable(&child_scope, "global"), Some(Value::Integer(100))));

        // Test undefined variable
        assert!(Scope::get_variable(&child_scope, "undefined").is_none());
    }

    #[test]
    fn test_vector_lifecycle() {
        let ctx = ExecutionContext::new();

        // Create a vector reference
        let vec_ref = VectorRef::owned(std::ptr::null_mut());
        assert!(vec_ref.is_owned());

        let borrowed_ref = VectorRef::borrowed(std::ptr::null_mut());
        assert!(!borrowed_ref.is_owned());
    }
}
