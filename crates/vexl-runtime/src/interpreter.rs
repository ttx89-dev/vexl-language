//! Bytecode Interpreter for VEXL
//!
//! Fast startup interpreter that executes VEXL bytecode directly.
//! Used for REPL and small scripts to avoid JIT compilation overhead.

use crate::vector::Vector;
use crate::context::Scope;
use vexl_ir as vir;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

/// VEXL bytecode instruction set
#[derive(Debug, Clone)]
pub enum BytecodeInstruction {
    /// Load constant integer: push i64
    LoadConstI64(i64),
    /// Load constant float: push f64
    LoadConstF64(f64),
    /// Load constant string: push String
    LoadConstString(String),

    /// Load vector from runtime
    LoadVector(*mut Vector),
    /// Store vector to runtime
    StoreVector(*mut Vector),

    /// Arithmetic operations (pop operands, push result)
    Add,
    Sub,
    Mul,
    Div,

    /// Vector operations
    VectorAdd,
    VectorMul,
    VectorScale(f64),

    /// Stack operations
    Dup,    // duplicate top of stack
    Pop,    // remove top of stack
    Swap,   // swap top two elements

    /// Control flow
    Jump(usize),          // unconditional jump to address
    JumpIfTrue(usize),    // jump if top of stack is true
    JumpIfFalse(usize),   // jump if top of stack is false

    /// Function calls
    Call(String),         // call function by name
    Return,               // return from function

    /// Scope management
    PushScope,             // create new child scope
    PopScope,              // return to parent scope
    StoreVar(String),      // store top of stack in variable
    LoadVar(String),       // load variable onto stack

    /// Runtime calls
    RuntimeCall(String),  // call runtime function

    /// Halt execution
    Halt,
}

/// VEXL bytecode program
#[derive(Debug)]
pub struct BytecodeProgram {
    pub instructions: Vec<BytecodeInstruction>,
    pub constants: Vec<Value>,
}

/// Interpreter value types
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    String(String),
    Vector(*mut Vector),
    Boolean(bool),
}

/// Bytecode interpreter
pub struct Interpreter {
    /// Program counter
    pc: usize,
    /// Value stack
    stack: Vec<Value>,
    /// Current scope for variable bindings
    current_scope: Rc<RefCell<Scope>>,
    /// Call stack for function calls
    call_stack: Vec<usize>,
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        Self {
            pc: 0,
            stack: Vec::new(),
            current_scope: Scope::root(),
            call_stack: Vec::new(),
        }
    }

    /// Execute a bytecode program
    pub fn execute(&mut self, program: &BytecodeProgram) -> Result<Option<Value>, String> {
        self.pc = 0;
        self.stack.clear();

        while self.pc < program.instructions.len() {
            match &program.instructions[self.pc] {
                BytecodeInstruction::LoadConstI64(value) => {
                    self.stack.push(Value::Integer(*value));
                }
                BytecodeInstruction::LoadConstF64(value) => {
                    self.stack.push(Value::Float(*value));
                }
                BytecodeInstruction::LoadConstString(value) => {
                    self.stack.push(Value::String(value.clone()));
                }

                BytecodeInstruction::LoadVector(vec_ptr) => {
                    self.stack.push(Value::Vector(*vec_ptr));
                }
                BytecodeInstruction::StoreVector(vec_ptr) => {
                    // Store operation - implementation depends on use case
                    // For now, just acknowledge
                }

                BytecodeInstruction::Add => self.execute_add()?,
                BytecodeInstruction::Sub => self.execute_sub()?,
                BytecodeInstruction::Mul => self.execute_mul()?,
                BytecodeInstruction::Div => self.execute_div()?,

                BytecodeInstruction::VectorAdd => self.execute_vector_add()?,
                BytecodeInstruction::VectorMul => self.execute_vector_mul()?,
                BytecodeInstruction::VectorScale(scale) => self.execute_vector_scale(*scale)?,

                BytecodeInstruction::Dup => self.execute_dup()?,
                BytecodeInstruction::Pop => self.execute_pop()?,
                BytecodeInstruction::Swap => self.execute_swap()?,

                BytecodeInstruction::Jump(target) => {
                    self.pc = *target;
                    continue; // Don't increment PC
                }
                BytecodeInstruction::JumpIfTrue(target) => {
                    if let Some(Value::Boolean(true)) = self.stack.last() {
                        self.pc = *target;
                        continue;
                    }
                }
                BytecodeInstruction::JumpIfFalse(target) => {
                    if let Some(Value::Boolean(false)) = self.stack.last() {
                        self.pc = *target;
                        continue;
                    }
                }

                BytecodeInstruction::Call(func_name) => {
                    self.call_stack.push(self.pc);
                    // Function lookup and execution would be implemented here
                    // For now, just skip
                }
                BytecodeInstruction::Return => {
                    if let Some(return_pc) = self.call_stack.pop() {
                        self.pc = return_pc;
                    } else {
                        // Return from main function
                        break;
                    }
                    continue;
                }

                BytecodeInstruction::PushScope => {
                    let new_scope = Scope::child(self.current_scope.clone());
                    self.current_scope = new_scope;
                }
                BytecodeInstruction::PopScope => {
                    // Check if we have a parent scope before popping
                    if self.current_scope.borrow().parent.is_some() {
                        // Clone the parent before dropping the borrow
                        let parent = self.current_scope.borrow().parent.as_ref().unwrap().clone();
                        self.current_scope = parent;
                    } else {
                        return Err("Cannot pop root scope".to_string());
                    }
                }
                BytecodeInstruction::StoreVar(name) => {
                    let value = self.stack.pop().ok_or("Stack underflow for StoreVar")?;
                    let ctx_value = self.convert_to_context_value(value)?;
                    Scope::set_local(&self.current_scope, name.clone(), ctx_value);
                }
                BytecodeInstruction::LoadVar(name) => {
                    let ctx_value = Scope::get_variable(&self.current_scope, name)
                        .ok_or_else(|| format!("Undefined variable: {}", name))?;
                    let value = self.convert_from_context_value(ctx_value)?;
                    self.stack.push(value);
                }

                BytecodeInstruction::RuntimeCall(func_name) => {
                    self.execute_runtime_call(func_name)?;
                }

                BytecodeInstruction::Halt => break,
            }

            self.pc += 1;
        }

        // Return top of stack as result
        Ok(self.stack.pop())
    }

    /// Execute addition
    fn execute_add(&mut self) -> Result<(), String> {
        let b = self.stack.pop().ok_or("Stack underflow")?;
        let a = self.stack.pop().ok_or("Stack underflow")?;

        match (a, b) {
            (Value::Integer(a_val), Value::Integer(b_val)) => {
                self.stack.push(Value::Integer(a_val + b_val));
            }
            (Value::Float(a_val), Value::Float(b_val)) => {
                self.stack.push(Value::Float(a_val + b_val));
            }
            _ => return Err("Type mismatch in addition".to_string()),
        }

        Ok(())
    }

    /// Execute subtraction
    fn execute_sub(&mut self) -> Result<(), String> {
        let b = self.stack.pop().ok_or("Stack underflow")?;
        let a = self.stack.pop().ok_or("Stack underflow")?;

        match (a, b) {
            (Value::Integer(a_val), Value::Integer(b_val)) => {
                self.stack.push(Value::Integer(a_val - b_val));
            }
            (Value::Float(a_val), Value::Float(b_val)) => {
                self.stack.push(Value::Float(a_val - b_val));
            }
            _ => return Err("Type mismatch in subtraction".to_string()),
        }

        Ok(())
    }

    /// Execute multiplication
    fn execute_mul(&mut self) -> Result<(), String> {
        let b = self.stack.pop().ok_or("Stack underflow")?;
        let a = self.stack.pop().ok_or("Stack underflow")?;

        match (a, b) {
            (Value::Integer(a_val), Value::Integer(b_val)) => {
                self.stack.push(Value::Integer(a_val * b_val));
            }
            (Value::Float(a_val), Value::Float(b_val)) => {
                self.stack.push(Value::Float(a_val * b_val));
            }
            _ => return Err("Type mismatch in multiplication".to_string()),
        }

        Ok(())
    }

    /// Execute division
    fn execute_div(&mut self) -> Result<(), String> {
        let b = self.stack.pop().ok_or("Stack underflow")?;
        let a = self.stack.pop().ok_or("Stack underflow")?;

        match (a, b) {
            (Value::Integer(a_val), Value::Integer(b_val)) => {
                if b_val == 0 {
                    return Err("Division by zero".to_string());
                }
                self.stack.push(Value::Integer(a_val / b_val));
            }
            (Value::Float(a_val), Value::Float(b_val)) => {
                self.stack.push(Value::Float(a_val / b_val));
            }
            _ => return Err("Type mismatch in division".to_string()),
        }

        Ok(())
    }

    /// Execute vector addition
    fn execute_vector_add(&mut self) -> Result<(), String> {
        let b = self.stack.pop().ok_or("Stack underflow")?;
        let a = self.stack.pop().ok_or("Stack underflow")?;

        match (a, b) {
            (Value::Vector(vec_a), Value::Vector(vec_b)) => {
                unsafe {
                    // Call runtime vector addition
                    let result_ptr = crate::vector::vexl_vec_add_i64(vec_a, vec_b);
                    self.stack.push(Value::Vector(result_ptr));
                }
            }
            _ => return Err("Vector addition requires two vectors".to_string()),
        }

        Ok(())
    }

    /// Execute vector multiplication
    fn execute_vector_mul(&mut self) -> Result<(), String> {
        let b = self.stack.pop().ok_or("Stack underflow")?;
        let a = self.stack.pop().ok_or("Stack underflow")?;

        match (a, b) {
            (Value::Vector(vec_a), Value::Vector(vec_b)) => {
                unsafe {
                    // Call runtime vector multiplication
                    // Note: This would need to be implemented in runtime
                    let result_ptr = vec_a; // Placeholder
                    self.stack.push(Value::Vector(result_ptr));
                }
            }
            _ => return Err("Vector multiplication requires two vectors".to_string()),
        }

        Ok(())
    }

    /// Execute vector scaling
    fn execute_vector_scale(&mut self, scale: f64) -> Result<(), String> {
        let vec = self.stack.pop().ok_or("Stack underflow")?;

        match vec {
            Value::Vector(vec_ptr) => {
                unsafe {
                    // Call runtime vector scaling
                    let result_ptr = crate::vector::vexl_vec_mul_scalar_i64(vec_ptr, scale as i64);
                    self.stack.push(Value::Vector(result_ptr));
                }
            }
            _ => return Err("Vector scaling requires a vector".to_string()),
        }

        Ok(())
    }

    /// Execute stack operations
    fn execute_dup(&mut self) -> Result<(), String> {
        let top = self.stack.last().ok_or("Stack underflow")?;
        self.stack.push(top.clone());
        Ok(())
    }

    fn execute_pop(&mut self) -> Result<(), String> {
        self.stack.pop().ok_or("Stack underflow")?;
        Ok(())
    }

    fn execute_swap(&mut self) -> Result<(), String> {
        let len = self.stack.len();
        if len < 2 {
            return Err("Stack underflow for swap".to_string());
        }
        self.stack.swap(len - 1, len - 2);
        Ok(())
    }

    /// Execute runtime function call
    fn execute_runtime_call(&mut self, func_name: &str) -> Result<(), String> {
        match func_name {
            "vexl_print_int" => {
                if let Some(Value::Integer(val)) = self.stack.last() {
                    println!("{}", val);
                }
            }
            "vexl_print_float" | "vexl_print_f64" => {
                if let Some(Value::Float(val)) = self.stack.last() {
                    println!("{}", val);
                }
            }
            "vexl_print_string" => {
                if let Some(Value::String(val)) = self.stack.last() {
                    println!("{}", val);
                }
            }
            // Handle the older naming for compatibility
            "print_int" => {
                if let Some(Value::Integer(val)) = self.stack.last() {
                    println!("{}", val);
                }
            }
            "print_float" => {
                if let Some(Value::Float(val)) = self.stack.last() {
                    println!("{}", val);
                }
            }
            "print_string" => {
                if let Some(Value::String(val)) = self.stack.last() {
                    println!("{}", val);
                }
            }
            _ => return Err(format!("Unknown runtime function: {}", func_name)),
        }
        Ok(())
    }

    /// Get current stack state (for debugging)
    pub fn stack_state(&self) -> &[Value] {
        &self.stack
    }

    /// Get current program counter
    pub fn program_counter(&self) -> usize {
        self.pc
    }

    /// Convert interpreter Value to context Value
    fn convert_to_context_value(&self, value: Value) -> Result<crate::context::Value, String> {
        match value {
            Value::Integer(i) => Ok(crate::context::Value::Integer(i)),
            Value::Float(f) => Ok(crate::context::Value::Float(f)),
            Value::String(s) => Ok(crate::context::Value::String(s)),
            Value::Boolean(b) => Ok(crate::context::Value::Boolean(b)),
            Value::Vector(ptr) => Ok(crate::context::Value::Vector(crate::context::VectorRef::owned(ptr))),
        }
    }

    /// Convert context Value to interpreter Value
    fn convert_from_context_value(&self, value: crate::context::Value) -> Result<Value, String> {
        match value {
            crate::context::Value::Integer(i) => Ok(Value::Integer(i)),
            crate::context::Value::Float(f) => Ok(Value::Float(f)),
            crate::context::Value::String(s) => Ok(Value::String(s)),
            crate::context::Value::Boolean(b) => Ok(Value::Boolean(b)),
            crate::context::Value::Vector(vec_ref) => Ok(Value::Vector(vec_ref.ptr())),
            _ => Err("Unsupported value type conversion".to_string()),
        }
    }
}

/// Bytecode compiler (VIR to bytecode)
pub struct BytecodeCompiler {
    instructions: Vec<BytecodeInstruction>,
    constants: Vec<Value>,
}

impl BytecodeCompiler {
    /// Create a new bytecode compiler
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
        }
    }

    /// Add a constant and return its index
    pub fn add_constant(&mut self, value: Value) -> usize {
        self.constants.push(value);
        self.constants.len() - 1
    }

    /// Add an instruction
    pub fn add_instruction(&mut self, instruction: BytecodeInstruction) {
        self.instructions.push(instruction);
    }

    /// Compile VIR instruction to bytecode
    pub fn compile_vir_instruction(&mut self, vir_instr: &vir::InstructionKind) {
        match vir_instr {
            vir::InstructionKind::ConstInt(n) => {
                self.add_instruction(BytecodeInstruction::LoadConstI64(*n));
            }
            vir::InstructionKind::ConstFloat(f) => {
                self.add_instruction(BytecodeInstruction::LoadConstF64(*f));
            }
            vir::InstructionKind::ConstString(s) => {
                self.add_instruction(BytecodeInstruction::LoadConstString(s.clone()));
            }
            vir::InstructionKind::Add(_, _) => {
                self.add_instruction(BytecodeInstruction::Add);
            }
            vir::InstructionKind::Sub(_, _) => {
                self.add_instruction(BytecodeInstruction::Sub);
            }
            vir::InstructionKind::Mul(_, _) => {
                self.add_instruction(BytecodeInstruction::Mul);
            }
            vir::InstructionKind::Div(_, _) => {
                self.add_instruction(BytecodeInstruction::Div);
            }
            vir::InstructionKind::RuntimeCall { function_name, .. } => {
                self.add_instruction(BytecodeInstruction::RuntimeCall(function_name.clone()));
            }
            vir::InstructionKind::PushScope => {
                self.add_instruction(BytecodeInstruction::PushScope);
            }
            vir::InstructionKind::PopScope => {
                self.add_instruction(BytecodeInstruction::PopScope);
            }
            vir::InstructionKind::StoreVar { name, .. } => {
                self.add_instruction(BytecodeInstruction::StoreVar(name.clone()));
            }
            vir::InstructionKind::LoadVar(name) => {
                self.add_instruction(BytecodeInstruction::LoadVar(name.clone()));
            }
            // Handle other VIR instructions as needed
            _ => {
                // For now, skip unsupported instructions
                // In a full implementation, all VIR instructions would be handled
            }
        }
    }

    /// Build the final bytecode program
    pub fn build(self) -> BytecodeProgram {
        BytecodeProgram {
            instructions: self.instructions,
            constants: self.constants,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpreter_creation() {
        let interpreter = Interpreter::new();
        assert_eq!(interpreter.program_counter(), 0);
        assert!(interpreter.stack_state().is_empty());
    }

    #[test]
    fn test_simple_arithmetic() {
        let mut compiler = BytecodeCompiler::new();

        // Program: 5 + 3
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(5));
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(3));
        compiler.add_instruction(BytecodeInstruction::Add);
        compiler.add_instruction(BytecodeInstruction::Halt);

        let program = compiler.build();

        let mut interpreter = Interpreter::new();
        let result = interpreter.execute(&program).unwrap();

        match result {
            Some(Value::Integer(8)) => {} // Expected result
            _ => panic!("Expected integer 8"),
        }
    }

    #[test]
    fn test_stack_operations() {
        let mut compiler = BytecodeCompiler::new();

        // Program: dup and add (5 + 5)
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(5));
        compiler.add_instruction(BytecodeInstruction::Dup);
        compiler.add_instruction(BytecodeInstruction::Add);
        compiler.add_instruction(BytecodeInstruction::Halt);

        let program = compiler.build();

        let mut interpreter = Interpreter::new();
        let result = interpreter.execute(&program).unwrap();

        match result {
            Some(Value::Integer(10)) => {} // Expected result
            _ => panic!("Expected integer 10"),
        }
    }

    #[test]
    fn test_variable_scoping() {
        let mut compiler = BytecodeCompiler::new();

        // Program: let x = 5 in x + 1
        compiler.add_instruction(BytecodeInstruction::PushScope);
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(5));
        compiler.add_instruction(BytecodeInstruction::StoreVar("x".to_string()));
        compiler.add_instruction(BytecodeInstruction::LoadVar("x".to_string()));
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(1));
        compiler.add_instruction(BytecodeInstruction::Add);
        compiler.add_instruction(BytecodeInstruction::PopScope);
        compiler.add_instruction(BytecodeInstruction::Halt);

        let program = compiler.build();

        let mut interpreter = Interpreter::new();
        let result = interpreter.execute(&program).unwrap();

        match result {
            Some(Value::Integer(6)) => {} // Expected result: 5 + 1 = 6
            _ => panic!("Expected integer 6, got {:?}", result),
        }
    }

    #[test]
    fn test_nested_scopes() {
        let mut compiler = BytecodeCompiler::new();

        // Program: let x = 5 in let y = x + 1 in y * 2
        compiler.add_instruction(BytecodeInstruction::PushScope); // Outer scope
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(5));
        compiler.add_instruction(BytecodeInstruction::StoreVar("x".to_string()));
        compiler.add_instruction(BytecodeInstruction::PushScope); // Inner scope
        compiler.add_instruction(BytecodeInstruction::LoadVar("x".to_string()));
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(1));
        compiler.add_instruction(BytecodeInstruction::Add);
        compiler.add_instruction(BytecodeInstruction::StoreVar("y".to_string()));
        compiler.add_instruction(BytecodeInstruction::LoadVar("y".to_string()));
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(2));
        compiler.add_instruction(BytecodeInstruction::Mul);
        compiler.add_instruction(BytecodeInstruction::PopScope); // End inner scope
        compiler.add_instruction(BytecodeInstruction::PopScope); // End outer scope
        compiler.add_instruction(BytecodeInstruction::Halt);

        let program = compiler.build();

        let mut interpreter = Interpreter::new();
        let result = interpreter.execute(&program).unwrap();

        match result {
            Some(Value::Integer(12)) => {} // Expected result: (5 + 1) * 2 = 12
            _ => panic!("Expected integer 12, got {:?}", result),
        }
    }

    #[test]
    fn test_variable_shadowing() {
        let mut compiler = BytecodeCompiler::new();

        // Program: let x = 5 in let x = x + 1 in x
        compiler.add_instruction(BytecodeInstruction::PushScope); // Outer scope
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(5));
        compiler.add_instruction(BytecodeInstruction::StoreVar("x".to_string()));
        compiler.add_instruction(BytecodeInstruction::PushScope); // Inner scope
        compiler.add_instruction(BytecodeInstruction::LoadVar("x".to_string())); // Should get 5 from outer scope
        compiler.add_instruction(BytecodeInstruction::LoadConstI64(1));
        compiler.add_instruction(BytecodeInstruction::Add);
        compiler.add_instruction(BytecodeInstruction::StoreVar("x".to_string())); // Shadow outer x
        compiler.add_instruction(BytecodeInstruction::LoadVar("x".to_string())); // Should get 6 from inner scope
        compiler.add_instruction(BytecodeInstruction::PopScope); // End inner scope
        compiler.add_instruction(BytecodeInstruction::PopScope); // End outer scope
        compiler.add_instruction(BytecodeInstruction::Halt);

        let program = compiler.build();

        let mut interpreter = Interpreter::new();
        let result = interpreter.execute(&program).unwrap();

        match result {
            Some(Value::Integer(6)) => {} // Expected result: shadowed x = 6
            _ => panic!("Expected integer 6, got {:?}", result),
        }
    }
}
