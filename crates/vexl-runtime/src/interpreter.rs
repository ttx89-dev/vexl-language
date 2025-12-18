//! Bytecode Interpreter for VEXL
//!
//! Fast startup interpreter that executes VEXL bytecode directly.
//! Used for REPL and small scripts to avoid JIT compilation overhead.

use crate::vector::Vector;
use std::collections::HashMap;

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
    /// Variables
    variables: HashMap<String, Value>,
    /// Call stack for function calls
    call_stack: Vec<usize>,
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        Self {
            pc: 0,
            stack: Vec::new(),
            variables: HashMap::new(),
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
}
