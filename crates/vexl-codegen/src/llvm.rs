//! LLVM code generation backend

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, FloatValue};
use inkwell::types::BasicTypeEnum;
use inkwell::IntPredicate;
use std::collections::HashMap;

use vexl_ir::{VirModule, VirFunction, BasicBlock, Instruction, InstructionKind, Terminator, ValueId, BlockId};

/// LLVM code generation context
pub struct LLVMCodegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    
    /// Map VIR value IDs to LLVM values
    values: HashMap<ValueId, BasicValueEnum<'ctx>>,
}

impl<'ctx> LLVMCodegen<'ctx> {
    /// Create new LLVM codegen context
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        
        Self {
            context,
            module,
            builder,
            values: HashMap::new(),
        }
    }
    
    /// Compile a VIR module to LLVM IR
    pub fn compile_module(mut self, vir_module: &VirModule) -> Result<Module<'ctx>, String> {
        // Compile all functions in the module
        for (name, func) in &vir_module.functions {
            self.compile_function(name, func)?;
        }
        
        // If no functions, create a simple main that returns 0
        if vir_module.functions.is_empty() {
            let i64_type = self.context.i64_type();
            let fn_type = i64_type.fn_type(&[], false);
            let function = self.module.add_function("main", fn_type, None);
            
            let entry = self.context.append_basic_block(function, "entry");
            self.builder.position_at_end(entry);
            
            let zero = i64_type.const_int(0, false);
            self.builder.build_return(Some(&zero)).unwrap();
        }
        
        Ok(self.module)
    }
    
    /// Compile a VIR function to LLVM
    fn compile_function(&mut self, name: &str, func: &VirFunction) -> Result<FunctionValue<'ctx>, String> {
        // Create function signature
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[], false);
        let function = self.module.add_function(name, fn_type, None);
        
        // Create entry block
        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);
        
        // Compile the entry basic block
        if let Some(block) = func.blocks.get(&func.entry_block) {
            self.compile_basic_block(block)?;
        }
        
        Ok(function)
    }
    
    /// Compile a basic block
    fn compile_basic_block(&mut self, block: &BasicBlock) -> Result<(), String> {
        // Compile all instructions
        for inst in &block.instructions {
            let value = self.compile_instruction(inst)?;
            self.values.insert(inst.result, value);
        }
        
        // Compile terminator
        self.compile_terminator(&block.terminator)?;
        
        Ok(())
    }
    
    /// Compile a single instruction
    fn compile_instruction(&mut self, inst: &Instruction) -> Result<BasicValueEnum<'ctx>, String> {
        match &inst.kind {
            InstructionKind::ConstInt(n) => {
                let i64_type = self.context.i64_type();
                let value = i64_type.const_int(*n as u64, true);
                Ok(value.into())
            }
            
            InstructionKind::ConstFloat(f) => {
                let f64_type = self.context.f64_type();
                let value = f64_type.const_float(*f);
                Ok(value.into())
            }
            
            InstructionKind::Add(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                
                // Both must be integers for now
                let left_int = left_val.into_int_value();
                let right_int = right_val.into_int_value();
                
                let result = self.builder.build_int_add(left_int, right_int, "add").unwrap();
                Ok(result.into())
            }
            
            InstructionKind::Sub(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let left_int = left_val.into_int_value();
                let right_int = right_val.into_int_value();
                let result = self.builder.build_int_sub(left_int, right_int, "sub").unwrap();
                Ok(result.into())
            }
            
            InstructionKind::Mul(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let left_int = left_val.into_int_value();
                let right_int = right_val.into_int_value();
                let result = self.builder.build_int_mul(left_int, right_int, "mul").unwrap();
                Ok(result.into())
            }
            
            InstructionKind::Div(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let left_int = left_val.into_int_value();
                let right_int = right_val.into_int_value();
                let result = self.builder.build_int_signed_div(left_int, right_int, "div").unwrap();
                Ok(result.into())
            }
            
            _ => Err(format!("Codegen not yet implemented for {:?}", inst.kind)),
        }
    }
    
    /// Compile a terminator instruction
    fn compile_terminator(&mut self, term: &Terminator) -> Result<(), String> {
        match term {
            Terminator::Return(value_id) => {
                let value = self.get_value(*value_id)?;
                self.builder.build_return(Some(&value)).unwrap();
                Ok(())
            }
            
            _ => Err(format!("Terminator not yet implemented: {:?}", term)),
        }
    }
    
    /// Get a compiled LLVM value by VIR ID
    fn get_value(&self, id: ValueId) -> Result<BasicValueEnum<'ctx>, String> {
        self.values
            .get(&id)
            .copied()
            .ok_or_else(|| format!("Value {:?} not found", id))
    }
}

/// Compile a VIR module to LLVM IR and return as string
pub fn codegen_to_string(vir_module: &VirModule) -> Result<String, String> {
    let context = Context::create();
    let codegen = LLVMCodegen::new(&context, "vexl_module");
    let module = codegen.compile_module(vir_module)?;
    
    Ok(module.print_to_string().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_constant() {
        let mut module = VirModule::new();
        let v1 = module.fresh_value();
        let block_id = module.fresh_block();
        
        let block = BasicBlock {
            id: block_id,
            instructions: vec![
                Instruction {
                    result: v1,
                    kind: InstructionKind::ConstInt(42),
                },
            ],
            terminator: Terminator::Return(v1),
        };
        
        let mut func = VirFunction {
            name: "main".to_string(),
            params: vec![],
            blocks: HashMap::from([(block_id, block)]),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
        };
        
        module.add_function("main".to_string(), func);
        
        let llvm_ir = codegen_to_string(&module).unwrap();
        assert!(llvm_ir.contains("ret i64 42"));
    }
    
    #[test]
    fn test_codegen_add() {
        let mut module = VirModule::new();
        let v1 = module.fresh_value();
        let v2 = module.fresh_value();
        let v3 = module.fresh_value();
        let block_id = module.fresh_block();
        
        let block = BasicBlock {
            id: block_id,
            instructions: vec![
                Instruction {
                    result: v1,
                    kind: InstructionKind::ConstInt(1),
                },
                Instruction {
                    result: v2,
                    kind: InstructionKind::ConstInt(2),
                },
                Instruction {
                    result: v3,
                    kind: InstructionKind::Add(v1, v2),
                },
            ],
            terminator: Terminator::Return(v3),
        };
        
        let mut func = VirFunction {
            name: "main".to_string(),
            params: vec![],
            blocks: HashMap::from([(block_id, block)]),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
        };
        
        module.add_function("main".to_string(), func);
        
        let llvm_ir = codegen_to_string(&module).unwrap();
        // Just verify it compiles and returns valid LLVM IR
        assert!(!llvm_ir.is_empty());
        assert!(llvm_ir.contains("define"));
    }
}
