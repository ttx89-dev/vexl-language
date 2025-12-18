//! AST to VIR lowering

use crate::{VirModule, VirFunction, VirType, FunctionSignature, BasicBlock, Terminator, Instruction, InstructionKind, ValueId, BlockId};
use vexl_syntax::ast::{Decl, Expr, BinOpKind, Type};
use std::collections::HashMap;

/// Context for lowering AST to VIR
pub struct LoweringContext {
    module: VirModule,
    current_instructions: Vec<Instruction>,
    variables: HashMap<String, ValueId>,
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            module: VirModule::new(),
            current_instructions: Vec::new(),
            variables: HashMap::new(),
        }
    }

    /// Check if a value came from a string constant
    fn is_string_value(&self, value_id: ValueId) -> bool {
        // Search through current instructions to see if this value came from ConstString
        for instr in &self.current_instructions {
            if instr.result == value_id {
                match &instr.kind {
                    InstructionKind::ConstString(_) => return true,
                    _ => return false,
                }
            }
        }
        // If not found in current instructions, assume it's not a string
        // In a full implementation, we'd track value types throughout
        false
    }

    /// Check if a value came from a float constant
    fn is_float_value(&self, value_id: ValueId) -> bool {
        // Search through current instructions to see if this value came from ConstFloat
        for instr in &self.current_instructions {
            if instr.result == value_id {
                match &instr.kind {
                    InstructionKind::ConstFloat(_) => return true,
                    _ => return false,
                }
            }
        }
        // If not found in current instructions, assume it's not a float
        false
    }

    /// Get the length of a string value (simplified for string constants)
    fn get_string_length(&mut self, value_id: ValueId) -> Result<ValueId, String> {
        // Search through current instructions to find the ConstString that produced this value
        for instr in &self.current_instructions {
            if instr.result == value_id {
                match &instr.kind {
                    InstructionKind::ConstString(s) => {
                        // Return the length as a constant
                        return Ok(self.emit(InstructionKind::ConstInt(s.len() as i64)));
                    }
                    _ => return Err("Cannot get length of non-string value".to_string()),
                }
            }
        }
        // If not found, assume it's a string variable and we need to call vexl_string_len
        // For now, return a placeholder length of 0 - this needs proper implementation
        Ok(self.emit(InstructionKind::ConstInt(0)))
    }
    
    /// Emit an instruction and return its result value
    fn emit(&mut self, kind: InstructionKind) -> ValueId {
        let result = self.module.fresh_value();
        self.current_instructions.push(Instruction { result, result_type: None, kind });
        result
    }
    
    /// Lower an expression to VIR, returning the result value
    pub fn lower_expr(&mut self, expr: &Expr) -> Result<ValueId, String> {
        match expr {
            Expr::Int(n, _) => {
                Ok(self.emit(InstructionKind::ConstInt(*n)))
            }
            
            Expr::Float(f, _) => {
                Ok(self.emit(InstructionKind::ConstFloat(*f)))
            }
            
            Expr::String(s, _) => {
                Ok(self.emit(InstructionKind::ConstString(s.clone())))
            }
            
            Expr::Ident(name, _) => {
                self.variables
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }
            
            Expr::Vector(elements, _) => {
                // 1. Lower all elements first
                let mut element_vals = Vec::new();
                for elem in elements {
                    element_vals.push(self.lower_expr(elem)?);
                }

                // 2. Allocate vector
                let count = elements.len() as i64;
                let count_val = self.emit(InstructionKind::ConstInt(count));

                // Call vexl_vec_alloc_i64 runtime function
                let vec_val = self.emit(InstructionKind::RuntimeCall {
                    function_name: "vexl_vec_alloc_i64".to_string(),
                    args: vec![count_val],
                });

                // 3. Populate vector
                for (i, val) in element_vals.into_iter().enumerate() {
                    let index_val = self.emit(InstructionKind::ConstInt(i as i64));
                    self.emit(InstructionKind::RuntimeCall {
                        function_name: "vexl_vec_set_i64".to_string(),
                        args: vec![vec_val, index_val, val], // vec, index, value
                    });
                }

                Ok(vec_val)
            }
            
            Expr::Range(start, end, _) => {
                let start_val = self.lower_expr(start)?;
                let end_val = self.lower_expr(end)?;
                
                Ok(self.emit(InstructionKind::Range {
                    start: start_val,
                    end: end_val,
                }))
            }
            
            Expr::InfiniteRange(start, _) => {
                let start_val = self.lower_expr(start)?;
                
                Ok(self.emit(InstructionKind::InfiniteRange {
                    start: start_val,
                }))
            }
            
            Expr::BinOp { op, left, right, .. } => {
                let left_val = self.lower_expr(left)?;
                let right_val = self.lower_expr(right)?;

                // Special handling for string concatenation
                if *op == BinOpKind::Add {
                    // For now, assume + always means string concatenation if either operand is a string
                    // In a full implementation, we'd have proper type information
                    Ok(self.emit(InstructionKind::RuntimeCall {
                        function_name: "vexl_string_concat".to_string(),
                        args: vec![left_val, right_val],
                    }))
                } else {
                    // Regular arithmetic operations
                    let kind = match op {
                        BinOpKind::Add => InstructionKind::Add(left_val, right_val),
                        BinOpKind::Sub => InstructionKind::Sub(left_val, right_val),
                        BinOpKind::Mul => InstructionKind::Mul(left_val, right_val),
                        BinOpKind::Div => InstructionKind::Div(left_val, right_val),
                        BinOpKind::MatMul => InstructionKind::MatMul(left_val, right_val),
                        BinOpKind::Outer => InstructionKind::Outer(left_val, right_val),
                        BinOpKind::Dot => InstructionKind::Dot(left_val, right_val),
                        BinOpKind::Eq => InstructionKind::Eq(left_val, right_val),
                        BinOpKind::NotEq => InstructionKind::NotEq(left_val, right_val),
                        BinOpKind::Lt => InstructionKind::Lt(left_val, right_val),
                        BinOpKind::Le => InstructionKind::Le(left_val, right_val),
                        BinOpKind::Gt => InstructionKind::Gt(left_val, right_val),
                        BinOpKind::Ge => InstructionKind::Ge(left_val, right_val),
                    };

                    Ok(self.emit(kind))
                }
            }
            
            Expr::App { func, args, .. } => {
                // Handle function calls
                match &**func {
                    Expr::Ident(name, _) => {
                        // Lower all arguments first
                        let mut lowered_args = Vec::new();
                        for arg in args {
                            lowered_args.push(self.lower_expr(arg)?);
                        }

                        match name.as_str() {
                            "sum" => {
                                if lowered_args.len() == 1 {
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_vec_sum".to_string(),
                                        args: lowered_args,
                                    }))
                                } else {
                                    Err("sum() expects one argument".to_string())
                                }
                            }
                            "print" => {
                                if lowered_args.len() == 1 {
                                    // Type-based dispatch for print
                                    // For now, we dispatch based on the instruction that produced the argument
                                    let arg_val = &lowered_args[0];
                                    // Check if this argument came from a string constant
                                    // This is a simplified approach - in a full implementation,
                                    // we'd have type information available during lowering
                                    let func_name = if self.is_string_value(*arg_val) {
                                        "vexl_print_string"
                                    } else if self.is_float_value(*arg_val) {
                                        "vexl_print_float"
                                    } else {
                                        "vexl_print_int"
                                    };

                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: func_name.to_string(),
                                        args: lowered_args,
                                    }))
                                } else {
                                    Err("print() expects one argument".to_string())
                                }
                            }
                            "print_int" => {
                                if lowered_args.len() == 1 {
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_print_int".to_string(),
                                        args: lowered_args,
                                    }))
                                } else {
                                    Err("print_int() expects one argument".to_string())
                                }
                            }
                            "print_string" => {
                                if lowered_args.len() == 1 {
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_print_string".to_string(),
                                        args: lowered_args,
                                    }))
                                } else {
                                    Err("print_string() expects one argument".to_string())
                                }
                            }
                            "print_float" => {
                                if lowered_args.len() == 1 {
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_print_float".to_string(),
                                        args: lowered_args,
                                    }))
                                } else {
                                    Err("print_float() expects one argument".to_string())
                                }
                            }
                            "len" => {
                                if lowered_args.len() == 1 {
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_vec_len".to_string(),
                                        args: lowered_args,
                                    }))
                                } else {
                                    Err("len() expects one argument".to_string())
                                }
                            }
                            "get" => {
                                if lowered_args.len() == 2 {
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_vec_get_i64".to_string(),
                                        args: lowered_args,
                                    }))
                                } else {
                                    Err("get() expects two arguments (vec, index)".to_string())
                                }
                            }
                            // User-defined function calls
                            _ => {
                                Ok(self.emit(InstructionKind::Call {
                                    func: name.clone(),
                                    args: lowered_args,
                                }))
                            }
                        }
                    }
                    _ => Err("Complex function calls not yet supported".to_string()),
                }
            }

            Expr::Let { name, value, body, .. } => {
                // For top-level let bindings, store the variable globally
                // Evaluate the value and store it
                let val = self.lower_expr(value)?;
                self.emit(InstructionKind::StoreVar {
                    name: name.clone(),
                    value: val,
                });

                // Store in the symbol table for subsequent lookups
                self.variables.insert(name.clone(), val);

                // Evaluate the body (usually dummy for top-level lets)
                let body_result = self.lower_expr(body)?;

                Ok(body_result)
            }

            Expr::Fix { name, body, .. } => {
                // For fix f => e, we create a recursive binding
                // For evaluation purposes, we can treat this as just the body
                // since we're not doing complex recursion analysis
                self.lower_expr(body)
            }

            Expr::Lambda { params, body, .. } => {
                // Create a new function with parameters
                let lambda_index = self.module.functions.len();
                let func_name = format!("lambda_{}", lambda_index);

                // Create parameter ValueIds
                let mut param_values = Vec::new();
                let mut param_map = HashMap::new();

                for param_name in params {
                    let param_value = self.module.fresh_value();
                    param_values.push(param_value);
                    param_map.insert(param_name.clone(), param_value);
                }

                // Save current context
                let old_variables = self.variables.clone();
                let old_instructions = self.current_instructions.clone();

                // Set up parameter bindings in new scope
                self.variables.extend(param_map);
                self.current_instructions.clear();

                // Lower the function body
                let body_result = self.lower_expr(body)?;

                // Create function signature
                let param_types = vec![VirType::Int64; params.len()]; // Assume all params are i64 for now
                let signature = FunctionSignature::new(param_types, VirType::Int64);

                // Create basic block
                let block_id = self.module.fresh_block();
                let block = BasicBlock {
                    id: block_id,
                    instructions: std::mem::take(&mut self.current_instructions),
                    terminator: Terminator::Return(body_result),
                };

                // Create function
                let mut blocks = HashMap::new();
                blocks.insert(block_id, block);

                let func = VirFunction {
                    name: func_name.clone(),
                    params: param_values,
                    blocks,
                    entry_block: block_id,
                    effect: vexl_core::Effect::Pure,
                    signature,
                };

                // Add function to module
                self.module.add_function(func_name.clone(), func);

                // Restore context
                self.variables = old_variables;
                self.current_instructions = old_instructions;

                // Return lambda index as identifier (0 for lambda_0, 1 for lambda_1, etc.)
                Ok(self.emit(InstructionKind::ConstInt(lambda_index as i64)))
            }
            
            
            Expr::Pipeline { stages, .. } => {
                // Pipeline: data |> f |> g
                // Lower left-to-right, threading value through stages
                if stages.is_empty() {
                    return Err("Empty pipeline".to_string());
                }

                // Lower first stage (the data)
                let mut current_value = self.lower_expr(&stages[0])?;

                // For subsequent stages, apply them as function calls
                for stage in &stages[1..] {
                    current_value = self.lower_pipeline_stage(current_value, stage)?;
                }

                Ok(current_value)
            }
        
        _ => Err(format!("Lowering not yet implemented for {:?}", expr)),
        }
    }
    
    /// Lower a pipeline stage (function application to current value)
    fn lower_pipeline_stage(&mut self, input: ValueId, stage: &Expr) -> Result<ValueId, String> {
        match stage {
            Expr::App { func, args, .. } => {
                // Check if this is a map/filter/reduce call
                if let Expr::Ident(func_name, _) = &**func {
                    match func_name.as_str() {
                        "map" => {
                            if args.len() == 1 {
                                // map(f) - generate parallel map call
                                let lambda_val = self.lower_expr(&args[0])?;
                                let zero_val = self.emit(InstructionKind::ConstInt(0)); // 0 = auto threads
                                return Ok(self.emit(InstructionKind::RuntimeCall {
                                    function_name: "vexl_vec_map_parallel".to_string(),
                                    args: vec![input, lambda_val, zero_val],
                                }));
                            }
                        }
                        "filter" => {
                            if args.len() == 1 {
                                // filter(pred) - generate filter call
                                let lambda_val = self.lower_expr(&args[0])?;
                                return Ok(self.emit(InstructionKind::RuntimeCall {
                                    function_name: "vexl_vec_filter".to_string(),
                                    args: vec![input, lambda_val],
                                }));
                            }
                        }
                        "reduce" => {
                            if args.len() == 2 {
                                // reduce(init, f) - generate parallel reduce call
                                let init_val = self.lower_expr(&args[0])?;
                                let lambda_val = self.lower_expr(&args[1])?;
                                let zero_val = self.emit(InstructionKind::ConstInt(0));
                                return Ok(self.emit(InstructionKind::RuntimeCall {
                                    function_name: "vexl_vec_reduce_parallel".to_string(),
                                    args: vec![input, init_val, lambda_val, zero_val],
                                }));
                            }
                        }
                        _ => {}
                    }
                }

                // Regular function call - extract function name
                let func_name = match &**func {
                    Expr::Ident(name, _) => name.clone(),
                    _ => return Err("Complex function calls in pipelines not yet supported".to_string()),
                };

                let mut call_args = vec![input]; // Input as first argument
                for arg in args {
                    call_args.push(self.lower_expr(arg)?);
                }

                Ok(self.emit(InstructionKind::Call {
                    func: func_name,
                    args: call_args,
                }))
            }

            _ => {
                // For now, treat as function call with single argument
                let func_name = match stage {
                    Expr::Ident(name, _) => name.clone(),
                    _ => return Err("Complex function calls in pipelines not yet supported".to_string()),
                };
                Ok(self.emit(InstructionKind::Call {
                    func: func_name,
                    args: vec![input],
                }))
            }
        }
    }

    /// Finish lowering and return the module
    pub fn finish(self) -> VirModule {
        self.module
    }
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower declarations to a VIR module
pub fn lower_decls_to_vir(decls: &[Decl]) -> Result<VirModule, String> {
    let mut ctx = LoweringContext::new();
    let mut has_main = false;
    let mut main_instructions = Vec::new();
    let mut last_result = None;

    for decl in decls {
        match decl {
            Decl::Function { name, params, return_type, body, .. } => {
                if name == "main" {
                    has_main = true;
                }

                // Convert AST types to VIR types
                let param_types: Vec<VirType> = params.iter()
                    .map(|(_, ty)| ast_type_to_vir_type(ty))
                    .collect();
                let ret_type = ast_type_to_vir_type(return_type);

                // Create parameter ValueIds and mapping
                let mut param_values = Vec::new();
                let mut param_map = HashMap::new();

                for (param_name, _) in params {
                    let param_value = ctx.module.fresh_value();
                    param_values.push(param_value);
                    param_map.insert(param_name.clone(), param_value);
                }

                // Save current context
                let old_variables = ctx.variables.clone();
                let old_instructions = ctx.current_instructions.clone();

                // Set up parameter bindings in new scope
                ctx.variables.extend(param_map);
                ctx.current_instructions.clear();

                // Lower the function body
                let body_result = ctx.lower_expr(body)?;

                // Create function signature
                let signature = FunctionSignature::new(param_types, ret_type);

                // Create basic block
                let block_id = ctx.module.fresh_block();
                let block = BasicBlock {
                    id: block_id,
                    instructions: std::mem::take(&mut ctx.current_instructions),
                    terminator: Terminator::Return(body_result),
                };

                // Create function
                let mut blocks = HashMap::new();
                blocks.insert(block_id, block);

                let func = VirFunction {
                    name: name.clone(),
                    params: param_values,
                    blocks,
                    entry_block: block_id,
                    effect: vexl_core::Effect::Pure,
                    signature,
                };

                // Add function to module
                ctx.module.add_function(name.clone(), func);

                // Restore context
                ctx.variables = old_variables;
                ctx.current_instructions = old_instructions;
            }
            Decl::Expr(expr) => {
                // For expression declarations, collect them for a single main function
                let result = ctx.lower_expr(expr)?;
                last_result = Some(result);

                // Add instructions to main function
                main_instructions.extend(std::mem::take(&mut ctx.current_instructions));

                has_main = true;
            }
        }
    }

    // Create main function with all expression declarations
    if has_main && !main_instructions.is_empty() {
        let block_id = ctx.module.fresh_block();
        let result_value = last_result.unwrap_or_else(|| ctx.emit(InstructionKind::ConstInt(0)));
        let block = BasicBlock {
            id: block_id,
            instructions: main_instructions,
            terminator: Terminator::Return(result_value),
        };

        let func = VirFunction {
            name: "main".to_string(),
            params: vec![],
            blocks: HashMap::from([(block_id, block)]),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
            signature: FunctionSignature::new(vec![], VirType::Int64),
        };

        ctx.module.add_function("main".to_string(), func);
    } else if !has_main {
        // If no main function was defined, create a default one that returns 0
        let block_id = ctx.module.fresh_block();
        let block = BasicBlock {
            id: block_id,
            instructions: vec![],
            terminator: Terminator::Return(ctx.emit(InstructionKind::ConstInt(0))),
        };

        let func = VirFunction {
            name: "main".to_string(),
            params: vec![],
            blocks: HashMap::from([(block_id, block)]),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
            signature: FunctionSignature::new(vec![], VirType::Int64),
        };

        ctx.module.add_function("main".to_string(), func);
    }

    Ok(ctx.module)
}

/// Convert AST type to VIR type
fn ast_type_to_vir_type(ast_type: &Type) -> VirType {
    match ast_type {
        Type::Int => VirType::Int64,
        Type::Float => VirType::Float64,
        Type::Bool => VirType::Int64, // Booleans as i64 for now
        Type::String => VirType::Pointer, // Strings as pointers
        Type::Vector { .. } => VirType::Pointer, // Vectors as pointers
        Type::Function { .. } => VirType::Pointer, // Function pointers
        Type::Named(name) => match name.as_str() {
            "i64" => VirType::Int64,
            "f64" => VirType::Float64,
            "bool" => VirType::Int64,
            "string" => VirType::Pointer,
            _ => VirType::Pointer, // Unknown named types as pointers
        }
    }
}

/// Lower an expression to a VIR module (backward compatibility)
pub fn lower_to_vir(expr: &Expr) -> Result<VirModule, String> {
    let mut ctx = LoweringContext::new();
    let result = ctx.lower_expr(expr)?;

    eprintln!("DEBUG: lower_to_vir - result value: {:?}", result);
    eprintln!("DEBUG: lower_to_vir - current_instructions count: {}", ctx.current_instructions.len());
    for (i, instr) in ctx.current_instructions.iter().enumerate() {
        eprintln!("DEBUG: lower_to_vir - instruction {}: {:?} -> {:?}", i, instr.kind, instr.result);
    }

    // Create a main function that returns the result
    // For JIT compatibility, main should return i32 in the VIR signature
    let block_id = ctx.module.fresh_block();
    let block = crate::BasicBlock {
        id: block_id,
        instructions: ctx.current_instructions,
        terminator: crate::Terminator::Return(result),
    };

    let func = crate::VirFunction {
        name: "main".to_string(),
        params: vec![],
        blocks: std::collections::HashMap::from([(block_id, block)]),
        entry_block: block_id,
        effect: vexl_core::Effect::Pure,
        signature: crate::FunctionSignature::new(vec![], crate::VirType::Int32), // Use i32 for JIT compatibility
    };

    ctx.module.add_function("main".to_string(), func);

    Ok(ctx.module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_syntax::ast::Span;

    #[test]
    fn test_lower_int() {
        let expr = Expr::Int(42, Span { start: 0, end: 2 });
        let mut ctx = LoweringContext::new();
        let result = ctx.lower_expr(&expr).unwrap();
        assert_eq!(result.0, 0);
    }
    
    #[test]
    fn test_lower_binop() {
        let span = Span { start: 0, end: 5 };
        let expr = Expr::BinOp {
            op: BinOpKind::Add,
            left: Box::new(Expr::Int(1, span)),
            right: Box::new(Expr::Int(2, span)),
            span,
        };
        
        let mut ctx = LoweringContext::new();
        let result = ctx.lower_expr(&expr).unwrap();
        assert_eq!(result.0, 2); // Two constants + one add = value id 2
    }
    
    #[test]
    fn test_lower_vector() {
        let span = Span { start: 0, end: 7 };
        let expr = Expr::Vector(
            vec![
                Expr::Int(1, span),
                Expr::Int(2, span),
                Expr::Int(3, span),
            ],
            span,
        );
        
        let result = lower_to_vir(&expr);
        assert!(result.is_ok());
    }
}
