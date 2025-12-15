//! AST to VIR lowering

use crate::{VirModule, Instruction, InstructionKind, ValueId};
use vexl_syntax::ast::{Expr, BinOpKind};
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
    
    /// Emit an instruction and return its result value
    fn emit(&mut self, kind: InstructionKind) -> ValueId {
        let result = self.module.fresh_value();
        self.current_instructions.push(Instruction { result, kind });
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
            
            Expr::App { func, args, .. } => {
                // Handle function calls
                match &**func {
                    Expr::Ident(name, _) => {
                        match name.as_str() {
                            "sum" => {
                                if args.len() == 1 {
                                    let vec_val = self.lower_expr(&args[0])?;
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_vec_sum".to_string(),
                                        args: vec![vec_val],
                                    }))
                                } else {
                                    Err("sum() expects one argument".to_string())
                                }
                            }
                            "print" => {
                                if args.len() == 1 {
                                    let arg_val = self.lower_expr(&args[0])?;
                                    // For now, assume all prints are integers
                                    // TODO: Type-based dispatch to print_int, print_string, etc.
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_print_int".to_string(),
                                        args: vec![arg_val],
                                    }))
                                } else {
                                    Err("print() expects one argument".to_string())
                                }
                            }
                            "len" => {
                                if args.len() == 1 {
                                    let vec_val = self.lower_expr(&args[0])?;
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_vec_len".to_string(),
                                        args: vec![vec_val],
                                    }))
                                } else {
                                    Err("len() expects one argument".to_string())
                                }
                            }
                            "get" => {
                                if args.len() == 2 {
                                    let vec_val = self.lower_expr(&args[0])?;
                                    let index_val = self.lower_expr(&args[1])?;
                                    Ok(self.emit(InstructionKind::RuntimeCall {
                                        function_name: "vexl_vec_get_i64".to_string(),
                                        args: vec![vec_val, index_val],
                                    }))
                                } else {
                                    Err("get() expects two arguments (vec, index)".to_string())
                                }
                            }
                            _ => Err(format!("Unknown function: {}", name)),
                        }
                    }
                    _ => Err("Complex function calls not yet supported".to_string()),
                }
            }

            Expr::Let { name, value, body, .. } => {
                let val = self.lower_expr(value)?;
                self.variables.insert(name.clone(), val);
                self.lower_expr(body)
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

                // Regular function call
                let func_val = self.lower_expr(func)?;
                let mut call_args = vec![input]; // Input as first argument
                for arg in args {
                    call_args.push(self.lower_expr(arg)?);
                }

                Ok(self.emit(InstructionKind::Call {
                    func: func_val,
                    args: call_args,
                }))
            }

            _ => {
                // For now, treat as function call with single argument
                let func_val = self.lower_expr(stage)?;
                Ok(self.emit(InstructionKind::Call {
                    func: func_val,
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

/// Lower an expression to a VIR module
pub fn lower_to_vir(expr: &Expr) -> Result<VirModule, String> {
    let mut ctx = LoweringContext::new();
    let result = ctx.lower_expr(expr)?;
    
    // Create a main function that returns the result
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
