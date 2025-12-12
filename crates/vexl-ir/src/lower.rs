//! AST to VIR lowering

use crate::{VirModule, VirFunction, BasicBlock, Instruction, InstructionKind, Terminator, ValueId, BlockId};
use vexl_syntax::ast::{Expr, BinOpKind};
use vexl_types::effect_check::{EffectEnv, infer_effect};
use std::collections::HashMap;

/// Context for lowering AST to VIR
pub struct LoweringContext {
    module: VirModule,
    current_block: Option<BlockId>,
    current_instructions: Vec<Instruction>,
    variables: HashMap<String, ValueId>,
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            module: VirModule::new(),
            current_block: None,
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
                let elem_vals: Result<Vec<_>, _> = elements
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect();
                
                let elem_vals = elem_vals?;
                let dimension = if elements.is_empty() { 0 } else { 1 };
                
                Ok(self.emit(InstructionKind::VectorNew {
                    elements: elem_vals,
                    dimension,
                }))
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
            
            // For subsequent stages, just lower them and use as current value
            // In full implementation, these would be function applications
            for stage in &stages[1..] {
                current_value = self.lower_expr(stage)?;
            }
            
            Ok(current_value)
        }
        
        _ => Err(format!("Lowering not yet implemented for {:?}", expr)),
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
        let result = self.lower_expr(&expr).unwrap();
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
        let result = self.lower_expr(&expr).unwrap();
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
