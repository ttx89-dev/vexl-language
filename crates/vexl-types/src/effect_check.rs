//! Effect type checking - tracking purity, I/O, mutability for parallelization

use vexl_core::Effect;
use vexl_syntax::ast::{Expr, BinOpKind};
use crate::inference::{TypeEnv, InferredType};
use std::collections::HashMap;

/// Effect environment tracking effects for variables and functions
pub struct EffectEnv {
    effects: HashMap<String, Effect>,
}

impl EffectEnv {
    pub fn new() -> Self {
        Self {
            effects: HashMap::new(),
        }
    }
    
    pub fn insert(&mut self, name: String, effect: Effect) {
        self.effects.insert(name, effect);
    }
    
    pub fn lookup(&self, name: &str) -> Option<&Effect> {
        self.effects.get(name)
    }
}

impl Default for EffectEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Infer the effect of an expression
pub fn infer_effect(expr: &Expr, env: &EffectEnv) -> Effect {
    match expr {
        // Literals are always pure
        Expr::Int(_, _) | Expr::Float(_, _) | Expr::String(_, _) | Expr::Bool(_, _) => Effect::Pure,
        
        // Variable lookup: depends on variable's effect
        Expr::Ident(name, _) => {
            env.lookup(name).copied().unwrap_or(Effect::Pure)
        }
        
        // Vectors: combine effects of all elements
        Expr::Vector(elements, _) => {
            elements.iter()
                .map(|e| infer_effect(e, env))
                .fold(Effect::Pure, |acc, eff| acc.combine(eff))
        }
        
        // Ranges are pure (generator-based, lazy evaluation)
        Expr::Range(_, _, _) | Expr::InfiniteRange(_, _) => Effect::Pure,
        
        // Comprehensions: combine effects of element, bindings, and filter
        Expr::Comprehension { element, bindings, filter, .. } => {
            let elem_effect = infer_effect(element, env);
            let binding_effect = bindings.iter()
                .map(|(_, expr)| infer_effect(expr, env))
                .fold(Effect::Pure, |acc, eff| acc.combine(eff));
            let filter_effect = filter.as_ref()
                .map(|f| infer_effect(f, env))
                .unwrap_or(Effect::Pure);
            
            elem_effect.combine(binding_effect).combine(filter_effect)
        }
        
        // Binary operations: combine left and right effects
        Expr::BinOp { left, right, op, .. } => {
            let left_eff = infer_effect(left, env);
            let right_eff = infer_effect(right, env);
            let combined = left_eff.combine(right_eff);
            
            // Some operations may introduce additional effects
            match op {
                BinOpKind::Div => {
                    // Division can fail (divide by zero)
                    combined.combine(Effect::Fail)
                }
                _ => combined,
            }
        }
        
        // Unary operations: effect of the inner expression
        Expr::UnOp { expr, .. } => infer_effect(expr, env),
        
        // Function application: depends on function's effect
        Expr::App { func, args, .. } => {
            let func_eff = infer_effect(func, env);
            let args_eff = args.iter()
                .map(|a| infer_effect(a, env))
                .fold(Effect::Pure, |acc, eff| acc.combine(eff));
            
            func_eff.combine(args_eff)
        }
        
        // Lambda: infer body effect (default to Pure)
        Expr::Lambda { body, .. } => {
            infer_effect(body, env)
        }
        
        // Let binding: combine value and body effects
        Expr::Let { value, body, .. } => {
            let val_eff = infer_effect(value, env);
            let body_eff = infer_effect(body, env);
            val_eff.combine(body_eff)
        }
        
        // If expression: combine all branches
        Expr::If { cond, then_branch, else_branch, .. } => {
            let cond_eff = infer_effect(cond, env);
            let then_eff = infer_effect(then_branch, env);
            let else_eff = else_branch.as_ref()
                .map(|e| infer_effect(e, env))
                .unwrap_or(Effect::Pure);
            
            cond_eff.combine(then_eff).combine(else_eff)
        }
        
        // Pipeline: combine all stages
        Expr::Pipeline { stages, .. } => {
            stages.iter()
                .map(|s| infer_effect(s, env))
                .fold(Effect::Pure, |acc, eff| acc.combine(eff))
        }
        
        // Fix (recursion): effect of the body
        Expr::Fix { body, .. } => {
            infer_effect(body, env)
        }
    }
}

/// Check if an expression can be parallelized
pub fn is_parallelizable(expr: &Expr, env: &EffectEnv) -> bool {
    infer_effect(expr, env).is_parallelizable()
}

/// Infer both type and effect simultaneously
pub fn infer_with_effect(
    expr: &Expr,
    type_env: &mut TypeEnv,
    effect_env: &EffectEnv,
) -> Result<(InferredType, Effect), String> {
    use crate::inference::infer;
    
    let (ty, _constraints) = infer(expr, type_env)?;
    let effect = infer_effect(expr, effect_env);
    
    Ok((ty, effect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_syntax::ast::Span;

    #[test]
    fn test_literal_is_pure() {
        let expr = Expr::Int(42, Span { start: 0, end: 2 });
        let env = EffectEnv::new();
        assert_eq!(infer_effect(&expr, &env), Effect::Pure);
    }
    
    #[test]
    fn test_vector_combines_effects() {
        let span = Span { start: 0, end: 7 };
        let expr = Expr::Vector(
            vec![
                Expr::Int(1, span),
                Expr::Int(2, span),
                Expr::Int(3, span),
            ],
            span,
        );
        let env = EffectEnv::new();
        assert_eq!(infer_effect(&expr, &env), Effect::Pure);
    }
    
    #[test]
    fn test_range_is_pure() {
        let span = Span { start: 0, end: 7 };
        let expr = Expr::Range(
            Box::new(Expr::Int(0, span)),
            Box::new(Expr::Int(10, span)),
            span,
        );
        let env = EffectEnv::new();
        assert_eq!(infer_effect(&expr, &env), Effect::Pure);
    }
    
    #[test]
    fn test_pure_is_parallelizable() {
        let expr = Expr::Int(42, Span { start: 0, end: 2 });
        let env = EffectEnv::new();
        assert!(is_parallelizable(&expr, &env));
    }
    
    #[test]
    fn test_binop_combines_effects() {
        let span = Span { start: 0, end: 5 };
        let expr = Expr::BinOp {
            op: BinOpKind::Add,
            left: Box::new(Expr::Int(1, span)),
            right: Box::new(Expr::Int(2, span)),
            span,
        };
        let env = EffectEnv::new();
        assert_eq!(infer_effect(&expr, &env), Effect::Pure);
        assert!(is_parallelizable(&expr, &env));
    }
    
    #[test]
    fn test_division_has_fail_effect() {
        let span = Span { start: 0, end: 5 };
        let expr = Expr::BinOp {
            op: BinOpKind::Div,
            left: Box::new(Expr::Int(10, span)),
            right: Box::new(Expr::Int(2, span)),
            span,
        };
        let env = EffectEnv::new();
        let effect = infer_effect(&expr, &env);
        
        // Division adds Fail effect
        assert!(matches!(effect, Effect::Fail));
    }
    
    #[test]
    fn test_pipeline_combines_stages() {
        let span = Span { start: 0, end: 10 };
        let expr = Expr::Pipeline {
            stages: vec![
                Expr::Int(1, span),
                Expr::Int(2, span),
                Expr::Int(3, span),
            ],
            span,
        };
        let env = EffectEnv::new();
        assert_eq!(infer_effect(&expr, &env), Effect::Pure);
    }
}
