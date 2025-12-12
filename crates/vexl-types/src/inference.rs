//! Type inference using Hindley-Milner + dimensional types + effect types

use vexl_syntax::ast::{Expr, BinOpKind};
use vexl_core::Effect;
use std::collections::HashMap;

/// Type variable for unification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVar(pub usize);

/// Inferred type with type variables and dimensional information
#[derive(Debug, Clone, PartialEq)]
pub enum InferredType {
    /// Concrete types
    Int,
    Float,
    String,
    Bool,
    
    /// Vector with element type and dimension
    Vector {
        element: Box<InferredType>,
        dimension: DimVar,
    },
    
    /// Function type
    Function {
        params: Vec<InferredType>,
        ret: Box<InferredType>,
        effect: Effect,
    },
    
    /// Type variable (for inference)
    Var(TypeVar),
}

/// Dimensional variable for dimensional polymorphism
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimVar {
    /// Concrete dimension
    Concrete(usize),
    /// Dimension variable (for inference)
    Var(usize),
    /// Any dimension (fully polymorphic)
    Any,
}

/// Type environment for variable bindings
pub struct TypeEnv {
    bindings: HashMap<String, InferredType>,
    next_type_var: usize,
    next_dim_var: usize,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            next_type_var: 0,
            next_dim_var: 0,
        }
    }
    
    pub fn insert(&mut self, name: String, ty: InferredType) {
        self.bindings.insert(name, ty);
    }
    
    pub fn lookup(&self, name: &str) -> Option<&InferredType> {
        self.bindings.get(name)
    }
    
    /// Generate fresh type variable
    pub fn fresh_type_var(&mut self) -> TypeVar {
        let var = TypeVar(self.next_type_var);
        self.next_type_var += 1;
        var
    }
    
    /// Generate fresh dimension variable
    pub fn fresh_dim_var(&mut self) -> DimVar {
        let var = DimVar::Var(self.next_dim_var);
        self.next_dim_var += 1;
        var
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Type constraints for unification
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Type equality
    Equal(InferredType, InferredType),
    /// Dimensional equality
    DimEqual(DimVar, DimVar),
}

/// Infer the type of an expression with dimensional checking
pub fn infer(expr: &Expr, env: &mut TypeEnv) -> Result<(InferredType, Vec<Constraint>), String> {
    match expr {
        Expr::Int(_, _) => Ok((InferredType::Int, vec![])),
        
        Expr::Float(_, _) => Ok((InferredType::Float, vec![])),
        
        Expr::String(_, _) => Ok((InferredType::String, vec![])),
        
        Expr::Ident(name, _) => {
            env.lookup(name)
                .cloned()
                .map(|ty| (ty, vec![]))
                .ok_or_else(|| format!("Undefined variable: {}", name))
        }
        
        Expr::Vector(elements, _) => {
            if elements.is_empty() {
                // Empty vector - polymorphic
                let elem_var = env.fresh_type_var();
                let dim_var = env.fresh_dim_var();
                Ok((
                    InferredType::Vector {
                        element: Box::new(InferredType::Var(elem_var)),
                        dimension: dim_var,
                    },
                    vec![]
                ))
            } else {
                // Infer element type from first element
                let (first_ty, mut constraints) = infer(&elements[0], env)?;
                
                // All elements must have same type
                for elem in &elements[1..] {
                    let (elem_ty, elem_constraints) = infer(elem, env)?;
                    constraints.push(Constraint::Equal(first_ty.clone(), elem_ty));
                    constraints.extend(elem_constraints);
                }
                
                // 1D vector
                Ok((
                    InferredType::Vector {
                        element: Box::new(first_ty),
                        dimension: DimVar::Concrete(1),
                    },
                    constraints
                ))
            }
        }
        
        Expr::Range(start, end, _) => {
            let (start_ty, mut constraints) = infer(start, env)?;
            let (end_ty, end_constraints) = infer(end, env)?;
            
            // Start and end must be Int
            constraints.push(Constraint::Equal(start_ty, InferredType::Int));
            constraints.push(Constraint::Equal(end_ty, InferredType::Int));
            constraints.extend(end_constraints);
            
            // Returns 1D vector of integers
            Ok((
                InferredType::Vector {
                    element: Box::new(InferredType::Int),
                    dimension: DimVar::Concrete(1),
                },
                constraints
            ))
        }
        
        Expr::InfiniteRange(start, _) => {
            let (start_ty, mut constraints) = infer(start, env)?;
            
            // Start must be Int
            constraints.push(Constraint::Equal(start_ty, InferredType::Int));
            
            // Returns 1D infinite vector of integers
            Ok((
                InferredType::Vector {
                    element: Box::new(InferredType::Int),
                    dimension: DimVar::Concrete(1),
                },
                constraints
            ))
        }
        
        Expr::BinOp { op, left, right, .. } => {
            infer_binop(*op, left, right, env)
        }
        
        Expr::Let { name, value, body, .. } => {
            let (val_ty, mut constraints) = infer(value, env)?;
            
            // Add binding to environment
            env.insert(name.clone(), val_ty);
            
            // Infer body type
            let (body_ty, body_constraints) = infer(body, env)?;
            constraints.extend(body_constraints);
            
            Ok((body_ty, constraints))
        }
        
        Expr::Lambda { params, body, .. } => {
            // Generate fresh type variables for parameters
            let param_types: Vec<InferredType> = params.iter()
                .map(|_| InferredType::Var(env.fresh_type_var()))
                .collect();
            
            // Add parameters to environment
            for (param, ty) in params.iter().zip(param_types.iter()) {
                env.insert(param.clone(), ty.clone());
            }
            
            // Infer body type
            let (body_ty, constraints) = infer(body, env)?;
            
            // Lambda is pure by default
            Ok((
                InferredType::Function {
                    params: param_types,
                    ret: Box::new(body_ty),
                    effect: Effect::Pure,
                },
                constraints
            ))
        }
        
        Expr::If { cond, then_branch, else_branch, .. } => {
            let (cond_ty, mut constraints) = infer(cond, env)?;
            
            // Condition must be Bool
            constraints.push(Constraint::Equal(cond_ty, InferredType::Bool));
            
            let (then_ty, then_constraints) = infer(then_branch, env)?;
            constraints.extend(then_constraints);
            
            if let Some(else_expr) = else_branch {
                let (else_ty, else_constraints) = infer(else_expr, env)?;
                constraints.extend(else_constraints);
                
                // Then and else must have same type
                constraints.push(Constraint::Equal(then_ty.clone(), else_ty));
            }
            
            Ok((then_ty, constraints))
        }
        
        
        Expr::Pipeline { stages, .. } => {
            // Pipeline: data |> f |> g
            // Type flows left to right: output of each stage is input to next
            if stages.is_empty() {
                return Err("Empty pipeline".to_string());
            }
            
            // Infer type of first stage (the data)
            let (mut current_ty, mut constraints) = infer(&stages[0], env)?;
            
            // For each subsequent stage, it should be a function that takes current type
            // For now, we just thread the type through (simplified pipeline typing)
            // In full implementation, we'd check function types match
            for stage in &stages[1..] {
                let (stage_ty, stage_constraints) = infer(stage, env)?;
                constraints.extend(stage_constraints);
                
                // Simplified: assume stage transforms current_ty
                // In practice, you'd unify stage_ty with Function(current_ty -> result)
                // For now, just use stage identifier's type from env if available
                match stage {
                    Expr::Ident(name, _) => {
                        if let Some(func_ty) = env.lookup(name) {
                            match func_ty {
                                InferredType::Function { ret, .. } => {
                                    current_ty = (**ret).clone();
                                }
                                _ => {
                                    // Not a function, keep current type
                                }
                            }
                        }
                    }
                    _ => {
                        // For complex expressions, use their inferred type
                        current_ty = stage_ty;
                    }
                }
            }
            
            Ok((current_ty, constraints))
        }
        
        _ => Err(format!("Type inference not yet implemented for {:?}", expr)),
    }
}

/// Infer type of binary operation with dimensional checking
fn infer_binop(
    op: BinOpKind,
    left: &Expr,
    right: &Expr,
    env: &mut TypeEnv,
) -> Result<(InferredType, Vec<Constraint>), String> {
    let (left_ty, mut constraints) = infer(left, env)?;
    let (right_ty, right_constraints) = infer(right, env)?;
    constraints.extend(right_constraints);
    
    match op {
        BinOpKind::Add | BinOpKind::Sub => {
            // Addition/subtraction: must have same type and dimension
            constraints.push(Constraint::Equal(left_ty.clone(), right_ty));
            Ok((left_ty, constraints))
        }
        
        BinOpKind::Mul | BinOpKind::Div => {
            // For now, require same type (will enhance for broadcasting later)
            constraints.push(Constraint::Equal(left_ty.clone(), right_ty));
            Ok((left_ty, constraints))
        }
        
        BinOpKind::MatMul => {
            // Matrix multiplication: dimensions must be compatible
            // [m, n] @ [n, p] => [m, p]
            // For now, simplified
            Ok((left_ty, constraints))
        }
        
        BinOpKind::Eq | BinOpKind::NotEq |
        BinOpKind::Lt | BinOpKind::Le |
        BinOpKind::Gt | BinOpKind::Ge => {
            // Comparison: operands must have same type, result is Bool
            constraints.push(Constraint::Equal(left_ty, right_ty));
            Ok((InferredType::Bool, constraints))
        }
        
        _ => Err(format!("Type inference not implemented for {:?}", op)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vexl_syntax::ast::Span;

    #[test]
    fn test_infer_int() {
        let expr = Expr::Int(42, Span { start: 0, end: 2 });
        let mut env = TypeEnv::new();
        let (ty, _) = infer(&expr, &mut env).unwrap();
        assert_eq!(ty, InferredType::Int);
    }

    #[test]
    fn test_infer_vector() {
        let span = Span { start: 0, end: 7 };
        let expr = Expr::Vector(
            vec![
                Expr::Int(1, span),
                Expr::Int(2, span),
                Expr::Int(3, span),
            ],
            span,
        );
        let mut env = TypeEnv::new();
        let (ty, _) = infer(&expr, &mut env).unwrap();
        
        match ty {
            InferredType::Vector { element, dimension } => {
                assert_eq!(*element, InferredType::Int);
                assert_eq!(dimension, DimVar::Concrete(1));
            }
            _ => panic!("Expected vector type"),
        }
    }
    
    #[test]
    fn test_infer_range() {
        let span = Span { start: 0, end: 7 };
        let expr = Expr::Range(
            Box::new(Expr::Int(0, span)),
            Box::new(Expr::Int(10, span)),
            span,
        );
        let mut env = TypeEnv::new();
        let (ty, _) = infer(&expr, &mut env).unwrap();
        
        match ty {
            InferredType::Vector { element, dimension } => {
                assert_eq!(*element, InferredType::Int);
                assert_eq!(dimension, DimVar::Concrete(1));
            }
            _ => panic!("Expected vector type"),
        }
    }
}
