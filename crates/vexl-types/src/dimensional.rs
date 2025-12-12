//! Dimensional type checking and constraint solving

use crate::inference::{DimVar, Constraint, InferredType, TypeVar};
use std::collections::HashMap;

/// Substitution map for type variables
pub struct Substitution {
    type_subst: HashMap<TypeVar, InferredType>,
    dim_subst: HashMap<usize, DimVar>,
}

impl Substitution {
    pub fn new() -> Self {
        Self {
            type_subst: HashMap::new(),
            dim_subst: HashMap::new(),
        }
    }
    
    /// Apply substitution to a type
    pub fn apply(&self, ty: &InferredType) -> InferredType {
        match ty {
            InferredType::Var(var) => {
                self.type_subst.get(var).cloned().unwrap_or_else(|| ty.clone())
            }
            InferredType::Vector { element, dimension } => {
                InferredType::Vector {
                    element: Box::new(self.apply(element)),
                    dimension: self.apply_dim(dimension),
                }
            }
            InferredType::Function { params, ret, effect } => {
                InferredType::Function {
                    params: params.iter().map(|p| self.apply(p)).collect(),
                    ret: Box::new(self.apply(ret)),
                    effect: *effect,
                }
            }
            _ => ty.clone(),
        }
    }
    
    /// Apply substitution to a dimensional variable
    pub fn apply_dim(&self, dim: &DimVar) -> DimVar {
        match dim {
            DimVar::Var(v) => {
                self.dim_subst.get(v).cloned().unwrap_or_else(|| dim.clone())
            }
            _ => dim.clone(),
        }
    }
    
    /// Add type variable binding
    pub fn bind_type(&mut self, var: TypeVar, ty: InferredType) {
        self.type_subst.insert(var, ty);
    }
    
    /// Add dimensional variable binding
    pub fn bind_dim(&mut self, var: usize, dim: DimVar) {
        self.dim_subst.insert(var, dim);
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

/// Unify two types, returning a substitution or an error
pub fn unify(t1: &InferredType, t2: &InferredType) -> Result<Substitution, String> {
    let mut subst = Substitution::new();
    unify_with_subst(t1, t2, &mut subst)?;
    Ok(subst)
}

fn unify_with_subst(
    t1: &InferredType,
    t2: &InferredType,
    subst: &mut Substitution,
) -> Result<(), String> {
    match (t1, t2) {
        // Same concrete types unify
        (InferredType::Int, InferredType::Int) |
        (InferredType::Float, InferredType::Float) |
        (InferredType::String, InferredType::String) |
        (InferredType::Bool, InferredType::Bool) => Ok(()),
        
        // Type variable unifies with anything
        (InferredType::Var(v), t) | (t, InferredType::Var(v)) => {
            if occurs_check(v, t) {
                Err(format!("Occurs check failed: {:?} occurs in {:?}", v, t))
            } else {
                subst.bind_type(v.clone(), t.clone());
                Ok(())
            }
        }
        
        // Vectors unify if elements and dimensions unify
        (
            InferredType::Vector { element: e1, dimension: d1 },
            InferredType::Vector { element: e2, dimension: d2 },
        ) => {
            unify_with_subst(e1, e2, subst)?;
            unify_dim(d1, d2, subst)?;
            Ok(())
        }
        
        // Functions unify if parameters and return types unify
        (
            InferredType::Function { params: p1, ret: r1, .. },
            InferredType::Function { params: p2, ret: r2, .. },
        ) => {
            if p1.len() != p2.len() {
                return Err(format!(
                    "Function arity mismatch: {} vs {}",
                    p1.len(),
                    p2.len()
                ));
            }
            
            for (param1, param2) in p1.iter().zip(p2.iter()) {
                unify_with_subst(param1, param2, subst)?;
            }
            
            unify_with_subst(r1, r2, subst)?;
            Ok(())
        }
        
        _ => Err(format!("Cannot unify {:?} with {:?}", t1, t2)),
    }
}

/// Unify two dimensional variables
fn unify_dim(
    d1: &DimVar,
    d2: &DimVar,
    subst: &mut Substitution,
) -> Result<(), String> {
    match (d1, d2) {
        // Same concrete dimensions unify
        (DimVar::Concrete(n1), DimVar::Concrete(n2)) => {
            if n1 == n2 {
                Ok(())
            } else {
                Err(format!("Dimension mismatch: {} vs {}", n1, n2))
            }
        }
        
        // Any dimension unifies with anything
        (DimVar::Any, _) | (_, DimVar::Any) => Ok(()),
        
        // Dimension variable unifies with concrete or another variable
        (DimVar::Var(v), d) | (d, DimVar::Var(v)) => {
            subst.bind_dim(*v, d.clone());
            Ok(())
        }
    }
}

/// Occurs check: does variable v occur in type t?
fn occurs_check(v: &TypeVar, t: &InferredType) -> bool {
    match t {
        InferredType::Var(var) => v == var,
        InferredType::Vector { element, .. } => occurs_check(v, element),
        InferredType::Function { params, ret, .. } => {
            params.iter().any(|p| occurs_check(v, p)) || occurs_check(v, ret)
        }
        _ => false,
    }
}

/// Solve a list of constraints, returning a substitution
pub fn solve_constraints(constraints: Vec<Constraint>) -> Result<Substitution, String> {
    let mut subst = Substitution::new();
    
    for constraint in constraints {
        match constraint {
            Constraint::Equal(t1, t2) => {
                let t1_subst = subst.apply(&t1);
                let t2_subst = subst.apply(&t2);
                unify_with_subst(&t1_subst, &t2_subst, &mut subst)?;
            }
            Constraint::DimEqual(d1, d2) => {
                let d1_subst = subst.apply_dim(&d1);
                let d2_subst = subst.apply_dim(&d2);
                unify_dim(&d1_subst, &d2_subst, &mut subst)?;
            }
        }
    }
    
    Ok(subst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_int() {
        let result = unify(&InferredType::Int, &InferredType::Int);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_unify_mismatch() {
        let result = unify(&InferredType::Int, &InferredType::Float);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_unify_var() {
        let var = TypeVar(0);
        let result = unify(&InferredType::Var(var), &InferredType::Int);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_unify_vectors_same_dim() {
        let v1 = InferredType::Vector {
            element: Box::new(InferredType::Int),
            dimension: DimVar::Concrete(2),
        };
        let v2 = InferredType::Vector {
            element: Box::new(InferredType::Int),
            dimension: DimVar::Concrete(2),
        };
        let result = unify(&v1, &v2);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_unify_vectors_diff_dim() {
        let v1 = InferredType::Vector {
            element: Box::new(InferredType::Int),
            dimension: DimVar::Concrete(2),
        };
        let v2 = InferredType::Vector {
            element: Box::new(InferredType::Int),
            dimension: DimVar::Concrete(3),
        };
        let result = unify(&v1, &v2);
        assert!(result.is_err());
    }
}
