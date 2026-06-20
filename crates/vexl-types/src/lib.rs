//! VEXL Type System - Type inference and checking

pub mod inference;
pub mod dimensional;
pub mod effect_check;

pub use inference::{infer, InferredType, TypeEnv, Constraint, DimVar};
pub use effect_check::{infer_effect, EffectEnv, infer_with_effect};
pub use dimensional::{solve_constraints, Substitution, unify};
