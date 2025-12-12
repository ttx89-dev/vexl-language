//! VEXL Type System - Type inference and checking

pub mod inference;
pub mod dimensional;
pub mod effect_check;

pub use inference::infer;
