//! VEXL Core - Universal Vector Type System
//!
//! This crate implements the foundational types for VEXL:
//! - `Vector<T, D>` - The universal vector type
//! - Generator trait and implementations
//! - Effect type system

pub mod vector;
pub mod generator;
pub mod effect;

pub use vector::Vector;
pub use generator::Generator;
pub use effect::Effect;
