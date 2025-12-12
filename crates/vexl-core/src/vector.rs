//! VEXL Vector Type
//!
//! Everything in VEXL is a vector. This module implements the universal
//! `Vector<T, D>` type with dimensional polymorphism.

use std::marker::PhantomData;

/// Universal vector type with element type T and dimensionality D
#[derive(Debug, Clone)]
pub struct Vector<T, const D: usize> {
    _phantom: PhantomData<T>,
}

impl<T, const D: usize> Vector<T, D> {
    /// Creates a new empty vector
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T, const D: usize> Default for Vector<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v: Vector<i32, 1> = Vector::new();
        assert!(true); // Placeholder test
    }
}
