//! VEXL Generator Trait
//!
//! Generators store algorithms, not data, enabling near-infinite logical storage
//! and lazy evaluation.

use std::sync::Arc;

/// Core generator trait for lazy evaluation
pub trait Generator: Send + Sync {
    /// Evaluate the generator at a specific index
    fn evaluate(&self, index: usize) -> Option<Box<dyn std::any::Any + Send + Sync>>;
    
    /// Get the bounds of this generator (if finite)
    fn bounds(&self) -> Option<(usize, usize)>;
    
    /// Check if this generator is pure (deterministic, no side effects)
    fn is_pure(&self) -> bool;
    
    /// Clone this generator into an Arc
    fn clone_generator(&self) -> Arc<dyn Generator>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleGenerator;
    
    impl Generator for SimpleGenerator {
        fn evaluate(&self, _index: usize) -> Option<Box<dyn std::any::Any + Send + Sync>> {
            None
        }
        
        fn bounds(&self) -> Option<(usize, usize)> {
            None
        }
        
        fn is_pure(&self) -> bool {
            true
        }
        
        fn clone_generator(&self) -> Arc<dyn Generator> {
            Arc::new(SimpleGenerator)
        }
    }

    #[test]
    fn test_generator_trait() {
        let gen = SimpleGenerator;
        assert!(gen.is_pure());
    }
}
