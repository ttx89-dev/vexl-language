//! VEXL Effect Type System
//!
//! Effect typing enables implicit parallelism by tracking function purity
//! and side effects at compile time.

use std::fmt;

/// Effect annotations for functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Pure function - no side effects, deterministic
    Pure,
    /// I/O operations
    Io,
    /// Mutable state access
    Mut,
    /// Asynchronous operations
    Async,
    /// Fallible operations
    Fail,
}

impl Effect {
    /// Check if this effect allows automatic parallelization
    pub fn is_parallelizable(&self) -> bool {
        matches!(self, Effect::Pure)
    }
    
    /// Combine two effects (used in effect inference)
    pub fn combine(self, other: Effect) -> Effect {
        use Effect::*;
        match (self, other) {
            (Pure, Pure) => Pure,
            (Pure, e) | (e, Pure) => e,
            (Io, _) | (_, Io) => Io,
            (Async, _) | (_, Async) => Async,
            (Mut, Mut) => Mut,
            (Mut, Fail) | (Fail, Mut) => Mut,
            (Fail, Fail) => Fail,
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Pure => write!(f, "pure"),
            Effect::Io => write!(f, "io"),
            Effect::Mut => write!(f, "mut"),
            Effect::Async => write!(f, "async"),
            Effect::Fail => write!(f, "fail"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_parallelizable() {
        assert!(Effect::Pure.is_parallelizable());
        assert!(!Effect::Io.is_parallelizable());
        assert!(!Effect::Mut.is_parallelizable());
    }

    #[test]
    fn test_effect_combine() {
        assert_eq!(Effect::Pure.combine(Effect::Pure), Effect::Pure);
        assert_eq!(Effect::Pure.combine(Effect::Io), Effect::Io);
        assert_eq!(Effect::Mut.combine(Effect::Fail), Effect::Mut);
    }
}
