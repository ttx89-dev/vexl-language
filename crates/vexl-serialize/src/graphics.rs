//! Graphics .xg format implementation
//!
//! SVG-like vector graphics for visual programming.

use crate::{Result, SerializeError};

/// Placeholder for graphics format implementation
pub fn placeholder() -> Result<()> {
    Err(SerializeError::NotImplemented("Graphics format not yet implemented".to_string()))
}
