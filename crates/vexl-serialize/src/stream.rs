//! Stream .xs format implementation
//!
//! Real-time vector streaming protocol.

use crate::{Result, SerializeError};

/// Placeholder for stream format implementation
pub fn placeholder() -> Result<()> {
    Err(SerializeError::NotImplemented("Stream format not yet implemented".to_string()))
}
