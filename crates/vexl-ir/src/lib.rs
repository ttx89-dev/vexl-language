//! VEXL Intermediate Representation (VIR)
//! 
//! VIR is a Single Static Assignment (SSA) form IR that enables optimizations
//! before lowering to LLVM IR. It preserves VEXL's dimensional and effect information.

use vexl_core::Effect;
use std::collections::HashMap;

pub mod lower;
pub mod optimize;

/// VIR instruction identifier (SSA value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

/// VIR basic block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

/// VIR function
#[derive(Debug, Clone)]
pub struct VirFunction {
    pub name: String,
    pub params: Vec<ValueId>,
    pub blocks: HashMap<BlockId, BasicBlock>,
    pub entry_block: BlockId,
    pub effect: Effect,
}

/// Basic block in SSA form
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

/// VIR instruction (SSA form)
#[derive(Debug, Clone)]
pub struct Instruction {
    pub result: ValueId,
    pub kind: InstructionKind,
}

/// VIR instruction kinds
#[derive(Debug, Clone)]
pub enum InstructionKind {
    // Constants
    ConstInt(i64),
    ConstFloat(f64),
    ConstString(String),
    
    // Vector operations
    VectorNew {
        elements: Vec<ValueId>,
        dimension: usize,
    },
    VectorGet {
        vector: ValueId,
        index: ValueId,
    },
    VectorSet {
        vector: ValueId,
        index: ValueId,
        value: ValueId,
    },
    
    // Arithmetic
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    
    // Matrix operations
    MatMul(ValueId, ValueId),
    Outer(ValueId, ValueId),
    Dot(ValueId, ValueId),
    
    // Comparisons
    Eq(ValueId, ValueId),
    NotEq(ValueId, ValueId),
    Lt(ValueId, ValueId),
    Le(ValueId, ValueId),
    Gt(ValueId, ValueId),
    Ge(ValueId, ValueId),
    
    // Function call
    Call {
        func: ValueId,
        args: Vec<ValueId>,
    },

    // Runtime function call (FFI to runtime library)
    RuntimeCall {
        function_name: String,
        args: Vec<ValueId>,
    },
    
    // Generator operations
    GeneratorNew {
        func: ValueId,
        bounds: Option<(ValueId, ValueId)>,
    },
    GeneratorEval {
        generator: ValueId,
        index: ValueId,
    },
    
    // Range
    Range {
        start: ValueId,
        end: ValueId,
    },
    InfiniteRange {
        start: ValueId,
    },
    
    // Phi node (for SSA)
    Phi(Vec<(ValueId, BlockId)>),
}

/// Block terminator
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return from function
    Return(ValueId),
    
    /// Conditional branch
    Branch {
        cond: ValueId,
        then_block: BlockId,
        else_block: BlockId,
    },
    
    /// Unconditional jump
    Jump(BlockId),
    
    /// Unreachable
    Unreachable,
}

/// VIR module (collection of functions)
#[derive(Debug, Clone)]
pub struct VirModule {
    pub functions: HashMap<String, VirFunction>,
    pub next_value_id: usize,
    pub next_block_id: usize,
}

impl VirModule {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            next_value_id: 0,
            next_block_id: 0,
        }
    }
    
    /// Generate fresh value ID
    pub fn fresh_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }
    
    /// Generate fresh block ID
    pub fn fresh_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }
    
    /// Add function to module
    pub fn add_function(&mut self, name: String, func: VirFunction) {
        self.functions.insert(name, func);
    }
}

impl Default for VirModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_module() {
        let mut module = VirModule::new();
        let v1 = module.fresh_value();
        let v2 = module.fresh_value();
        assert_eq!(v1.0, 0);
        assert_eq!(v2.0, 1);
    }
    
    #[test]
    fn test_create_block() {
        let mut module = VirModule::new();
        let b1 = module.fresh_block();
        let b2 = module.fresh_block();
        assert_eq!(b1.0, 0);
        assert_eq!(b2.0, 1);
    }
    
    #[test]
    fn test_add_function() {
        let mut module = VirModule::new();
        let entry = module.fresh_block();
        
        let func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: entry,
            effect: Effect::Pure,
        };
        
        module.add_function("test".to_string(), func);
        assert!(module.functions.contains_key("test"));
    }
}
