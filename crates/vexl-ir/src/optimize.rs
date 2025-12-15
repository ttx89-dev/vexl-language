//! VIR optimization passes

use crate::{VirModule, VirFunction, Instruction, InstructionKind, ValueId};
use std::collections::{HashMap, HashSet};

/// Constant folding optimization
/// Evaluates constant expressions at compile time
pub fn constant_fold(module: &mut VirModule) {
    for func in module.functions.values_mut() {
        constant_fold_function(func);
    }
}

fn constant_fold_function(func: &mut VirFunction) {
    let mut constants: HashMap<ValueId, ConstValue> = HashMap::new();
    
    for block in func.blocks.values_mut() {
        let mut new_instructions = Vec::new();
        
        for inst in &block.instructions {
            match try_fold_instruction(&inst.kind, &constants) {
                Some(const_val) => {
                    // Replace with constant
                    let new_kind = match const_val {
                        ConstValue::Int(n) => InstructionKind::ConstInt(n),
                        ConstValue::Float(f) => InstructionKind::ConstFloat(f),
                        ConstValue::String(ref s) => InstructionKind::ConstString(s.clone()),
                    };
                    constants.insert(inst.result, const_val);
                    new_instructions.push(Instruction {
                        result: inst.result,
                        kind: new_kind,
                    });
                }
                None => {
                    // Track if this produces a constant
                    if let Some(val) = extract_constant(&inst.kind) {
                        constants.insert(inst.result, val);
                    }
                    new_instructions.push(inst.clone());
                }
            }
        }
        
        block.instructions = new_instructions;
    }
}

/// Constant value types
#[derive(Debug, Clone)]
enum ConstValue {
    Int(i64),
    Float(f64),
    String(String),
}

/// Try to fold an instruction to a constant
fn try_fold_instruction(kind: &InstructionKind, constants: &HashMap<ValueId, ConstValue>) -> Option<ConstValue> {
    match kind {
        InstructionKind::Add(left, right) => {
            match (constants.get(left)?, constants.get(right)?) {
                (ConstValue::Int(a), ConstValue::Int(b)) => Some(ConstValue::Int(a + b)),
                (ConstValue::Float(a), ConstValue::Float(b)) => Some(ConstValue::Float(a + b)),
                _ => None,
            }
        }
        InstructionKind::Sub(left, right) => {
            match (constants.get(left)?, constants.get(right)?) {
                (ConstValue::Int(a), ConstValue::Int(b)) => Some(ConstValue::Int(a - b)),
                (ConstValue::Float(a), ConstValue::Float(b)) => Some(ConstValue::Float(a - b)),
                _ => None,
            }
        }
        InstructionKind::Mul(left, right) => {
            match (constants.get(left)?, constants.get(right)?) {
                (ConstValue::Int(a), ConstValue::Int(b)) => Some(ConstValue::Int(a * b)),
                (ConstValue::Float(a), ConstValue::Float(b)) => Some(ConstValue::Float(a * b)),
                _ => None,
            }
        }
        InstructionKind::Div(left, right) => {
            match (constants.get(left)?, constants.get(right)?) {
                (ConstValue::Int(a), ConstValue::Int(b)) if *b != 0 => Some(ConstValue::Int(a / b)),
                (ConstValue::Float(a), ConstValue::Float(b)) if *b != 0.0 => Some(ConstValue::Float(a / b)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract constant value from instruction
fn extract_constant(kind: &InstructionKind) -> Option<ConstValue> {
    match kind {
        InstructionKind::ConstInt(n) => Some(ConstValue::Int(*n)),
        InstructionKind::ConstFloat(f) => Some(ConstValue::Float(*f)),
        InstructionKind::ConstString(s) => Some(ConstValue::String(s.clone())),
        _ => None,
    }
}

/// Dead code elimination
/// Removes instructions that are never used
pub fn eliminate_dead_code(module: &mut VirModule) {
    for func in module.functions.values_mut() {
        eliminate_dead_code_function(func);
    }
}

fn eliminate_dead_code_function(func: &mut VirFunction) {
    // Find all used values
    let mut used = HashSet::new();
    
    // Mark terminator values as used
    for block in func.blocks.values() {
        mark_terminator_uses(&block.terminator, &mut used);
        
        // Mark all instruction operands as used
        for inst in &block.instructions {
            mark_instruction_uses(&inst.kind, &mut used);
        }
    }
    
    // Remove unused instructions
    for block in func.blocks.values_mut() {
        block.instructions.retain(|inst| {
            // Keep if result is used or has side effects
            used.contains(&inst.result) || has_side_effects(&inst.kind)
        });
    }
}

fn mark_terminator_uses(terminator: &crate::Terminator, used: &mut HashSet<ValueId>) {
    match terminator {
        crate::Terminator::Return(val) => {
            used.insert(*val);
        }
        crate::Terminator::Branch { cond, .. } => {
            used.insert(*cond);
        }
        _ => {}
    }
}

fn mark_instruction_uses(kind: &InstructionKind, used: &mut HashSet<ValueId>) {
    match kind {
        InstructionKind::Add(a, b) |
        InstructionKind::Sub(a, b) |
        InstructionKind::Mul(a, b) |
        InstructionKind::Div(a, b) |
        InstructionKind::MatMul(a, b) |
        InstructionKind::Outer(a, b) |
        InstructionKind::Dot(a, b) |
        InstructionKind::Eq(a, b) |
        InstructionKind::NotEq(a, b) |
        InstructionKind::Lt(a, b) |
        InstructionKind::Le(a, b) |
        InstructionKind::Gt(a, b) |
        InstructionKind::Ge(a, b) => {
            used.insert(*a);
            used.insert(*b);
        }
        InstructionKind::VectorNew { elements, .. } => {
            for elem in elements {
                used.insert(*elem);
            }
        }
        InstructionKind::VectorGet { vector, index } |
        InstructionKind::GeneratorEval { generator: vector, index } => {
            used.insert(*vector);
            used.insert(*index);
        }
        InstructionKind::VectorSet { vector, index, value } => {
            used.insert(*vector);
            used.insert(*index);
            used.insert(*value);
        }
        InstructionKind::RuntimeCall { args, .. } => {
            for arg in args {
                used.insert(*arg);
            }
        }
        InstructionKind::Call { func, args } => {
            used.insert(*func);
            for arg in args {
                used.insert(*arg);
            }
        }
        InstructionKind::Range { start, end } => {
            used.insert(*start);
            used.insert(*end);
        }
        InstructionKind::InfiniteRange { start } => {
            used.insert(*start);
        }
        InstructionKind::GeneratorNew { func, bounds } => {
            used.insert(*func);
            if let Some((start, end)) = bounds {
                used.insert(*start);
                used.insert(*end);
            }
        }
        InstructionKind::Phi(incoming) => {
            for (val, _) in incoming {
                used.insert(*val);
            }
        }
        _ => {}
    }
}

fn has_side_effects(kind: &InstructionKind) -> bool {
    matches!(kind,
        InstructionKind::Call { .. } |
        InstructionKind::RuntimeCall { .. } |
        InstructionKind::VectorSet { .. }
    )
}

/// Run all optimization passes
pub fn optimize(module: &mut VirModule) {
    // Run passes multiple times until fixpoint
    for _ in 0..3 {
        constant_fold(module);
        eliminate_dead_code(module);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Terminator, BlockId, BasicBlock};

    #[test]
    fn test_constant_fold_add() {
        let mut module = VirModule::new();
        
        // Create: 1 + 2
        let v1 = module.fresh_value();
        let v2 = module.fresh_value();
        let v3 = module.fresh_value();
        
        let block_id = module.fresh_block();
        let mut block = BasicBlock {
            id: block_id,
            instructions: vec![
                Instruction {
                    result: v1,
                    kind: InstructionKind::ConstInt(1),
                },
                Instruction {
                    result: v2,
                    kind: InstructionKind::ConstInt(2),
                },
                Instruction {
                    result: v3,
                    kind: InstructionKind::Add(v1, v2),
                },
            ],
            terminator: Terminator::Return(v3),
        };
        
        let mut func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
        };
        
        func.blocks.insert(block_id, block);
        module.add_function("test".to_string(), func);
        
        constant_fold(&mut module);
        
        let func = module.functions.get("test").unwrap();
        let block = func.blocks.get(&block_id).unwrap();
        
        // Should fold to ConstInt(3)
        assert!(matches!(
            block.instructions.last().unwrap().kind,
            InstructionKind::ConstInt(3)
        ));
    }
    
    #[test]
    fn test_dead_code_elimination() {
        let mut module = VirModule::new();
        
        let v1 = module.fresh_value();
        let v2 = module.fresh_value();  // Unused
        let v3 = module.fresh_value();
        
        let block_id = module.fresh_block();
        let block = BasicBlock {
            id: block_id,
            instructions: vec![
                Instruction {
                    result: v1,
                    kind: InstructionKind::ConstInt(1),
                },
                Instruction {
                    result: v2,  // This should be eliminated
                    kind: InstructionKind::ConstInt(2),
                },
                Instruction {
                    result: v3,
                    kind: InstructionKind::Add(v1, v1),
                },
            ],
            terminator: Terminator::Return(v3),
        };
        
        let mut func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::from([(block_id, block)]),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
        };
        
        eliminate_dead_code_function(&mut func);
        
        // Should only have 2 instructions now (v1 and v3, not v2)
        assert_eq!(func.blocks.get(&block_id).unwrap().instructions.len(), 2);
    }
}
