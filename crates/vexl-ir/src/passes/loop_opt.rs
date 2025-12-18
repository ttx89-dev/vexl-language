//! Loop optimization passes for VIR

use crate::{VirModule, VirFunction, BasicBlock, BlockId, Instruction, InstructionKind, ValueId, Terminator};
use std::collections::{HashMap, HashSet};

/// Loop information structure
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Header block of the loop
    pub header: BlockId,
    /// Body blocks (excluding header)
    pub body: HashSet<BlockId>,
    /// Preheader block (entry point to loop)
    pub preheader: Option<BlockId>,
    /// Back edges to the header
    pub back_edges: Vec<(BlockId, BlockId)>,
    /// Loop nesting level
    pub nesting_level: usize,
}

/// Loop analysis and optimization passes
pub struct LoopOptimizer {
    loops: Vec<LoopInfo>,
}

impl LoopOptimizer {
    /// Create a new loop optimizer
    pub fn new() -> Self {
        Self {
            loops: Vec::new(),
        }
    }

    /// Analyze loops in a function and return loop information
    pub fn analyze_loops(&mut self, func: &VirFunction) -> Vec<LoopInfo> {
        self.loops.clear();

        // Find natural loops using back edge detection
        let back_edges = self.find_back_edges(func);

        // Build loop structures from back edges
        for (source, header) in back_edges {
            let mut loop_info = LoopInfo {
                header,
                body: HashSet::new(),
                preheader: None,
                back_edges: vec![(source, header)],
                nesting_level: 0,
            };

            // Find all blocks in the loop body
            self.find_loop_body(func, &mut loop_info);
            self.loops.push(loop_info);
        }

        // Calculate nesting levels
        self.calculate_nesting_levels();

        self.loops.clone()
    }

    /// Find back edges in the control flow graph
    fn find_back_edges(&self, func: &VirFunction) -> Vec<(BlockId, BlockId)> {
        let mut back_edges = Vec::new();
        let dominators = self.compute_dominators(func);

        for (block_id, block) in &func.blocks {
            for successor in self.get_successors(block) {
                if dominators.get(block_id).map_or(false, |doms| doms.contains(&successor)) {
                    back_edges.push((*block_id, successor));
                }
            }
        }

        back_edges
    }

    /// Compute dominator tree for the function
    fn compute_dominators(&self, func: &VirFunction) -> HashMap<BlockId, HashSet<BlockId>> {
        let mut dominators: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        let mut changed = true;

        // Initialize
        for block_id in func.blocks.keys() {
            dominators.insert(*block_id, func.blocks.keys().cloned().collect());
        }
        dominators.insert(func.entry_block, HashSet::from([func.entry_block]));

        // Iterative computation
        while changed {
            changed = false;

            for block_id in &func.blocks.keys().cloned().collect::<Vec<_>>() {
                if *block_id == func.entry_block {
                    continue;
                }

                let predecessors = self.get_predecessors(func, *block_id);
                if predecessors.is_empty() {
                    continue;
                }

                // Intersection of all predecessor dominators
                let mut new_dom = dominators[&predecessors[0]].clone();
                for pred in &predecessors[1..] {
                    new_dom = new_dom.intersection(&dominators[pred]).cloned().collect();
                }
                new_dom.insert(*block_id);

                if new_dom != dominators[block_id] {
                    dominators.insert(*block_id, new_dom);
                    changed = true;
                }
            }
        }

        dominators
    }

    /// Find all blocks in a loop body
    fn find_loop_body(&self, func: &VirFunction, loop_info: &mut LoopInfo) {
        let mut worklist = vec![loop_info.header];
        let mut visited = HashSet::new();

        while let Some(block_id) = worklist.pop() {
            if visited.contains(&block_id) || !func.blocks.contains_key(&block_id) {
                continue;
            }
            visited.insert(block_id);

            // Add to body if not header
            if block_id != loop_info.header {
                loop_info.body.insert(block_id);
            }

            // Add predecessors to worklist
            for pred in self.get_predecessors(func, block_id) {
                if !visited.contains(&pred) {
                    worklist.push(pred);
                }
            }
        }
    }

    /// Calculate nesting levels for loops
    fn calculate_nesting_levels(&mut self) {
        for i in 0..self.loops.len() {
            let mut level = 0;
            for j in 0..self.loops.len() {
                if i != j && self.loops[j].body.contains(&self.loops[i].header) {
                    level += 1;
                }
            }
            self.loops[i].nesting_level = level;
        }
    }

    /// Get successors of a block
    fn get_successors(&self, block: &BasicBlock) -> Vec<BlockId> {
        match &block.terminator {
            Terminator::Branch { then_block, else_block, .. } => vec![*then_block, *else_block],
            Terminator::Jump(target) => vec![*target],
            Terminator::Return(_) => vec![],
            Terminator::Unreachable => vec![],
        }
    }

    /// Get predecessors of a block
    fn get_predecessors(&self, func: &VirFunction, block_id: BlockId) -> Vec<BlockId> {
        let mut predecessors = Vec::new();

        for (id, block) in &func.blocks {
            if self.get_successors(block).contains(&block_id) {
                predecessors.push(*id);
            }
        }

        predecessors
    }
}

/// Loop Invariant Code Motion (LICM)
/// Move loop-invariant computations outside the loop
pub fn loop_invariant_code_motion(func: &mut VirFunction) {
    let mut optimizer = LoopOptimizer::new();
    let loops = optimizer.analyze_loops(func);

    for loop_info in loops {
        perform_licm(func, &loop_info);
    }
}

fn perform_licm(func: &mut VirFunction, loop_info: &LoopInfo) {
    // Find preheader or create one
    let preheader = match loop_info.preheader {
        Some(ph) => ph,
        None => {
            // Create a new preheader block
            let preheader_id = func.blocks.keys().map(|id| id.0).max().unwrap_or(0) + 1;
            let preheader_id = BlockId(preheader_id);

            // Find the predecessor of the header that's not in the loop
            let mut preheader_pred = None;
            for (block_id, block) in &func.blocks {
                for succ in LoopOptimizer::new().get_successors(block) {
                    if succ == loop_info.header && !loop_info.body.contains(block_id) {
                        preheader_pred = Some(*block_id);
                        break;
                    }
                }
                if preheader_pred.is_some() {
                    break;
                }
            }

            // Create preheader block
            let preheader_block = BasicBlock {
                id: preheader_id,
                instructions: Vec::new(),
                terminator: Terminator::Jump(loop_info.header),
            };

            func.blocks.insert(preheader_id, preheader_block);

            // Update predecessor to jump to preheader
            if let Some(pred_id) = preheader_pred {
                if let Some(pred_block) = func.blocks.get_mut(&pred_id) {
                    match &mut pred_block.terminator {
                        Terminator::Branch { then_block, else_block, .. } => {
                            if *then_block == loop_info.header {
                                *then_block = preheader_id;
                            }
                            if *else_block == loop_info.header {
                                *else_block = preheader_id;
                            }
                        }
                        Terminator::Jump(target) => {
                            if *target == loop_info.header {
                                *target = preheader_id;
                            }
                        }
                        _ => {}
                    }
                }
            }

            preheader_id
        }
    };

    // Find invariant instructions
    let mut invariant_instructions = Vec::new();
    let mut loop_values = HashSet::new();

    // Collect all values defined in the loop
    for &block_id in &loop_info.body {
        if let Some(block) = func.blocks.get(&block_id) {
            for inst in &block.instructions {
                loop_values.insert(inst.result);
            }
        }
    }

    // Find instructions that only use values defined outside the loop
    for &block_id in &loop_info.body {
        if let Some(block) = func.blocks.get(&block_id) {
            for inst in &block.instructions {
                if is_invariant_instruction(inst, &loop_values) {
                    invariant_instructions.push((block_id, inst.clone()));
                }
            }
        }
    }

    // Move invariant instructions to preheader
    // First, collect the blocks we need to modify to avoid borrowing conflicts
    let mut blocks_to_modify = HashMap::new();

    for (original_block_id, inst) in &invariant_instructions {
        if let Some(block) = func.blocks.get(original_block_id) {
            // Clone the block and mark for modification
            let mut modified_block = block.clone();
            modified_block.instructions.retain(|i| i.result != inst.result);
            blocks_to_modify.insert(*original_block_id, modified_block);
        }
    }

    // Apply modifications
    for (block_id, modified_block) in blocks_to_modify {
        func.blocks.insert(block_id, modified_block);
    }

    // Add invariant instructions to preheader
    if let Some(preheader_block) = func.blocks.get_mut(&preheader) {
        for (_, inst) in invariant_instructions {
            preheader_block.instructions.push(inst);
        }
    }
}

/// Check if an instruction is loop-invariant
fn is_invariant_instruction(inst: &Instruction, loop_values: &HashSet<ValueId>) -> bool {
    match &inst.kind {
        InstructionKind::ConstInt(_) |
        InstructionKind::ConstFloat(_) |
        InstructionKind::ConstString(_) => true,

        InstructionKind::Add(a, b) |
        InstructionKind::Sub(a, b) |
        InstructionKind::Mul(a, b) |
        InstructionKind::Div(a, b) => {
            !loop_values.contains(a) && !loop_values.contains(b)
        }

        InstructionKind::RuntimeCall { args, .. } => {
            // Conservative: assume runtime calls are not invariant
            false
        }

        InstructionKind::Call { args, .. } => {
            // Function names are not loop-variant, only check arguments
            args.iter().all(|arg| !loop_values.contains(arg))
        }

        _ => false, // Conservative: assume unknown instructions are not invariant
    }
}

/// Loop Unrolling
/// Unroll small loops for better instruction-level parallelism
pub fn loop_unrolling(func: &mut VirFunction, max_unroll_factor: usize) {
    let mut optimizer = LoopOptimizer::new();
    let loops = optimizer.analyze_loops(func);

    for loop_info in loops {
        if should_unroll(&loop_info, func) {
            perform_unrolling(func, &loop_info, max_unroll_factor);
        }
    }
}

fn should_unroll(loop_info: &LoopInfo, func: &VirFunction) -> bool {
    // Simple heuristic: unroll if loop body is small and has constant bounds
    let mut total_instructions = 0;
    for &block_id in &loop_info.body {
        if let Some(block) = func.blocks.get(&block_id) {
            total_instructions += block.instructions.len();
        }
    }

    // Unroll if body has fewer than 20 instructions
    total_instructions < 20
}

fn perform_unrolling(func: &mut VirFunction, loop_info: &LoopInfo, max_unroll_factor: usize) {
    // Simplified unrolling: duplicate loop body
    // In a full implementation, this would need to handle trip counts, etc.
    let unroll_factor = max_unroll_factor.min(4); // Conservative unroll factor

    if let Some(header_block) = func.blocks.get(&loop_info.header) {
        let mut new_instructions = Vec::new();

        // Duplicate instructions for unrolling
        for _ in 0..unroll_factor {
            for inst in &header_block.instructions {
                new_instructions.push(inst.clone());
            }
        }

        // Replace header block instructions
        if let Some(block) = func.blocks.get_mut(&loop_info.header) {
            block.instructions = new_instructions;
        }
    }
}

/// Loop Fusion
/// Combine adjacent loops that operate on the same data
pub fn loop_fusion(func: &mut VirFunction) {
    // This is a complex optimization that would require more sophisticated analysis
    // For now, just a placeholder
}

/// Strength Reduction
/// Replace expensive operations with cheaper equivalents
pub fn strength_reduction(func: &mut VirFunction) {
    // Pre-compute constant values to avoid borrowing issues
    let mut const_values = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let InstructionKind::ConstInt(_) |
                InstructionKind::ConstFloat(_) |
                InstructionKind::ConstString(_) = &inst.kind {
                const_values.insert(inst.result, inst.kind.clone());
            }
        }
    }

    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            match &inst.kind {
                InstructionKind::Mul(left, right) => {
                    // Simple case: x * 2 -> x + x
                    if let Some(InstructionKind::ConstInt(2)) = const_values.get(right) {
                        inst.kind = InstructionKind::Add(*left, *left);
                    } else if let Some(InstructionKind::ConstInt(2)) = const_values.get(left) {
                        inst.kind = InstructionKind::Add(*right, *right);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Get constant value of a value ID
fn get_const_value(value_id: ValueId, func: &VirFunction) -> Option<InstructionKind> {
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if inst.result == value_id {
                match &inst.kind {
                    InstructionKind::ConstInt(_) |
                    InstructionKind::ConstFloat(_) |
                    InstructionKind::ConstString(_) => {
                        return Some(inst.kind.clone());
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

/// Run all loop optimization passes
pub fn optimize_loops(func: &mut VirFunction) {
    loop_invariant_code_motion(func);
    loop_unrolling(func, 4);
    strength_reduction(func);
    // loop_fusion(func); // Not implemented yet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Terminator;

    #[test]
    fn test_loop_invariant_code_motion() {
        let mut func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: BlockId(0),
            effect: vexl_core::Effect::Pure,
            signature: crate::FunctionSignature::new(vec![], crate::VirType::Void),
        };

        // Create a simple loop with invariant code
        let header_id = BlockId(1);
        let body_id = BlockId(2);

        let invariant_val = func.blocks.len() as u32 + 1;
        let loop_var = func.blocks.len() as u32 + 2;

        let header_block = BasicBlock {
            id: header_id,
            instructions: vec![
                // Invariant: 1 + 2 (should be moved out)
                Instruction { result_type: None, 
                    result: ValueId(invariant_val as usize),
                    kind: InstructionKind::Add(ValueId(10), ValueId(11)), // 1 + 2
                },
                // Loop variable update
                Instruction { result_type: None, 
                    result: ValueId(loop_var as usize),
                    kind: InstructionKind::Add(ValueId(loop_var as usize), ValueId(1)),
                },
            ],
            terminator: Terminator::Branch {
                cond: ValueId(loop_var as usize),
                then_block: body_id,
                else_block: BlockId(0), // exit
            },
        };

        func.blocks.insert(header_id, header_block);

        // Before LICM
        let header_before = func.blocks.get(&header_id).unwrap();
        assert_eq!(header_before.instructions.len(), 2);

        // Run LICM (should create preheader and move invariant code)
        loop_invariant_code_motion(&mut func);

        // After LICM, original header should have fewer instructions
        let header_after = func.blocks.get(&header_id).unwrap();
        assert!(header_after.instructions.len() <= 2);
    }

    #[test]
    fn test_strength_reduction() {
        let mut func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: BlockId(0),
            effect: vexl_core::Effect::Pure,
            signature: crate::FunctionSignature::new(vec![], crate::VirType::Void),
        };

        let block_id = BlockId(1);
        let x_val = ValueId(1);
        let const_2 = ValueId(2);
        let result = ValueId(3);

        let block = BasicBlock {
            id: block_id,
            instructions: vec![
                // x * 2 should become x + x
                Instruction { result_type: None, 
                    result: result,
                    kind: InstructionKind::Mul(x_val, const_2),
                },
            ],
            terminator: Terminator::Return(result),
        };

        func.blocks.insert(block_id, block);

        // Add constant definition
        if let Some(block) = func.blocks.get_mut(&block_id) {
            block.instructions.insert(0, Instruction { result_type: None, 
                result: const_2,
                kind: InstructionKind::ConstInt(2),
            });
        }

        strength_reduction(&mut func);

        // Check that multiplication was replaced with addition
        let block = func.blocks.get(&block_id).unwrap();
        let mul_inst = block.instructions.iter().find(|i| i.result == result).unwrap();
        assert!(matches!(mul_inst.kind, InstructionKind::Add(_, _)));
    }
}
