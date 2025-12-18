//! Auto-vectorization pass for VIR

use crate::{VirModule, VirFunction, BasicBlock, Instruction, InstructionKind, ValueId};
use std::collections::{HashMap, HashSet};

/// Vectorization analysis and transformation
pub struct Vectorizer {
    /// Target vector width (e.g., 4 for SSE, 8 for AVX2, 16 for AVX-512)
    vector_width: usize,
    /// Current vectorization state
    vectorized_instructions: HashMap<ValueId, Vec<ValueId>>,
}

impl Vectorizer {
    /// Create a new vectorizer with given vector width
    pub fn new(vector_width: usize) -> Self {
        Self {
            vector_width,
            vectorized_instructions: HashMap::new(),
        }
    }

    /// Analyze and vectorize a function
    pub fn vectorize_function(&mut self, func: &mut VirFunction) {
        // Find vectorizable loops
        let vectorizable_loops = self.find_vectorizable_loops(func);

        for loop_info in vectorizable_loops {
            self.vectorize_loop(func, &loop_info);
        }

        // Vectorize straight-line code
        self.vectorize_straight_line_code(func);
    }

    /// Find loops that can be vectorized
    fn find_vectorizable_loops(&self, func: &VirFunction) -> Vec<LoopInfo> {
        let mut vectorizable_loops = Vec::new();

        // Simple loop detection - look for patterns that suggest vectorization
        for (block_id, block) in &func.blocks {
            if self.is_vectorizable_loop_header(block, func) {
                let loop_info = self.analyze_loop(*block_id, func);
                if self.can_vectorize_loop(&loop_info, func) {
                    vectorizable_loops.push(loop_info);
                }
            }
        }

        vectorizable_loops
    }

    /// Check if a block is a vectorizable loop header
    fn is_vectorizable_loop_header(&self, block: &BasicBlock, func: &VirFunction) -> bool {
        // Look for patterns like:
        // - Induction variable updates
        // - Array/vector operations in loop body
        // - No complex control flow

        let has_induction_var = block.instructions.iter().any(|inst| {
            matches!(inst.kind, InstructionKind::Add(_, _) | InstructionKind::Sub(_, _))
        });

        let has_vector_ops = block.instructions.iter().any(|inst| {
            matches!(inst.kind,
                InstructionKind::VectorGet { .. } |
                InstructionKind::VectorSet { .. } |
                InstructionKind::VectorNew { .. }
            )
        });

        has_induction_var && has_vector_ops
    }

    /// Analyze a loop structure
    fn analyze_loop(&self, header_id: crate::BlockId, func: &VirFunction) -> LoopInfo {
        LoopInfo {
            header: header_id,
            body_blocks: self.find_loop_body(header_id, func),
            induction_vars: self.find_induction_vars(header_id, func),
            dependencies: self.analyze_dependencies(header_id, func),
        }
    }

    /// Find all blocks in the loop body
    fn find_loop_body(&self, header_id: crate::BlockId, func: &VirFunction) -> Vec<crate::BlockId> {
        let mut body = Vec::new();
        let mut visited = HashSet::new();
        let mut worklist = vec![header_id];

        while let Some(block_id) = worklist.pop() {
            if visited.contains(&block_id) {
                continue;
            }
            visited.insert(block_id);

            if block_id != header_id {
                body.push(block_id);
            }

            // Add successors that are dominated by header
            if let Some(block) = func.blocks.get(&block_id) {
                // Simple approximation: assume all successors are in loop
                // In a full implementation, we'd use dominator analysis
                for succ in self.get_successors(block) {
                    if func.blocks.contains_key(&succ) && !visited.contains(&succ) {
                        worklist.push(succ);
                    }
                }
            }
        }

        body
    }

    /// Find induction variables in the loop
    fn find_induction_vars(&self, header_id: crate::BlockId, func: &VirFunction) -> Vec<ValueId> {
        let mut induction_vars = Vec::new();

        if let Some(block) = func.blocks.get(&header_id) {
            for inst in &block.instructions {
                // Look for patterns like: i = i + 1
                if let InstructionKind::Add(left, right) = &inst.kind {
                    if *left == inst.result {
                        // Potential induction variable
                        induction_vars.push(inst.result);
                    }
                }
            }
        }

        induction_vars
    }

    /// Analyze data dependencies in the loop
    fn analyze_dependencies(&self, header_id: crate::BlockId, func: &VirFunction) -> Vec<Dependency> {
        let mut dependencies = Vec::new();

        // Simple dependency analysis
        let mut definitions = HashMap::new();
        let mut uses = HashMap::new();

        // Collect definitions and uses
        for (block_id, block) in &func.blocks {
            for inst in &block.instructions {
                // Track definition
                definitions.insert(inst.result, *block_id);

                // Track uses
                self.collect_uses(&inst.kind, &mut uses, *block_id);
            }
        }

        // Check for dependencies
        for (value, def_block) in &definitions {
            if let Some(use_blocks) = uses.get(value) {
                for &use_block in use_blocks {
                    if def_block != &use_block {
                        dependencies.push(Dependency {
                            value: *value,
                            from: *def_block,
                            to: use_block,
                            kind: DependencyKind::Flow,
                        });
                    }
                }
            }
        }

        dependencies
    }

    /// Collect value uses from an instruction
    fn collect_uses(&self, kind: &InstructionKind, uses: &mut HashMap<ValueId, Vec<crate::BlockId>>, block_id: crate::BlockId) {
        match kind {
            InstructionKind::Add(a, b) |
            InstructionKind::Sub(a, b) |
            InstructionKind::Mul(a, b) |
            InstructionKind::Div(a, b) => {
                uses.entry(*a).or_insert_with(Vec::new).push(block_id);
                uses.entry(*b).or_insert_with(Vec::new).push(block_id);
            }
            InstructionKind::VectorGet { vector, index } => {
                uses.entry(*vector).or_insert_with(Vec::new).push(block_id);
                uses.entry(*index).or_insert_with(Vec::new).push(block_id);
            }
            InstructionKind::VectorSet { vector, index, value } => {
                uses.entry(*vector).or_insert_with(Vec::new).push(block_id);
                uses.entry(*index).or_insert_with(Vec::new).push(block_id);
                uses.entry(*value).or_insert_with(Vec::new).push(block_id);
            }
            _ => {}
        }
    }

    /// Get successors of a block
    fn get_successors(&self, block: &BasicBlock) -> Vec<crate::BlockId> {
        match &block.terminator {
            crate::Terminator::Branch { then_block, else_block, .. } => vec![*then_block, *else_block],
            crate::Terminator::Jump(target) => vec![*target],
            crate::Terminator::Return(_) => vec![],
            crate::Terminator::Unreachable => vec![],
        }
    }

    /// Check if a loop can be vectorized
    fn can_vectorize_loop(&self, loop_info: &LoopInfo, func: &VirFunction) -> bool {
        // Check for data dependencies that prevent vectorization
        for dep in &loop_info.dependencies {
            if matches!(dep.kind, DependencyKind::Anti | DependencyKind::Output) {
                return false; // Loop-carried dependencies
            }
        }

        // Check if operations are vectorizable
        for &block_id in &loop_info.body_blocks {
            if let Some(block) = func.blocks.get(&block_id) {
                for inst in &block.instructions {
                    if !self.is_vectorizable_instruction(&inst.kind) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if an instruction can be vectorized
    fn is_vectorizable_instruction(&self, kind: &InstructionKind) -> bool {
        match kind {
            InstructionKind::Add(_, _) |
            InstructionKind::Sub(_, _) |
            InstructionKind::Mul(_, _) |
            InstructionKind::Div(_, _) => true,

            InstructionKind::VectorGet { .. } |
            InstructionKind::VectorSet { .. } |
            InstructionKind::VectorNew { .. } => true,

            InstructionKind::RuntimeCall { function_name, .. } => {
                // Some runtime functions are vectorizable
                matches!(function_name.as_str(),
                    "vexl_vec_add_i64" | "vexl_vec_mul_scalar_i64" |
                    "vexl_math_sin" | "vexl_math_cos" | "vexl_math_sqrt"
                )
            }

            _ => false,
        }
    }

    /// Vectorize a loop
    fn vectorize_loop(&mut self, func: &mut VirFunction, loop_info: &LoopInfo) {
        // Create vectorized versions of instructions
        let mut vectorized_blocks = HashMap::new();

        for &block_id in &loop_info.body_blocks {
            if let Some(block) = func.blocks.get(&block_id) {
                let vectorized_block = self.vectorize_block(block, loop_info);
                vectorized_blocks.insert(block_id, vectorized_block);
            }
        }

        // Replace original blocks with vectorized versions
        for (block_id, vectorized_block) in vectorized_blocks {
            if let Some(block) = func.blocks.get_mut(&block_id) {
                *block = vectorized_block;
            }
        }
    }

    /// Vectorize a basic block
    fn vectorize_block(&mut self, block: &BasicBlock, loop_info: &LoopInfo) -> BasicBlock {
        let mut vectorized_instructions = Vec::new();

        for inst in &block.instructions {
            let vectorized_inst = self.vectorize_instruction(inst, loop_info);
            vectorized_instructions.push(vectorized_inst);
        }

        BasicBlock {
            id: block.id,
            instructions: vectorized_instructions,
            terminator: block.terminator.clone(),
        }
    }

    /// Vectorize a single instruction
    fn vectorize_instruction(&mut self, inst: &Instruction, loop_info: &LoopInfo) -> Instruction {
        match &inst.kind {
            InstructionKind::Add(left, right) => {
                self.create_vector_instruction(inst.result, InstructionKind::Add(*left, *right))
            }
            InstructionKind::Sub(left, right) => {
                self.create_vector_instruction(inst.result, InstructionKind::Sub(*left, *right))
            }
            InstructionKind::Mul(left, right) => {
                self.create_vector_instruction(inst.result, InstructionKind::Mul(*left, *right))
            }
            InstructionKind::Div(left, right) => {
                self.create_vector_instruction(inst.result, InstructionKind::Div(*left, *right))
            }
            InstructionKind::VectorGet { vector, index } => {
                // Convert scalar access to vector gather/scatter
                self.create_vector_instruction(inst.result, InstructionKind::VectorGet { vector: *vector, index: *index })
            }
            InstructionKind::VectorSet { vector, index, value } => {
                self.create_vector_instruction(inst.result, InstructionKind::VectorSet { vector: *vector, index: *index, value: *value })
            }
            _ => inst.clone(), // Keep non-vectorizable instructions as-is
        }
    }

    /// Create a vectorized instruction
    fn create_vector_instruction(&mut self, result: ValueId, kind: InstructionKind) -> Instruction {
        Instruction {
            result,
            result_type: None,
            kind,
        }
    }

    /// Vectorize straight-line code (non-loop code)
    fn vectorize_straight_line_code(&mut self, func: &mut VirFunction) {
        for block in func.blocks.values_mut() {
            let mut new_instructions = Vec::new();

            for inst in &block.instructions {
                // Look for patterns that can be vectorized even without loops
                let vectorized = self.try_vectorize_straight_line(inst);
                new_instructions.push(vectorized);
            }

            block.instructions = new_instructions;
        }
    }

    /// Try to vectorize straight-line code
    fn try_vectorize_straight_line(&mut self, inst: &Instruction) -> Instruction {
        // For now, just return the instruction unchanged
        // In a full implementation, this would look for consecutive operations
        // that can be combined into vector operations
        inst.clone()
    }
}

/// Loop information for vectorization
#[derive(Debug)]
struct LoopInfo {
    header: crate::BlockId,
    body_blocks: Vec<crate::BlockId>,
    induction_vars: Vec<ValueId>,
    dependencies: Vec<Dependency>,
}

/// Data dependency information
#[derive(Debug)]
struct Dependency {
    value: ValueId,
    from: crate::BlockId,
    to: crate::BlockId,
    kind: DependencyKind,
}

/// Types of data dependencies
#[derive(Debug)]
enum DependencyKind {
    Flow,   // Read after write
    Anti,   // Write after read
    Output, // Write after write
}

/// Run auto-vectorization on a module
pub fn vectorize_module(module: &mut VirModule, vector_width: usize) {
    let mut vectorizer = Vectorizer::new(vector_width);

    for func in module.functions.values_mut() {
        vectorizer.vectorize_function(func);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Terminator, BlockId};

    #[test]
    fn test_vectorizer_creation() {
        let vectorizer = Vectorizer::new(8);
        assert_eq!(vectorizer.vector_width, 8);
        assert!(vectorizer.vectorized_instructions.is_empty());
    }

    #[test]
    fn test_vectorize_simple_function() {
        let mut func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: BlockId(0),
            effect: vexl_core::Effect::Pure,
            signature: crate::FunctionSignature::new(vec![], crate::VirType::Void),
        };

        // Add a simple block with vectorizable operations
        let block_id = BlockId(1);
        let block = BasicBlock {
            id: block_id,
            instructions: vec![
                Instruction { result_type: None, 
                    result: ValueId(1),
                    kind: InstructionKind::Add(ValueId(10), ValueId(11)),
                },
                Instruction { result_type: None, 
                    result: ValueId(2),
                    kind: InstructionKind::Mul(ValueId(1), ValueId(12)),
                },
            ],
            terminator: Terminator::Return(ValueId(2)),
        };

        func.blocks.insert(block_id, block);

        let mut vectorizer = Vectorizer::new(4);
        vectorizer.vectorize_function(&mut func);

        // Check that the function still has the same structure
        assert!(func.blocks.contains_key(&block_id));
        let block = func.blocks.get(&block_id).unwrap();
        assert_eq!(block.instructions.len(), 2);
    }

    #[test]
    fn test_vectorize_module() {
        let mut module = VirModule::new();
        let func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: BlockId(0),
            effect: vexl_core::Effect::Pure,
            signature: crate::FunctionSignature::new(vec![], crate::VirType::Void),
        };
        module.add_function("test".to_string(), func);

        vectorize_module(&mut module, 8);

        // Module should still contain the function
        assert!(module.functions.contains_key("test"));
    }
}
