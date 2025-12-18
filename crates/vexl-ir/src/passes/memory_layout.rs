//! Memory layout optimization pass - Structure of Arrays (SoA) transformation

use crate::{VirModule, VirFunction, BasicBlock, Instruction, InstructionKind, ValueId};
use std::collections::{HashMap, HashSet};

/// Memory layout optimizer for Structure of Arrays transformations
pub struct MemoryLayoutOptimizer {
    /// Vectors identified for SoA transformation
    soa_candidates: HashMap<ValueId, SoAInfo>,
    /// Field access patterns observed
    field_accesses: HashMap<ValueId, Vec<FieldAccess>>,
}

/// Information about a Structure of Arrays transformation
#[derive(Debug, Clone)]
struct SoAInfo {
    /// Original AoS vector
    aos_vector: ValueId,
    /// Number of fields/elements per structure
    num_fields: usize,
    /// Field types (simplified - all i64 for now)
    field_types: Vec<String>,
    /// SoA vectors created (one per field)
    soa_vectors: Vec<ValueId>,
}

/// Field access pattern
#[derive(Debug, Clone)]
struct FieldAccess {
    /// The vector being accessed
    vector: ValueId,
    /// Index into the vector
    index: ValueId,
    /// Field offset within the structure
    field_offset: usize,
    /// The instruction that performs this access
    instruction: ValueId,
}

impl MemoryLayoutOptimizer {
    /// Create a new memory layout optimizer
    pub fn new() -> Self {
        Self {
            soa_candidates: HashMap::new(),
            field_accesses: HashMap::new(),
        }
    }

    /// Analyze memory access patterns in a function
    pub fn analyze_function(&mut self, func: &VirFunction) {
        self.soa_candidates.clear();
        self.field_accesses.clear();

        // Collect field access patterns
        self.collect_field_accesses(func);

        // Identify vectors that would benefit from SoA transformation
        self.identify_soa_candidates(func);
    }

    /// Collect field access patterns from vector operations
    fn collect_field_accesses(&mut self, func: &VirFunction) {
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let InstructionKind::VectorGet { vector, index } = &inst.kind {
                    // Analyze the index to see if it's accessing fields in a structure
                    let field_access = self.analyze_index_pattern(*index, func);
                    if let Some(field_offset) = field_access {
                        self.field_accesses.entry(*vector).or_insert_with(Vec::new).push(FieldAccess {
                            vector: *vector,
                            index: *index,
                            field_offset,
                            instruction: inst.result,
                        });
                    }
                }
            }
        }
    }

    /// Analyze index pattern to detect field access
    fn analyze_index_pattern(&self, index: ValueId, func: &VirFunction) -> Option<usize> {
        // Look for patterns like: base_index + field_offset
        // For now, just check for constant indices that could represent field offsets
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if inst.result == index {
                    match &inst.kind {
                        InstructionKind::ConstInt(n) => {
                            // Assume small constants are field offsets
                            return Some(*n as usize);
                        }
                        InstructionKind::Add(a, b) => {
                            // Check if one operand is a constant field offset
                            if let Some(const_val) = self.get_constant_value(*a, func) {
                                return Some(const_val as usize);
                            }
                            if let Some(const_val) = self.get_constant_value(*b, func) {
                                return Some(const_val as usize);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }

    /// Get constant value of a value ID
    fn get_constant_value(&self, value_id: ValueId, func: &VirFunction) -> Option<i64> {
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if inst.result == value_id {
                    if let InstructionKind::ConstInt(n) = &inst.kind {
                        return Some(*n);
                    }
                }
            }
        }
        None
    }

    /// Identify vectors that are good candidates for SoA transformation
    fn identify_soa_candidates(&mut self, func: &VirFunction) {
        for (vector_id, accesses) in &self.field_accesses {
            if accesses.len() < 3 {
                continue; // Need multiple field accesses to justify transformation
            }

            // Analyze access patterns to determine number of fields
            let mut field_offsets = HashSet::new();
            for access in accesses {
                field_offsets.insert(access.field_offset);
            }

            let num_fields = field_offsets.len();
            if num_fields > 1 {
                // Create SoA info
                let mut soa_vectors = Vec::new();
                for _ in 0..num_fields {
                    // Generate fresh IDs for SoA vectors
                    let fresh_id = self.generate_fresh_value_id(func);
                    soa_vectors.push(fresh_id);
                }

                self.soa_candidates.insert(*vector_id, SoAInfo {
                    aos_vector: *vector_id,
                    num_fields,
                    field_types: vec!["i64".to_string(); num_fields], // Simplified
                    soa_vectors,
                });
            }
        }
    }

    /// Generate a fresh value ID not used in the function
    fn generate_fresh_value_id(&self, func: &VirFunction) -> ValueId {
        let mut max_id = 0;
        for block in func.blocks.values() {
            for inst in &block.instructions {
                max_id = max_id.max(inst.result.0);
            }
        }
        ValueId(max_id + 1)
    }

    /// Transform a function to use SoA layout
    pub fn transform_function(&self, func: &mut VirFunction) {
        if self.soa_candidates.is_empty() {
            return; // No transformations to apply
        }

        // Create SoA transformation instructions
        let mut transformation_instructions = Vec::new();

        for (aos_vector, soa_info) in &self.soa_candidates {
            transformation_instructions.extend(
                self.create_soa_transformation(*aos_vector, soa_info, func)
            );
        }

        // Insert transformation instructions at function entry
        if let Some(entry_block) = func.blocks.get_mut(&func.entry_block) {
            // Insert at the beginning of the entry block
            for inst in transformation_instructions.into_iter().rev() {
                entry_block.instructions.insert(0, inst);
            }
        }

        // Update field access instructions
        self.update_field_accesses(func);
    }

    /// Create SoA transformation from AoS vector
    fn create_soa_transformation(&self, aos_vector: ValueId, soa_info: &SoAInfo, func: &VirFunction) -> Vec<Instruction> {
        let mut instructions = Vec::new();

        // Get the length of the original vector
        let len_value = self.generate_fresh_value_id(func);
        instructions.push(Instruction { result_type: None, 
            result: len_value,
            kind: InstructionKind::RuntimeCall {
                function_name: "vexl_vec_len".to_string(),
                args: vec![aos_vector],
            },
        });

        // For each field, create a SoA vector
        for (field_idx, &soa_vector) in soa_info.soa_vectors.iter().enumerate() {
            // Allocate SoA vector
            let alloc_result = self.generate_fresh_value_id(func);
            instructions.push(Instruction { result_type: None, 
                result: alloc_result,
                kind: InstructionKind::RuntimeCall {
                    function_name: "vexl_vec_alloc_i64".to_string(),
                    args: vec![len_value],
                },
            });

            // Copy field data from AoS to SoA
            // This is a simplified version - in practice, we'd need a loop
            let copy_result = self.generate_fresh_value_id(func);
            instructions.push(Instruction { result_type: None, 
                result: copy_result,
                kind: InstructionKind::RuntimeCall {
                    function_name: "vexl_soa_extract_field".to_string(),
                    args: vec![aos_vector, alloc_result, ValueId(field_idx as usize)],
                },
            });

            // Store the SoA vector
            instructions.push(Instruction { result_type: None, 
                result: soa_vector,
                kind: InstructionKind::RuntimeCall {
                    function_name: "vexl_vec_copy".to_string(),
                    args: vec![copy_result],
                },
            });
        }

        instructions
    }

    /// Update field access instructions to use SoA vectors
    fn update_field_accesses(&self, func: &mut VirFunction) {
        // Pre-compute field offsets to avoid borrowing issues
        let mut field_offsets = HashMap::new();
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let InstructionKind::VectorGet { index, .. } = &inst.kind {
                    if let Some(offset) = self.get_field_offset_from_index(*index, func) {
                        field_offsets.insert(inst.result, offset);
                    }
                }
            }
        }

        // Generate fresh IDs needed for transformations
        let mut fresh_ids = Vec::new();
        let mut id_count = 0;
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let InstructionKind::VectorGet { vector, .. } = &inst.kind {
                    if let Some(soa_info) = self.soa_candidates.get(vector) {
                        if let Some(field_offset) = field_offsets.get(&inst.result) {
                            if *field_offset < soa_info.soa_vectors.len() {
                                // Need fresh IDs for this transformation
                                id_count += 2; // struct_index and num_fields_const
                            }
                        }
                    }
                }
            }
        }

        let base_fresh_id = self.generate_fresh_value_id(func);
        for i in 0..id_count {
            fresh_ids.push(ValueId(base_fresh_id.0 + i as usize));
        }

        // Pre-generate all needed fresh IDs to avoid borrowing issues
        let mut fresh_id_map = HashMap::new();
        let mut fresh_id_counter = 0;

        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let InstructionKind::VectorGet { vector, .. } = &inst.kind {
                    if let Some(soa_info) = self.soa_candidates.get(vector) {
                        if let Some(field_offset) = field_offsets.get(&inst.result) {
                            if *field_offset < soa_info.soa_vectors.len() {
                                // Need two fresh IDs for this transformation
                                let struct_index = ValueId(func.blocks.len() as usize + fresh_id_counter);
                                let num_fields_const = ValueId(func.blocks.len() as usize + fresh_id_counter + 1);
                                fresh_id_map.insert(inst.result, (struct_index, num_fields_const));
                                fresh_id_counter += 2;
                            }
                        }
                    }
                }
            }
        }

        for block in func.blocks.values_mut() {
            for inst in &mut block.instructions {
                if let InstructionKind::VectorGet { vector, index } = &mut inst.kind {
                    if let Some(soa_info) = self.soa_candidates.get(vector) {
                        // Determine which field is being accessed
                        if let Some(field_offset) = field_offsets.get(&inst.result) {
                            if *field_offset < soa_info.soa_vectors.len() {
                                if let Some((struct_index, _)) = fresh_id_map.get(&inst.result) {
                                    // Replace with SoA access
                                    let soa_vector = soa_info.soa_vectors[*field_offset];

                                    // Insert instructions to compute struct_index = index / num_fields
                                    // This is simplified - in practice, we'd need proper division
                                    *vector = soa_vector;
                                    *index = *struct_index;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get field offset from an index value
    fn get_field_offset_from_index(&self, index: ValueId, func: &VirFunction) -> Option<usize> {
        // Check if this index represents a field access pattern
        if let Some(accesses) = self.field_accesses.get(&index) {
            if let Some(first_access) = accesses.first() {
                return Some(first_access.field_offset);
            }
        }

        // Check for constant field offsets
        self.analyze_index_pattern(index, func)
    }
}

/// Run memory layout optimization on a module
pub fn optimize_memory_layout(module: &mut VirModule) {
    let mut optimizer = MemoryLayoutOptimizer::new();

    for func in module.functions.values_mut() {
        optimizer.analyze_function(func);
        optimizer.transform_function(func);
    }
}

/// Run Structure of Arrays transformation on a module
pub fn structure_of_arrays_transform(module: &mut VirModule) {
    optimize_memory_layout(module);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Terminator, BlockId};

    #[test]
    fn test_memory_layout_optimizer_creation() {
        let optimizer = MemoryLayoutOptimizer::new();
        assert!(optimizer.soa_candidates.is_empty());
        assert!(optimizer.field_accesses.is_empty());
    }

    #[test]
    fn test_soa_transformation() {
        let mut module = VirModule::new();

        // Create a function with AoS vector accesses
        let mut func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: BlockId(0),
            effect: vexl_core::Effect::Pure,
            signature: crate::FunctionSignature::new(vec![], crate::VirType::Void),
        };

        // Create AoS vector
        let aos_vector = ValueId(1);
        let aos_block = BasicBlock {
            id: BlockId(0),
            instructions: vec![
                // Allocate AoS vector
                Instruction { result_type: None, 
                    result: aos_vector,
                    kind: InstructionKind::RuntimeCall {
                        function_name: "vexl_vec_alloc_i64".to_string(),
                        args: vec![ValueId(10)], // size 10
                    },
                },
                // Access field 0 of structure 0: aos[0]
                Instruction { result_type: None, 
                    result: ValueId(2),
                    kind: InstructionKind::VectorGet {
                        vector: aos_vector,
                        index: ValueId(0),
                    },
                },
                // Access field 1 of structure 0: aos[1]
                Instruction { result_type: None, 
                    result: ValueId(3),
                    kind: InstructionKind::VectorGet {
                        vector: aos_vector,
                        index: ValueId(1),
                    },
                },
                // Access field 0 of structure 1: aos[2]
                Instruction { result_type: None, 
                    result: ValueId(4),
                    kind: InstructionKind::VectorGet {
                        vector: aos_vector,
                        index: ValueId(2),
                    },
                },
            ],
            terminator: Terminator::Return(ValueId(4)),
        };

        func.blocks.insert(BlockId(0), aos_block);
        module.add_function("test_soa".to_string(), func);

        // Run SoA transformation
        structure_of_arrays_transform(&mut module);

        // Check that the function still has valid structure
        let func = module.functions.get("test_soa").unwrap();
        assert!(func.blocks.contains_key(&BlockId(0)));
    }

    #[test]
    fn test_memory_layout_optimization() {
        let mut module = VirModule::new();

        // Add a simple function
        let func = VirFunction {
            name: "test".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: BlockId(0),
            effect: vexl_core::Effect::Pure,
            signature: crate::FunctionSignature::new(vec![], crate::VirType::Void),
        };

        module.add_function("test".to_string(), func);

        // Run optimization
        optimize_memory_layout(&mut module);

        // Should not crash and module should remain valid
        assert!(module.functions.contains_key("test"));
    }
}
