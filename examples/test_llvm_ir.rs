//! Quick test to see LLVM IR output

use vexl_ir::*;
use vexl_codegen::codegen_to_string;
use std::collections::HashMap;

fn main() {
    // Test 1 + 2
    let mut module = VirModule::new();
    let v1 = module.fresh_value();
    let v2 = module.fresh_value();
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
    
    let func = VirFunction {
        name: "main".to_string(),
        params: vec![],
        blocks: HashMap::from([(block_id, block)]),
        entry_block: block_id,
        effect: vexl_core::Effect::Pure,
    };
    
    module.add_function("main".to_string(), func);
    
    let llvm_ir = codegen_to_string(&module).unwrap();
    println!("=== LLVM IR for '1 + 2' ===");
    println!("{}", llvm_ir);
}
