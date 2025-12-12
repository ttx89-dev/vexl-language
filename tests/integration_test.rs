//! End-to-end integration tests: Parse → Type Check → Lower → Optimize

use vexl_syntax::parser::parse;
use vexl_types::inference::TypeEnv;
use vexl_ir::lower::lower_to_vir;
use vexl_ir::optimize::optimize;

#[test]
fn test_e2e_simple_arithmetic() {
    // Complete pipeline: "1 + 2"
    let source = "1 + 2";
    
    // Parse
    let ast = parse(source).expect("Parse failed");
    
    // Type check
    let mut type_env = TypeEnv::new();
    let result = vexl_types::inference::infer(&ast, &mut type_env);
    assert!(result.is_ok(), "Type inference failed");
    
    // Lower to VIR
    let mut module = lower_to_vir(&ast).expect("Lowering failed");
    
    // Optimize
    optimize(&mut module);
    
    // Success - complete pipeline works!
    assert!(module.next_value_id > 0);
}

#[test]
fn test_e2e_vector() {
    // Parse: "[1, 2, 3]"
    let source = "[1, 2, 3]";
    let ast = parse(source).expect("Parse failed");
    
    // Type check with dimensional inference
    let mut type_env = TypeEnv::new();
    let result = vexl_types::inference::infer(&ast, &mut type_env);
    assert!(result.is_ok());
    
    let (ty, _) = result.unwrap();
    // Should be Vector<Int, 1>
    assert!(matches!(ty, vexl_types::inference::InferredType::Vector { .. }));
    
    // Lower and verify
    let module = lower_to_vir(&ast).expect("Lowering failed");
    assert!(module.next_value_id >= 4); // 3 elements + vector
}

#[test]
fn test_e2e_let_binding() {
    // Parse: "let x = 5 x"
    let source = "let x = 5 x";
    let ast = parse(source).expect("Parse failed");
    
    // Type check
    let mut type_env = TypeEnv::new();
    let result = vexl_types::inference::infer(&ast, &mut type_env);
    assert!(result.is_ok());
    
    // Lower
    let module = lower_to_vir(&ast).expect("Lowering failed");
    assert!(module.next_value_id >= 1);
}

#[test]
fn test_e2e_pipeline() {
    // Parse VEXL's signature feature: "data |> process"
    let source = "data |> process";
    let ast = parse(source).expect("Parse failed");
    
    // Verify pipeline structure
    assert!(matches!(ast, vexl_syntax::ast::Expr::Pipeline { .. }));
}

#[test]
fn test_e2e_range() {
    // Parse: "[0..10]"
    let source = "[0..10]";
    let ast = parse(source).expect("Parse failed");
    
    // Type check
    let mut type_env = TypeEnv::new();
    let result = vexl_types::inference::infer(&ast, &mut type_env);
    assert!(result.is_ok());
    
    // Lower
    let module = lower_to_vir(&ast).expect("Lowering failed");
    assert!(module.next_value_id >= 2); // start + end
}
