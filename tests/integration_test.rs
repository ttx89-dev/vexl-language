//! End-to-end integration tests: Parse -> Type Infer -> Lower -> Optimize -> Codegen
//!
//! These tests exercise the full VEXL pipeline using the new type inference
//! integration (infer_module_types, type_check_module, lower_decls_to_vir).
//! Run with: cargo test -p vexl-integration-tests

use vexl_syntax::parser::parse;
use vexl_ir::lower::{lower_to_vir, lower_decls_to_vir, type_check_module, infer_module_types};
use vexl_ir::optimize::optimize;
use vexl_types::InferredType;

// ─── Module-level type inference ───────────────────────────────────────────

#[test]
fn test_infer_module_simple_function() {
    // A function returning an integer constant
    let source = "fn main() -> int { 42 }";
    let ast = parse(source).expect("Parse failed");

    let results = infer_module_types(&ast).expect("Type inference failed");
    assert_eq!(results.len(), 1, "Expected one function result");

    let (name, ty, effect) = &results[0];
    assert_eq!(name, "main");
    assert!(matches!(ty, InferredType::Int), "Expected Int, got {:?}", ty);
    assert_eq!(*effect, vexl_core::Effect::Pure);
}

#[test]
fn test_infer_module_binop_types() {
    // Expression type inference for arithmetic
    let source = "fn add(a: int, b: int) -> int { a + b }";
    let ast = parse(source).expect("Parse failed");

    let results = infer_module_types(&ast).expect("Type inference failed");
    assert_eq!(results.len(), 1);

    let (name, ty, _) = &results[0];
    assert_eq!(name, "add");
    assert!(matches!(ty, InferredType::Int), "Expected Int, got {:?}", ty);
}

#[test]
fn test_infer_module_float() {
    let source = "fn pi() -> float { 3.14159 }";
    let ast = parse(source).expect("Parse failed");

    let results = infer_module_types(&ast).expect("Type inference failed");
    let (name, ty, _) = &results[0];
    assert_eq!(name, "pi");
    assert!(matches!(ty, InferredType::Float), "Expected Float, got {:?}", ty);
}

// ─── type_check_module ─────────────────────────────────────────────────────

#[test]
fn test_type_check_module_passes() {
    let source = "fn main() -> int { 0 }";
    let ast = parse(source).expect("Parse failed");
    let results = type_check_module(&ast).expect("Type check failed");

    assert_eq!(results.len(), 1);
    let (name, result) = &results[0];
    assert_eq!(name, "main");
    assert!(result.is_ok(), "Expected OK, got {:?}", result);
}

#[test]
fn test_type_check_multiple_functions() {
    let source = "\
        fn add(a: int, b: int) -> int { a + b }
        fn main() -> int { add(1, 2) }
    ";
    let ast = parse(source).expect("Parse failed");
    let results = type_check_module(&ast).expect("Type check failed");

    assert_eq!(results.len(), 2);
    for (name, result) in &results {
        assert!(result.is_ok(), "Function {} failed type check: {:?}", name, result);
    }
}

// ─── lower_decls_to_vir with type inference ────────────────────────────────

#[test]
fn test_lower_decls_function_types() {
    let source = "fn add(a: int, b: int) -> int { a + b }";
    let ast = parse(source).expect("Parse failed");
    let module = lower_decls_to_vir(&ast).expect("Lowering failed");

    let add_fn = module.functions.get("add")
        .expect("add function not found in module");

    // Ret type should be Int64 (from inferred_type_to_vir_type(InferredType::Int))
    assert_eq!(add_fn.signature.return_type, vexl_ir::VirType::Int64,
        "Expected Int64 return type");
    assert_eq!(add_fn.signature.param_types.len(), 2,
        "Expected 2 parameters");

    // Effect should have been set from inference (Pure for arithmetic)
    assert_eq!(add_fn.effect, vexl_core::Effect::Pure,
        "Expected Pure effect for arithmetic function");
}

#[test]
fn test_lower_decls_main_function() {
    let source = "fn main() -> int { 42 }";
    let ast = parse(source).expect("Parse failed");
    let module = lower_decls_to_vir(&ast).expect("Lowering failed");

    let main_fn = module.functions.get("main")
        .expect("main function not found");
    assert_eq!(main_fn.signature.return_type, vexl_ir::VirType::Int64);
    assert!(!main_fn.blocks.is_empty());
}

#[test]
fn test_lower_decls_with_effect() {
    // A function that calls print should have Io effect
    let source = "fn greet() -> int { print(42) }";
    let ast = parse(source).expect("Parse failed");
    let module = lower_decls_to_vir(&ast).expect("Lowering failed");

    let greet_fn = module.functions.get("greet")
        .expect("greet function not found");
    // print has Io effect, so the function should propagate it
    assert_eq!(greet_fn.effect, vexl_core::Effect::Io,
        "print function should have Io effect");
}

// ─── Full pipeline: parse -> infer -> lower -> codegen ─────────────────────

#[test]
fn test_e2e_simple_arithmetic() {
    let source = "1 + 2";
    let ast = parse(source).expect("Parse failed");

    // Type inference (on single expression)
    let results = infer_module_types(&ast).expect("Type inference failed");
    assert!(!results.is_empty());

    // Lower to VIR
    let mut module = lower_decls_to_vir(&ast).expect("Lowering failed");

    // Optimize
    optimize(&mut module);

    assert!(module.next_value_id > 0, "Module should have values");
}

#[test]
fn test_e2e_vector_literal() {
    let source = "[1, 2, 3]";
    let ast = parse(source).expect("Parse failed");

    // Type inference should produce Vector type
    let results = infer_module_types(&ast).expect("Type inference failed");
    let (name, ty, _) = &results[0];
    assert!(matches!(ty, InferredType::Vector { .. }),
        "Expected Vector type, got {:?}", ty);

    // Lower should produce at least a few values
    let module = lower_decls_to_vir(&ast).expect("Lowering failed");
    assert!(module.next_value_id >= 4, "Expected >= 4 values, got {}", module.next_value_id);
}

#[test]
fn test_e2e_let_binding() {
    let source = "let x = 5 x";
    let ast = parse(source).expect("Parse failed");

    // Type inference
    let _results = infer_module_types(&ast).expect("Type inference failed");

    // Lower
    let module = lower_decls_to_vir(&ast).expect("Lowering failed");
    assert!(module.next_value_id >= 1);
}

#[test]
fn test_e2e_function_call() {
    let source = "\
        fn double(x: int) -> int { x * 2 }
        fn main() -> int { double(21) }
    ";
    let ast = parse(source).expect("Parse failed");

    // Full pipeline: infer -> lower -> optimize
    let results = type_check_module(&ast).expect("Type check failed");
    assert_eq!(results.len(), 2);
    assert!(results[0].1.is_ok(), "double type check failed");
    assert!(results[1].1.is_ok(), "main type check failed");

    let mut module = lower_decls_to_vir(&ast).expect("Lowering failed");
    assert!(module.functions.contains_key("double"));
    assert!(module.functions.contains_key("main"));

    optimize(&mut module);
}

#[test]
fn test_e2e_string_concatenation() {
    let source = r#""hello, " + "world""#;
    let ast = parse(source).expect("Parse failed");

    let results = infer_module_types(&ast).expect("Type inference failed");
    let (name, ty, _) = &results[0];
    assert!(matches!(ty, InferredType::String),
        "String concat should produce String, got {:?}", ty);

    let module = lower_decls_to_vir(&ast).expect("Lowering failed");
    assert!(module.next_value_id > 0);
}

// ─── Effect inference ──────────────────────────────────────────────────────

#[test]
fn test_effect_pure_arithmetic() {
    let source = "fn calc() -> int { 1 + 2 * 3 }";
    let ast = parse(source).expect("Parse failed");
    let results = infer_module_types(&ast).expect("Inference failed");

    let (_, _, effect) = &results[0];
    assert_eq!(*effect, vexl_core::Effect::Pure,
        "Pure arithmetic should have Pure effect");
}

#[test]
fn test_effect_io_print() {
    let source = "fn say_hi() -> int { print(99) }";
    let ast = parse(source).expect("Parse failed");
    let results = infer_module_types(&ast).expect("Inference failed");

    let (_, _, effect) = &results[0];
    assert_eq!(*effect, vexl_core::Effect::Io,
        "print should produce Io effect");
}

// ─── Edge cases ────────────────────────────────────────────────────────────

#[test]
fn test_empty_module() {
    let source = "";
    let ast = parse(source).expect("Parse failed");

    let results = infer_module_types(&ast).expect("Type inference failed on empty module");
    assert!(results.is_empty(), "Empty module should have no results");
}

#[test]
fn test_identity_function() {
    let source = "fn id(x: int) -> int { x }";
    let ast = parse(source).expect("Parse failed");

    let results = type_check_module(&ast).expect("Type check failed");
    assert_eq!(results.len(), 1);

    let (name, result) = &results[0];
    assert!(result.is_ok(), "id function should type check: {:?}", result);
}
