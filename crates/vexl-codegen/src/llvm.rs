//! LLVM code generation backend

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::types::BasicType;
use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock as LLVMBasicBlock;
use std::collections::HashMap;

// Import function registry
use crate::{FunctionRegistry, CallingConvention};

use vexl_ir::{VirModule, VirFunction, BasicBlock, Instruction, InstructionKind, Terminator, ValueId, BlockId};
use vexl_ir::VirType;

/// CPU SIMD capabilities
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub has_avx: bool,
    pub has_sse4_2: bool,
    pub has_sse4_1: bool,
    pub has_ssse3: bool,
    pub has_sse3: bool,
    pub has_sse2: bool,
    pub has_sse: bool,
    pub has_mmx: bool,
    pub preferred_vector_width: usize,
}

impl SimdCapabilities {
    /// Detect CPU SIMD capabilities at runtime
    pub fn detect() -> Self {
        // Use std::arch detection when available
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64()
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::fallback()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        // Safe CPU feature detection
        let has_avx512 = std::arch::is_x86_feature_detected!("avx512f");
        let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
        let has_avx = std::arch::is_x86_feature_detected!("avx");
        let has_sse4_2 = std::arch::is_x86_feature_detected!("sse4.2");
        let has_sse4_1 = std::arch::is_x86_feature_detected!("sse4.1");
        let has_ssse3 = std::arch::is_x86_feature_detected!("ssse3");
        let has_sse3 = std::arch::is_x86_feature_detected!("sse3");
        let has_sse2 = std::arch::is_x86_feature_detected!("sse2");
        let has_sse = std::arch::is_x86_feature_detected!("sse");
        let has_mmx = std::arch::is_x86_feature_detected!("mmx");

        let preferred_vector_width = if has_avx512 { 64 } // 512 bits = 64 bytes
            else if has_avx2 { 32 } // 256 bits = 32 bytes
            else if has_avx { 32 }
            else if has_sse { 16 } // 128 bits = 16 bytes
            else { 8 }; // Minimum vector width

        Self {
            has_avx512,
            has_avx2,
            has_avx,
            has_sse4_2,
            has_sse4_1,
            has_ssse3,
            has_sse3,
            has_sse2,
            has_sse,
            has_mmx,
            preferred_vector_width,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64() -> Self {
        // ARM NEON is always available on AArch64
        Self {
            has_avx512: false,
            has_avx2: false,
            has_avx: false,
            has_sse4_2: false,
            has_sse4_1: false,
            has_ssse3: false,
            has_sse3: false,
            has_sse2: false,
            has_sse: true, // NEON is similar to SSE
            has_mmx: false,
            preferred_vector_width: 16, // 128-bit NEON
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn fallback() -> Self {
        Self {
            has_avx512: false,
            has_avx2: false,
            has_avx: false,
            has_sse4_2: false,
            has_sse4_1: false,
            has_ssse3: false,
            has_sse3: false,
            has_sse2: false,
            has_sse: false,
            has_mmx: false,
            preferred_vector_width: 8,
        }
    }
}

    /// LLVM code generation context with SIMD support
    pub struct LLVMCodegen<'ctx> {
        context: &'ctx Context,
        module: Module<'ctx>,
        builder: Builder<'ctx>,

        /// Map VIR value IDs to LLVM values
        values: HashMap<ValueId, BasicValueEnum<'ctx>>,

        /// Runtime functions (on-demand declaration)
        runtime_functions: RuntimeFunctions<'ctx>,

        /// Function registry for type-safe function management
        function_registry: FunctionRegistry,

        /// SIMD capabilities
        simd_caps: SimdCapabilities,

        /// Current function being compiled (for special handling)
        current_function_name: String,

        /// Map of function names to LLVM function values (for forward references)
        function_map: HashMap<String, FunctionValue<'ctx>>,
    }

/// Runtime function declarations - declare on demand
struct RuntimeFunctions<'ctx> {
    context: &'ctx Context,
    // Cache for declared functions
    functions: HashMap<String, FunctionValue<'ctx>>,
}

impl<'ctx> RuntimeFunctions<'ctx> {
    fn new(context: &'ctx Context) -> Self {
        Self {
            context,
            functions: HashMap::new(),
        }
    }

    /// Get or declare a runtime function by name
    fn get_or_declare(&mut self, name: &str, module: &Module<'ctx>) -> Result<FunctionValue<'ctx>, String> {
        if let Some(&func) = self.functions.get(name) {
            return Ok(func);
        }

        // Declare the function based on its name
        let func = match name {
            "vexl_vec_alloc_i64" => {
                let i64_type = self.context.i64_type();
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let vec_alloc_type = ptr_type.fn_type(&[i64_type.into()], false);
                module.add_function("vexl_vec_alloc_i64", vec_alloc_type, None)
            }
            "vexl_vec_get_i64" => {
                let i64_type = self.context.i64_type();
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let vec_get_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
                module.add_function("vexl_vec_get_i64", vec_get_type, None)
            }
            "vexl_vec_set_i64" => {
                let i64_type = self.context.i64_type();
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let void_type = self.context.void_type();
                let vec_set_type = void_type.fn_type(&[ptr_type.into(), i64_type.into(), i64_type.into()], false);
                module.add_function("vexl_vec_set_i64", vec_set_type, None)
            }
            "vexl_vec_len" => {
                let i64_type = self.context.i64_type();
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let vec_len_type = i64_type.fn_type(&[ptr_type.into()], false);
                module.add_function("vexl_vec_len", vec_len_type, None)
            }
            "vexl_vec_free" => {
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let void_type = self.context.void_type();
                let vec_free_type = void_type.fn_type(&[ptr_type.into()], false);
                module.add_function("vexl_vec_free", vec_free_type, None)
            }
            "vexl_vec_from_i64_array" => {
                let i64_type = self.context.i64_type();
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let vec_from_array_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
                module.add_function("vexl_vec_from_i64_array", vec_from_array_type, None)
            }
            "vexl_vec_add_i64" => {
                let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
                let vec_add_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
                module.add_function("vexl_vec_add_i64", vec_add_type, None)
            }
            "vexl_print_int" => {
                let i64_type = self.context.i64_type();
                let void_type = self.context.void_type();
                let print_int_type = void_type.fn_type(&[i64_type.into()], false);
                module.add_function("vexl_print_int", print_int_type, None)
            }
            // For simple expressions, we don't need most runtime functions
            // Add more as needed when implementing advanced features
            _ => return Err(format!("Runtime function '{}' not implemented", name)),
        };

        self.functions.insert(name.to_string(), func);
        Ok(func)
    }
}

impl<'ctx> LLVMCodegen<'ctx> {
    /// Create new LLVM codegen context with SIMD support
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        let runtime_functions = RuntimeFunctions::new(context);
        let function_registry = FunctionRegistry::default();
        let simd_caps = SimdCapabilities::detect();

        Self {
            context,
            module,
            builder,
            values: HashMap::new(),
            runtime_functions,
            function_registry,
            simd_caps,
            current_function_name: String::new(),
            function_map: HashMap::new(),
        }
    }

    // Removed duplicate function declaration
    
    /// Compile a VIR module to LLVM IR
    pub fn compile_module(mut self, vir_module: &VirModule) -> Result<Module<'ctx>, String> {
        // First pass: Create forward declarations for all functions
        for (name, func) in &vir_module.functions {
            // Special handling for main function - always return i32 for JIT compatibility
            let return_type = if name == "main" {
                self.context.i32_type().into()
            } else {
                self.vir_type_to_llvm_type(&func.signature.return_type)
            };

            let param_types: Vec<inkwell::types::BasicTypeEnum> = func.signature.param_types
                .iter()
                .map(|vir_type| self.vir_type_to_llvm_type(vir_type))
                .collect();

            let param_metadata_types: Vec<inkwell::types::BasicMetadataTypeEnum> = param_types
                .into_iter()
                .map(|t| t.into())
                .collect();
            let fn_type = return_type.fn_type(&param_metadata_types, false);

            // Store the function in our map for later reference
            let llvm_func = self.module.add_function(name, fn_type, None);
            self.function_map.insert(name.clone(), llvm_func);
        }

        // Second pass: Compile all function bodies
        for (name, func) in &vir_module.functions {
            self.compile_function_body(name, func)?;
        }

        // If no functions, create a simple main that returns 0
        if vir_module.functions.is_empty() {
            let i32_type = self.context.i32_type();
            let fn_type = i32_type.fn_type(&[], false);
            let function = self.module.add_function("main", fn_type, None);

            let entry = self.context.append_basic_block(function, "entry");
            self.builder.position_at_end(entry);

            let zero = i32_type.const_int(0, false);
            self.builder.build_return(Some(&zero)).unwrap();
        }

        Ok(self.module)
    }
    
    /// Compile a VIR function body (assumes function declaration already exists)
    fn compile_function_body(&mut self, name: &str, func: &VirFunction) -> Result<(), String> {
        // Set current function name for special handling
        self.current_function_name = name.to_string();

        // Get the already declared function
        let function = self.module.get_function(name)
            .ok_or_else(|| format!("Function '{}' not declared", name))?;

        // Map function parameters to VIR values
        for (i, &param_id) in func.params.iter().enumerate() {
            let param_value = function.get_nth_param(i as u32).unwrap();
            self.values.insert(param_id, param_value);
        }

        // Create all basic blocks first
        let mut block_map = HashMap::new();
        for (block_id, _) in &func.blocks {
            let bb_name = format!("block_{}", block_id.0);
            let bb = self.context.append_basic_block(function, &bb_name);
            block_map.insert(*block_id, bb);
        }

        // Compile each basic block
        for (block_id, block) in &func.blocks {
            let bb = block_map[block_id];
            self.builder.position_at_end(bb);
            self.compile_basic_block(block, &block_map)?;
        }

        Ok(())
    }

    /// Compile a VIR function to LLVM
    fn compile_function(&mut self, name: &str, func: &VirFunction) -> Result<FunctionValue<'ctx>, String> {
        // Set current function name for special handling
        self.current_function_name = name.to_string();

        // Create function signature from VIR function signature
        let return_type = self.vir_type_to_llvm_type(&func.signature.return_type);
        let param_types: Vec<inkwell::types::BasicTypeEnum> = func.signature.param_types
            .iter()
            .map(|vir_type| self.vir_type_to_llvm_type(vir_type))
            .collect();

        let param_metadata_types: Vec<inkwell::types::BasicMetadataTypeEnum> = param_types
            .into_iter()
            .map(|t| t.into())
            .collect();
        let fn_type = return_type.fn_type(&param_metadata_types, false);
        let function = self.module.add_function(name, fn_type, None);

        // Map function parameters to VIR values
        for (i, &param_id) in func.params.iter().enumerate() {
            let param_value = function.get_nth_param(i as u32).unwrap();
            self.values.insert(param_id, param_value);
        }

        // Create all basic blocks first
        let mut block_map = HashMap::new();
        for (block_id, _) in &func.blocks {
            let bb_name = format!("block_{}", block_id.0);
            let bb = self.context.append_basic_block(function, &bb_name);
            block_map.insert(*block_id, bb);
        }

        // Compile each basic block
        for (block_id, block) in &func.blocks {
            let bb = block_map[block_id];
            self.builder.position_at_end(bb);
            self.compile_basic_block(block, &block_map)?;
        }

        Ok(function)
    }
    
    /// Compile a basic block
    fn compile_basic_block(&mut self, block: &BasicBlock, block_map: &HashMap<BlockId, LLVMBasicBlock<'ctx>>) -> Result<(), String> {
        // Compile all instructions
        for inst in &block.instructions {
            let value = self.compile_instruction(inst)?;
            self.values.insert(inst.result, value);
        }

        // Compile terminator
        self.compile_terminator(&block.terminator, block_map)?;

        Ok(())
    }
    
    /// Compile a single instruction
    fn compile_instruction(&mut self, inst: &Instruction) -> Result<BasicValueEnum<'ctx>, String> {
        match &inst.kind {
            InstructionKind::ConstInt(n) => {
                let i64_type = self.context.i64_type();
                let value = i64_type.const_int(*n as u64, true);
                Ok(value.into())
            }

            InstructionKind::ConstFloat(f) => {
                let f64_type = self.context.f64_type();
                let value = f64_type.const_float(*f);
                Ok(value.into())
            }

            InstructionKind::ConstString(s) => {
                let string_val = self.builder.build_global_string_ptr(s, "str").unwrap();
                Ok(string_val.as_pointer_value().into())
            }

            // Vector operations
            InstructionKind::VectorNew { elements, dimension } => {
                if *dimension == 1 {
                    // Create array for elements
                    let i64_type = self.context.i64_type();
                    let _ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());

                    // Allocate array
                    let count = elements.len() as u64;
                    let alloc_size = i64_type.const_int(count * 8, false); // 8 bytes per i64
                    let alloc_func = self.get_runtime_function("vexl_vec_alloc_i64")?;
                    let data_ptr = self.builder.build_call(
                        alloc_func,
                        &[alloc_size.into()],
                        "vec_data"
                    ).unwrap().try_as_basic_value().left().unwrap();

                    // Store elements
                    for (i, &elem_id) in elements.iter().enumerate() {
                        let elem_val = self.get_value(elem_id)?;
                        let idx_val = i64_type.const_int(i as u64, false);
                        let set_func = self.get_runtime_function("vexl_vec_set_i64")?;
                        self.builder.build_call(
                            set_func,
                            &[data_ptr.into(), idx_val.into(), elem_val.into()],
                            ""
                        ).unwrap();
                    }

                    // Create vector from array
                    let from_array_func = self.get_runtime_function("vexl_vec_from_i64_array")?;
                    let vec_ptr = self.builder.build_call(
                        from_array_func,
                        &[data_ptr.into(), i64_type.const_int(count, false).into()],
                        "vec"
                    ).unwrap().try_as_basic_value().left().unwrap();

                    Ok(vec_ptr)
                } else {
                    // Multi-dimensional vectors not yet implemented
                    Err(format!("Multi-dimensional vectors not yet implemented"))
                }
            }

            InstructionKind::VectorGet { vector, index } => {
                let vec_val = self.get_value(*vector)?;
                let idx_val = self.get_value(*index)?;
                let func = self.get_runtime_function("vexl_vec_get_i64")?;
                let result = self.builder.build_call(
                    func,
                    &[vec_val.into(), idx_val.into()],
                    "vec_get"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::VectorSet { vector, index, value } => {
                let vec_val = self.get_value(*vector)?;
                let idx_val = self.get_value(*index)?;
                let val_val = self.get_value(*value)?;
                let func = self.get_runtime_function("vexl_vec_set_i64")?;
                self.builder.build_call(
                    func,
                    &[vec_val.into(), idx_val.into(), val_val.into()],
                    ""
                ).unwrap();
                // Return void (0)
                let i64_type = self.context.i64_type();
                Ok(i64_type.const_int(0, false).into())
            }

            // Arithmetic operations with SIMD support
            InstructionKind::Add(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;

                if left_val.is_pointer_value() && right_val.is_pointer_value() {
                    // Vector + Vector - use SIMD if possible
                    if self.simd_caps.has_avx2 || self.simd_caps.has_sse {
                        self.generate_simd_vector_add(left_val, right_val)
                    } else {
                        let func = self.get_runtime_function("vexl_vec_add_i64")?;
                        let args = &[left_val.into(), right_val.into()];
                        let call = self.builder.build_call(func, args, "vec_add").unwrap();
                        let result = call.try_as_basic_value().left().unwrap();
                        Ok(result)
                    }
                } else if left_val.is_int_value() && right_val.is_int_value() {
                    // Int + Int
                    let left_int = left_val.into_int_value();
                    let right_int = right_val.into_int_value();
                    let result = self.builder.build_int_add(left_int, right_int, "add").unwrap();
                    Ok(result.into())
                } else if left_val.is_float_value() && right_val.is_float_value() {
                    // Float + Float
                    let left_float = left_val.into_float_value();
                    let right_float = right_val.into_float_value();
                    let result = self.builder.build_float_add(left_float, right_float, "fadd").unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Add: {:?} + {:?}", left_val, right_val))
                }
            }

            InstructionKind::Sub(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let left_int = left_val.into_int_value();
                    let right_int = right_val.into_int_value();
                    let result = self.builder.build_int_sub(left_int, right_int, "sub").unwrap();
                    Ok(result.into())
                } else if left_val.is_float_value() && right_val.is_float_value() {
                    let left_float = left_val.into_float_value();
                    let right_float = right_val.into_float_value();
                    let result = self.builder.build_float_sub(left_float, right_float, "fsub").unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Sub"))
                }
            }

            InstructionKind::Mul(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let left_int = left_val.into_int_value();
                    let right_int = right_val.into_int_value();
                    let result = self.builder.build_int_mul(left_int, right_int, "mul").unwrap();
                    Ok(result.into())
                } else if left_val.is_float_value() && right_val.is_float_value() {
                    let left_float = left_val.into_float_value();
                    let right_float = right_val.into_float_value();
                    let result = self.builder.build_float_mul(left_float, right_float, "fmul").unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Mul"))
                }
            }

            InstructionKind::Div(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let left_int = left_val.into_int_value();
                    let right_int = right_val.into_int_value();
                    let result = self.builder.build_int_signed_div(left_int, right_int, "div").unwrap();
                    Ok(result.into())
                } else if left_val.is_float_value() && right_val.is_float_value() {
                    let left_float = left_val.into_float_value();
                    let right_float = right_val.into_float_value();
                    let result = self.builder.build_float_div(left_float, right_float, "fdiv").unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Div"))
                }
            }

            // Matrix operations
            InstructionKind::MatMul(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let func = self.get_runtime_function("vexl_mat_mul")?;
                let result = self.builder.build_call(
                    func,
                    &[left_val.into(), right_val.into()],
                    "mat_mul"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::Outer(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let func = self.get_runtime_function("vexl_mat_outer")?;
                let result = self.builder.build_call(
                    func,
                    &[left_val.into(), right_val.into()],
                    "mat_outer"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::Dot(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let func = self.get_runtime_function("vexl_mat_dot")?;
                let result = self.builder.build_call(
                    func,
                    &[left_val.into(), right_val.into()],
                    "mat_dot"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            // Comparisons
            InstructionKind::Eq(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let result = self.builder.build_int_compare(
                        inkwell::IntPredicate::EQ,
                        left_val.into_int_value(),
                        right_val.into_int_value(),
                        "eq"
                    ).unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Eq"))
                }
            }

            InstructionKind::NotEq(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let result = self.builder.build_int_compare(
                        inkwell::IntPredicate::NE,
                        left_val.into_int_value(),
                        right_val.into_int_value(),
                        "neq"
                    ).unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for NotEq"))
                }
            }

            InstructionKind::Lt(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let result = self.builder.build_int_compare(
                        inkwell::IntPredicate::SLT,
                        left_val.into_int_value(),
                        right_val.into_int_value(),
                        "lt"
                    ).unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Lt"))
                }
            }

            InstructionKind::Le(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let result = self.builder.build_int_compare(
                        inkwell::IntPredicate::SLE,
                        left_val.into_int_value(),
                        right_val.into_int_value(),
                        "le"
                    ).unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Le"))
                }
            }

            InstructionKind::Gt(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let result = self.builder.build_int_compare(
                        inkwell::IntPredicate::SGT,
                        left_val.into_int_value(),
                        right_val.into_int_value(),
                        "gt"
                    ).unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Gt"))
                }
            }

            InstructionKind::Ge(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                if left_val.is_int_value() && right_val.is_int_value() {
                    let result = self.builder.build_int_compare(
                        inkwell::IntPredicate::SGE,
                        left_val.into_int_value(),
                        right_val.into_int_value(),
                        "ge"
                    ).unwrap();
                    Ok(result.into())
                } else {
                    Err(format!("Unsupported operand types for Ge"))
                }
            }

            // Function calls
            InstructionKind::Call { func, args } => {
                // For now, we assume func is a string constant representing the function name
                // This is a simplification - proper function value handling would be more complex

                // Since we don't have proper function value tracking, we need to extract
                // the function name from the VIR. For now, we'll use a placeholder approach
                // where function names are stored as global string constants.

                // In the current VIR lowering, function names are not properly tracked as values.
                // We need to modify the lowering to store function names properly.

                // For this implementation, we'll assume the function name can be derived
                // from the VIR instruction that produced the func value, but since we don't
                // have that information here, we'll use a temporary approach:

                // Check if we can find a function with a matching name in the module
                // This is a hack - in a real implementation, function values would be tracked properly

                let arg_vals: Vec<inkwell::values::BasicMetadataValueEnum> = args.iter()
                    .map(|&arg| self.get_value(arg).map(|v| v.into()))
                    .collect::<Result<Vec<_>, _>>()?;

                // Try to find a function by name
                let called_function = self.module.get_function(&func);

                if let Some(function) = called_function {
                    let call = self.builder.build_call(function, &arg_vals, "func_call").unwrap();
                    if let Some(return_val) = call.try_as_basic_value().left() {
                        Ok(return_val)
                    } else {
                        // Void return - return 0
                        let i64_type = self.context.i64_type();
                        Ok(i64_type.const_int(0, false).into())
                    }
                } else {
                    // Function not found - this indicates a bug in our function reference tracking
                    Err(format!("Function '{}' not found in module", func))
                }
            }

            // Runtime function calls
            InstructionKind::RuntimeCall { function_name, args } => {
                let func = self.get_runtime_function(function_name)?;
                let arg_vals: Vec<inkwell::values::BasicMetadataValueEnum> = args.iter()
                    .map(|&arg| self.get_value(arg).map(|v| v.into()))
                    .collect::<Result<Vec<_>, _>>()?;

                let call = self.builder.build_call(func, &arg_vals, "runtime_call").unwrap();

                // Handle different return types based on function
                match function_name.as_str() {
                    // Functions that return i64
                    "vexl_vec_sum" | "vexl_vec_product" | "vexl_vec_max" | "vexl_vec_min" |
                    "vexl_vec_len" | "vexl_vec_get_i64" => {
                        Ok(call.try_as_basic_value().left().unwrap())
                    }

                    // Functions that return pointers
                    "vexl_vec_alloc_i64" | "vexl_vec_from_i64_array" |
                    "vexl_vec_add_i64" | "vexl_vec_mul_scalar_i64" |
                    "vexl_vec_map_parallel" | "vexl_vec_filter" |
                    "vexl_vec_reduce_parallel" | "vexl_vec_map_sequential" |
                    "vexl_vec_reduce_sequential" => {
                        Ok(call.try_as_basic_value().left().unwrap())
                    }

                    // Math functions that return f64
                    "vexl_math_sin" | "vexl_math_cos" | "vexl_math_sqrt" => {
                        Ok(call.try_as_basic_value().left().unwrap())
                    }

                    // Functions that return void (return 0)
                    "vexl_print_int" | "vexl_print_string" | "vexl_print_float" |
                    "vexl_vec_set_i64" | "vexl_vec_free" => {
                        let i64_type = self.context.i64_type();
                        Ok(i64_type.const_int(0, false).into())
                    }

                    _ => Err(format!("Unknown return type for runtime function: {}", function_name)),
                }
            }

            // Generator operations
            InstructionKind::GeneratorNew { func, bounds: _ } => {
                let func_val = self.get_value(*func)?;
                let func = self.get_runtime_function("vexl_gen_new")?;
                let result = self.builder.build_call(
                    func,
                    &[func_val.into()],
                    "gen_new"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::GeneratorEval { generator, index } => {
                let gen_val = self.get_value(*generator)?;
                let idx_val = self.get_value(*index)?;
                let func = self.get_runtime_function("vexl_gen_eval")?;
                let result = self.builder.build_call(
                    func,
                    &[gen_val.into(), idx_val.into()],
                    "gen_eval"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            // Range operations
            InstructionKind::Range { start, end } => {
                let start_val = self.get_value(*start)?;
                let end_val = self.get_value(*end)?;
                let func = self.get_runtime_function("vexl_range_new")?;

                let result = self.builder.build_call(
                    func,
                    &[start_val.into(), end_val.into()],
                    "range"
                ).unwrap().try_as_basic_value().left().unwrap();

                Ok(result)
            }

            InstructionKind::InfiniteRange { start } => {
                let start_val = self.get_value(*start)?;
                let func = self.get_runtime_function("vexl_range_infinite")?;
                let result = self.builder.build_call(
                    func,
                    &[start_val.into()],
                    "infinite_range"
                ).unwrap().try_as_basic_value().left().unwrap();

                Ok(result)
            }

            // Phi nodes (SSA)
            InstructionKind::Phi(_) => {
                let i64_type = self.context.i64_type();
                let phi = self.builder.build_phi(i64_type, "phi").unwrap();

                // For now, just return the phi node
                // TODO: Properly set up incoming edges
                Ok(phi.as_basic_value())
            }
        }
    }
    
    /// Compile a terminator instruction
    fn compile_terminator(&mut self, term: &Terminator, block_map: &HashMap<BlockId, LLVMBasicBlock<'ctx>>) -> Result<(), String> {
        match term {
            Terminator::Return(value_id) => {
                let value = self.get_value(*value_id)?;
                // For main function, truncate i64 to i32 for JIT compatibility
                let return_value = if self.current_function_name == "main" && value.is_int_value() {
                    let int_val = value.into_int_value();
                    let i32_type = self.context.i32_type();
                    self.builder.build_int_truncate(int_val, i32_type, "trunc_result").unwrap().into()
                } else {
                    value
                };
                self.builder.build_return(Some(&return_value)).unwrap();
                Ok(())
            }

            Terminator::Branch { cond, then_block, else_block } => {
                let cond_value = self.get_value(*cond)?;
                let then_bb = block_map.get(then_block)
                    .ok_or_else(|| format!("Unknown block {:?}", then_block))?;
                let else_bb = block_map.get(else_block)
                    .ok_or_else(|| format!("Unknown block {:?}", else_block))?;

                self.builder.build_conditional_branch(
                    cond_value.into_int_value(),
                    *then_bb,
                    *else_bb
                ).unwrap();
                Ok(())
            }

            Terminator::Jump(target_block) => {
                let target_bb = block_map.get(target_block)
                    .ok_or_else(|| format!("Unknown block {:?}", target_block))?;
                self.builder.build_unconditional_branch(*target_bb).unwrap();
                Ok(())
            }

            Terminator::Unreachable => {
                self.builder.build_unreachable().unwrap();
                Ok(())
            }
        }
    }
    
    /// Get a compiled LLVM value by VIR ID
    fn get_value(&self, id: ValueId) -> Result<BasicValueEnum<'ctx>, String> {
        self.values.get(&id)
            .copied()
            .ok_or_else(|| format!("Value {:?} not found", id))
    }

    /// Generate SIMD vector addition
    fn generate_simd_vector_add(&mut self, left_vec: BasicValueEnum<'ctx>, right_vec: BasicValueEnum<'ctx>) -> Result<BasicValueEnum<'ctx>, String> {
        // For now, delegate to runtime function with SIMD hint
        // In a full implementation, this would generate LLVM SIMD intrinsics
        let func = self.get_runtime_function("vexl_vec_add_i64")?;
        let args = &[left_vec.into(), right_vec.into()];
        let call = self.builder.build_call(func, args, "simd_vec_add").unwrap();
        let result = call.try_as_basic_value().left().unwrap();
        Ok(result)
    }

    /// Generate SIMD vector operation using LLVM intrinsics
    fn generate_simd_intrinsic(&mut self, operation: &str, left: BasicValueEnum<'ctx>, right: BasicValueEnum<'ctx>) -> Result<BasicValueEnum<'ctx>, String> {
        // This would generate actual SIMD intrinsics like @llvm.x86.sse2.padd.d
        // For now, fall back to regular operations
        match operation {
            "add" => {
                if left.is_int_value() && right.is_int_value() {
                    let result = self.builder.build_int_add(
                        left.into_int_value(),
                        right.into_int_value(),
                        "simd_add"
                    ).unwrap();
                    Ok(result.into())
                } else if left.is_float_value() && right.is_float_value() {
                    let result = self.builder.build_float_add(
                        left.into_float_value(),
                        right.into_float_value(),
                        "simd_fadd"
                    ).unwrap();
                    Ok(result.into())
                } else {
                    Err("Unsupported types for SIMD add".to_string())
                }
            }
            _ => Err(format!("Unsupported SIMD operation: {}", operation)),
        }
    }

    /// Get runtime function by name (on-demand declaration)
    fn get_runtime_function(&mut self, name: &str) -> Result<FunctionValue<'ctx>, String> {
        self.runtime_functions.get_or_declare(name, &self.module)
    }

    /// Convert VIR type to LLVM type
    fn vir_type_to_llvm_type(&self, vir_type: &VirType) -> inkwell::types::BasicTypeEnum<'ctx> {
        match vir_type {
            VirType::Int32 => self.context.i32_type().into(),
            VirType::Int64 => self.context.i64_type().into(),
            VirType::Float64 => self.context.f64_type().into(),
            VirType::Pointer => self.context.i8_type().ptr_type(AddressSpace::default()).into(),
            VirType::Vector { .. } => self.context.i8_type().ptr_type(AddressSpace::default()).into(), // Vectors are pointers
            VirType::Void => panic!("Cannot convert Void to LLVM type"), // Void is only for return types
        }
    }

    /// Extract function name from a VIR value ID (for function calls)
    /// This is a simplified implementation that assumes function names are stored as string constants
    fn extract_function_name(&self, func_value_id: ValueId) -> Option<String> {
        // In a more complete implementation, this would look up the instruction that produced
        // this value and extract the function name. For now, we assume it's a global function name.

        // Since the VIR lowering currently doesn't properly handle function references,
        // we'll use a simple heuristic: check if this value ID corresponds to a known function
        // in the module. This is a temporary solution until proper function value handling is implemented.

        // For the current implementation, we'll return None and let the caller handle it
        // In a real implementation, we'd track which value IDs correspond to which functions
        None
    }
}

/// Compile a VIR module to LLVM IR and return as string
pub fn codegen_to_string(vir_module: &VirModule) -> Result<String, String> {
    let context = Context::create();
    let codegen = LLVMCodegen::new(&context, "vexl_module");
    let module = codegen.compile_module(vir_module)?;
    
    Ok(module.print_to_string().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_constant() {
        let mut module = VirModule::new();
        let v1 = module.fresh_value();
        let block_id = module.fresh_block();

        let block = BasicBlock {
            id: block_id,
            instructions: vec![
                Instruction { result_type: None,
                    result: v1,
                    kind: InstructionKind::ConstInt(42),
                },
            ],
            terminator: Terminator::Return(v1),
        };

        let mut func = VirFunction {
            name: "main".to_string(), // Use "main" to trigger i32 return type
            params: vec![],
            blocks: HashMap::new(),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
            signature: vexl_ir::FunctionSignature::new(vec![], vexl_ir::VirType::Int64),
        };
        func.blocks.insert(block_id, block);

        module.add_function("main".to_string(), func);

        let llvm_ir = codegen_to_string(&module).unwrap();
        assert!(llvm_ir.contains("ret i32 42"));
    }
    
    #[test]
    fn test_codegen_add() {
        let mut module = VirModule::new();
        let v1 = module.fresh_value();
        let v2 = module.fresh_value();
        let v3 = module.fresh_value();
        let block_id = module.fresh_block();

        let block = BasicBlock {
            id: block_id,
            instructions: vec![
                Instruction { result_type: None,
                    result: v1,
                    kind: InstructionKind::ConstInt(1),
                },
                Instruction { result_type: None,
                    result: v2,
                    kind: InstructionKind::ConstInt(2),
                },
                Instruction { result_type: None,
                    result: v3,
                    kind: InstructionKind::Add(v1, v2),
                },
            ],
            terminator: Terminator::Return(v3),
        };

        let mut func = VirFunction {
            name: "main".to_string(),
            params: vec![],
            blocks: HashMap::new(),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
            signature: vexl_ir::FunctionSignature::new(vec![], vexl_ir::VirType::Int64),
        };
        func.blocks.insert(block_id, block);

        module.add_function("main".to_string(), func);

        let llvm_ir = codegen_to_string(&module).unwrap();
        // Just verify it compiles and returns valid LLVM IR
        assert!(!llvm_ir.is_empty());
        assert!(llvm_ir.contains("define"));
    }
}
