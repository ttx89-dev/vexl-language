//! LLVM code generation backend

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock as LLVMBasicBlock;
use std::collections::HashMap;

use vexl_ir::{VirModule, VirFunction, BasicBlock, Instruction, InstructionKind, Terminator, ValueId, BlockId};

/// LLVM code generation context
pub struct LLVMCodegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    /// Map VIR value IDs to LLVM values
    values: HashMap<ValueId, BasicValueEnum<'ctx>>,

    /// Runtime functions
    runtime_functions: RuntimeFunctions<'ctx>,
}

/// Runtime function declarations
struct RuntimeFunctions<'ctx> {
    vexl_vec_alloc_i64: FunctionValue<'ctx>,
    vexl_vec_get_i64: FunctionValue<'ctx>,
    vexl_vec_set_i64: FunctionValue<'ctx>,
    vexl_vec_len: FunctionValue<'ctx>,
    vexl_vec_free: FunctionValue<'ctx>,
    vexl_vec_from_i64_array: FunctionValue<'ctx>,
    vexl_vec_add_i64: FunctionValue<'ctx>,
    vexl_vec_mul_scalar_i64: FunctionValue<'ctx>,
    vexl_print_int: FunctionValue<'ctx>,
    vexl_print_string: FunctionValue<'ctx>,
    vexl_vec_map_parallel: FunctionValue<'ctx>,
    vexl_vec_filter: FunctionValue<'ctx>,
    vexl_vec_reduce_parallel: FunctionValue<'ctx>,
    vexl_vec_map_sequential: FunctionValue<'ctx>,
    vexl_vec_reduce_sequential: FunctionValue<'ctx>,
    // Matrix operations
    vexl_mat_mul: FunctionValue<'ctx>,
    vexl_mat_outer: FunctionValue<'ctx>,
    vexl_mat_dot: FunctionValue<'ctx>,
    // Generator operations
    vexl_gen_new: FunctionValue<'ctx>,
    vexl_gen_eval: FunctionValue<'ctx>,
    vexl_range_new: FunctionValue<'ctx>,
    vexl_range_infinite: FunctionValue<'ctx>,
    // Standard library functions
    vexl_vec_sum: FunctionValue<'ctx>,
    vexl_vec_product: FunctionValue<'ctx>,
    vexl_vec_max: FunctionValue<'ctx>,
    vexl_vec_min: FunctionValue<'ctx>,
    vexl_math_sin: FunctionValue<'ctx>,
    vexl_math_cos: FunctionValue<'ctx>,
    vexl_math_sqrt: FunctionValue<'ctx>,
    vexl_math_pow: FunctionValue<'ctx>,
    vexl_print_float: FunctionValue<'ctx>,
    // I/O functions
    vexl_read_line: FunctionValue<'ctx>,
    vexl_read_file: FunctionValue<'ctx>,
    vexl_write_file: FunctionValue<'ctx>,
    vexl_file_exists: FunctionValue<'ctx>,
    vexl_file_size: FunctionValue<'ctx>,
    // System functions
    vexl_current_time: FunctionValue<'ctx>,
    vexl_current_time_ns: FunctionValue<'ctx>,
    vexl_sleep_ms: FunctionValue<'ctx>,
    vexl_getenv: FunctionValue<'ctx>,
    vexl_setenv: FunctionValue<'ctx>,
    vexl_get_args: FunctionValue<'ctx>,
    vexl_exit: FunctionValue<'ctx>,
    vexl_getpid: FunctionValue<'ctx>,
    vexl_random: FunctionValue<'ctx>,
    // Memory functions
    vexl_alloc: FunctionValue<'ctx>,
    vexl_free: FunctionValue<'ctx>,
    // String functions
    vexl_string_compare: FunctionValue<'ctx>,
    vexl_string_substring: FunctionValue<'ctx>,
}

impl<'ctx> LLVMCodegen<'ctx> {
    /// Create new LLVM codegen context
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        let runtime_functions = Self::declare_runtime_functions(context, &module);

        Self {
            context,
            module,
            builder,
            values: HashMap::new(),
            runtime_functions,
        }
    }

    /// Declare runtime functions
    fn declare_runtime_functions(context: &'ctx Context, module: &Module<'ctx>) -> RuntimeFunctions<'ctx> {
        let i64_type = context.i64_type();
        let ptr_type = context.i8_type().ptr_type(AddressSpace::default());
        let void_type = context.void_type();

        // vexl_vec_alloc_i64(count: u64) -> *mut Vector
        let vec_alloc_type = ptr_type.fn_type(&[i64_type.into()], false);
        let vexl_vec_alloc_i64 = module.add_function("vexl_vec_alloc_i64", vec_alloc_type, None);

        // vexl_vec_get_i64(vec: *mut Vector, index: u64) -> i64
        let vec_get_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_vec_get_i64 = module.add_function("vexl_vec_get_i64", vec_get_type, None);

        // vexl_vec_set_i64(vec: *mut Vector, index: u64, value: i64)
        let vec_set_type = void_type.fn_type(&[ptr_type.into(), i64_type.into(), i64_type.into()], false);
        let vexl_vec_set_i64 = module.add_function("vexl_vec_set_i64", vec_set_type, None);

        // vexl_vec_len(vec: *mut Vector) -> u64
        let vec_len_type = i64_type.fn_type(&[ptr_type.into()], false);
        let vexl_vec_len = module.add_function("vexl_vec_len", vec_len_type, None);

        // vexl_vec_free(vec: *mut Vector)
        let vec_free_type = void_type.fn_type(&[ptr_type.into()], false);
        let vexl_vec_free = module.add_function("vexl_vec_free", vec_free_type, None);

        // vexl_vec_from_i64_array(data: *const i64, count: u64) -> *mut Vector
        let vec_from_array_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_vec_from_i64_array = module.add_function("vexl_vec_from_i64_array", vec_from_array_type, None);

        // vexl_vec_add_i64(a: *mut Vector, b: *mut Vector) -> *mut Vector
        let vec_add_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let vexl_vec_add_i64 = module.add_function("vexl_vec_add_i64", vec_add_type, None);

        // vexl_vec_mul_scalar_i64(vec: *mut Vector, scalar: i64) -> *mut Vector
        let vec_mul_scalar_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_vec_mul_scalar_i64 = module.add_function("vexl_vec_mul_scalar_i64", vec_mul_scalar_type, None);

        // vexl_print_int(n: i64)
        let print_int_type = void_type.fn_type(&[i64_type.into()], false);
        let vexl_print_int = module.add_function("vexl_print_int", print_int_type, None);

        // vexl_print_string(ptr: *const u8, len: u64)
        let print_string_type = void_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_print_string = module.add_function("vexl_print_string", print_string_type, None);

        // vexl_vec_map_parallel(vec: *mut Vector, fn: *const (), threads: u64) -> *mut Vector
        let vec_map_parallel_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into(), i64_type.into()], false);
        let vexl_vec_map_parallel = module.add_function("vexl_vec_map_parallel", vec_map_parallel_type, None);

        // vexl_vec_filter(vec: *mut Vector, pred: *const ()) -> *mut Vector
        let vec_filter_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let vexl_vec_filter = module.add_function("vexl_vec_filter", vec_filter_type, None);

        // vexl_vec_reduce_parallel(vec: *mut Vector, init: *const u8, fn: *const (), threads: u64) -> *mut u8
        let vec_reduce_parallel_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into(), i64_type.into()], false);
        let vexl_vec_reduce_parallel = module.add_function("vexl_vec_reduce_parallel", vec_reduce_parallel_type, None);

        // Sequential versions
        let vexl_vec_map_sequential = module.add_function("vexl_vec_map_sequential", vec_map_parallel_type, None);
        let vec_reduce_sequential_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let vexl_vec_reduce_sequential = module.add_function("vexl_vec_reduce_sequential", vec_reduce_sequential_type, None);

        // Matrix operations
        let mat_mul_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let vexl_mat_mul = module.add_function("vexl_mat_mul", mat_mul_type, None);

        let mat_outer_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let vexl_mat_outer = module.add_function("vexl_mat_outer", mat_outer_type, None);

        let mat_dot_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let vexl_mat_dot = module.add_function("vexl_mat_dot", mat_dot_type, None);

        let mat_transpose_type = ptr_type.fn_type(&[ptr_type.into()], false);
        let _vexl_mat_transpose = module.add_function("vexl_mat_transpose", mat_transpose_type, None);

        // Generator operations
        let gen_new_type = ptr_type.fn_type(&[ptr_type.into()], false);
        let vexl_gen_new = module.add_function("vexl_gen_new", gen_new_type, None);

        let gen_eval_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_gen_eval = module.add_function("vexl_gen_eval", gen_eval_type, None);

        let gen_take_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let _vexl_gen_take = module.add_function("vexl_gen_take", gen_take_type, None);

        // Range operations
        let range_new_type = ptr_type.fn_type(&[i64_type.into(), i64_type.into()], false);
        let vexl_range_new = module.add_function("vexl_range_new", range_new_type, None);

        let range_infinite_type = ptr_type.fn_type(&[i64_type.into()], false);
        let vexl_range_infinite = module.add_function("vexl_range_infinite", range_infinite_type, None);

        // Standard library functions
        let vec_sum_type = i64_type.fn_type(&[ptr_type.into()], false);
        let vexl_vec_sum = module.add_function("vexl_vec_sum", vec_sum_type, None);

        let vec_product_type = i64_type.fn_type(&[ptr_type.into()], false);
        let vexl_vec_product = module.add_function("vexl_vec_product", vec_product_type, None);

        let vec_max_type = i64_type.fn_type(&[ptr_type.into()], false);
        let vexl_vec_max = module.add_function("vexl_vec_max", vec_max_type, None);

        let vec_min_type = i64_type.fn_type(&[ptr_type.into()], false);
        let vexl_vec_min = module.add_function("vexl_vec_min", vec_min_type, None);

        let f64_type = context.f64_type();
        let math_sin_type = f64_type.fn_type(&[f64_type.into()], false);
        let vexl_math_sin = module.add_function("vexl_math_sin", math_sin_type, None);

        let math_cos_type = f64_type.fn_type(&[f64_type.into()], false);
        let vexl_math_cos = module.add_function("vexl_math_cos", math_cos_type, None);

        let math_sqrt_type = f64_type.fn_type(&[f64_type.into()], false);
        let vexl_math_sqrt = module.add_function("vexl_math_sqrt", math_sqrt_type, None);

        let math_pow_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
        let vexl_math_pow = module.add_function("vexl_math_pow", math_pow_type, None);

        let print_float_type = void_type.fn_type(&[f64_type.into()], false);
        let vexl_print_float = module.add_function("vexl_print_float", print_float_type, None);

        // I/O functions
        let read_line_type = ptr_type.fn_type(&[], false);
        let vexl_read_line = module.add_function("vexl_read_line", read_line_type, None);

        let read_file_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into()], false);
        let vexl_read_file = module.add_function("vexl_read_file", read_file_type, None);

        let write_file_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into(), i64_type.into()], false);
        let vexl_write_file = module.add_function("vexl_write_file", write_file_type, None);

        let file_exists_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_file_exists = module.add_function("vexl_file_exists", file_exists_type, None);

        let file_size_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_file_size = module.add_function("vexl_file_size", file_size_type, None);

        // System functions
        let current_time_type = i64_type.fn_type(&[], false);
        let vexl_current_time = module.add_function("vexl_current_time", current_time_type, None);

        let current_time_ns_type = i64_type.fn_type(&[], false);
        let vexl_current_time_ns = module.add_function("vexl_current_time_ns", current_time_ns_type, None);

        let sleep_ms_type = void_type.fn_type(&[i64_type.into()], false);
        let vexl_sleep_ms = module.add_function("vexl_sleep_ms", sleep_ms_type, None);

        let getenv_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let vexl_getenv = module.add_function("vexl_getenv", getenv_type, None);

        let setenv_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into(), i64_type.into()], false);
        let vexl_setenv = module.add_function("vexl_setenv", setenv_type, None);

        let get_args_type = ptr_type.fn_type(&[], false);
        let vexl_get_args = module.add_function("vexl_get_args", get_args_type, None);

        let exit_type = void_type.fn_type(&[context.i32_type().into()], true); // noreturn
        let vexl_exit = module.add_function("vexl_exit", exit_type, None);

        let getpid_type = context.i32_type().fn_type(&[], false);
        let vexl_getpid = module.add_function("vexl_getpid", getpid_type, None);

        let random_type = f64_type.fn_type(&[], false);
        let vexl_random = module.add_function("vexl_random", random_type, None);

        // Memory functions
        let alloc_type = ptr_type.fn_type(&[i64_type.into()], false);
        let vexl_alloc = module.add_function("vexl_alloc", alloc_type, None);

        let free_type = void_type.fn_type(&[ptr_type.into()], false);
        let vexl_free = module.add_function("vexl_free", free_type, None);

        // String functions
        let string_compare_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into(), i64_type.into()], false);
        let vexl_string_compare = module.add_function("vexl_string_compare", string_compare_type, None);

        let string_substring_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into(), i64_type.into(), i64_type.into()], false);
        let vexl_string_substring = module.add_function("vexl_string_substring", string_substring_type, None);

        RuntimeFunctions {
            vexl_vec_alloc_i64,
            vexl_vec_get_i64,
            vexl_vec_set_i64,
            vexl_vec_len,
            vexl_vec_free,
            vexl_vec_from_i64_array,
            vexl_vec_add_i64,
            vexl_vec_mul_scalar_i64,
            vexl_print_int,
            vexl_print_string,
            vexl_vec_map_parallel,
            vexl_vec_filter,
            vexl_vec_reduce_parallel,
            vexl_vec_map_sequential,
            vexl_vec_reduce_sequential,
            vexl_mat_mul,
            vexl_mat_outer,
            vexl_mat_dot,
            vexl_gen_new,
            vexl_gen_eval,
            vexl_range_new,
            vexl_range_infinite,
            vexl_vec_sum,
            vexl_vec_product,
            vexl_vec_max,
            vexl_vec_min,
            vexl_math_sin,
            vexl_math_cos,
            vexl_math_sqrt,
            vexl_math_pow,
            vexl_print_float,
            vexl_read_line,
            vexl_read_file,
            vexl_write_file,
            vexl_file_exists,
            vexl_file_size,
            vexl_current_time,
            vexl_current_time_ns,
            vexl_sleep_ms,
            vexl_getenv,
            vexl_setenv,
            vexl_get_args,
            vexl_exit,
            vexl_getpid,
            vexl_random,
            vexl_alloc,
            vexl_free,
            vexl_string_compare,
            vexl_string_substring,
        }
    }
    
    /// Compile a VIR module to LLVM IR
    pub fn compile_module(mut self, vir_module: &VirModule) -> Result<Module<'ctx>, String> {
        // Compile all functions in the module
        for (name, func) in &vir_module.functions {
            self.compile_function(name, func)?;
        }
        
        // If no functions, create a simple main that returns 0
        if vir_module.functions.is_empty() {
            let i64_type = self.context.i64_type();
            let fn_type = i64_type.fn_type(&[], false);
            let function = self.module.add_function("main", fn_type, None);
            
            let entry = self.context.append_basic_block(function, "entry");
            self.builder.position_at_end(entry);
            
            let zero = i64_type.const_int(0, false);
            self.builder.build_return(Some(&zero)).unwrap();
        }
        
        Ok(self.module)
    }
    
    /// Compile a VIR function to LLVM
    fn compile_function(&mut self, name: &str, func: &VirFunction) -> Result<FunctionValue<'ctx>, String> {
        // Create function signature
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[], false);
        let function = self.module.add_function(name, fn_type, None);

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
        for (i, inst) in block.instructions.iter().enumerate() {
            eprintln!("DEBUG: Processing instruction {}: {:?}", i, inst);
            let value = self.compile_instruction(inst)?;
            self.values.insert(inst.result, value);
            eprintln!("DEBUG: Stored value for {:?}", inst.result);
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
                    let data_ptr = self.builder.build_call(
                        self.runtime_functions.vexl_vec_alloc_i64,
                        &[alloc_size.into()],
                        "vec_data"
                    ).unwrap().try_as_basic_value().left().unwrap();

                    // Store elements
                    for (i, &elem_id) in elements.iter().enumerate() {
                        let elem_val = self.get_value(elem_id)?;
                        let idx_val = i64_type.const_int(i as u64, false);
                        self.builder.build_call(
                            self.runtime_functions.vexl_vec_set_i64,
                            &[data_ptr.into(), idx_val.into(), elem_val.into()],
                            ""
                        ).unwrap();
                    }

                    // Create vector from array
                    let vec_ptr = self.builder.build_call(
                        self.runtime_functions.vexl_vec_from_i64_array,
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
                let result = self.builder.build_call(
                    self.runtime_functions.vexl_vec_get_i64,
                    &[vec_val.into(), idx_val.into()],
                    "vec_get"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::VectorSet { vector, index, value } => {
                let vec_val = self.get_value(*vector)?;
                let idx_val = self.get_value(*index)?;
                let val_val = self.get_value(*value)?;
                self.builder.build_call(
                    self.runtime_functions.vexl_vec_set_i64,
                    &[vec_val.into(), idx_val.into(), val_val.into()],
                    ""
                ).unwrap();
                // Return void (0)
                let i64_type = self.context.i64_type();
                Ok(i64_type.const_int(0, false).into())
            }

            // Arithmetic operations
            InstructionKind::Add(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;

                if left_val.is_pointer_value() && right_val.is_pointer_value() {
                    // Vector + Vector
                    let func = self.runtime_functions.vexl_vec_add_i64;
                    let args = &[left_val.into(), right_val.into()];
                    let call = self.builder.build_call(func, args, "vec_add").unwrap();
                    let result = call.try_as_basic_value().left().unwrap();
                    Ok(result)
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
                let result = self.builder.build_call(
                    self.runtime_functions.vexl_mat_mul,
                    &[left_val.into(), right_val.into()],
                    "mat_mul"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::Outer(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let result = self.builder.build_call(
                    self.runtime_functions.vexl_mat_outer,
                    &[left_val.into(), right_val.into()],
                    "mat_outer"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::Dot(left, right) => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                let result = self.builder.build_call(
                    self.runtime_functions.vexl_mat_dot,
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
                let _func_val = self.get_value(*func)?;
                let _arg_vals: Vec<inkwell::values::BasicMetadataValueEnum> = args.iter()
                    .map(|&arg| self.get_value(arg).map(|v| v.into()))
                    .collect::<Result<Vec<_>, _>>()?;

                // For now, assume all functions return i64 and take no args
                let i64_type = self.context.i64_type();
                let _fn_type = i64_type.fn_type(&[], false);

                // For function calls, we need to handle this differently
                // For now, just return a placeholder value
                Ok(i64_type.const_int(0, false).into())
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
                let result = self.builder.build_call(
                    self.runtime_functions.vexl_gen_new,
                    &[func_val.into()],
                    "gen_new"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            InstructionKind::GeneratorEval { generator, index } => {
                let gen_val = self.get_value(*generator)?;
                let idx_val = self.get_value(*index)?;
                let result = self.builder.build_call(
                    self.runtime_functions.vexl_gen_eval,
                    &[gen_val.into(), idx_val.into()],
                    "gen_eval"
                ).unwrap().try_as_basic_value().left().unwrap();
                Ok(result)
            }

            // Range operations
            InstructionKind::Range { start, end } => {
                let start_val = self.get_value(*start)?;
                let end_val = self.get_value(*end)?;

                let result = self.builder.build_call(
                    self.runtime_functions.vexl_range_new,
                    &[start_val.into(), end_val.into()],
                    "range"
                ).unwrap().try_as_basic_value().left().unwrap();

                Ok(result)
            }

            InstructionKind::InfiniteRange { start } => {
                let start_val = self.get_value(*start)?;
                let result = self.builder.build_call(
                    self.runtime_functions.vexl_range_infinite,
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
                self.builder.build_return(Some(&value)).unwrap();
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
        match self.values.get(&id) {
            Some(&value) => Ok(value),
            None => {
                eprintln!("DEBUG: Available values: {:?}", self.values.keys().collect::<Vec<_>>());
                Err(format!("Value {:?} not found", id))
            }
        }
    }

    /// Get runtime function by name
    fn get_runtime_function(&self, name: &str) -> Result<FunctionValue<'ctx>, String> {
        match name {
            "vexl_vec_alloc_i64" => Ok(self.runtime_functions.vexl_vec_alloc_i64),
            "vexl_vec_get_i64" => Ok(self.runtime_functions.vexl_vec_get_i64),
            "vexl_vec_set_i64" => Ok(self.runtime_functions.vexl_vec_set_i64),
            "vexl_vec_len" => Ok(self.runtime_functions.vexl_vec_len),
            "vexl_vec_free" => Ok(self.runtime_functions.vexl_vec_free),
            "vexl_vec_from_i64_array" => Ok(self.runtime_functions.vexl_vec_from_i64_array),
            "vexl_vec_add_i64" => Ok(self.runtime_functions.vexl_vec_add_i64),
            "vexl_vec_mul_scalar_i64" => Ok(self.runtime_functions.vexl_vec_mul_scalar_i64),
            "vexl_print_int" => Ok(self.runtime_functions.vexl_print_int),
            "vexl_print_string" => Ok(self.runtime_functions.vexl_print_string),
            "vexl_vec_map_parallel" => Ok(self.runtime_functions.vexl_vec_map_parallel),
            "vexl_vec_filter" => Ok(self.runtime_functions.vexl_vec_filter),
            "vexl_vec_reduce_parallel" => Ok(self.runtime_functions.vexl_vec_reduce_parallel),
            "vexl_vec_map_sequential" => Ok(self.runtime_functions.vexl_vec_map_sequential),
            "vexl_vec_reduce_sequential" => Ok(self.runtime_functions.vexl_vec_reduce_sequential),
            // Standard library functions
            "vexl_vec_sum" => Ok(self.runtime_functions.vexl_vec_sum),
            "vexl_vec_product" => Ok(self.runtime_functions.vexl_vec_product),
            "vexl_vec_max" => Ok(self.runtime_functions.vexl_vec_max),
            "vexl_vec_min" => Ok(self.runtime_functions.vexl_vec_min),
            "vexl_math_sin" => Ok(self.runtime_functions.vexl_math_sin),
            "vexl_math_cos" => Ok(self.runtime_functions.vexl_math_cos),
            "vexl_math_sqrt" => Ok(self.runtime_functions.vexl_math_sqrt),
            "vexl_math_pow" => Ok(self.runtime_functions.vexl_math_pow),
            "vexl_print_float" => Ok(self.runtime_functions.vexl_print_float),
            // I/O functions
            "vexl_read_line" => Ok(self.runtime_functions.vexl_read_line),
            "vexl_read_file" => Ok(self.runtime_functions.vexl_read_file),
            "vexl_write_file" => Ok(self.runtime_functions.vexl_write_file),
            "vexl_file_exists" => Ok(self.runtime_functions.vexl_file_exists),
            "vexl_file_size" => Ok(self.runtime_functions.vexl_file_size),
            // System functions
            "vexl_current_time" => Ok(self.runtime_functions.vexl_current_time),
            "vexl_current_time_ns" => Ok(self.runtime_functions.vexl_current_time_ns),
            "vexl_sleep_ms" => Ok(self.runtime_functions.vexl_sleep_ms),
            "vexl_getenv" => Ok(self.runtime_functions.vexl_getenv),
            "vexl_setenv" => Ok(self.runtime_functions.vexl_setenv),
            "vexl_get_args" => Ok(self.runtime_functions.vexl_get_args),
            "vexl_exit" => Ok(self.runtime_functions.vexl_exit),
            "vexl_getpid" => Ok(self.runtime_functions.vexl_getpid),
            "vexl_random" => Ok(self.runtime_functions.vexl_random),
            // Memory functions
            "vexl_alloc" => Ok(self.runtime_functions.vexl_alloc),
            "vexl_free" => Ok(self.runtime_functions.vexl_free),
            // String functions
            "vexl_string_compare" => Ok(self.runtime_functions.vexl_string_compare),
            "vexl_string_substring" => Ok(self.runtime_functions.vexl_string_substring),
            _ => Err(format!("Unknown runtime function: {}", name)),
        }
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
                Instruction {
                    result: v1,
                    kind: InstructionKind::ConstInt(42),
                },
            ],
            terminator: Terminator::Return(v1),
        };
        
        let mut func = VirFunction {
            name: "main".to_string(),
            params: vec![],
            blocks: HashMap::from([(block_id, block)]),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
        };
        
        module.add_function("main".to_string(), func);
        
        let llvm_ir = codegen_to_string(&module).unwrap();
        assert!(llvm_ir.contains("ret i64 42"));
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
            name: "main".to_string(),
            params: vec![],
            blocks: HashMap::from([(block_id, block)]),
            entry_block: block_id,
            effect: vexl_core::Effect::Pure,
        };
        
        module.add_function("main".to_string(), func);
        
        let llvm_ir = codegen_to_string(&module).unwrap();
        // Just verify it compiles and returns valid LLVM IR
        assert!(!llvm_ir.is_empty());
        assert!(llvm_ir.contains("define"));
    }
}
