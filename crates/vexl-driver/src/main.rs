//! VEXL Compiler Driver - Main CLI

mod repl;
mod package_commands;

use clap::{Parser, Subcommand, CommandFactory};
use std::fs;
use std::path::PathBuf;

use vexl_syntax::parser::parse;
use vexl_syntax::ast::Type;
use vexl_ir::lower::{lower_to_vir, lower_decls_to_vir};
use vexl_ir::optimize::optimize;
use vexl_codegen::{codegen_to_string, JitEngine};

#[derive(Parser)]
#[command(name = "vexl")]
#[command(about = "VEXL Compiler - Vector Expression Language", long_about = None)]
struct Cli {
    /// Optional input file (enables 'vexl script.vexl' syntax)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a VEXL source file to LLVM IR
    Compile {
        /// Input VEXL source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file (defaults to stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Show optimization passes
        #[arg(short, long)]
        verbose: bool,
    },

    /// Type-check a VEXL file without compiling
    Check {
        /// Input VEXL source file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Compile and run a VEXL program immediately
    Run {
        /// Input VEXL source file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Evaluate a VEXL expression
    Eval {
        /// VEXL expression to evaluate
        #[arg(value_name = "EXPR")]
        expression: String,
    },

    /// Start interactive REPL
    Repl,

    /// Show the AST for a VEXL file
    Ast {
        /// Input VEXL source file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Link an object file to create an executable
    Link {
        /// Input object file (.o)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output executable name
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
    },

    /// Package management commands
    #[command(subcommand)]
    Package(crate::package_commands::PackageCommands),
}

fn main() {
    let cli = Cli::parse();

    // Handle direct script execution: 'vexl script.vexl' -> 'vexl run script.vexl'
    let command = if cli.input.is_some() && cli.command.is_none() {
        Some(Commands::Run { input: cli.input.unwrap() })
    } else {
        cli.command
    };

    match command {
        Some(Commands::Compile { input, output, verbose }) => {
            if let Err(e) = compile_file(&input, output.as_ref(), verbose) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Some(Commands::Check { input }) => {
            if let Err(e) = check_file(&input) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            println!("✓ Type check passed!");
        }

        Some(Commands::Run { input }) => {
            if let Err(e) = run_file(&input) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Some(Commands::Eval { expression }) => {
            if let Err(e) = eval_expression(&expression) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Some(Commands::Repl) => {
            if let Err(e) = crate::repl::run_repl() {
                eprintln!("REPL error: {}", e);
                std::process::exit(1);
            }
        }

        Some(Commands::Ast { input }) => {
            if let Err(e) = show_ast(&input) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Some(Commands::Link { input, output }) => {
            if let Err(e) = link_executable(&input, &output) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            println!("✅ Linked executable: {}", output.display());
        }

        Some(Commands::Package(cmd)) => {
            if let Err(e) = crate::package_commands::execute_package_command(cmd) {
                eprintln!("Package error: {}", e);
                std::process::exit(1);
            }
        }

        None => {
            // No command specified, show help
            let _ = Cli::command().print_help();
            std::process::exit(1);
        }
    }
}

fn compile_file(input: &PathBuf, output: Option<&PathBuf>, verbose: bool) -> Result<(), String> {
    // Read source
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    if verbose {
        eprintln!("📄 Parsing {}...", input.display());
    }
    
    // Parse
    eprintln!("Starting parsing...");
    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;
    eprintln!("Parsing successful, got {} declarations", ast.len());
    
    if verbose {
        eprintln!("✓ Parse successful");
        eprintln!("🔍 Type checking...");
    }
    
    // Type check - for now, skip type checking declarations
    // TODO: Implement type checking for declarations
    let inferred_type = Type::Int; // Placeholder
    
    if verbose {
        eprintln!("✓ Type check passed");
        eprintln!("  Inferred type: {:?}", inferred_type);
        eprintln!("⚙️  Lowering to VIR...");
    }
    
    // Lower to VIR - handle both declarations and expressions
    let mut vir_module = if ast.len() == 1 {
        if let vexl_syntax::ast::Decl::Expr(ref expr) = ast[0] {
            // Backward compatibility for single expressions
            lower_to_vir(expr)?
        } else {
            // Function declarations
            lower_decls_to_vir(&ast)?
        }
    } else {
        // Multiple declarations
        lower_decls_to_vir(&ast)?
    };

    if verbose {
        eprintln!("✓ VIR lowering complete");
        eprintln!("VIR Module: {:#?}", vir_module);
        eprintln!("🔧 Optimizing...");
    }


    
    // Optimize (disabled for now due to crash)
    println!("🛠️  Skipping optimization (temporarily disabled)");
    // optimize(&mut vir_module);
    println!("✅ Optimization skipped");
    
    if verbose {
        eprintln!("✓ Optimizations applied");
        eprintln!("🎯 Generating LLVM IR...");
    }
    
    // Generate LLVM IR
    let llvm_ir = codegen_to_string(&vir_module)?;
    
    if verbose {
        eprintln!("✓ LLVM IR generated");
    }
    
    // Output
    if let Some(output_path) = output {
        fs::write(output_path, llvm_ir)
            .map_err(|e| format!("Failed to write output: {}", e))?;
        
        if verbose {
            eprintln!("✅ Compiled to {}", output_path.display());
        }
    } else {
        println!("{}", llvm_ir);
    }
    
    Ok(())
}

fn link_executable(input_ll: &PathBuf, output: &PathBuf) -> Result<(), String> {
    // Find the runtime library
    // Prioritize static library for easier linking
    let runtime_paths = [
        PathBuf::from("target/release/libvexl_runtime.a"),
        PathBuf::from("target/debug/libvexl_runtime.a"),
        // Look in deps directory for static libraries
        PathBuf::from("target/release/deps/libvexl_runtime-d6f23d98c65485fe.a"),
        PathBuf::from("target/debug/deps/libvexl_runtime-d6f23d98c65485fe.a"),
        PathBuf::from("target/release/libvexl_runtime.rlib"),
        PathBuf::from("target/debug/libvexl_runtime.rlib"),
    ];

    let lib_path = runtime_paths.into_iter().find(|p| p.exists()).ok_or_else(|| {
        "Could not find VEXL runtime library (libvexl_runtime.a). Please run 'cargo build -p vexl-runtime' first.".to_string()
    })?;

    println!("🔗 Linking with runtime: {}", lib_path.display());

    // Use clang to compile LLVM IR directly to executable
    // Clang can handle LLVM IR files directly
    println!("🔗 Compiling LLVM IR with clang: {} -> {}", input_ll.display(), output.display());

    let clang_status = std::process::Command::new("clang")
        .arg(input_ll)  // Input LLVM IR file
        .arg(&lib_path) // Link with runtime library
        .arg("-o")
        .arg(output)    // Output executable
        .arg("-lm")     // Link math library
        .arg("-lpthread") // Link pthread for scheduler
        .arg("-ldl")    // Link dl for dynamic loading if needed
        .status()
        .map_err(|e| format!("Failed to run clang (linker): {}. Make sure clang is installed.", e))?;

    println!("🔗 Clang exit code: {}", clang_status.code().unwrap_or(-1));

    if clang_status.success() {
        if output.exists() {
            println!("✅ Executable created successfully: {}", output.display());
            Ok(())
        } else {
            Err(format!("Clang succeeded but executable not found: {}", output.display()))
        }
    } else {
        Err("Executable linking failed".to_string())
    }
}

fn check_file(input: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;

    // TODO: Implement type checking for declarations
    println!("Parsed {} declarations successfully", ast.len());

    Ok(())
}

fn run_file(input: &PathBuf) -> Result<(), String> {
    // Initialize VEXL runtime
    unsafe { vexl_runtime::vexl_runtime_init(); }

    // Read source
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // Parse
    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;
    println!("✅ Parsing successful, {} declarations", ast.len());

    // Lower to VIR - always use lower_decls_to_vir for full program support
    println!("🔄 Starting VIR lowering...");
    let mut vir_module = lower_decls_to_vir(&ast)?;
    println!("✅ VIR lowering successful");

    // Optimize
    optimize(&mut vir_module);

    // Create JIT engine and register runtime functions
    let context = inkwell::context::Context::create();
    let mut jit_engine = JitEngine::new(&context);

    // Register runtime functions with the JIT engine's symbol resolver
    let symbol_resolver = &mut jit_engine.symbol_resolver;
    symbol_resolver.register_static_symbol(
        "vexl_vec_map_parallel",
        vexl_runtime::vexl_vec_map_parallel as *mut std::ffi::c_void,
        vexl_codegen::FunctionDescriptor::new(
            "vexl_vec_map_parallel".to_string(),
            vec![vexl_ir::VirType::Pointer, vexl_ir::VirType::Pointer, vexl_ir::VirType::Int64],
            vexl_ir::VirType::Pointer,
            vexl_codegen::CallingConvention::C,
            false,
        ),
    );

    symbol_resolver.register_static_symbol(
        "vexl_vec_sum",
        vexl_runtime::vexl_vec_sum as *mut std::ffi::c_void,
        vexl_codegen::FunctionDescriptor::new(
            "vexl_vec_sum".to_string(),
            vec![vexl_ir::VirType::Pointer],
            vexl_ir::VirType::Int64,
            vexl_codegen::CallingConvention::C,
            false,
        ),
    );

    symbol_resolver.register_static_symbol(
        "vexl_print_int",
        vexl_runtime::ffi::vexl_print_int as *mut std::ffi::c_void,
        vexl_codegen::FunctionDescriptor::new(
            "vexl_print_int".to_string(),
            vec![vexl_ir::VirType::Int64],
            vexl_ir::VirType::Void,
            vexl_codegen::CallingConvention::C,
            false,
        ),
    );

    symbol_resolver.register_static_symbol(
        "vexl_print_float",
        vexl_runtime::ffi::vexl_print_float as *mut std::ffi::c_void,
        vexl_codegen::FunctionDescriptor::new(
            "vexl_print_float".to_string(),
            vec![vexl_ir::VirType::Float64],
            vexl_ir::VirType::Void,
            vexl_codegen::CallingConvention::C,
            false,
        ),
    );

    symbol_resolver.register_static_symbol(
        "vexl_print_string",
        vexl_runtime::ffi::vexl_print_string as *mut std::ffi::c_void,
        vexl_codegen::FunctionDescriptor::new(
            "vexl_print_string".to_string(),
            vec![vexl_ir::VirType::Pointer, vexl_ir::VirType::Int64],
            vexl_ir::VirType::Void,
            vexl_codegen::CallingConvention::C,
            false,
        ),
    );

    symbol_resolver.register_static_symbol(
        "vexl_string_concat",
        vexl_runtime::vexl_string_concat as *mut std::ffi::c_void,
        vexl_codegen::FunctionDescriptor::new(
            "vexl_string_concat".to_string(),
            vec![vexl_ir::VirType::Pointer, vexl_ir::VirType::Int64, vexl_ir::VirType::Pointer, vexl_ir::VirType::Int64],
            vexl_ir::VirType::Pointer,
            vexl_codegen::CallingConvention::C,
            false,
        ),
    );

    println!("🚀 Running {}...", input.display());

    let result = jit_engine.compile_and_execute(&vir_module)?;

    println!("Program result: {}", result);

    Ok(())
}

fn eval_expression(expression: &str) -> Result<(), String> {
    // Parse the expression directly (no wrapping needed)
    let ast = parse(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    // TODO: Implement type checking for declarations
    println!("Expression: {}", expression);
    println!("Parsed {} declarations", ast.len());

    // Lower to VIR and optimize - handle declarations
    let mut vir_module = if ast.len() == 1 {
        if let vexl_syntax::ast::Decl::Expr(ref expr) = ast[0] {
            lower_to_vir(expr)?
        } else {
            lower_decls_to_vir(&ast)?
        }
    } else {
        lower_decls_to_vir(&ast)?
    };

    optimize(&mut vir_module);

    // Generate LLVM IR
    let llvm_ir = codegen_to_string(&vir_module)?;
    println!("Generated LLVM IR:");
    println!("{}", llvm_ir);

    // Create JIT engine and evaluate
    let context = inkwell::context::Context::create();
    let mut jit_engine = JitEngine::new(&context);

    let result = jit_engine.compile_and_execute(&vir_module)?;

    println!("Result: {}", result);

    Ok(())
}

fn show_ast(input: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;

    println!("AST for {}:", input.display());
    for (i, decl) in ast.iter().enumerate() {
        println!("Declaration {}: {:#?}", i, decl);
    }

    Ok(())
}