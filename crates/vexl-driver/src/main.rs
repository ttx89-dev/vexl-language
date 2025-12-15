//! VEXL Compiler Driver - Main CLI

use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

use vexl_syntax::parser::parse;
use vexl_types::inference::TypeEnv;
use vexl_ir::lower::lower_to_vir;
use vexl_ir::optimize::optimize;
use vexl_codegen::codegen_to_string;

#[derive(Parser)]
#[command(name = "vexl")]
#[command(about = "VEXL Compiler - Vector Expression Language", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { input, output, verbose } => {
            if let Err(e) = compile_file(&input, output.as_ref(), verbose) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Commands::Check { input } => {
            if let Err(e) = check_file(&input) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            println!("✓ Type check passed!");
        }

        Commands::Run { input } => {
            if let Err(e) = run_file(&input) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Commands::Eval { expression } => {
            if let Err(e) = eval_expression(&expression) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Commands::Ast { input } => {
            if let Err(e) = show_ast(&input) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }

        Commands::Link { input, output } => {
            if let Err(e) = link_executable(&input, &output) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            println!("✅ Linked executable: {}", output.display());
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
    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;
    
    if verbose {
        eprintln!("✓ Parse successful");
        eprintln!("🔍 Type checking...");
    }
    
    // Type check
    let mut type_env = TypeEnv::new();
    let (inferred_type, _) = vexl_types::inference::infer(&ast, &mut type_env)?;
    
    if verbose {
        eprintln!("✓ Type check passed");
        eprintln!("  Inferred type: {:?}", inferred_type);
        eprintln!("⚙️  Lowering to VIR...");
    }
    
    // Lower to VIR
    let mut vir_module = lower_to_vir(&ast)?;

    if verbose {
        eprintln!("✓ VIR lowering complete");
        eprintln!("VIR Module: {:#?}", vir_module);
        eprintln!("🔧 Optimizing...");
    }

    // Debug: print VIR before optimization
    eprintln!("DEBUG: VIR before optimization:");
    for func in vir_module.functions.values() {
        eprintln!("  Function: {}", func.name);
        if let Some(block) = func.blocks.get(&func.entry_block) {
            for (i, inst) in block.instructions.iter().enumerate() {
                eprintln!("    Instruction {}: {:?}", i, inst);
            }
        }
    }
    
    // Optimize
    optimize(&mut vir_module);
    
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

fn link_executable(input: &PathBuf, output: &PathBuf) -> Result<(), String> {
    // Find the runtime library
    // Prioritize static library for easier linking
    let runtime_paths = [
        PathBuf::from("target/release/libvexl_runtime.a"),
        PathBuf::from("target/debug/libvexl_runtime.a"),
        PathBuf::from("target/release/libvexl_runtime.rlib"),
        PathBuf::from("target/debug/libvexl_runtime.rlib"),
    ];

    let lib_path = runtime_paths.into_iter().find(|p| p.exists()).ok_or_else(|| {
        "Could not find VEXL runtime library (libvexl_runtime.a). Please run 'cargo build -p vexl-runtime' first.".to_string()
    })?;

    println!("🔗 Linking with runtime: {}", lib_path.display());

    // Invoke Clang to link
    // input.o + libvexl_runtime.rlib -> output
    let status = std::process::Command::new("clang")
        .arg(input)
        .arg(&lib_path)
        .arg("-o")
        .arg(output)
        .arg("-lm") // Link math library
        .arg("-lpthread") // Link pthread for scheduler
        .arg("-ldl") // Link dl for dynamic loading if needed
        .status()
        .map_err(|e| format!("Failed to run linker (cc): {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err("Linking failed".to_string())
    }
}

fn check_file(input: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;
    
    let mut type_env = TypeEnv::new();
    let (inferred_type, _) = vexl_types::inference::infer(&ast, &mut type_env)?;
    
    println!("Type: {:?}", inferred_type);
    
    Ok(())
}

fn run_file(input: &PathBuf) -> Result<(), String> {
    // 1. Compile to LLVM IR (temp file)
    let temp_ll = input.with_extension("tmp.ll");
    compile_file(input, Some(&temp_ll), false)?;

    // 2. Link to executable (temp file)
    let temp_exe = input.with_extension("tmp");
    if let Err(e) = link_executable(&temp_ll, &temp_exe) {
        let _ = fs::remove_file(&temp_ll);
        return Err(e);
    }

    // 3. Run
    println!("🚀 Running {}...", input.display());
    let status = std::process::Command::new(&temp_exe)
        .status()
        .map_err(|e| format!("Failed to run executable: {}", e));

    // Cleanup
    let _ = fs::remove_file(&temp_ll);
    let _ = fs::remove_file(&temp_exe);

    match status {
        Ok(s) => {
            if !s.success() {
                Err(format!("Program exited with status: {}", s))
            } else {
                Ok(())
            }
        }
        Err(e) => Err(e),
    }
}

fn eval_expression(expression: &str) -> Result<(), String> {
    // Wrap expression for evaluation
    let wrapped = format!("let _result = {}", expression);

    let ast = parse(&wrapped).map_err(|e| format!("Parse error: {:?}", e))?;

    let mut type_env = TypeEnv::new();
    let (inferred_type, _) = vexl_types::inference::infer(&ast, &mut type_env)?;

    println!("Expression: {}", expression);
    println!("Type: {:?}", inferred_type);

    // For now, just type-check - full evaluation would require JIT
    println!("Note: Full evaluation requires JIT compilation (planned feature)");

    Ok(())
}

fn show_ast(input: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;

    println!("AST for {}:", input.display());
    println!("{:#?}", ast);

    Ok(())
}
