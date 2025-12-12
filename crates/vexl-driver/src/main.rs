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
    
    /// Show the AST for a VEXL file
    Ast {
        /// Input VEXL source file
        #[arg(value_name = "FILE")]
        input: PathBuf,
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
        
        Commands::Ast { input } => {
            if let Err(e) = show_ast(&input) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
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
        eprintln!("🔧 Optimizing...");
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

fn check_file(input: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;
    
    let mut type_env = TypeEnv::new();
    let (inferred_type, _) = vexl_types::inference::infer(&ast, &mut type_env)?;
    
    println!("Type: {:?}", inferred_type);
    
    Ok(())
}

fn show_ast(input: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(input)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    let ast = parse(&source).map_err(|e| format!("Parse error: {:?}", e))?;
    
    println!("{:#?}", ast);
    
    Ok(())
}
