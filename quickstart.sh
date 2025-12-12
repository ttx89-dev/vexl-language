#!/usr/bin/env bash
# Quick start guide for using the VEXL compiler

echo "🎯 VEXL Compiler - Quick Start"
echo "==============================="
echo ""

# Build the compiler if not already built
if [ ! -f target/release/vexl ]; then
    echo "📦 Building VEXL compiler (first time setup)..."
    cargo build --release -p vexl-driver
    echo ""
fi

echo "✅ VEXL compiler ready!"
echo ""
echo "📋 Usage Examples:"
echo ""
echo "1. Compile a file to LLVM IR:"
echo "   ./target/release/vexl compile examples/simple.vexl -o output.ll"
echo ""
echo "2. Type-check a file:"
echo "   ./target/release/vexl check examples/simple.vexl"
echo ""
echo "3. Show AST:"
echo "   ./target/release/vexl ast examples/simple.vexl"
echo ""
echo "4. With verbose output:"
echo "   ./target/release/vexl compile examples/simple.vexl --verbose"
echo ""
echo "💡 To install globally (optional):"
echo "   cargo install --path crates/vexl-driver"
echo "   # Then use: vexl compile file.vexl"
echo ""
