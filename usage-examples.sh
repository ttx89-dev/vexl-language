#!/usr/bin/env bash
# VEXL Compiler Usage Examples

echo "🎯 VEXL Compiler - Usage Guide"
echo "==============================="
echo ""

cd /home/ryan/code/vexl

echo "📝 Example 1: Type-check a VEXL file"
echo "-------------------------------------"
echo "Command: ./target/release/vexl check examples/simple.vexl"
echo ""
./target/release/vexl check examples/simple.vexl
echo ""

echo "📝 Example 2: Compile to LLVM IR (verbose)"
echo "-------------------------------------------"
echo "Command: ./target/release/vexl compile examples/simple.vexl --verbose"
echo ""
echo "1 + 2 * 3" | tee /tmp/example.vexl
echo ""
./target/release/vexl compile /tmp/example.vexl --verbose
echo ""

echo "📝 Example 3: Save LLVM IR to file"
echo "-----------------------------------"
echo "Command: ./target/release/vexl compile /tmp/example.vexl -o /tmp/output.ll"
echo ""
./target/release/vexl compile /tmp/example.vexl -o /tmp/output.ll
echo "✅ Saved to /tmp/output.ll"
echo ""

echo "📝 Example 4: View generated LLVM IR"
echo "-------------------------------------"
head -n 15 /tmp/output.ll
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Your VEXL compiler is working!"
echo ""
echo "Try it yourself:"
echo "  echo 'let x = 5 in x + 1' > test.vexl"
echo "  ./target/release/vexl check test.vexl"
echo "  ./target/release/vexl compile test.vexl --verbose"
echo ""
