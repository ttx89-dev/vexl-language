# VEXL Working Examples

These examples use **only implemented features** and will compile successfully!

## ✅ Working Examples

All these examples parse, type-check, and compile:

- **`01_arithmetic.vexl`** - Basic math operations
- **`02_let_bindings.vexl`** - Variable bindings
- **`03_vectors.vexl`** - Vector creation
- **`04_ranges.vexl`** - Finite and infinite ranges
- **`05_lambdas.vexl`** - Anonymous functions
- **`06_pipelines.vexl`** - Data flow with `|>`
- **`simple.vexl`** - Mix of features

## 🧪 Testing

```bash
# Type-check all examples
for f in examples/0*.vexl; do
  ./target/release/vexl check "$f"
done

# Compile to LLVM IR
./target/release/vexl compile examples/01_arithmetic.vexl --verbose
```

## ⚠️ Not Yet Implemented

The following syntax is **planned but not yet in the parser**:

- Records/Objects: `{key: value}`
- Nested arrays: `[[1, 2], [3, 4]]`
- String interpolation
- Pattern matching
- Standard library functions (map, filter, etc.)

These will be added in future iterations!

## 📝 What Works Now

Current parser supports:
- ✅ Literals (int, float, string, ident)
- ✅ Binary operations (+, -, *, /, @, ==, !=, <, >, <=, >=)
- ✅ Vectors: `[1, 2, 3]`
- ✅ Ranges: `[0..10]`, `[0..]`
- ✅ Let bindings: `let x = 5 in x + 1`
- ✅ Lambdas: `|x| x + 1`, `(x, y) => x * y`
- ✅ If expressions
- ✅ Pipelines: `data |> f |> g`
- ✅ Function calls: `f(x, y)`
- ✅ Comprehensions: `[x * 2 | x <- xs]`
- ✅ Fix (recursion): `fix f => ...`
