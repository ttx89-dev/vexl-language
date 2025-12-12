# Creating Executables from VEXL

## ✅ What Works Now

VEXL compiles programs to LLVM IR with **real computed values**!

### Example

**Input (`program.vexl`):**
```vexl
let x = 40
let y = 2
x + y
```

**Output (LLVM IR):**
```llvm
define i64 @main() {
entry:
  ret i64 42
}
```

## 🔗 How to Create an Executable

### Prerequisites

Install LLVM tools:
```bash
sudo apt install llvm
```

### Compilation Pipeline

```bash
# 1. Compile VEXL to LLVM IR
./target/release/vexl compile program.vexl -o program.ll

# 2. Compile LLVM IR to object file  
llc program.ll -o program.o

# 3. Link to executable
gcc program.o -o program

# 4. Run!
./program
echo $?  # Should print 42!
```

### What Each Step Does

1. **VEXL → LLVM IR** ✅ WORKING!
   - Parse VEXL source
   - Type-check (with dimensional safety!)
   - Optimize (constant folding, DCE)
   - Generate LLVM IR

2. **LLVM IR → Object File** (requires `llc`)
   - LLVM compiles IR to native machine code
   - Creates `.o` object file

3. **Object → Executable** (requires `gcc`)
   - Links object file
   - Creates final binary

4. **Execute**
   - Run the program
   - Exit code = return value

## 🎯 Current Status

**Working:**
- ✅ VEX

L → LLVM IR (with real values!)
- ✅ Type checking
- ✅ Optimizations
- ✅ Arithmetic computations

**Needs:**
- Install LLVM (`sudo apt install llvm`)
- Standard library for I/O
- Control flow compilation

## 📝 Example Programs

### Simple Return
```vexl
42
```
Compiles to: `ret i64 42`

### Arithmetic
```vexl
1 + 2 * 3
```
Compiles to: `ret i64 7`

### Variables
```vexl
let x = 10
let y = 32
x + y
```
Compiles to: `ret i64 42`

### Vectors (type-checks!)
```vexl
[1, 2, 3]
```
Type checks correctly, generates IR structure.

## 🚀 Next Steps

To make this production-ready:

1. **Install LLVM** (one-time setup)
2. **Runtime library** (for I/O, printing)
3. **Standard library** (map, filter, etc.)
4. **Control flow** (if/else compilation)

**You're 80% there!** The compiler works, just needs runtime support.
