# VEXL Documentation

> **Complete Documentation for the VEXL Programming Language**

## Welcome to VEXL

VEXL is a revolutionary vector programming language that treats **everything as a vector**. This documentation provides comprehensive guides, examples, and references for learning and using VEXL.

## Quick Navigation

### 🚀 Getting Started
- **[Installation Guide](getting-started/installation.md)** - Install VEXL on your system
- **[Your First Program](getting-started/first-program.md)** - Write and run your first VEXL code
- **[Basic Concepts](getting-started/basic-concepts.md)** - Understand VEXL fundamentals
- **[Complete Tutorial](getting-started/tutorial.md)** - Step-by-step learning path

### 📚 Language Guide
- **[Syntax Reference](language-guide/syntax-reference.md)** - Complete language grammar
- **[Type System](language-guide/type-system.md)** - VEXL's universal vector types
- **[Vector Operations](language-guide/vector-operations.md)** - Master vector manipulation
- **[Functions](language-guide/functions.md)** - Function types and effects
- **[Control Flow](language-guide/control-flow.md)** - Pattern matching and flow control
- **[Effects](language-guide/effects.md)** - Pure, IO, and state functions

### 📖 Reference
- **[Complete Syntax](reference/complete-syntax.md)** - Every language construct
- **[Built-in Functions](reference/built-in-functions.md)** - Core function reference
- **[Standard Library](reference/standard-library.md)** - Standard modules and utilities

### 💡 Examples
- **[Beginner Examples](examples/beginner-examples.md)** - Simple, practical programs
- **[Intermediate Examples](examples/intermediate-examples.md)** - More complex applications
- **[Advanced Examples](examples/advanced-examples.md)** - Expert-level demonstrations

### ⚡ Advanced Topics
- **[Performance](advanced/performance.md)** - Optimization techniques
- **[Parallelism](advanced/parallelism.md)** - Concurrent programming
- **[GPU Computing](advanced/gpu-computing.md)** - High-performance GPU acceleration with safety monitoring
- **[Optimization](advanced/optimization.md)** - Advanced optimizations

### 🔧 Development
- **[Contributing](development/contributing.md)** - How to contribute to VEXL
- **[Architecture](development/architecture.md)** - VEXL system design
- **[Roadmap](development/roadmap.md)** - Future development plans

### 👥 Community
- **[FAQ](community/faq.md)** - Frequently asked questions
- **[Troubleshooting](community/troubleshooting.md)** - Common issues and solutions
- **[Resources](community/resources.md)** - Additional learning materials

## What is VEXL?

VEXL is a programming language built on a simple but powerful idea: **everything is a vector**. 

In traditional languages:
```python
# Python - different types everywhere
number = 42          # int
list_data = [1, 2, 3]     # list  
matrix = [[1, 2], [3, 4]]  # nested list
```

In VEXL:
```vexl
// Everything follows the same rules
let scalar = 42                     // Vector<Int, 1>
let vector = [1, 2, 3]              // Vector<Int, 1>
let matrix = [[1, 2], [3, 4]]       // Vector<Vector<Int, 1>, 1>
```

## Key Features

### ✨ Universal Vector Type
Everything from single numbers to complex data structures follows the same vector rules.

### ⚡ Automatic Parallelism
The compiler automatically identifies parallelizable code and optimizes it for multiple cores.

### 📐 Shape Safety
VEXL prevents impossible operations at compile time through dimensional checking.

### ∞ Infinite Storage
Generators allow you to work with infinite datasets without using infinite memory.

### 🎯 Effect Types
Clear separation between pure functions (parallelizable), IO functions, and state functions.

## Why VEXL?

1. **Consistency**: Same operations work at every level
2. **Safety**: Compile-time shape checking prevents errors
3. **Performance**: Automatic parallelization and optimization
4. **Simplicity**: Learn once, apply everywhere
5. **Power**: From simple scripts to complex algorithms

## Getting Started

### Installation

1. **Install Rust** (required for building VEXL):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. **Build VEXL**:
```bash
git clone https://github.com/your-org/vexl-language.git
cd vexl-language
cargo build --release
```

3. **Test Installation**:
```bash
echo 'let result = 2 + 2 * 3' > test.vexl
./target/release/vexl run test.vexl
```

### Your First Program

Create `hello.vexl`:
```vexl
// Your first VEXL program!
print("Hello, World!")

// Working with vectors
let numbers = [1, 2, 3, 4, 5]
let doubled = numbers |> map(|x| x * 2)
let sum = doubled |> sum()

print("Doubled: " + doubled)
print("Sum: " + sum)
```

Run it:
```bash
./target/release/vexl run hello.vexl
```

## Learning Path

### Beginner
1. Read [Basic Concepts](getting-started/basic-concepts.md)
2. Follow the [Tutorial](getting-started/tutorial.md)
3. Try [Beginner Examples](examples/beginner-examples.md)

### Intermediate
1. Study the [Type System](language-guide/type-system.md)
2. Learn [Vector Operations](language-guide/vector-operations.md)
3. Explore [Functions](language-guide/functions.md)
4. Work through [Intermediate Examples](examples/intermediate-examples.md)

### Advanced
1. Understand [Effects](language-guide/effects.md)
2. Study [Performance](advanced/performance.md)
3. Learn [Parallelism](advanced/parallelism.md)
4. Try [Advanced Examples](examples/advanced-examples.md)

## Documentation Status

### ✅ Complete
- [x] Installation and setup
- [x] Basic concepts and tutorial
- [x] Core syntax reference
- [x] Type system guide
- [x] Built-in functions
- [x] Beginner examples
- [x] **GPU Computing** - High-performance GPU acceleration with safety monitoring
- [x] **Performance** - OPTIBEST optimization strategies and benchmarking

### 🚧 In Progress
- [ ] Vector operations guide
- [ ] Functions and effects
- [ ] Control flow patterns
- [ ] Standard library reference
- [ ] Intermediate examples

### 📋 Planned
- [ ] Advanced optimization guides
- [ ] Contributing guidelines
- [ ] Architecture documentation
- [ ] Community resources

## Community

- **GitHub**: https://github.com/your-org/vexl-language
- **Discord**: [Join our community](https://discord.gg/vexl)
- **Forum**: https://forum.vexl.dev
- **Issues**: [Report bugs or request features](https://github.com/your-org/vexl-language/issues)

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to help improve VEXL documentation.

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

VEXL is dual licensed under MIT or Apache-2.0.

---

**Built with ❤️ using the OPTIBEST framework**

*VEXL: Where mathematics meets programming*
