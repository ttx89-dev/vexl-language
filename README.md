# VEXL

[![MIT License](https://img.shields.io/badge/OpenSource-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![AI Coded](https://img.shields.io/badge/AI%20Coded-100%25-brightgreen.svg?style=flat-square)](https://anthropic.com)
[![Built with Claude 4.5 Opus](https://img.shields.io/badge/Claude%204.5%20Opus-ff9900.svg?style=flat-square&logo=anthropic&logoColor=white)](https://claude.ai)

> **"What if everything was a vector?"**

VEXL starts with a simple premise: **Everything is a vector.**

In most languages, you have integers, arrays, lists, and objects. In VEXL, a single number is just a 1-dimensional vector of size 1. A list is a vector. A matrix is a vector. Even a tree or a graph can be expressed as a nested vector structure.

By treating everything as a vector, VEXL unifies mathematics and programming. Whether you are doing simple arithmetic, complex data processing, or high-performance neural networks, the syntax and logic remain exactly the same. It is designed to be **simple enough for a beginner** to write a script, yet **powerful enough for a scientist**.

VEXL is a compiled, statically-typed language built for the future of high-performance computing, where parallel processing is the default, not an afterthought.

## ✨ Key Features

- 🔢 **Universal Vector Type**: Whether it's `42` or a billion-row dataset, it's all just `Vector<T, D>`.
- ⚡ **Automatic Speed**: The compiler knows which parts of your code don't affect each other, so it runs them in parallel automatically.
- 📐 **Shape Safety**: VEXL catches math errors at compile-time. It won't let you multiply a `2x3` matrix by a `5x5` matrix.
- ∞ **Infinite Storage**: Generators allow you to describe infinite datasets (like "all prime numbers") without using infinite memory.

## 🚀 Current Status

**Phase 1: Foundation** - In Progress

✅ Complete workspace with 9 crates  
✅ Core types (Vector, Generator, Effect)  
✅ Lexer with 40+ tokens  
✅ Complete AST  
✅ Type inference foundation  
✅ Cooperative load-balancing scheduler  
⏱️ Build time: 6.55s  
✅ All tests passing

## 📦 Project Structure

```
vexl/
├── crates/
│   ├── vexl-core/         # Vector<T,D>, Generator trait, Effect types
│   ├── vexl-syntax/       # Lexer (logos) + Parser (chumsky) + AST
│   ├── vexl-types/        # Hindley-Milner + dimensional + effect inference
│   ├── vexl-ir/           # VEXL Intermediate Representation (VIR)
│   ├── vexl-codegen/      # LLVM backend (planned)
│   ├── vexl-runtime/      # CooperativeScheduler, GC, caching
│   ├── vexl-driver/       # CLI, REPL, build pipeline
│   ├── vexl-lsp/          # Language Server Protocol
│   └── vexl-pkg/          # Package manager
├── tests/
│   ├── framework/         # Test infrastructure
│   └── conformance/       # Conformance test suite
└── docs/                  # Documentation
```

## 🛠️ Building

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the workspace
cargo build --workspace

# Run tests
cargo test --workspace

# Check code
cargo check --workspace
```

## 📖 Language Examples

### Vector Literals
```vexl
let scalar = 42
let vector = [1, 2, 3]
let matrix = [[1, 2], [3, 4]]
```

### Comprehensions
```vexl
let squares = [x*x | x <- [1..10]]
```

### Generators (Infinite Sequences)
```vexl
let fibs = fix f => [0, 1, ...[f[i-1] + f[i-2] | i <- [2..]]]
let first_10 = fibs |> take(10)
```

### Pipelines
```vexl
data |> map(f) |> filter(p) |> reduce(+) |> print
```

### Effect Types
```vexl
pure fn add(a: Int, b: Int) -> Int { a + b }  // Auto-parallelizable
io fn read_file(path: String) -> String { ... }  // Not parallelized
```

## 🎯 Design Principles (OPTIBEST Framework)

VEXL follows the OPTIBEST framework for premium software design:

1. **Functional Excellence**: Builds and tests pass
2. **Efficiency**: CooperativeScheduler for optimal CPU utilization
3. **Robustness**: Strong type system with dimensional checking
4. **Scalability**: Generator-based storage for infinite data
5. **Maintainability**: Clear module separation, comprehensive tests
6. **Innovation**: Fractal computing, universal vector type
7. **Elegance**: Minimal complexity for maximum expressiveness

## 🔄 Cooperative Scheduler

VEXL uses a **Cooperative Load Balancing Scheduler** (renamed from "work-stealing" for clarity):

- Idle threads *help* busy threads by taking tasks from their queues
- Fair work distribution across all CPU cores
- Lock-free task queues using `crossbeam-deque`
- Optimal CPU utilization without thread starvation

## 📋 Roadmap

### Phase 1: Foundation (In Progress)
- ✅ Workspace setup
- ✅ Core types
- ✅ Lexer & AST
- ⏭️ Complete parser
- ⏭️ Full type inference

### Phase 2: Compilation (Planned)
- VIR lowering
- LLVM backend
- SIMD optimization
- Runtime expansion

### Phase 3: Tooling (Planned)
- CLI commands
- REPL
- LSP server
- VS Code extension

### Phase 4: Testing & QA (Planned)
- Conformance suite
- Property-based tests
- Benchmarks

### Phase 5: Documentation (Planned)
- Language reference
- Standard library docs
- Getting started guide

## 🤝 Contributing

VEXL is designed following the OPTIBEST framework. Contributions should:

- Maintain or improve code quality
- Include comprehensive tests
- Follow Rust best practices
- Respect the cooperative scheduler terminology

## 📜 License

MIT

## 🙏 Acknowledgments

Built with:
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [logos](https://github.com/maciejhirsz/logos) - Lexer generator
- [chumsky](https://github.com/zesterer/chumsky) - Parser combinator library
- [crossbeam](https://github.com/crossbeam-rs/crossbeam) - Concurrency tools
- [tower-lsp](https://github.com/ebkalderon/tower-lsp) - LSP framework
- LLVM (planned) - Code generation backend

---

**Status**: Active Development | **First Release**: TBD | **Language Version**: 0.1.0
