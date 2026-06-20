# VEXL Bootstrapping Architecture

## Purpose

VEXL aims to be a **self-hosting** compiler: a VEXL compiler eventually
written in VEXL itself. This document outlines the staged bootstrapping
plan.

---

## Bootstrapping Stages

### Stage 0: Rust Implementation (current)

The entire VEXL compiler is implemented in Rust across these crates:

| Crate | Role |
|-------|------|
| `vexl-syntax` | Lexer, parser (chumsky), AST |
| `vexl-types` | Type system, type inference, effect tracking |
| `vexl-ir` | VEXL Intermediate Representation, optimization passes |
| `vexl-codegen` | LLVM IR generation, JIT engine |
| `vexl-runtime` | Runtime library (GC, vector ops, I/O, FFI bridge) |
| `vexl-core` | Core compiler library (pipeline orchestration) |
| `vexl-driver` | CLI driver (compile/run/check/eval/repl/link/pkg) |
| `vexl-gpu` | GPU compute kernel generation (SPIR-V/CUDA) |
| `vexl-lsp` | Language Server Protocol implementation |
| `vexl-pkg` | Package manager (fetch, build, resolve deps) |
| `vexl-serialize` | Serialization (JSON, BSON, MessagePack) |
| `vexl-stdlib` | Standard library runtime support (Rust side) |
| `vexl-storage` | Persistent storage engine |

All stage 0 crates are substantially complete with real implementations,
tests, and documentation. The compiler can parse, type-check, lower to an
IR, optimize, generate LLVM IR, and JIT-execute or link to native
binaries.

### Stage 1: VEXL Standard Library in VEXL

Write often-used modules in `.vexl` source, compiled by the Rust
compiler. These serve as both the stdlib and as "reference code" for the
self-hosted compiler.

**Current `.vexl` files in `stdlib/`:**

| File | Lines | Purpose |
|------|-------|---------|
| `io.vexl` | 431 | I/O library (print, format, debug, pretty-print) |
| `vector.vexl` | 203 | Vector utilities (map, filter, reduce, sort, zip, partition) |
| `math.vexl` | 37 | Math constants and trigonometric intrinsics |
| `string.vexl` | ~145 | String manipulation (case, search, split, join, formatting) |
| `random.vexl` | ~105 | Random number generation (uniform, normal, shuffle, sample) |
| `time.vexl` | ~90 | Time measurement (timestamp, duration, sleep, retry) |
| `linalg.vexl` | ~195 | Linear algebra (matrix ops, dot/cross/norm, reshape) |
| `stats.vexl` | ~195 | Statistics (descriptive, correlation, z-score, window functions) |

**Intrinsic functions** (`__intrinsic_*`) are the bridging mechanism:
the `.vexl` stdlib declares pure/io functions, and the Rust runtime
supplies their implementations. This allows the stdlib to be written
entirely in VEXL while depending on a small set of native intrinsics.

### Stage 2: VEXL Parser in VEXL

Replace the chumsky-based Rust parser with a VEXL parser written in
VEXL.

**Required language features:**
- Pattern matching on AST nodes (algebraic data types)
- Recursive descent parsing combinators
- String operations (lexer) — via `string.vexl`
- I/O (file reading) — via `io.vexl`
- Error reporting with source locations

**Strategy:**
1. Write `lexer.vexl` — tokenizer producing token streams
2. Write `parser.vexl` — recursive descent parser consuming tokens
3. Validate that the VEXL parser can parse all `.vexl` stdlib files
4. Cross-validate output against the Rust parser's AST output

### Stage 3: VEXL Type Checker in VEXL

Port the type inference and effect tracking from `vexl-types` to VEXL.

**Required language features:**
- Hindley-Milner type inference (unification algorithm)
- Effect tracking (pure/io)
- Constraint generation and solving
- Recursive data structures (type representations)

**Strategy:**
1. Define type representations in VEXL
2. Implement unification algorithm
3. Implement constraint generator and solver
4. Validate against the Rust type checker on stdlib files

### Stage 4: VEXL IR Lowering and Code Generation

Port the VIR lowering and LLVM IR generation to VEXL.

**Required language features:**
- VEXL IR definitions (module, function, block, instruction)
- Pattern matching on IR nodes
- LLVM IR text generation
- Memory management for IR nodes

**Strategy:**
1. Lower VEXL AST → VIR using VEXL (replace `vexl-ir/src/lower.rs`)
2. Generate LLVM IR text from VIR using VEXL (replace `vexl-codegen`)
3. Keep the JIT engine in Rust (Stage 0) — the C FFI is complex
4. Cross-validate generated LLVM IR against Rust-produced output

### Stage 5: VEXL Compiler Bootstrapped

Once the VEXL compiler can compile itself:
1. Use the Rust compiler to compile the VEXL compiler to a native binary
2. Use that binary to compile the VEXL compiler again
3. Verify both outputs are byte-identical (reproducible build)

---

## Key Design Principles for Bootstrap

### 1. Intrinsic Bridge

Intrinsic functions provide the interface between VEXL stdlib code and
Rust runtime implementations. Each intrinsic is:

- Declared in `.vexl` as `pure fn __intrinsic_foo(...) -> T = { }`
- Implemented in Rust under `vexl-runtime` or `vexl-stdlib`
- Resolved during lowering in `vexl-ir/src/lower.rs`

**Current intrinsics used by stdlib:**
- `__intrinsic_random_float`, `__intrinsic_random_int`, `__intrinsic_seeded_random_float`
- `__intrinsic_shuffle`
- `__intrinsic_timestamp`, `__intrinsic_timestamp_ms`, `__intrinsic_sleep`

### 2. Minimal Surface Area

Keep the intrinsic surface area small. Every intrinsic is a bootstrap
dependency that must be re-implemented during self-hosting.

**Target intrinsic count for bootstrap:**
- Stage 2: 0 new intrinsics (pure VEXL parsing)
- Stage 3: 0 new intrinsics (pure VEXL type checking)
- Stage 4: 0 new intrinsics (pure VEXL code generation — text output via io.vexl)

### 3. Test-Driven Bootstrap

Each stage must have a test suite that validates output against the
Stage 0 reference implementation:

```text
Stage N output == Stage 0 output  (for all test inputs)
```

### 4. Incremental Replacement

Each stage replaces exactly one compiler phase. The rest of the pipeline
remains in Rust until its stage arrives. This ensures the compiler is
always functional.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VEXL lacks ADT support | Medium | High | Add `type` keyword for algebraic data types before Stage 2 |
| VEXL parser too slow | Low | Medium | Profile and optimize hot paths; parser speed is ~1M tokens/sec target |
| Recursive depth limits | Medium | Medium | TCO (tail call optimization) support in VEXL; iterative alternatives |
| LLVM IR generation complex | Low | Low | Generate text IR (not binary); LLVM tools handle parsing |
| Chumsky parser produces different AST | Low | Medium | Shared AST type definitions; test against same .vexl files |
| Missing features block bootstrap | Medium | High | Prioritize ADT, pattern matching, and string ops in language roadmap |

---

## Milestone Timeline

```
Stage 0 ─── Rust compiler (COMPLETE)
              │
Stage 1 ─── VEXL stdlib in VEXL (IN PROGRESS)
              │
Stage 2 ─── VEXL parser in VEXL
              │
Stage 3 ─── VEXL type checker in VEXL
              │
Stage 4 ─── VEXL codegen in VEXL
              │
Stage 5 ─── COMPILER SELF-HOSTING VERIFIED
```

Each stage is gated by the language features it requires. The language
roadmap (see `ROADMAP.md`) prioritizes features needed for bootstrap.
