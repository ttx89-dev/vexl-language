# VEXL Type System

> **Understanding VEXL's Revolutionary Universal Vector Type System**

## What You'll Learn

This guide explains VEXL's unique type system that treats everything as vectors, making programming more consistent and powerful.

## The Universal Type Philosophy

### Traditional Programming Types

In most languages, you have separate types:
```python
# Python - different types for different concepts
number = 42          # int
list_data = [1, 2, 3]     # list  
matrix = [[1, 2], [3, 4]]  # nested list
text = "hello"       # str
```

### VEXL's Unified Approach

In VEXL, everything is a vector:
```vexl
// Everything is Vector<T, D> where T=type, D=dimensions
let scalar = 42                     // Vector<Int, 1>
let vector = [1, 2, 3]              // Vector<Int, 1>
let matrix = [[1, 2], [3, 4]]       // Vector<Vector<Int, 1>, 1>
let tensor = [[[1, 2]], [[3, 4]]]   // Vector<Vector<Vector<Int, 1>, 1>, 1>
let text = "Hello"                  // Vector<Char, 1>
```

## Type Notation

### Vector Types

```vexl
// Syntax: Vector<ElementType, Dimensions>
let integers: Vector<Int, 1> = [1, 2, 3, 4, 5]
let floats: Vector<Float, 1> = [1.5, 2.7, 3.8]
let booleans: Vector<Bool, 1> = [true, false, true]
let texts: Vector<String, 1> = ["hello", "world", "vexl"]

// Multi-dimensional types
let matrix_2x2: Vector<Vector<Int, 1>, 1> = [[1, 2], [3, 4]]
let tensor_3d: Vector<Vector<Vector<Float, 1>, 1>, 1> = [[[1.0, 2.0]], [[3.0, 4.0]]]
```

### Primitive Types

```vexl
// Integer types
let small_int: Int = 42
let large_int: BigInt = 999999999999999999999999999999

// Float types
let single: Float = 3.14
let double: Double = 3.141592653589793

// Boolean
let flag: Bool = true

// String (which is Vector<Char, 1>)
let message: String = "Hello, World!"

// Unit (empty vector)
let empty: Unit = []
```

## Type Inference

### Automatic Inference

VEXL automatically figures out types in most cases:

```vexl
// VEXL infers these automatically
let number = 42                                    // Int
let decimal = 3.14                                 // Float  
let flag = true                                    // Bool
let list_data = [1, 2, 3]                          // Vector<Int, 1>
let matrix = [[1.0, 2.0], [3.0, 4.0]]             // Vector<Vector<Float, 1>, 1>
let text = "hello"                                 // String (Vector<Char, 1>)

// Complex inference
let result = [1, 2, 3] |> map(|x| x * 2) |> sum()  // Int
```

### When to Use Type Annotations

```vexl
// When type might be unclear
let data: Vector<Int, 1> = []  // Empty vector needs annotation

// When you want to be explicit
fn process_data(data: Vector<Float, 1>) -> Vector<Float, 1> {
    data |> map(|x| x * 2.0)
}

// For generic functions
fn identity<T>(x: T) -> T {
    x
}
```

## Shape Safety

### Dimensional Checking

VEXL prevents impossible operations at compile time:

```vexl
// This works - same dimensions
let vector_a = [1, 2, 3]
let vector_b = [4, 5, 6]
let sum = vector_a + vector_b  // [5, 7, 9]

// This fails - different dimensions
let vector_c = [1, 2, 3, 4]
let vector_d = [5, 6, 7]
// let invalid_sum = vector_c + vector_d  // Compile error!

// Matrix multiplication - shape must match
let matrix_a = [[1, 2], [3, 4]]        // 2x2
let matrix_b = [[5, 6], [7, 8]]        // 2x2
let product = matrix_a @ matrix_b      // 2x2 result

// This fails - incompatible matrix dimensions
let matrix_c = [[1, 2, 3]]             // 1x3
let matrix_d = [[4, 5], [6, 7]]        // 2x2
// let invalid_product = matrix_c @ matrix_d  // Compile error!
```

### Broadcasting (Future)

```vexl
// Future: automatic broadcasting
let vector = [1, 2, 3]
let scalar = 10
let result = vector + scalar  // [11, 12, 13] (broadcasts scalar)
```

## Generic Types

### Generic Functions

```vexl
// Type parameter T
fn identity<T>(x: T) -> T {
    x
}

// Generic vector operations
fn vector_length<T>(vector: Vector<T, 1>) -> Int {
    vector |> length()
}

fn vector_sum<T>(vector: Vector<T, 1>) -> T {
    vector |> sum()
}

// Multiple type parameters
fn map_vector<T, U>(vector: Vector<T, 1>, func: fn(T) -> U) -> Vector<U, 1> {
    vector |> map(func)
}
```

### Generic Constraints (Future)

```vexl
// Type class constraints
fn add<T: Addable>(a: T, b: T) -> T {
    a + b
}

fn max<T: Comparable>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Multiple constraints
fn sort_vector<T: Addable + Comparable>(vector: Vector<T, 1>) -> Vector<T, 1> {
    vector |> sort()
}
```

## Type Aliases

### Useful Abbreviations

```vexl
// Common type aliases
type Vector2D = Vector<Float, 2>
type Vector3D = Vector<Float, 3>
type Matrix2x2 = Vector<Vector<Float, 1>, 1>
type Matrix3x3 = Vector<Vector<Float, 1>, 1>
type IntList = Vector<Int, 1>
type StringList = Vector<String, 1>
type BoolMatrix = Vector<Vector<Bool, 1>, 1>

// Using aliases
let position: Vector2D = [10.5, 20.3]
let rotation: Vector3D = [0.0, 0.0, 90.0]
let transform: Matrix2x2 = [[1, 0], [0, 1]]
```

## Effect Types

### Pure Functions

```vexl
// Pure functions have no side effects
pure fn calculate(x: Int) -> Int {
    x * x + 10
}

pure fn process_data(data: Vector<Int, 1>) -> Vector<Int, 1> {
    data |> map(|x| x * 2)
    |> filter(|x| x > 0)
}
```

### IO Functions

```vexl
// IO functions have side effects
io fn read_file(path: String) -> String {
    // Implementation reads from filesystem
}

io fn print_data(data: Vector<Int, 1>) -> Unit {
    // Implementation prints to screen
}
```

### State Functions

```vexl
// State functions maintain mutable state
state fn counter() -> Int {
    static mut count = 0
    count += 1
    count
}

state fn cache<T>(key: String, func: fn() -> T) -> T {
    // Implementation caches results
}
```

## Type Checking Rules

### Compatible Operations

```vexl
// Arithmetic requires same numeric type
let ints = [1, 2, 3] + [4, 5, 6]          // Vector<Int, 1>
let floats = [1.0, 2.0] + [3.0, 4.0]      // Vector<Float, 1>

// Comparison requires same type
let same_type = [1, 2] == [1, 2]          // Bool
// let different_type = [1, 2] == [1.0, 2.0]  // Compile error!

// Logical operations require Bool
let bool_result = true && false           // Bool
// let mixed_types = 1 && true             // Compile error!
```

### Type Promotion

```vexl
// Integer to Float promotion in mixed expressions
let result = [1, 2, 3] |> map(|x| x * 2.5)  // Vector<Float, 1>

// Automatic conversion
let int_vector = [1, 2, 3]
let float_vector = [1.0, 2.0, 3.0]
let promoted = int_vector + float_vector   // Vector<Float, 1>
```

## Advanced Type Features

### Recursive Types (Future)

```vexl
// Self-referential types
type BinaryTree<T> = 
    | Node(T, BinaryTree<T>, BinaryTree<T>)
    | Leaf

type List<T> = 
    | Cons(T, List<T>)
    | Nil
```

### Sum Types (Future)

```vexl
// Algebraic data types
type Result<T, E> = 
    | Ok(T)
    | Err(E)

type Option<T> = 
    | Some(T)
    | None
```

### Existential Types (Future)

```vexl
// Type existential quantification
type Container = exists T. { value: T, process: fn(T) -> String }
```

## Best Practices

### When to Use Type Annotations

```vexl
// Good: Function signatures
fn calculate_mean(data: Vector<Float, 1>) -> Float {
    data |> sum() / (data |> length())
}

// Good: Empty collections
let empty_ints: Vector<Int, 1> = []

// Good: Generic functions
fn map_vector<T, U>(vector: Vector<T, 1>, func: fn(T) -> U) -> Vector<U, 1> {
    vector |> map(func)
}

// Avoid: Unnecessary annotations
let x: Int = 42  // Redundant - VEXL infers this
```

### Type Safety Guidelines

```vexl
// Use shape safety to catch errors early
fn matrix_multiply(a: Vector<Vector<Float, 1>, 1>, b: Vector<Vector<Float, 1>, 1>) -> Vector<Vector<Float, 1>, 1> {
    // VEXL will catch dimension mismatches at compile time
    let rows_a = a |> length()
    let cols_a = a[0] |> length()
    let cols_b = b[0] |> length()
    
    // This will fail to compile if dimensions don't match
    [[a[i][k] * b[k][j] | k <- [0..cols_a], j <- [0..cols_b]] | i <- [0..rows_a]]
}
```

## Troubleshooting

### Common Type Errors

```vexl
// Error: Cannot add vectors of different lengths
let a = [1, 2, 3]
let b = [1, 2]
// let sum = a + b  // Error!

// Solution: Ensure same dimensions
let c = [1, 2, 3]
let d = [4, 5, 6]
let result = c + d  // OK

// Error: Type mismatch in operations
let text = "42"
let number = 42
// let sum = text + number  // Error!

// Solution: Convert types explicitly
let text_number = "42" |> parse_int()  // Future: parse function
let sum = text_number + number  // OK

// Error: Cannot index with wrong type
let data = [1, 2, 3, 4, 5]
// let value = data["hello"]  // Error!

// Solution: Use valid index type
let value = data[2]  // OK - index 2
```

## What's Next?

Now that you understand VEXL's type system:

1. [Learn Vector Operations](vector-operations.md) - Master vector manipulation
2. [Explore Functions](functions.md) - Understand function types and effects
3. [Study Control Flow](control-flow.md) - Learn about pattern matching and control
4. [Understand Effects](effects.md) - Master pure, IO, and state functions

---

**Next:** [Vector Operations](vector-operations.md)
