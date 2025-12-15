# Basic Concepts

> **Understanding VEXL Fundamentals: Everything is a Vector**

## What You'll Learn

This guide introduces the fundamental concepts that make VEXL unique. By the end, you'll understand:
- The universal vector concept
- Type system basics
- Basic operations and syntax
- How VEXL thinks about data

## The Universal Vector

### The Core Idea

In traditional programming languages, you have different types for different things:
- **Integers** for whole numbers
- **Arrays** for lists
- **Matrices** for 2D data
- **Objects** for complex structures

In VEXL, **everything is a vector**. This is not just a slogan—it's a fundamental design principle that simplifies programming.

### Understanding Vectors

A **vector** is a container that holds elements in order. But in VEXL, vectors are more flexible than in other languages:

```vexl
// A single number (scalar) is a 1-dimensional vector of size 1
let single_number = 42           // Vector<Int, 1>

// A list is a 1-dimensional vector  
let numbers = [1, 2, 3, 4, 5]    // Vector<Int, 1>

// A matrix is a 2-dimensional vector
let matrix = [[1, 2], [3, 4]]    // Vector<Vector<Int, 1>, 1>

// Even complex structures are vectors
let data_table = [
    ["Alice", 25, "Engineer"],
    ["Bob", 30, "Designer"],
    ["Carol", 28, "Manager"]
]  // Vector<Vector<String|Int, 1>, 1>
```

### Why This Matters

**Same Operations Work Everywhere:**
```vexl
// These all work the same way!
let doubled_single = 5 |> map(|x| x * 2)        // 10
let doubled_vector = [1, 2, 3] |> map(|x| x * 2)  // [2, 4, 6]
let doubled_matrix = [[1, 2], [3, 4]] |> map(|row| row |> map(|x| x * 2))  // [[2, 4], [6, 8]]
```

**No Special Cases:**
- No separate functions for scalars vs arrays
- Same logic applies at every level
- Mathematical consistency throughout

## Type System Basics

### Primitive Types

```vexl
// Integers
let whole_number = 42
let negative = -17

// Floating-point numbers  
let decimal = 3.14159
let scientific = 1.5e-4

// Boolean values
let is_true = true
let is_false = false

// Text strings
let greeting = "Hello, World!"
let multi_line = "This is a
multi-line
string"
```

### Vector Types

```vexl
// Vector notation: Vector<ElementType, Dimensions>
let integers = [1, 2, 3]                     // Vector<Int, 1>
let floats = [1.5, 2.7, 3.8]                 // Vector<Float, 1>
let booleans = [true, false, true]           // Vector<Bool, 1>
let texts = ["hello", "world"]               // Vector<String, 1>

// Nested vectors (higher dimensions)
let matrix = [[1, 2], [3, 4]]                // Vector<Vector<Int, 1>, 1>
let tensor = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  // Vector<Vector<Vector<Int, 1>, 1>, 1>
```

### Type Inference

VEXL automatically figures out types—you usually don't need to write them:

```vexl
// VEXL infers these types automatically
let x = 42                                    // Int
let pi = 3.14                                 // Float
let name = "Alice"                            // String
let numbers = [1, 2, 3]                       // Vector<Int, 1>
let result = numbers |> sum()                 // Int

// You can still specify types explicitly if needed
let explicit_int: Int = 42
let explicit_vector: Vector<Int, 1> = [1, 2, 3]
```

## Basic Operations

### Arithmetic Operations

```vexl
// Basic math
1 + 2                 // 3 (addition)
5 - 3                 // 2 (subtraction)
4 * 6                 // 24 (multiplication)
15 / 3                // 5 (division)
17 % 5                // 2 (modulo)

// Vector arithmetic
[1, 2, 3] + [4, 5, 6]     // [5, 7, 9]
[10, 20, 30] * 2          // [20, 40, 60]
```

### Comparison Operations

```vexl
// Equality
1 == 1                 // true
"hello" == "hello"     // true
[1, 2] == [1, 2]       // true

// Inequality  
5 != 3                 // true
"abc" != "xyz"         // true

// Ordering
5 > 3                  // true
10 < 20                // true
7 >= 7                 // true
15 <= 15               // true
```

### Logical Operations

```vexl
// Boolean logic
true && false          // false (AND)
true || false          // true (OR)
!true                  // false (NOT)

// Combining conditions
let age = 25
let is_adult = age >= 18 && age < 65
```

## Variables and Bindings

### Let Bindings

```vexl
// Simple assignment
let name = "Alice"
let age = 25

// Vector assignment
let scores = [85, 92, 78, 90]

// Computed values
let full_name = "Alice " + "Johnson"
let is_even = 42 % 2 == 0

// Multiple bindings
let width = 10
let height = 20
let area = width * height
```

### Shadowing

```vexl
let x = 5
let x = x * 2        // x is now 10 (previous x is replaced)
let x = x + 1        // x is now 11
```

## Functions

### Named Functions

```vexl
// Simple function
fn greet(name) {
    "Hello, " + name + "!"
}

let message = greet("Alice")     // "Hello, Alice!"

// Function with multiple parameters
fn add(a, b) {
    a + b
}

let sum = add(5, 3)              // 8

// Function with calculations
fn calculate_area(width, height) {
    width * height
}

let rectangle_area = calculate_area(10, 5)  // 50
```

### Anonymous Functions (Lambdas)

```vexl
// Single parameter
|x| x + 1

// Multiple parameters  
|a, b| a + b

// Used immediately
let result = [1, 2, 3] |> map(|x| x * 2)  // [2, 4, 6]

// Stored in variables
let double = |x| x * 2
let doubled = double(5)                    // 10
```

## Control Flow

### If Expressions

```vexl
// Simple if
let x = 10
let result = if x > 5 {
    "greater than 5"
} else {
    "not greater than 5"
}

// If-else if-else
let grade = 85
let letter = if grade >= 90 {
    "A"
} else if grade >= 80 {
    "B"  
} else if grade >= 70 {
    "C"
} else {
    "F"
}
```

### Pattern Matching (Coming Soon)

```vexl
// This syntax is planned for future versions
match value {
    0 => "zero",
    1 => "one", 
    _ => "other"
}
```

## Data Flow with Pipes

### The Pipe Operator

The `|>` operator is one of VEXL's most powerful features. It passes data through a series of functions:

```vexl
// Without pipes (nested function calls)
let result = sum(map(double, filter(even, [1, 2, 3, 4, 5])))

// With pipes (data flows left to right)
let result = [1, 2, 3, 4, 5]
    |> filter(|x| x % 2 == 0)     // [2, 4]
    |> map(|x| x * 2)             // [4, 8]  
    |> sum()                      // 12

// Complex example
let processed = data
    |> filter(|x| x > 0)
    |> map(|x| x * x)
    |> sort()
    |> take(10)
    |> sum()
```

## Ranges and Sequences

### Finite Ranges

```vexl
// Simple ranges
[1..10]                    // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
[0..5]                     // [0, 1, 2, 3, 4, 5]

// Ranges with different starts
[5..10]                    // [5, 6, 7, 8, 9, 10]
[-2..2]                    // [-2, -1, 0, 1, 2]
```

### Infinite Ranges

```vexl
// Infinite ranges (generators)
[1..]                      // 1, 2, 3, 4, 5, ...
[0..]                      // 0, 1, 2, 3, 4, ...

// Using infinite ranges
let first_10 = [1..] |> take(10)           // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let evens = [0..] |> filter(|x| x % 2 == 0)  // 0, 2, 4, 6, 8, ...
```

### Step Ranges

```vexl
// Future syntax (planned)
[0..10, step: 2]           // [0, 2, 4, 6, 8, 10]
[1..10, step: 3]           // [1, 4, 7, 10]
```

## Common Patterns

### Map-Filter-Reduce

```vexl
// Transform, filter, then combine
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

let result = numbers
    |> filter(|x| x % 2 == 0)          // Keep only evens: [2, 4, 6, 8, 10]
    |> map(|x| x * x)                  // Square them: [4, 16, 36, 64, 100]
    |> sum()                           // Add them up: 220
```

### Statistics Calculation

```vexl
fn calculate_stats(data) {
    let n = data |> length()
    let sum = data |> sum()
    let mean = sum / n
    
    let squared_diffs = data 
        |> map(|x| (x - mean) * (x - mean))
    let variance = squared_diffs |> sum() / n
    let std_dev = variance |> sqrt()
    
    {count: n, sum: sum, mean: mean, variance: variance, std_dev: std_dev}
}

let stats = calculate_stats([1, 2, 3, 4, 5])
```

### Nested Operations

```vexl
// Working with matrices
let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

// Sum of each row
let row_sums = matrix |> map(|row| row |> sum())  // [6, 15, 24]

// Sum of each column
let col_sums = [0, 1, 2] |> map(|col| 
    matrix |> map(|row| row[col]) |> sum()
)  // [12, 15, 18]
```

## Best Practices

### Use Descriptive Names

```vexl
// Good
let student_scores = [85, 92, 78, 90]
let average_score = student_scores |> sum() / (student_scores |> length())

// Avoid
let x = [85, 92, 78, 90]
let y = x |> sum() / (x |> length())
```

### Break Complex Operations

```vexl
// Complex operation
let result = data |> filter(|x| x > 0) |> map(|x| x * 2) |> sum()

// Better: Break into steps
let positive_data = data |> filter(|x| x > 0)
let doubled_data = positive_data |> map(|x| x * 2)
let final_sum = doubled_data |> sum()
```

### Use Type Annotations When Helpful

```vexl
// When type might be unclear
let numbers: Vector<Int, 1> = [1, 2, 3]

// When you want to be explicit
fn process_data(data: Vector<Float, 1>) -> Vector<Float, 1> {
    data |> map(|x| x * 2.0)
}
```

## Common Mistakes

### Forgetting Brackets

```vexl
// Wrong
let numbers = (1, 2, 3)

// Right
let numbers = [1, 2, 3]
```

### Mixing Up Pipe Direction

```vexl
// Wrong
map(|x| x * 2, [1, 2, 3])

// Right  
[1, 2, 3] |> map(|x| x * 2)
```

### Type Mismatches

```vexl
// Wrong
let result = "42" + 8

// Right
let result = 42 + 8  // 50
// OR
let result = "42" + "8"  // "42"
```

## What's Next?

Now that you understand the basics:

1. [Follow the Tutorial](tutorial.md) - Step-by-step learning
2. [Explore Language Features](../language-guide/) - Deep dive into VEXL
3. [Try Examples](../examples/) - See real-world usage

---

**Next:** [Tutorial](tutorial.md)
