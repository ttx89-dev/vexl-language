# Complete VEXL Syntax Reference

> **Every Language Construct, Every Rule, Every Possibility**

## Table of Contents

1. [Literals and Basic Types](#literals-and-basic-types)
2. [Vector Construction and Manipulation](#vector-construction-and-manipulation)
3. [Functions and Lambdas](#functions-and-lambdas)
4. [Control Flow](#control-flow)
5. [Operators](#operators)
6. [Comprehensions and Generators](#comprehensions-and-generators)
7. [Type System](#type-system)
8. [Effect System](#effect-system)
9. [Modules and Imports](#modules-and-imports)
10. [Error Handling](#error-handling)

---

## Literals and Basic Types

### Integer Literals

```vexl
// Decimal (base 10)
42
-17
0
999999999999999999999999999999

// Positive and negative integers
let positive = 123
let negative = -456

// Underscores for readability (future)
let readable = 1_000_000
```

### Float Literals

```vexl
// Standard notation
3.14
-2.5
0.0
0.123

// Scientific notation
1.5e-4
-1.2e+10
6.022e23

// Infinity and NaN (future)
let inf = Infinity
let neg_inf = -Infinity
let nan = NaN
```

### String Literals

```vexl
// Basic strings
"Hello, World!"
""

 // Multi-line strings
"This is a
multi-line
string in VEXL"

// Escape sequences
"Line 1\nLine 2"
"Tab\there"
"Quote \" inside string"
"Backslash \\ in string"
"Unicode: \u{1F600}"

// Raw strings (future)
r"This \n is not escaped"
```

### Boolean Literals

```vexl
true
false
```

### Null Literals (Future)

```vexl
null  // Represents absence of value
```

---

## Vector Construction and Manipulation

### Vector Literals

```vexl
// Empty vector
[]

// Single element
[42]

// Multiple elements
[1, 2, 3, 4, 5]

// Nested vectors
[[1, 2], [3, 4]]
[[[1]]]

// Mixed types (future)
[1, "hello", true]

// Vector with spread operator
let base = [1, 2, 3]
let extended = [...base, 4, 5]
```

### Range Expressions

```vexl
// Inclusive ranges
[1..10]              // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
[0..5]               // [0, 1, 2, 3, 4, 5]

// Open-ended ranges
[1..]                // Infinite: 1, 2, 3, 4, ...
[0..]                // Infinite: 0, 1, 2, 3, ...

// Negative ranges
[-5..5]              // [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

// Step ranges (future)
[0..10, step: 2]     // [0, 2, 4, 6, 8, 10]
[1..10, step: 3]     // [1, 4, 7, 10]
[10..0, step: -1]    // [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

### Vector Indexing

```vexl
let vector = [10, 20, 30, 40, 50]

// Positive indexing
vector[0]            // 10
vector[1]            // 20
vector[4]            // 50

// Negative indexing (future)
vector[-1]           // 50 (last element)
vector[-2]           // 40 (second to last)

// Slicing (future)
vector[1..3]         // [20, 30, 40]
vector[1..]          // [20, 30, 40, 50]
vector[..3]          // [10, 20, 30]

// Multi-dimensional indexing
let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix[0][0]         // 1
matrix[1][2]         // 6
matrix[2][1]         // 8
```

---

## Functions and Lambdas

### Function Declarations

```vexl
// Simple function
fn greet(name) {
    "Hello, " + name + "!"
}

// Multiple parameters
fn add(a, b) {
    a + b
}

// With type annotations
fn multiply(x: Int, y: Int) -> Int {
    x * y
}

fn divide(x: Float, y: Float) -> Float {
    if y == 0.0 {
        0.0  // Simplified error handling
    } else {
        x / y
    }
}

// Function with multiple expressions
fn calculate(x) {
    let squared = x * x
    let cubed = squared * x
    squared + cubed
}

// Recursive function
fn factorial(n) {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
```

### Anonymous Functions (Lambdas)

```vexl
// Single parameter
|x| x + 1
|x| x * x

// Multiple parameters
|x, y| x + y
|a, b, c| a + b + c

// With type annotations
|x: Int| x * 2
|x: Float, y: Float| x + y

// Block body
|x| {
    let y = x * 2
    y + 1
}

|x, y| {
    let sum = x + y
    let product = x * y
    sum + product
}

// Used immediately
[1, 2, 3] |> map(|x| x * 2)

// Stored in variables
let double = |x| x * 2
let add = |a, b| a + b

// Higher-order functions
let make_adder = |addend| |x| x + addend
let add_five = make_adder(5)
let result = add_five(10)  // 15
```

### Function Calls

```vexl
// Basic call
greet("Alice")

// Multiple arguments
add(5, 3)

// Named arguments (future)
create_user(name: "Bob", age: 25, email: "bob@example.com")

// Variadic functions (future)
sum(1, 2, 3, 4, 5)
max(10, 20, 30, 40, 50)

// Method calls (future)
[1, 2, 3].map(|x| x * 2)
"hello".length()
```

---

## Control Flow

### If Expressions

```vexl
// Simple if-else
if condition {
    "true branch"
} else {
    "false branch"
}

// If-else if-else chain
if x > 90 {
    "A"
} else if x > 80 {
    "B"
} else if x > 70 {
    "C"
} else {
    "F"
}

// If as expression (returns value)
let grade = if score >= 90 {
    "A"
} else if score >= 80 {
    "B"
} else {
    "C"
}

// Nested if
if x > 0 {
    if y > 0 {
        "Both positive"
    } else {
        "x positive, y negative"
    }
} else {
    "x not positive"
}

// If with complex condition
if (x > 0 && y > 0) || (x < 0 && y < 0) {
    "Same sign"
} else {
    "Different signs"
}
```

### For Expressions (Future)

```vexl
// Simple for loop
for i in [0..10] {
    print(i)
}

// For with step
for i in [0..10, step: 2] {
    print(i)
}

// For with condition
for item in items where item.active {
    process(item)
}

// For with index
for (index, item) in enumerate(items) {
    print("Item " + index + ": " + item)
}

// Nested for loops
for i in [0..3] {
    for j in [0..3] {
        print("(" + i + ", " + j + ")")
    }
}
```

### While Expressions (Future)

```vexl
// Basic while loop
let mut count = 0
while count < 10 {
    print(count)
    count = count + 1
}

// While with break
let mut i = 0
while i < 100 {
    if i == 50 {
        break
    }
    i = i + 1
}

// While with continue
let mut j = 0
while j < 10 {
    j = j + 1
    if j % 2 == 0 {
        continue
    }
    print(j)
}
```

---

## Operators

### Arithmetic Operators

```vexl
// Binary operators
1 + 2         // Addition
5 - 3         // Subtraction  
4 * 6         // Multiplication
15 / 3        // Division
17 % 5        // Modulo (remainder)

// Unary operators
-42           // Negation
+5            // Unary plus

// Operator precedence
2 + 3 * 4     // 14 (multiplication before addition)
(2 + 3) * 4   // 20 (parentheses override precedence)

// Assignment operators (future)
let mut x = 5
x = x + 3     // 8
x += 2        // 10 (future)
x -= 1        // 9 (future)
x *= 2        // 18 (future)
x /= 3        // 6 (future)
```

### Comparison Operators

```vexl
// Equality
1 == 1        // true
"hello" == "hello"  // true
[1, 2] == [1, 2]    // true

// Inequality
5 != 3        // true
"abc" != "xyz"  // true

// Ordering
5 > 3         // true
10 < 20       // true
7 >= 7        // true
15 <= 15      // true

// Chained comparisons (future)
1 < x && x < 10    // Traditional way
1 < x < 10         // Future: chained comparison
```

### Logical Operators

```vexl
// Boolean logic
true && false  // false (AND)
true || false  // true (OR)
!true          // false (NOT)

// Short-circuit evaluation
let result = dangerous_function() || safe_fallback()

// Complex logical expressions
(x > 0 && x < 10) || (x > 20 && x < 30)
!(x == 0 || x == 1)
```

### Vector Operators

```vexl
// Element-wise operations
[1, 2, 3] + [4, 5, 6]    // [5, 7, 9]
[10, 20, 30] - [1, 2, 3] // [9, 18, 27]
[1, 2, 3] * [2, 2, 2]    // [2, 4, 6]

// Scalar operations
[10, 20, 30] * 2         // [20, 40, 60]
[10, 20, 30] / 2         // [5, 10, 15]

// Broadcasting (future)
[1, 2, 3] + 10           // [11, 12, 13]

// Matrix operations
[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]  // Matrix multiplication
[1, 2, 3] · [4, 5, 6]                 // Dot product (future)
```

---

## Comprehensions and Generators

### List Comprehensions

```vexl
// Simple comprehension
[x * x | x <- [1..5]]           // [1, 4, 9, 16, 25]

// With condition
[x * x | x <- [1..10], x % 2 == 0]  // [4, 16, 36, 64, 100]

// Multiple generators
[a + b | a <- [1, 2], b <- [10, 20]]  // [11, 21, 12, 22]

// Complex condition
[person.name | person <- people, person.age >= 18, person.active]

// Nested comprehension
[x + y | x <- [1, 2], y <- [1, 2]]    // [2, 3, 3, 4]
```

### Generator Comprehensions

```vexl
// Generator expression
(x * x | x <- [1..])            // Infinite generator

// With multiple conditions
(prime | prime <- [2..], is_prime(prime))

// Generator with filters
(even_squared | even_squared <- [x * x | x <- [1..]], even_squared % 2 == 0)
```

---

## Type System

### Primitive Types

```vexl
// Integer types
let integer: Int = 42
let large_int: BigInt = 999999999999999999999999999999

// Float types
let floating: Float = 3.14
let precise: Double = 3.141592653589793

// Boolean types
let boolean: Bool = true

// String types
let text: String = "Hello"

// Unit type
let unit: Unit = ()
```

### Vector Types

```vexl
// Syntax: Vector<ElementType, Dimensions>
let vector_int: Vector<Int, 1> = [1, 2, 3]
let matrix_int: Vector<Vector<Int, 1>, 1> = [[1, 2], [3, 4]]
let tensor_3d: Vector<Vector<Vector<Float, 1>, 1>, 1> = [[[1.0]]]

// Type inference (usually automatic)
let inferred = [1, 2, 3]  // Vector<Int, 1>
let also_inferred = [[1.0, 2.0], [3.0, 4.0]]  // Vector<Vector<Float, 1>, 1>

// Generic vector type
let generic_vector: Vector<T, 1> where T: Addable = [1, 2, 3]
```

### Generic Types

```vexl
// Generic function syntax
fn identity<T>(x: T) -> T {
    x
}

fn generic_add<T: Addable>(a: T, b: T) -> T {
    a + b
}

// Generic constraints
fn max<T: Comparable>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Multiple type parameters
fn map_vector<T, U>(vector: Vector<T, 1>, func: fn(T) -> U) -> Vector<U, 1> {
    // Implementation
}
```

### Type Aliases

```vexl
// Type aliases
type Vector2D = Vector<Float, 2>
type Matrix2x2 = Vector<Vector<Float, 1>, 1>
type IntList = Vector<Int, 1>
type StringMatrix = Vector<Vector<String, 1>, 1>
```

### Record Types (Future)

```vexl
// Record type definition
type Person = {
    name: String,
    age: Int,
    email: String,
    active: Bool
}

// Record construction
let person = {
    name: "Alice",
    age: 30,
    email: "alice@example.com",
    active: true
}

// Record access
person.name
person.age

// Record with methods
type Person = {
    name: String,
    age: Int,
    
    fn greet() -> String {
        "Hello, I am " + self.name
    }
}
```

---

## Effect System

### Pure Functions

```vexl
// Pure function - no side effects, can be parallelized
pure fn calculate(x: Int) -> Int {
    x * x + 10
}

pure fn add(a: Int, b: Int) -> Int {
    a + b
}

pure fn process_data(data: Vector<Int, 1>) -> Vector<Int, 1> {
    data |> map(|x| x * 2)
    |> filter(|x| x > 0)
    |> sort()
}
```

### IO Functions

```vexl
// IO function - has side effects, runs on main thread
io fn read_file(path: String) -> String {
    // File reading implementation
}

io fn print_line(text: String) -> Unit {
    // Print to stdout
}

io fn write_file(path: String, content: String) -> Unit {
    // File writing implementation
}
```

### State Functions

```vexl
// State function - maintains mutable state
state fn counter() -> Int {
    static mut count = 0
    count += 1
    count
}

state fn cache_computation<T>(key: String, func: fn() -> T) ->
