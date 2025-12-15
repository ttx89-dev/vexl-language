# VEXL Syntax Reference

> **Complete Language Grammar and Syntax Rules**

## Overview

This reference covers every syntactic element in VEXL. Use this as a comprehensive guide to the language grammar and structure.

## Lexical Structure

### Comments

```vexl
// Single-line comment
/* Multi-line comment
   that can span
   multiple lines */
```

### Identifiers

```vexl
// Valid identifiers
let variable_name = 42
let snake_case = "hello"
let camelCase = 123
let _underscore_start = true
let UPPERCASE = 999

// Invalid identifiers
let 123invalid = 1          // Starts with number
let invalid-name = 2        // Contains hyphen
let let = 3                 // Reserved keyword
```

### Keywords

```
fn, let, if, else, for, in, match, pure, io, state
true, false, null
```

### Literals

#### Integer Literals

```vexl
// Decimal integers
42
-17
0
999999

// Binary literals (future)
0b1010
0B1111

// Hexadecimal literals (future)
0xFF
0x1234ABCD
```

#### Float Literals

```vexl
// Decimal floats
3.14
-2.5
0.0
1.5e-4
-1.2e+10
```

#### String Literals

```vexl
// Basic strings
"Hello, World!"
"Multi-line
string is supported"

// Escape sequences
"Line 1\nLine 2"
"Tab\there"
"Quote \" inside string"
```

#### Boolean Literals

```vexl
true
false
```

## Types and Type Annotations

### Primitive Types

```vexl
// Integer types
let integer: Int = 42

// Float types  
let floating: Float = 3.14

// Boolean types
let boolean: Bool = true

// String types
let text: String = "Hello"
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
```

### Generic Types

```vexl
// Generic function syntax
fn identity<T>(x: T) -> T {
    x
}

// Generic constraints (future)
fn add<T: Addable>(a: T, b: T) -> T {
    a + b
}
```

## Expressions

### Primary Expressions

#### Literals

```vexl
42                  // Integer literal
3.14                // Float literal
"Hello"             // String literal
true                // Boolean literal
[1, 2, 3]           // Vector literal
```

#### Identifiers

```vexl
variable_name
function_name
Module.submodule
```

#### Parenthesized Expressions

```vexl
(42)                    // Just wraps a value
(1 + 2) * 3             // Controls precedence
(let x = 5 in x + 1)    // Scoped expression
```

### Vector Expressions

#### Vector Construction

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
```

#### Range Expressions

```vexl
// Finite ranges
[1..10]                // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
[0..5]                 // [0, 1, 2, 3, 4, 5]

// Infinite ranges
[1..]                  // 1, 2, 3, 4, ...
[0..]                  // 0, 1, 2, 3, ...

// Step ranges (future)
[0..10, step: 2]       // [0, 2, 4, 6, 8, 10]
[1..10, step: 3]       // [1, 4, 7, 10]
```

#### Vector Indexing

```vexl
let vector = [10, 20, 30, 40]
vector[0]              // 10
vector[1]              // 20
vector[-1]             // 40 (future: negative indexing)

let matrix = [[1, 2], [3, 4]]
matrix[0][0]           // 1
matrix[1][1]           // 4
```

### Function Expressions

#### Named Functions

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

// With body expressions
fn calculate(x) {
    let y = x * 2
    y + 1
}
```

#### Anonymous Functions (Lambdas)

```vexl
// Single parameter
|x| x + 1

// Multiple parameters
|x, y| x + y

// With type annotations
|x: Int| x * 2

// Complex body
|x| {
    let y = x * 2
    y + 1
}
```

#### Function Calls

```vexl
// Basic call
greet("Alice")

// Multiple arguments
add(5, 3)

// With vector argument
sum([1, 2, 3])

// Chained calls
map(double, filter(is_even, [1, 2, 3, 4]))

// Method-style call (future)
[1, 2, 3].map(|x| x * 2)
```

### Control Flow Expressions

#### If Expressions

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
} else {
    "B"
}
```

#### For Expressions (Future)

```vexl
// Traditional for loop (planned)
for i in [0..10] {
    print(i)
}

// For with condition (planned)
for item in items where item.active {
    process(item)
}
```

#### While Expressions (Future)

```vexl
// While loop (planned)
let mut count = 0
while count < 10 {
    count = count + 1
}
```

### Pattern Matching (Future)

```vexl
// Match expression (planned)
match value {
    0 => "zero",
    1 => "one",
    n if n > 10 => "large",
    _ => "other"
}

// Match with vectors (planned)
match numbers {
    [] => "empty",
    [x] => "single: " + x,
    [x, y] => "pair: " + x + ", " + y,
    _ => "many"
}
```

### Let Expressions

```vexl
// Simple binding
let x = 42

// Multiple bindings
let a = 1
let b = 2
let c = a + b

// With type annotation
let name: String = "Alice"

// Scoped binding
let x = 5 in x * 2

// Pattern matching (future)
let [first, second, ...rest] = [1, 2, 3, 4, 5]
```

### Operator Expressions

#### Arithmetic Operators

```vexl
// Binary operators
1 + 2         // Addition
5 - 3         // Subtraction
4 * 6         // Multiplication
15 / 3        // Division
17 % 5        // Modulo

// Unary operators
-42           // Negation
+5            // Positive (unary plus)
```

#### Comparison Operators

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
```

#### Logical Operators

```vexl
// Boolean logic
true && false  // false (AND)
true || false  // true (OR)
!true          // false (NOT)

// Short-circuit evaluation
let result = dangerous_function() || safe_fallback()
```

#### Vector Operators

```vexl
// Element-wise operations
[1, 2, 3] + [4, 5, 6]    // [5, 7, 9]
[10, 20, 30] * 2         // [20, 40, 60]

// Broadcasting (future)
[1, 2, 3] + 10           // [11, 12, 13]

// Matrix operations
[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]  // Matrix multiplication
```

### Comprehensions

#### List Comprehensions

```vexl
// Simple comprehension
[x * x | x <- [1..5]]           // [1, 4, 9, 16, 25]

// With condition
[x * x | x <- [1..10], x % 2 == 0]  // [4, 16, 36, 64, 100]

// Multiple generators
[a + b | a <- [1, 2], b <- [10, 20]]  // [11, 21, 12, 22]

// Complex condition
[person.name | person <- people, person.age >= 18, person.active]
```

#### Generator Comprehensions

```vexl
// Generator expression
(x * x | x <- [1..])            // Infinite generator

// With multiple conditions
(prime | prime <- [2..], is_prime(prime))
```

### Pipe Expressions

```vexl
// Basic pipe
data |> map(|x| x * 2)          // Pass data through map

// Chain operations
[1, 2, 3, 4, 5]
    |> filter(|x| x % 2 == 0)
    |> map(|x| x * x)
    |> sum()

// Function composition
(f |> g |> h)(input)            // Apply functions in sequence
```

### Effect Expressions

#### Pure Functions

```vexl
pure fn calculate(x) {
    x * x + 10
}

pure fn add(a, b) {
    a + b
}
```

#### IO Functions

```vexl
io fn read_file(path) {
    // File reading operations
}

io fn print_line(text) {
    // Print to stdout
}
```

#### State Functions

```vexl
state fn counter() {
    static mut count = 0
    count += 1
    count
}
```

## Statements

### Declaration Statements

#### Function Declarations

```vexl
fn function_name(parameters) {
    // function body
}

fn typed_function(x: Int, y: Float) -> String {
    // typed body
}
```

#### Type Declarations (Future)

```vexl
// Type alias
type Vector2D = Vector<Float, 2>

// Record type (planned)
type Person = {
    name: String,
    age: Int,
    email: String
}
```

#### Import Statements (Future)

```vexl
// Module import
import "math.vexl"

// Specific item import
import { sin, cos, pi } from "math.vexl"

// Rename import
import { sqrt as square_root } from "math.vexl"
```

### Binding Statements

```vexl
// Simple binding
let variable = value

// Type annotation
let typed_variable: Type = value

// Multiple binding (future)
let (x, y) = coordinate
let [first, ...rest] = list
```

### Expression Statements

```vexl
// Any expression can be a statement
42;
"Hello";
[1, 2, 3];
function_call();
```

## Modules and Organization

### Module Structure (Future)

```vexl
// Module file: math.vexl
pub fn add(a, b) { a + b }
pub fn subtract(a, b) { a - b }

let pi = 3.14159
let e = 2.71828

// Private function (not exported)
fn internal_function(x) { x * 2 }
```

### Module Usage

```vexl
// Import entire module
import "math.vexl"
let result = math.add(5, 3)

// Import specific items
import { add, pi } from "math.vexl"
let result = add(5, 3)

// Module alias
import { add as plus } from "math.vexl"
let result = plus(5, 3)
```

## Error Handling (Future)

### Result Type

```vexl
// Result type syntax (planned)
type Result<T, E> = Ok(T) | Err(E)

// Using Result
fn divide(a: Int, b: Int) -> Result<Int, String> {
    if b == 0 {
        Err("Division by zero")
    } else {
        Ok(a / b)
    }
}

// Pattern matching on Result
match divide(10, 2) {
    Ok(result) => print("Result: " + result),
    Err(error) => print("Error: " + error)
}
```

### Option Type

```vexl
// Option type syntax (planned)
type Option<T> = Some(T) | None

// Using Option
fn find_item(items, target) -> Option<Int> {
    // Search logic
    if found {
        Some(index)
    } else {
        None
    }
}

// Pattern matching on Option
match find_item([1, 2, 3], 2) {
    Some(index) => print("Found at index: " + index),
    None => print("Not found")
}
```

## Advanced Features

### Recursion

```vexl
// Recursive function
fn factorial(n) {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

// Mutual recursion (future)
fn is_even(n) {
    if n == 0 { true } else { is_odd(n - 1) }
}

fn is_odd(n) {
    if n == 0 { false } else { is_even(n - 1) }
}
```

### Higher-Order Functions

```vexl
// Function returning function
fn make_multiplier(factor) {
    |x| x * factor
}

let double = make_multiplier(2)
let triple = make_multiplier(3)

let result = double(5)    // 10
let result2 = triple(5)   // 15

// Function taking function
fn apply_twice(f, x) {
    f(f(x))
}

let result = apply_twice(|x| x + 1, 5)  // 7
```

### Closures

```vexl
// Closure capturing environment
let multiplier = 3
let times_three = |x| x * multiplier

let result = times_three(5)  // 15

// Closure with mutable capture (future)
let mut counter = 0
let increment = || {
    counter += 1
    counter
}
```

## Grammar Summary

```
program ::= statement*

statement ::= declaration
            | binding
            | expression ";"

declaration ::= "fn" identifier "(" parameters? ")" ("->" type)? "{" expression* "}"
              | "pub" declaration
              | "pure" declaration
              | "io" declaration
              | "state" declaration

binding ::= "let" identifier (":" type)? "=" expression
          | "let" pattern "=" expression

expression ::= literal
             | identifier
             | vector_literal
             | range_literal
             | "(" expression ")"
             | function_call
             | lambda_expression
             | if_expression
             | let_expression
             | binary_expression
             | unary_expression
             | pipe_expression
             | comprehension
             | match_expression

literal ::= integer_literal
          | float_literal
          | string_literal
          | boolean_literal

vector_literal ::= "[" (expression ("," expression)*)? "]"

range_literal ::= "[" expression ".." expression? "]"

function_call ::= expression "(" arguments? ")"

lambda_expression ::= "|" parameters? "|" expression
                    | "|" parameters? "|" "{" expression* "}"

if_expression ::= "if" expression "{" expression* "}" 
                 ("else" "{" expression* "}")?
                 ("else" "if" expression "{" expression* "}" )*

let_expression ::= "let" binding "in" expression

binary_expression ::= expression operator expression

unary_expression ::= ("-" | "+" | "!") expression

pipe_expression ::= expression "|>" expression

comprehension ::= "[" expression "|" generators "]"
                | "(" expression "|" generators ")"

generators ::= generator ("," generator)*
generator ::= identifier "<-" expression ("," condition)*

match_expression ::= "match" expression "{" match_arm* "}"
match_arm ::= pattern "=>" expression ("," if_expression
