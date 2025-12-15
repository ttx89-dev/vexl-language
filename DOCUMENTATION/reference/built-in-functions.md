# VEXL Built-in Functions Reference

> **Complete Reference to VEXL's Core Functions**

## Overview

This reference covers all built-in functions available in VEXL. These functions are automatically available in all VEXL programs without needing imports.

## Vector Operations

### Creation Functions

```vexl
// Create empty vector
empty() -> Vector<Unit, 1>
let empty_vec = empty()  // []

// Create vector with repeated value
repeat(value, count) -> Vector<T, 1>
let fives = repeat(5, 3)  // [5, 5, 5]

// Create vector from range
range(start, end) -> Vector<Int, 1>
let numbers = range(1, 5)  // [1, 2, 3, 4, 5]

// Create vector with custom step
range_with_step(start, end, step) -> Vector<Int, 1>
let evens = range_with_step(0, 10, 2)  // [0, 2, 4, 6, 8]

// Generate vector from function
generate(count, func) -> Vector<T, 1>
let squares = generate(5, |i| i * i)  // [0, 1, 4, 9, 16]
```

### Transformation Functions

```vexl
// Apply function to each element
map(vector, func) -> Vector<U, 1>
let doubled = map([1, 2, 3], |x| x * 2)  // [2, 4, 6]

// Filter elements by predicate
filter(vector, predicate) -> Vector<T, 1>
let evens = filter([1, 2, 3, 4], |x| x % 2 == 0)  // [2, 4]

// Transform and filter in one pass
map_filter(vector, func) -> Vector<U, 1>
let positive_squares = map_filter([-2, -1, 0, 1, 2], |x| if x > 0 { Some(x * x) } else { None })

// Flatten nested vectors
flatten(vector) -> Vector<T, 1>
let flat = flatten([[1, 2], [3, 4], [5, 6]])  // [1, 2, 3, 4, 5, 6]

// Sort vector
sort(vector) -> Vector<T, 1>
let sorted = sort([3, 1, 4, 1, 5])  // [1, 1, 3, 4, 5]

// Sort by custom key
sort_by(vector, key_func) -> Vector<T, 1>
let sorted_by_length = sort_by(["a", "abc", "ab"], |s| s |> length())  // ["a", "ab", "abc"]

// Reverse vector
reverse(vector) -> Vector<T, 1>
let reversed = reverse([1, 2, 3])  // [3, 2, 1]
```

### Reduction Functions

```vexl
// Sum all elements
sum(vector) -> T
let total = sum([1, 2, 3, 4, 5])  // 15

// Product of all elements
product(vector) -> T
let prod = product([2, 3, 4])  // 24

// Count elements
length(vector) -> Int
let len = length([1, 2, 3, 4])  // 4

// Find minimum element
min(vector) -> T
let minimum = min([3, 1, 4, 1, 5])  // 1

// Find maximum element
max(vector) -> T
let maximum = max([3, 1, 4, 1, 5])  // 5

// Reduce with binary function
reduce(vector, initial, func) -> T
let sum_alt = reduce([1, 2, 3, 4], 0, |acc, x| acc + x)  // 10

// Check if any element satisfies predicate
any(vector, predicate) -> Bool
let has_even = any([1, 3, 5, 7], |x| x % 2 == 0)  // false

// Check if all elements satisfy predicate
all(vector, predicate) -> Bool
let all_positive = all([1, 2, 3, 4], |x| x > 0)  // true

// Find first element matching predicate
find(vector, predicate) -> Option<T>
let first_even = find([1, 3, 5, 7, 8], |x| x % 2 == 0)  // Some(8)

// Find index of first matching element
find_index(vector, predicate) -> Option<Int>
let index_of_even = find_index([1, 3, 5, 7, 8], |x| x % 2 == 0)  // Some(4)
```

### Indexing Functions

```vexl
// Get element at index
at(vector, index) -> T
let third = at([1, 2, 3, 4, 5], 2)  // 3

// Get first element
first(vector) -> T
let head = first([1, 2, 3])  // 1

// Get last element
last(vector) -> T
let tail = last([1, 2, 3])  // 3

// Get element or default
get_or_default(vector, index, default) -> T
let safe = get_or_default([1, 2, 3], 10, 0)  // 0

// Slice vector
slice(vector, start, end) -> Vector<T, 1>
let sub = slice([1, 2, 3, 4, 5], 1, 3)  // [2, 3]
```

## Mathematical Functions

### Basic Math

```vexl
// Absolute value
abs(x) -> T
let pos = abs(-5)  // 5

// Square root
sqrt(x) -> Float
let root = sqrt(16)  // 4.0

// Power
pow(base, exponent) -> Float
let powered = pow(2, 3)  // 8.0

// Round to nearest integer
round(x) -> Int
let rounded = round(3.7)  // 4

// Round down (floor)
floor(x) -> Int
let floored = floor(3.7)  // 3

// Round up (ceiling)
ceil(x) -> Int
let ceiled = ceil(3.2)  // 4

// Truncate (remove decimal part)
truncate(x) -> Int
let truncated = truncate(3.7)  // 3

// Sign of number
sign(x) -> Int
let sgn = sign(-5)  // -1
```

### Trigonometric Functions

```vexl
// Sine
sin(angle) -> Float
let s = sin(3.14159 / 2)  // ~1.0

// Cosine
cos(angle) -> Float
let c = cos(0)  // 1.0

// Tangent
tan(angle) -> Float
let t = tan(0.785398)  // ~1.0

// Inverse sine
asin(value) -> Float
let arcs = asin(1.0)  // ~1.5708

// Inverse cosine
acos(value) -> Float
let arcc = acos(1.0)  // 0.0

// Inverse tangent
atan(value) -> Float
let arct = atan(1.0)  // ~0.7854

// Convert degrees to radians
to_radians(degrees) -> Float
let rad = to_radians(180)  // ~3.14159

// Convert radians to degrees
to_degrees(radians) -> Float
let deg = to_degrees(3.14159)  // ~180.0
```

### Statistical Functions

```vexl
// Calculate mean
mean(vector) -> Float
let avg = mean([1, 2, 3, 4, 5])  // 3.0

// Calculate median
median(vector) -> Float
let mid = median([1, 3, 5])  // 3.0

// Calculate variance
variance(vector) -> Float
let var = variance([1, 2, 3, 4, 5])  // 2.0

// Calculate standard deviation
std_dev(vector) -> Float
let sd = std_dev([1, 2, 3, 4, 5])  // ~1.414
```

## String Functions

### String Creation

```vexl
// Create empty string
empty_string() -> String
let empty_str = empty_string()  // ""

// Convert to string
to_string(value) -> String
let str = to_string(42)  // "42"

// Repeat string
repeat_string(text, count) -> String
let repeated = repeat_string("ha", 3)  // "hahaha"
```

### String Operations

```vexl
// Get string length
string_length(text) -> Int
let len = string_length("hello")  // 5

// Convert to uppercase
to_uppercase(text) -> String
let upper = to_uppercase("hello")  // "HELLO"

// Convert to lowercase
to_lowercase(text) -> String
let lower = to_lowercase("HELLO")  // "hello"

// Trim whitespace
trim(text) -> String
let trimmed = trim("  hello  ")  // "hello"

// Get substring
substring(text, start, length) -> String
let sub = substring("hello world", 0, 5)  // "hello"

// Find substring
find_substring(text, pattern) -> Option<Int>
let pos = find_substring("hello world", "world")  // Some(6)

// Replace substring
replace(text, old, new) -> String
let replaced = replace("hello world", "world", "VEXL")  // "hello VEXL"

// Split string
split(text, delimiter) -> Vector<String, 1>
let parts = split("a,b,c", ",")  // ["a", "b", "c"]
```

## Utility Functions

### Type Conversion

```vexl
// Convert string to integer
parse_int(text) -> Option<Int>
let num = parse_int("42")  // Some(42)

// Convert string to float
parse_float(text) -> Option<Float>
let decimal = parse_float("3.14")  // Some(3.14)

// Convert to boolean
to_bool(value) -> Bool
let flag = to_bool(1)  // true

// Check type
is_int(value) -> Bool
let check = is_int(42)  // true

is_float(value) -> Bool
let check2 = is_float(3.14)  // true

is_bool(value) -> Bool
let check3 = is_bool(true)  // true

is_string(value) -> Bool
let check4 = is_string("hello")  // true

is_vector(value) -> Bool
let check5 = is_vector([1, 2, 3])  // true
```

### Comparison Functions

```vexl
// Equality check
equals(a, b) -> Bool
let eq = equals(42, 42)  // true

// Less than
less_than(a, b) -> Bool
let lt = less_than(3, 5)  // true

// Greater than
greater_than(a, b) -> Bool
let gt = greater_than(5, 3)  // true

// Compare values
compare(a, b) -> Int
let cmp = compare(3, 5)  // -1
```

### Container Functions

```vexl
// Check if vector is empty
is_empty(vector) -> Bool
let empty_check = is_empty([])  // true

// Check if value is in vector
contains(vector, value) -> Bool
let has_element = contains([1, 2, 3], 2)  // true

// Get unique elements
unique(vector) -> Vector<T, 1>
let uniq = unique([1, 2, 2, 3, 3, 3])  // [1, 2, 3]

// Count occurrences
frequencies(vector) -> Vector<Vector<T, 1>, 1>
let freq = frequencies([1, 2, 2, 3, 3, 3])  // [[1, 1], [2, 2], [3, 3]]

// Take first N elements
take(vector, count) -> Vector<T, 1>
let first_five = take([1, 2, 3, 4, 5, 6, 7], 3)  // [1, 2, 3]

// Skip first N elements
skip(vector, count) -> Vector<T, 1>
let skipped = skip([1, 2, 3, 4, 5], 2)  // [3, 4, 5]
```

## Advanced Functions

### Pipeline Utilities

```vexl
// Compose functions
compose(f, g) -> fn(T) -> U
let double_then_add_one = compose(|x| x + 1, |x| x * 2)
let result = double_then_add_one(5)  // 11

// Pipe value through function
pipe(value, func) -> U
let piped = pipe(5, |x| x * 2)  // 10

// Conditional function application
if_then(condition, func) -> Option<fn(T) -> T>
let maybe_double = if_then(true, |x| x * 2)
let doubled = maybe_double(5)  // 10
```

### Higher-Order Functions

```vexl
// Apply function to each element and return results
map_with_index(vector, func) -> Vector<U, 1>
let indexed = map_with_index(["a", "b", "c"], |i, x| x + to_string(i))  // ["a0", "b1", "c2"]

// Filter with index
filter_with_index(vector, predicate) -> Vector<T, 1>
let filtered = filter_with_index([1, 2, 3, 4], |i, x| i % 2 == 0)  // [1, 3]

// Group by key function
group_by(vector, key_func) -> Vector<Vector<T, 1>, 1>
let grouped = group_by(["apple", "banana", "apricot"], |s| s[0])  // [["apple", "apricot"], ["banana"]]
```

### Generator Functions

```vexl
// Create infinite generator
generator(func) -> Generator<T>
let naturals = generator(|i| i)  // 0, 1, 2, 3, ...

// Take from generator
take_from(generator, count) -> Vector<T, 1>
let first_ten = take_from(naturals, 10)  // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

// Filter generator
filter_generator(generator, predicate) -> Generator<T>
let even_naturals = filter_generator(naturals, |x| x % 2 == 0)  // 0, 2, 4, 6, ...
```

## Error Handling Functions

```vexl
// Try function with fallback
try_or_else(func, fallback) -> T
let result = try_or_else(|| dangerous_function(), 0)

// Check for errors
has_error(result) -> Bool
let safe = has_error(try_result)

// Get error message
error_message(result) -> String
let msg = error_message(error_result)
```

## Performance Tips

### Efficient Usage

```vexl
// Use map instead of loops for transformations
let doubled = data |> map(|x| x * 2)  // Fast

// Use filter to reduce data early
let filtered = data |> filter(|x| x > 0) |> map(|x| x * 2)  // Efficient

// Use reduce for aggregations
let sum = data |> reduce(0, |acc, x| acc + x)  // Memory efficient

// Chain operations for better performance
let result = data 
    |> filter(|x| x % 2 == 0)      // Filter first
    |> map(|x| x * x)              // Then transform
    |> take(10)                    // Then limit
```

### Memory Management

```vexl
// Use generators for large datasets
let large_data = generator(|i| i * 2)  // No memory used

// Process in chunks
let chunks = data |> chunk(1000)       // Process 1000 at a time

// Use lazy evaluation
let computed = lazy(|| expensive_calculation())  // Only compute when needed
```

## Function Reference Table

| Function | Type | Description | Example |
|----------|------|-------------|---------|
| `length` | `Vector<T,1> -> Int` | Get vector length | `length([1,2,3]) -> 3` |
| `sum` | `Vector<Num,1> -> Num` | Sum all elements | `sum([1,2,3]) -> 6` |
| `map` | `(Vector<T,1>, fn(T)->U) -> Vector<U,1>` | Transform elements | `map([1,2,3], \|x\| x*2) -> [2,4,6]` |
| `filter` | `(Vector<T,1>, fn(T)->Bool) -> Vector
