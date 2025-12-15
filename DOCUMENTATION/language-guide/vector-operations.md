# VEXL Vector Operations

> **Master the Art of Vector Manipulation in VEXL**

## What You'll Learn

This comprehensive guide covers all vector operations in VEXL, from basic transformations to advanced mathematical operations. VEXL's universal vector approach means these operations work consistently at every level.

## Vector Creation

### Basic Vector Construction

```vexl
// Empty vector
let empty = []

// Single element
let single = [42]

// Multiple elements
let numbers = [1, 2, 3, 4, 5]
let mixed = [1, "hello", true]

// Nested vectors (matrices, tensors)
let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
let tensor = [[[1, 2]], [[3, 4]]]
```

### Range-Based Creation

```vexl
// Finite ranges
let range_1_to_10 = [1..10]              // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let range_0_to_5 = [0..5]                // [0, 1, 2, 3, 4, 5]

// Infinite ranges
let naturals = [1..]                     // 1, 2, 3, 4, 5, ...
let all_integers = [(-100)..]            // -100, -99, -98, ...

// Using infinite ranges with limits
let first_100 = [1..] |> take(100)       // First 100 natural numbers
let evens_0_to_20 = [0..] 
    |> filter(|x| x % 2 == 0) 
    |> take(11)                          // [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```

### Generator-Based Creation

```vexl
// Generate vectors from functions
let squares = [0..10] |> map(|x| x * x)  // [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

let even_squares = [0..20] 
    |> filter(|x| x % 2 == 0)
    |> map(|x| x * x)                    // [0, 4, 16, 36, 64, 100, 144, 196, 256, 324, 400]

// Generate from complex expressions
let fibonacci_like = [0..10] |> map(|n| 
    if n <= 1 { n }
    else if n == 2 { 1 }
    else { 0 }  // Simplified for example
)
```

## Element Access

### Indexing

```vexl
let vector = [10, 20, 30, 40, 50]

// Positive indexing
let first = vector[0]                    // 10
let second = vector[1]                   // 20
let last = vector[4]                     // 50

// Multi-dimensional indexing
let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
let top_left = matrix[0][0]              // 1
let center = matrix[1][1]                // 5
let bottom_right = matrix[2][2]          // 9

// Safe indexing with bounds checking
let safe_get = |vec, idx| 
    if idx >= 0 && idx < (vec |> length()) {
        Some(vec[idx])
    } else {
        None
    }

let value = safe_get(vector, 2)          // Some(30)
let invalid = safe_get(vector, 10)       // None
```

### Slicing (Future Feature)

```vexl
// Planned slicing syntax
let slice_1_to_3 = vector[1..3]          // [20, 30, 40] (future)
let slice_to_3 = vector[..3]             // [10, 20, 30] (future)
let slice_from_2 = vector[2..]           // [30, 40, 50] (future)
```

## Transformation Operations

### Map Operations

```vexl
let data = [1, 2, 3, 4, 5]

// Basic transformation
let doubled = data |> map(|x| x * 2)     // [2, 4, 6, 8, 10]

// Complex transformations
let squared = data |> map(|x| x * x)     // [1, 4, 9, 16, 25]

// Transform with index
let indexed = data |> map_with_index(|i, x| i * 10 + x)  // [10, 21, 32, 43, 54]

// Nested transformation
let matrix = [[1, 2, 3], [4, 5, 6]]
let doubled_matrix = matrix |> map(|row| row |> map(|x| x * 2))  // [[2, 4, 6], [8, 10, 12]]

// Flatten after mapping
let flattened = matrix 
    |> map(|row| row |> map(|x| x * 2))
    |> flatten()                       // [2, 4, 6, 8, 10, 12]
```

### Filter Operations

```vexl
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Basic filtering
let evens = numbers |> filter(|x| x % 2 == 0)    // [2, 4, 6, 8, 10]
let odds = numbers |> filter(|x| x % 2 == 1)     // [1, 3, 5, 7, 9]

// Complex filtering conditions
let greater_than_5 = numbers |> filter(|x| x > 5)  // [6, 7, 8, 9, 10]
let prime_like = numbers |> filter(|x| 
    x == 2 || x == 3 || x == 5 || x == 7 || (x > 7 && x % 2 != 0 && x % 3 != 0)
)  // [2, 3, 5, 7]

// Filter with index
let even_positions = numbers |> filter_with_index(|i, x| i % 2 == 0)  // [1, 3, 5, 7, 9]
```

### Combined Operations

```vexl
let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Map then filter
let even_squares = data 
    |> map(|x| x * x)
    |> filter(|x| x % 2 == 0)          // [4, 16, 36, 64, 100]

// Filter then map
let doubled_evens = data 
    |> filter(|x| x % 2 == 0)
    |> map(|x| x * 2)                  // [4, 8, 12, 16, 20]

// Complex pipeline
let result = data
    |> filter(|x| x > 2)               // [3, 4, 5, 6, 7, 8, 9, 10]
    |> map(|x| x * x)                  // [9, 16, 25, 36, 49, 64, 81, 100]
    |> filter(|x| x < 50)              // [9, 16, 25, 36, 49]
    |> sum()                           // 135
```

## Reduction Operations

### Basic Reductions

```vexl
let numbers = [1, 2, 3, 4, 5]

// Sum and product
let total = numbers |> sum()            // 15
let product = numbers |> product()      // 120

// Min and max
let minimum = numbers |> min()          // 1
let maximum = numbers |> max()          // 5

// Count and length
let count = numbers |> length()         // 5
```

### Statistical Operations

```vexl
let data = [23, 45, 56, 78, 32, 45, 67, 89, 12, 34, 56, 78, 90, 23, 45]

// Mean and median
let sum = data |> sum()
let count = data |> length()
let mean = sum / count                  // 52.6...

let sorted = data |> sort()
let median = if count % 2 == 0 {
    let mid = count / 2
    (sorted[mid - 1] + sorted[mid]) / 2
} else {
    sorted[count / 2]
}

// Variance and standard deviation
let squared_diffs = data |> map(|x| (x - mean) * (x - mean))
let variance = squared_diffs |> sum() / count
let std_dev = variance |> sqrt()
```

### Custom Reductions

```vexl
let numbers = [1, 2, 3, 4, 5]

// Custom reduce with initial value
let sum_custom = numbers |> reduce(0, |acc, x| acc + x)  // 15
let product_custom = numbers |> reduce(1, |acc, x| acc * x)  // 120

// Find operations
let first_even = numbers |> find(|x| x % 2 == 0)        // Some(2)
let first_large = numbers |> find(|x| x > 10)           // None

// Check operations
let has_even = numbers |> any(|x| x % 2 == 0)           // true
let all_positive = numbers |> all(|x| x > 0)            // true
let all_small = numbers |> all(|x| x < 10)              // false
```

## Sorting and Ordering

### Basic Sorting

```vexl
let unsorted = [3, 1, 4, 1, 5, 9, 2, 6]

// Ascending sort
let sorted_asc = unsorted |> sort()                     // [1, 1, 2, 3, 4, 5, 6, 9]

// Descending sort
let sorted_desc = unsorted |> sort() |> reverse()       // [9, 6, 5, 4, 3, 2, 1, 1]

// Sort strings
let words = ["zebra", "apple", "banana", "cherry"]
let sorted_words = words |> sort()                      // ["apple", "banana", "cherry", "zebra"]
```

### Custom Sorting

```vexl
let people = [
    ["Alice", 25],
    ["Bob", 30],
    ["Carol", 20],
    ["David", 35]
]

// Sort by age (ascending)
let by_age = people |> sort_by(|person| person[1])      // [[Carol, 20], [Alice, 25], [Bob, 30], [David, 35]]

// Sort by name length
let by_name_length = people |> sort_by(|person| person[0] |> length())  // [[Bob, 30], [Alice, 25], [Carol, 20], [David, 35]]

// Sort by age (descending)
let by_age_desc = people |> sort_by(|person| person[1]) |> reverse()
```

### Unique and Frequency Operations

```vexl
let numbers = [1, 2, 2, 3, 3, 3, 4, 4, 5]

// Get unique elements
let unique = numbers |> unique()                        // [1, 2, 3, 4, 5]

// Count frequencies
let frequencies = numbers |> frequencies()              // [[1, 1], [2, 2], [3, 3], [4, 2], [5, 1]]

// Most common element
let most_common = frequencies |> sort_by(|freq| freq[1]) |> last()  // [3, 3]
```

## Vector Mathematics

### Element-wise Operations

```vexl
let vector_a = [1, 2, 3, 4, 5]
let vector_b = [10, 20, 30, 40, 50]

// Element-wise addition
let sum = vector_a + vector_b                          // [11, 22, 33, 44, 55]

// Element-wise subtraction
let diff = vector_b - vector_a                         // [9, 18, 27, 36, 45]

// Element-wise multiplication
let product = vector_a * vector_b                      // [10, 40, 90, 160, 250]

// Scalar operations
let doubled = vector_a * 2                             // [2, 4, 6, 8, 10]
let incremented = vector_a + 1                         // [2, 3, 4, 5, 6]
```

### Matrix Operations

```vexl
let matrix_a = [[1, 2], [3, 4]]
let matrix_b = [[5, 6], [7, 8]]

// Matrix addition
let matrix_sum = matrix_a + matrix_b                   // [[6, 8], [10, 12]]

// Matrix multiplication
let matrix_product = matrix_a @ matrix_b               // [[19, 22], [43, 50]]

// Transpose (manual implementation)
fn transpose(matrix) {
    if matrix |> length() == 0 {
        []
    } else {
        let rows = matrix |> length()
        let cols = matrix[0] |> length()
        [[matrix[i][j] | i <- [0..rows]] | j <- [0..cols]]
    }
}

let transposed = transpose(matrix_a)                   // [[1, 3], [2, 4]]
```

### Dot Product and Vector Operations

```vexl
let vector_u = [1, 2, 3]
let vector_v = [4, 5, 6]

// Dot product
let dot_product = (vector_u * vector_v) |> sum()       // 32

// Vector magnitude
fn magnitude(vector) {
    (vector * vector |> sum()) |> sqrt()
}

let mag_u = magnitude(vector_u)                        // ~3.74
let mag_v = magnitude(vector_v)                        // ~8.77

// Normalize vector
fn normalize(vector) {
    let mag = magnitude(vector)
    if mag > 0 {
        vector / mag
    } else {
        vector
    }
}

let normalized_u = normalize(vector_u)                 // [0.27, 0.53, 0.80]
```

## Advanced Vector Operations

### Grouping and Aggregation

```vexl
let data = [
    ["Product A", "Electronics", 1200],
    ["Product B", "Books", 800],
    ["Product C", "Electronics", 2500],
    ["Product D", "Clothing", 600],
    ["Product E", "Electronics", 1800],
