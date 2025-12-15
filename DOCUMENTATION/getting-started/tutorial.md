# VEXL Tutorial

> **Complete Learning Path: From Beginner to Competent VEXL Programmer**

## What This Tutorial Covers

This comprehensive tutorial takes you from zero programming experience to writing substantial VEXL programs. Each lesson builds on the previous one, with practical exercises and real-world examples.

## Tutorial Structure

```
Lesson 1: Getting Started
Lesson 2: Working with Numbers  
Lesson 3: Vector Fundamentals
Lesson 4: Functions and Reusability
Lesson 5: Data Processing Pipelines
Lesson 6: Advanced Vector Operations
Lesson 7: Real-World Applications
Lesson 8: Performance and Optimization
```

---

## Lesson 1: Getting Started

### Learning Objectives
By the end of this lesson, you will:
- Set up your development environment
- Write and run your first VEXL programs
- Understand basic syntax and structure

### Setup Verification

**Step 1: Test Installation**
```bash
./target/release/vexl --version
```

**Step 2: Create Your First Program**

Create `lesson1.vexl`:
```vexl
// Lesson 1: Getting Started
print("Welcome to VEXL!")
print("Your journey begins now.")
```

**Run it:**
```bash
./target/release/vexl run lesson1.vexl
```

**Expected Output:**
```
Welcome to VEXL!
Your journey begins now.
```

### Exercise 1.1
Create a program that:
1. Prints your name
2. Prints your age
3. Calculates what your age will be in 10 years

### Exercise 1.2
Create a program that calculates the area of a rectangle:
```vexl
// Given width = 10, height = 5
// Calculate area and print the result
```

---

## Lesson 2: Working with Numbers

### Learning Objectives
- Master basic arithmetic operations
- Understand operator precedence
- Work with different number types

### Arithmetic Basics

Create `lesson2.vexl`:
```vexl
// Basic arithmetic
let a = 10
let b = 3

print("Addition: " + a + " + " + b + " = " + (a + b))
print("Subtraction: " + a + " - " + b + " = " + (a - b))
print("Multiplication: " + a + " * " + b + " = " + (a * b))
print("Division: " + a + " / " + b + " = " + (a / b))
print("Modulo: " + a + " % " + b + " = " + (a % b))
```

### Operator Precedence

```vexl
// Understanding precedence
let result1 = 2 + 3 * 4        // 14 (not 20)
let result2 = (2 + 3) * 4      // 20

print("2 + 3 * 4 = " + result1)
print("(2 + 3) * 4 = " + result2)
```

### Exercise 2.1
Create a program that calculates:
1. The perimeter of a rectangle (width = 8, height = 12)
2. The volume of a cube (side = 5)
3. The area of a circle (radius = 7)

### Exercise 2.2
Create a simple calculator that performs:
- Addition, subtraction, multiplication, division
- Takes two numbers as input (hardcoded for now)
- Prints all four results

---

## Lesson 3: Vector Fundamentals

### Learning Objectives
- Create and manipulate vectors
- Understand vector indexing
- Perform operations on entire vectors

### Creating Vectors

Create `lesson3.vexl`:
```vexl
// Creating vectors
let numbers = [1, 2, 3, 4, 5]
let grades = [85, 92, 78, 90, 88]
let names = ["Alice", "Bob", "Carol", "David"]

print("Numbers: " + numbers)
print("Grades: " + grades)
print("Names: " + names)

// Vector properties
let count = numbers |> length()
let sum = numbers |> sum()
let average = sum / count

print("Count: " + count)
print("Sum: " + sum)
print("Average: " + average)
```

### Vector Operations

```vexl
// Transform vectors
let doubled = numbers |> map(|x| x * 2)
print("Doubled: " + doubled)

// Filter vectors
let evens = numbers |> filter(|x| x % 2 == 0)
print("Evens: " + evens)

// Combine operations
let even_squares = numbers 
    |> filter(|x| x % 2 == 0)
    |> map(|x| x * x)
print("Even squares: " + even_squares)
```

### Exercise 3.1
Given the vector `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`:
1. Calculate the sum of all numbers
2. Find the sum of only odd numbers
3. Create a vector of squares
4. Find the average

### Exercise 3.2
Working with grade data `[85, 92, 78, 90, 88, 95, 87, 91]`:
1. Find the highest grade
2. Find the lowest grade
3. Calculate the class average
4. Count how many students scored above 90

---

## Lesson 4: Functions and Reusability

### Learning Objectives
- Define and use functions
- Pass parameters and return values
- Create reusable code components

### Defining Functions

Create `lesson4.vexl`:
```vexl
// Simple function
fn greet(name) {
    "Hello, " + name + "!"
}

fn add(a, b) {
    a + b
}

// Test the functions
let message = greet("Alice")
let result = add(5, 3)

print(message)
print("5 + 3 = " + result)
```

### Functions with Complex Logic

```vexl
// Calculate statistics
fn calculate_mean(data) {
    let count = data |> length()
    let sum = data |> sum()
    sum / count
}

fn calculate_median(data) {
    let sorted = data |> sort()
    let count = sorted |> length()
    let middle = count / 2
    
    if count % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2
    } else {
        sorted[middle]
    }
}

// Test with sample data
let scores = [85, 92, 78, 90, 88, 95, 87, 91]
let mean = calculate_mean(scores)
let median = calculate_median(scores)

print("Scores: " + scores)
print("Mean: " + mean)
print("Median: " + median)
```

### Exercise 4.1
Create functions for:
1. `calculate_area_of_circle(radius)` - returns π * r²
2. `is_prime(n)` - returns true if n is prime
3. `fibonacci(n)` - returns nth Fibonacci number

### Exercise 4.2
Create a temperature conversion module:
```vexl
fn celsius_to_fahrenheit(c) {
    // Your code here
}

fn fahrenheit_to_celsius(f) {
    // Your code here  
}

fn kelvin_to_celsius(k) {
    // Your code here
}
```

---

## Lesson 5: Data Processing Pipelines

### Learning Objectives
- Master the pipe operator
- Build complex data processing chains
- Handle real-world data transformations

### Building Pipelines

Create `lesson5.vexl`:
```vexl
// Simulate processing student data
let raw_scores = [85, 92, 78, 90, 88, 95, 87, 91, 76, 89, 93, 84]

// Build a processing pipeline
let processed = raw_scores
    |> filter(|x| x >= 0 && x <= 100)                    // Valid scores only
    |> sort()                                            // Sort ascending
    |> map(|x| if x >= 90 { "A" } else if x >= 80 { "B" } else if x >= 70 { "C" } else { "F" })
    |> frequencies()                                     // Count each grade
    |> sort_by_key()                                     // Sort by grade letter

print("Original scores: " + raw_scores)
print("Grade distribution: " + processed)
```

### Complex Data Analysis

```vexl
// Analyze sales data
let sales_data = [
    ["Product A", 1200, "Electronics"],
    ["Product B", 800, "Books"], 
    ["Product C", 2500, "Electronics"],
    ["Product D", 600, "Clothing"],
    ["Product E", 1800, "Electronics"],
    ["Product F", 400, "Books"]
]

// Calculate total sales by category
let sales_by_category = sales_data
    |> group_by(|item| item[2])                          // Group by category
    |> map(|group| {
        category: group[0][2],
        total_sales: group |> map(|item| item[1]) |> sum(),
        count: group |> length()
    })
    |> sort_by(|x| x.total_sales)                        // Sort by sales

print("Sales by Category:")
sales_by_category |> for_each(|cat| 
    print("- " + cat.category + ": $" + cat.total_sales + " (" + cat.count + " products)")
)
```

### Exercise 5.1
Process this customer data:
```vexl
let customers = [
    ["Alice", 25, "Premium"],
    ["Bob", 35, "Standard"], 
    ["Carol", 28, "Premium"],
    ["David", 42, "Standard"],
    ["Eve", 31, "Premium"]
]
```

Create a pipeline that:
1. Filters only Premium customers
2. Calculates their average age
3. Sorts them by age

### Exercise 5.2
Analyze this time series data:
```vexl
let temperatures = [68, 72, 70, 75, 78, 73, 69, 71, 74, 76]
```

Create a pipeline that:
1. Finds all temperatures above 70
2. Converts them to Celsius (subtract 32, multiply by 5/9)
3. Rounds to 1 decimal place
4. Calculates the average

---

## Lesson 6: Advanced Vector Operations

### Learning Objectives
- Work with multi-dimensional vectors
- Implement algorithms using vectors
- Understand performance considerations

### Working with Matrices

Create `lesson6.vexl`:
```vexl
// Creating and manipulating matrices
let matrix_a = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

let matrix_b = [
    [9, 8, 7],
    [6, 5, 4], 
    [3, 2, 1]
]

// Matrix addition
fn matrix_add(a, b) {
    let rows = a |> length()
    let cols = a[0] |> length()
    
    [[a[i][j] + b[i][j] | j <- [0..cols]] | i <- [0..rows]]
}

// Test matrix addition
let sum_matrix = matrix_add(matrix_a, matrix_b)
print("Matrix A + Matrix B:")
sum_matrix |> for_each(|row| print(row))
```

### Matrix Multiplication

```vexl
// Matrix multiplication
fn matrix_multiply(a, b) {
    let rows_a = a |> length()
    let cols_a = a[0] |> length()
    let cols_b = b[0] |> length()
    
    let result = [[0 | _ <- [0..cols_b]] | _ <- [0..rows_a]]
    
    for i in [0..rows_a] {
        for j in [0..cols_b] {
            let sum = 0
            for k in [0..cols_a] {
                result[i][j] = result[i][j] + (a[i][k] * b[k][j])
            }
        }
    }
    result
}

// Test
let small_a = [[1, 2], [3, 4]]
let small_b = [[5, 6], [7, 8]]
let product = matrix_multiply(small_a, small_b)

print("Matrix multiplication result:")
product |> for_each(|row| print(row))
```

### Exercise 6.1
Implement these matrix operations:
1. `matrix_transpose(matrix)` - Swap rows and columns
2. `matrix_scalar_multiply(matrix, scalar)` - Multiply by scalar
3. `matrix_determinant_2x2(matrix)` - Calculate determinant of 2x2 matrix

### Exercise 6.2
Create a function to solve systems of linear equations:
```vexl
// Solve Ax = b where A is 2x2 matrix and b is 2-element vector
fn solve_2x2_system(a, b) {
    // Your implementation here
}
```

---

## Lesson 7: Real-World Applications

### Learning Objectives
- Apply VEXL to practical problems
- Build complete applications
- Handle error cases and edge conditions

### Application 1: Grade Calculator

Create `lesson7_grade_calculator.vexl`:
```vexl
// Complete grade calculator application

fn calculate_letter_grade(score) {
    if score >= 97 {
        "A+"
    } else if score >= 93 {
        "A"
    } else if score >= 90 {
        "A-"
    } else if score >= 87 {
        "B+"
    } else if score >= 83 {
        "B"
    } else if score >= 80 {
        "B-"
    } else if score >= 77 {
        "C+"
    } else if score >= 73 {
        "C"
    } else if score >= 70 {
        "C-"
    } else if score >= 67 {
        "D+"
    } else if score >= 63 {
        "D"
    } else if score >= 60 {
        "D-"
    } else {
        "F"
    }
}

fn process_grades(scores) {
    let valid_scores = scores |> filter(|x| x >= 0 && x <= 100)
    let invalid_count = (scores |> length()) - (valid_scores |> length())
    
    let stats = {
        total_scores: valid_scores |> length(),
        invalid_scores: invalid_count,
        average: valid_scores |> sum() / (valid_scores |> length()),
        highest: valid_scores |> max(),
        lowest: valid_scores |> min(),
        letter_distribution: valid_scores 
            |> map(|x| calculate_letter_grade(x))
            |> frequencies()
            |> sort_by_key()
    }
    
    stats
}

// Sample data
let student_scores = [95, 87, 92, 78, 89, 96, 84, 91, 88, 93, 85, 90, 82, 94, 86]

// Process and display
let results = process_grades(student_scores)

print("=== GRADE REPORT ===")
print("Total valid scores: " + results.total_scores)
print("Invalid scores: " + results.invalid_scores)
print("Average: " + results.average)
print("Highest: " + results.highest)
print("Lowest: " + results.lowest)
print("")
print("Letter Grade Distribution:")
results.letter_distribution |> for_each(|grade| 
    print("- " + grade[0] + ": " + grade[1] + " students")
)
```

### Application 2: Data Analysis Tool

```vexl
// Sales analysis tool
fn analyze_sales_data(sales_records) {
    // Calculate various metrics
    let total_revenue = sales_records |> map(|record| record[2]) |> sum()
    let total_transactions = sales_records |> length()
    let average_transaction = total_revenue / total_transactions
    
    // Find best and worst performing products
    let product_totals = sales_records
        |> group_by(|record| record[0])              // Group by product
        |> map(|group| {
            product: group[0][0],
            total_sales: group |> map(|r| r[2]) |> sum(),
            transactions: group |> length()
        })
    
    let best_product = product_totals |> sort_by(|x| x.total_sales) |> last()
    let worst_product = product_totals |> sort_by(|x| x.total_sales) |> first()
    
    // Monthly breakdown
    let monthly_sales = sales_records
        |> group_by(|record| record[1])              // Group by month
        |> map(|group| {
            month: group[0][1],
            revenue: group |> map(|r| r[2]) |> sum(),
            transactions: group |> length()
        })
        |> sort_by(|x| x.month)
    
    {
        total_revenue: total_revenue,
        total_transactions: total_transactions,
        average_transaction: average_transaction,
        best_product: best_product,
        worst_product: worst_product,
        monthly_breakdown: monthly_sales
    }
}

// Sample data: [Product, Month, Sales Amount]
let sales_data = [
    ["Laptop", "Jan", 2500],
    ["Mouse", "Jan", 150],
    ["Keyboard", "Jan", 200],
    ["Laptop", "Feb", 2800],
    ["Mouse", "Feb", 180],
    ["Keyboard", "Feb", 220],
    ["Laptop", "Mar", 3200],
    ["Mouse", "Mar", 160],
    ["Keyboard", "Mar", 240]
]

// Analyze the data
let analysis = analyze_sales_data(sales_data)

print("=== SALES ANALYSIS ===")
print("Total Revenue: $" + analysis.total_revenue)
print("Total Transactions: " + analysis.total_transactions)
