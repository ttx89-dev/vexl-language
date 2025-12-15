# Your First VEXL Program

> **Writing and Running Your First Lines of VEXL Code**

## What You'll Learn

After reading this guide, you'll be able to:
- Write your first VEXL program
- Understand the basic structure of VEXL code
- Run VEXL programs and see the results
- Make sense of simple VEXL syntax

## The Traditional "Hello, World!" Program

Let's start with the classic first program. Create a file called `hello.vexl`:

```vexl
// Your first VEXL program!
print("Hello, World!")
```

**What does this do?**
- `print(...)` - A function that displays text on screen
- `"Hello, World!"` - A text string (put in quotes)

**Run it:**
```bash
./target/release/vexl run hello.vexl
```

**Expected output:**
```
Hello, World!
```

## Numbers and Simple Math

VEXL can handle numbers just like a calculator:

```vexl
// Basic arithmetic
let result = 2 + 3 * 4
print("2 + 3 * 4 = " + result)

// Try parentheses
let complicated = (2 + 3) * 4
print("(2 + 3) * 4 = " + complicated)
```

**Run it:**
```bash
./target/release/vexl run math.vexl
```

**Expected output:**
```
2 + 3 * 4 = 14
(2 + 3) * 4 = 20
```

## Working with Vectors (Lists of Numbers)

In VEXL, everything is a vector. Let's create some vectors:

```vexl
// A simple list of numbers
let numbers = [1, 2, 3, 4, 5]

// Let's see what's in it
print("Numbers: " + numbers)

// Let's do math with all of them
let doubled = numbers |> map(|x| x * 2)
print("Doubled: " + doubled)

// Let's find the sum
let total = doubled |> sum()
print("Sum: " + total)
```

**What are these new concepts?**
- `[1, 2, 3, 4, 5]` - A vector (list) of numbers
- `|>` - The pipe operator (feeds data through functions)
- `map(|x| x * 2)` - Applies a function to every element
- `sum()` - Adds all elements together

## A Complete Example

Let's put it all together in one program:

```vexl
// Complete program: Calculate statistics
let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Calculate mean
let count = data |> length()
let sum = data |> sum()
let mean = sum / count

// Calculate sum of squares
let squared_diffs = data 
    |> map(|x| (x - mean) * (x - mean))
let variance = squared_diffs |> sum() / count
let std_dev = variance |> sqrt()

// Display results
print("Data: " + data)
print("Count: " + count)
print("Sum: " + sum)
print("Mean: " + mean)
print("Variance: " + variance)
print("Standard Deviation: " + std_dev)
```

## Understanding VEXL Syntax

### Comments
```vexl
// Single line comment
/* Multi-line
   comment */
```

### Variables
```vexl
let name = "Alice"           // Text variable
let age = 25                 // Number variable
let scores = [85, 92, 78]    // Vector variable
```

### Functions
```vexl
// Named function
fn greet(name) {
    "Hello, " + name + "!"
}

// Anonymous function (lambda)
|x| x + 1
```

### Operations
```vexl
// Arithmetic
1 + 2 - 3 * 4 / 5

// Comparison
1 == 2        // false
3 > 2         // true
"hello" != "world"  // true

// Logic
true && false  // false
true || false  // true
!true          // false
```

## Running VEXL Programs

### Method 1: Run a File
```bash
./target/release/vexl run program.vexl
```

### Method 2: Check Syntax (Don't Run)
```bash
./target/release/vexl check program.vexl
```

### Method 3: Compile to LLVM IR
```bash
./target/release/vexl compile program.vexl --verbose
```

## Common Mistakes and Solutions

### Mistake 1: Forgetting Quotes
```vexl
// Wrong
print(Hello, World!)

// Right
print("Hello, World!")
```

### Mistake 2: Wrong Brackets
```vexl
// Wrong
let numbers = (1, 2, 3)

// Right
let numbers = [1, 2, 3]
```

### Mistake 3: Missing Pipe Operator
```vexl
// Wrong
map(|x| x * 2, numbers)

// Right
numbers |> map(|x| x * 2)
```

## Experiment Ideas

Try these exercises to practice:

1. **Calculator**: Create a program that calculates area of a circle
2. **Grade Calculator**: Calculate average of test scores
3. **Vector Math**: Create vectors and perform operations on them
4. **Text Processing**: Work with strings and text

## Debugging Tips

### Use print() to Debug
```vexl
let result = complicated_calculation()
print("Intermediate result: " + result)
result = result * 2
print("Final result: " + result)
```

### Check Your Types
```vexl
// If something doesn't work, check if you're mixing types
let text = "42"
let number = 42

// This works
let sum = number + 10

// This doesn't (text + number)
let mixed = text + number  // Error!
```

## What's Next?

Now that you've written your first programs:

1. [Learn Basic Concepts](basic-concepts.md) - Understand VEXL fundamentals
2. [Follow the Tutorial](tutorial.md) - Step-by-step learning path
3. [Explore Examples](../examples/) - See more complex programs

## Summary

You now know how to:
- ✓ Write basic VEXL programs
- ✓ Use print() to display output
- ✓ Create variables and vectors
- ✓ Perform mathematical operations
- ✓ Use the pipe operator for data flow
- ✓ Run VEXL programs from the command line

**Keep practicing!** The best way to learn programming is by writing programs.

---

**Next:** [Basic Concepts](basic-concepts.md)
