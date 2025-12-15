## EXAMPLES

These examples are designed for beginners who want to see VEXL in action. Each example is complete and runnable, showing real-world applications of VEXL's vector-based approach.

## Getting Started Examples

### Example 1: Basic Calculator

```vexl
// A simple calculator that performs various operations
fn calculator_demo() {
    let a = 10
    let b = 3
    
    print("=== Basic Calculator ===")
    print("a = " + a + ", b = " + b)
    print("Addition: " + a + " + " + b + " = " + (a + b))
    print("Subtraction: " + a + " - " + b + " = " + (a - b))
    print("Multiplication: " + a + " * " + b + " = " + (a * b))
    print("Division: " + a + " / " + b + " = " + (a / b))
    print("Modulo: " + a + " % " + b + " = " + (a % b))
}

calculator_demo()
```

**Output:**
```
=== Basic Calculator ===
a = 10, b = 3
Addition: 10 + 3 = 13
Subtraction: 10 - 3 = 7
Multiplication: 10 * 3 = 30
Division: 10 / 3 = 3
Modulo: 10 % 3 = 1
```

### Example 2: Working with Lists

```vexl
// Demonstrating vector operations
fn list_operations() {
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print("=== List Operations ===")
    print("Original list: " + numbers)
    print("Length: " + (numbers |> length()))
    print("Sum: " + (numbers |> sum()))
    print("Average: " + (numbers |> sum() / (numbers |> length())))
    
    // Transform the list
    let doubled = numbers |> map(|x| x * 2)
    print("Doubled: " + doubled)
    
    // Filter the list
    let evens = numbers |> filter(|x| x % 2 == 0)
    print("Even numbers: " + evens)
    
    // Complex operation
    let result = numbers
        |> filter(|x| x > 3)
        |> map(|x| x * x)
        |> filter(|x| x < 50)
        |> sum()
    print("Sum of squares > 3 and < 50: " + result)
}

list_operations()
```

### Example 3: Grade Calculator

```vexl
// Calculate statistics for student grades
fn grade_calculator() {
    let grades = [85, 92, 78, 90, 88, 95, 87, 91, 76, 89, 93, 84]
    
    print("=== Grade Calculator ===")
    print("Grades: " + grades)
    
    let count = grades |> length()
    let total = grades |> sum()
    let average = total / count
    let highest = grades |> max()
    let lowest = grades |> min()
    
    print("Count: " + count)
    print("Total: " + total)
    print("Average: " + average)
    print("Highest: " + highest)
    print("Lowest: " + lowest)
    
    // Count grades by letter
    let letter_grades = grades |> map(|x| 
        if x >= 90 { "A" }
        else if x >= 80 { "B" }
        else if x >= 70 { "C" }
        else { "F" }
    )
    
    let a_count = letter_grades |> filter(|g| g == "A") |> length()
    let b_count = letter_grades |> filter(|g| g == "B") |> length()
    let c_count = letter_grades |> filter(|g| g == "C") |> length()
    let f_count = letter_grades |> filter(|g| g == "F") |> length()
    
    print("A grades: " + a_count)
    print("B grades: " + b_count)
    print("C grades: " + c_count)
    print("F grades: " + f_count)
}

grade_calculator()
```

## Mathematical Examples

### Example 4: Fibonacci Sequence

```vexl
// Generate Fibonacci numbers
fn fibonacci_demo() {
    fn fibonacci(n) {
        if n <= 1 {
            n
        } else {
            fibonacci(n - 1) + fibonacci(n - 2)
        }
    }
    
    print("=== Fibonacci Sequence ===")
    let fib_numbers = [0..10] |> map(|i| fibonacci(i))
    print("First 10 Fibonacci numbers: " + fib_numbers)
    
    // Using a more efficient approach with vectors
    fn fibonacci_vector(count) {
        if count == 0 {
            []
        } else if count == 1 {
            [0]
        } else {
            let sequence = [0, 1]
            // This would need loop constructs (future feature)
            sequence
        }
    }
}

fibonacci_demo()
```

### Example 5: Prime Number Checker

```vexl
// Check for prime numbers
fn prime_examples() {
    fn is_prime(n) {
        if n <= 1 {
            false
        } else if n == 2 {
            true
        } else if n % 2 == 0 {
            false
        } else {
            let limit = n |> sqrt() |> truncate()
            let factors = [3..limit, step: 2] |> filter(|x| n % x == 0)
            (factors |> length()) == 0
        }
    }
    
    print("=== Prime Numbers ===")
    let test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    let primes = test_numbers |> filter(|x| is_prime(x))
    let non_primes = test_numbers |> filter(|x| !is_prime(x))
    
    print("Prime numbers: " + primes)
    print("Non-prime numbers: " + non_primes)
    
    // Count primes in a range
    let large_range = [1..100]
    let prime_count = large_range |> filter(|x| is_prime(x)) |> length()
    print("Primes between 1 and 100: " + prime_count)
}

prime_examples()
```

### Example 6: Statistics Calculator

```vexl
// Calculate comprehensive statistics
fn statistics_demo() {
    let data = [23, 45, 56, 78, 32, 45, 67, 89, 12, 34, 56, 78, 90, 23, 45]
    
    fn calculate_stats(numbers) {
        let count = numbers |> length()
        let sum = numbers |> sum()
        let mean = sum / count
        
        let sorted = numbers |> sort()
        let median = if count % 2 == 0 {
            let mid = count / 2
            (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            sorted[count / 2]
        }
        
        let squared_diffs = numbers |> map(|x| (x - mean) * (x - mean))
        let variance = squared_diffs |> sum() / count
        let std_dev = variance |> sqrt()
        
        {
            count: count,
            sum: sum,
            mean: mean,
            median: median,
            min: sorted[0],
            max: sorted[count - 1],
            variance: variance,
            std_dev: std_dev
        }
    }
    
    print("=== Statistics Calculator ===")
    print("Data: " + data)
    
    let stats = calculate_stats(data)
    print("Count: " + stats.count)
    print("Sum: " + stats.sum)
    print("Mean: " + stats.mean)
    print("Median: " + stats.median)
    print("Min: " + stats.min)
    print("Max: " + stats.max)
    print("Variance: " + stats.variance)
    print("Standard Deviation: " + stats.std_dev)
}

statistics_demo()
```

## Data Processing Examples

### Example 7: Sales Analysis

```vexl
// Analyze sales data
fn sales_analysis() {
    // Sample sales data: [Product, Month, Sales Amount]
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
    
    print("=== Sales Analysis ===")
    print("Sales data: ")
    sales_data |> for_each(|sale| 
        print("- " + sale[0] + " (" + sale[1] + "): $" + sale[2])
    )
    
    // Calculate total sales by product
    let product_totals = sales_data
        |> group_by(|sale| sale[0])
        |> map(|group| {
            product: group[0][0],
            total_sales: group |> map(|sale| sale[2]) |> sum(),
            transactions: group |> length()
        })
    
    print("\nSales by Product:")
    product_totals |> for_each(|product|
        print("- " + product.product + ": $" + product.total_sales + 
              " (" + product.transactions + " transactions)")
    )
    
    // Find best performing product
    let best_product = product_totals |> sort_by(|p| p.total_sales) |> last()
    print("\nBest performing product: " + best_product.product + 
          " with $" + best_product.total_sales)
}

sales_analysis()
```

### Example 8: Temperature Converter

```vexl
// Temperature conversion utilities
fn temperature_converter() {
    fn celsius_to_fahrenheit(c) {
        c * 9/5 + 32
    }
    
    fn fahrenheit_to_celsius(f) {
        (f - 32) * 5/9
    }
    
    fn celsius_to_kelvin(c) {
        c + 273.15
    }
    
    print("=== Temperature Converter ===")
    
    // Convert a range of Celsius temperatures
    let celsius_temps = [0, 10, 20, 30, 40, 50, 100]
    print("Celsius temperatures: " + celsius_temps)
    
    let fahrenheit_temps = celsius_temps |> map(|c| celsius_to_fahrenheit(c))
    print("Fahrenheit equivalents: " + fahrenheit_temps)
    
    let kelvin_temps = celsius_temps |> map(|c| celsius_to_kelvin(c))
    print("Kelvin equivalents: " + kelvin_temps)
    
    // Create a temperature conversion table
    let temp_table = celsius_temps |> map(|c| {
        celsius: c,
        fahrenheit: celsius_to_fahrenheit(c),
        kelvin: celsius_to_kelvin(c)
    })
    
    print("\nTemperature Conversion Table:")
    print("C°\t\tF°\t\tK°")
    temp_table |> for_each(|temp|
        print(to_string(temp.celsius) + "\t\t" + 
              to_string(temp.fahrenheit) + "\t\t" + 
              to_string(temp.kelvin))
    )
}

temperature_converter()
```

## Text Processing Examples

### Example 9: Word Counter

```vexl
// Count words in text
fn word_counter_demo() {
    fn count_words(text) {
        // Split by spaces and filter empty strings
        let words = text |> split(" ") |> filter(|w| w |> length() > 0)
        let word_count = words |> length()
        
        // Count frequency of each word
        let frequencies = words |> frequencies()
        
        // Sort by frequency (descending)
        let sorted_freq = frequencies |> sort_by(|freq| freq[1])
        
        {
            total_words: word_count,
            unique_words: frequencies |> length(),
            most_common: if frequencies |> length() > 0 {
                Some(sorted_freq |> last())
            } else {
                None
            }
        }
    }
    
    let text = "the quick brown fox jumps over the lazy dog the fox is quick"
    
    print("=== Word Counter ===")
    print("Text: \"" + text + "\"")
    
    let stats = count_words(text)
    print("Total words: " + stats.total_words)
    print("Unique words: " + stats.unique_words)
    
    match stats.most_common {
        Some(common) => print("Most common word: \"" + common[0] + "\" appears " + common[1] + " times"),
        None => print("No words found")
    }
}

word_counter_demo()
```

### Example 10: Text Analysis

```vexl
// Analyze text characteristics
fn text_analyzer() {
    fn analyze_text(text) {
        let chars = text |> length()
        let words = text |> split(" ") |> filter(|w| w |> length() > 0) |> length()
        let lines = text |> split("\n") |> length()
        
        // Count vowels and consonants
        let vowels = "aeiouAEIOU"
        let vowel_count = text |> filter(|c| vowels |> contains(c)) |> length()
        let consonant_count = chars - vowel_count
        
        // Calculate average word length
        let word_lengths = text |> split(" ") 
            |> filter(|w| w |> length() > 0)
            |> map(|w| w |> length())
        let avg_word_length = if word_lengths |> length() > 0 {
            word_lengths |> sum() / (word_lengths |> length())
        } else {
            0
        }
        
        {
            characters: chars,
            words: words,
            lines: lines,
            vowels: vowel_count,
            consonants: consonant_count,
            avg_word_length: avg_word_length
        }
    }
    
    let sample_text = "VEXL is a revolutionary programming language.\nIt treats everything as vectors.\nThis makes programming more consistent and powerful."
    
    print("=== Text Analyzer ===")
    print("Sample text:")
    print(sample_text)
    print("")
    
    let analysis = analyze_text(sample_text)
    print("Characters: " + analysis.characters)
    print("Words: " + analysis.words)
    print("Lines: " + analysis.lines)
    print("Vowels: " + analysis.vowels)
    print("Consonants: " + analysis.consonants)
    print("Average word length: " + analysis.avg_word_length)
}

text_analyzer()
```

## Pattern Examples

### Example 11: Pattern Matching with Data

```vexl
// Work with structured data
fn pattern_matching_demo() {
    // Simulate pattern matching with conditional logic
    fn categorize_age(age) {
        if age < 13 {
            "Child"
        } else if age < 20 {
            "Teenager"
        } else if age < 65 {
            "Adult"
        } else {
            "Senior"
        }
    }
    
    fn categorize_score(score) {
        if score >= 90 {
            "Excellent"
        } else if score >= 80 {
            "Good"
        } else if score >= 70 {
            "Average"
        } else if score >= 60 {
            "Below Average"
        } else {
            "Poor"
        }
    }
    
    let people = [
        ["Alice", 25, 95],
        ["Bob", 17, 78],
        ["Carol", 35, 82],
        ["David", 12, 88],
        ["E
