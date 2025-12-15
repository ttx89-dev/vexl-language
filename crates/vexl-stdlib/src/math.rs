/// Calculate dot product of two vectors
pub fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

/// Matrix multiplication (naive implementation for reference)
/// A is m x n, B is n x p, Result is m x p
pub fn mat_mul(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut result = vec![0.0; m * p];
    
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * p + j];
            }
            result[i * p + j] = sum;
        }
    }
    
    result
}

/// Transpose matrix
/// M is m x n, Result is n x m
pub fn transpose(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut result = vec![0.0; m * n];
    
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = matrix[i * n + j];
        }
    }
    
    result
}
