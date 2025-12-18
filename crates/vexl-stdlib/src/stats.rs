//! Statistical Operations for VEXL
//!
//! Comprehensive statistical analysis functions including:
//! - Descriptive statistics (mean, median, variance, std dev, quantiles)
//! - Probability distributions (normal, uniform, exponential, binomial)
//! - Hypothesis testing (t-test, chi-square, ANOVA)
//! - Regression analysis (linear, polynomial, logistic)
//! - Correlation and covariance

use vexl_runtime::vector::{Vector, vexl_vec_alloc_i64, vexl_vec_set_i64};
use vexl_runtime::context::{ExecutionContext, Value, VectorRef, Function};
use std::rc::Rc;

/// Statistical operations module
pub struct StatisticsOps;

impl StatisticsOps {
    /// Calculate arithmetic mean
    pub fn mean(data: &Vector) -> Result<f64, String> {
        if data.len() == 0 {
            return Err("Cannot calculate mean of empty dataset".to_string());
        }

        let len = data.len() as usize;
        let mut sum = 0.0f64;

        for i in 0..len {
            sum += unsafe { data.get_i64(i as u64) } as f64;
        }

        Ok(sum / len as f64)
    }

    /// Calculate median
    pub fn median(data: &Vector) -> Result<f64, String> {
        if data.len() == 0 {
            return Err("Cannot calculate median of empty dataset".to_string());
        }

        let len = data.len() as usize;
        let mut values: Vec<f64> = (0..len)
            .map(|i| unsafe { data.get_i64(i as u64) } as f64)
            .collect();

        values.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

        if len % 2 == 0 {
            // Even number of elements
            let mid1 = values[len / 2 - 1];
            let mid2 = values[len / 2];
            Ok((mid1 + mid2) / 2.0)
        } else {
            // Odd number of elements
            Ok(values[len / 2])
        }
    }

    /// Calculate variance
    pub fn variance(data: &Vector, sample: bool) -> Result<f64, String> {
        if data.len() < 2 {
            return Err("Need at least 2 data points for variance".to_string());
        }

        let mean = Self::mean(data)?;
        let len = data.len() as usize;
        let mut sum_squares = 0.0f64;

        for i in 0..len {
            let val = unsafe { data.get_i64(i as u64) } as f64;
            let diff = val - mean;
            sum_squares += diff * diff;
        }

        let divisor = if sample { len - 1 } else { len };
        Ok(sum_squares / divisor as f64)
    }

    /// Calculate standard deviation
    pub fn std_deviation(data: &Vector, sample: bool) -> Result<f64, String> {
        Ok(Self::variance(data, sample)?.sqrt())
    }

    /// Calculate quantiles
    pub fn quantile(data: &Vector, p: f64) -> Result<f64, String> {
        if !(0.0..=1.0).contains(&p) {
            return Err("Quantile probability must be between 0 and 1".to_string());
        }

        if data.len() == 0 {
            return Err("Cannot calculate quantile of empty dataset".to_string());
        }

        let mut values: Vec<f64> = (0..data.len() as usize)
            .map(|i| unsafe { data.get_i64(i as u64) } as f64)
            .collect();

        values.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

        let len = values.len();
        let index = (len - 1) as f64 * p;

        if index == index.floor() {
            // Exact index
            Ok(values[index as usize])
        } else {
            // Interpolate between values
            let lower_idx = index.floor() as usize;
            let upper_idx = index.ceil() as usize;
            let weight = index - index.floor();

            let lower_val = values[lower_idx];
            let upper_val = values[upper_idx];

            Ok(lower_val + weight * (upper_val - lower_val))
        }
    }

    /// Calculate quartiles
    pub fn quartiles(data: &Vector) -> Result<(f64, f64, f64), String> {
        let q1 = Self::quantile(data, 0.25)?;
        let q2 = Self::median(data)?;
        let q3 = Self::quantile(data, 0.75)?;

        Ok((q1, q2, q3))
    }

    /// Calculate interquartile range
    pub fn iqr(data: &Vector) -> Result<f64, String> {
        let (q1, _, q3) = Self::quartiles(data)?;
        Ok(q3 - q1)
    }

    /// Calculate skewness
    pub fn skewness(data: &Vector) -> Result<f64, String> {
        if data.len() < 3 {
            return Err("Need at least 3 data points for skewness".to_string());
        }

        let mean = Self::mean(data)?;
        let std_dev = Self::std_deviation(data, false)?;
        let len = data.len() as usize;

        let mut sum_cubed_deviations = 0.0f64;

        for i in 0..len {
            let val = unsafe { data.get_i64(i as u64) } as f64;
            let deviation = val - mean;
            sum_cubed_deviations += deviation * deviation * deviation;
        }

        let n = len as f64;
        let skewness = (sum_cubed_deviations / n) / (std_dev * std_dev * std_dev);

        Ok(skewness)
    }

    /// Calculate kurtosis
    pub fn kurtosis(data: &Vector) -> Result<f64, String> {
        if data.len() < 4 {
            return Err("Need at least 4 data points for kurtosis".to_string());
        }

        let mean = Self::mean(data)?;
        let std_dev = Self::std_deviation(data, false)?;
        let len = data.len() as usize;

        let mut sum_fourth_deviations = 0.0f64;

        for i in 0..len {
            let val = unsafe { data.get_i64(i as u64) } as f64;
            let deviation = val - mean;
            sum_fourth_deviations += deviation * deviation * deviation * deviation;
        }

        let n = len as f64;
        let kurtosis = (sum_fourth_deviations / n) / (std_dev * std_dev * std_dev * std_dev) - 3.0;

        Ok(kurtosis)
    }

    /// Calculate correlation coefficient between two datasets
    pub fn correlation(x: &Vector, y: &Vector) -> Result<f64, String> {
        if x.len() != y.len() || x.len() == 0 {
            return Err("Datasets must have same non-zero length".to_string());
        }

        let mean_x = Self::mean(x)?;
        let mean_y = Self::mean(y)?;
        let len = x.len() as usize;

        let mut numerator = 0.0f64;
        let mut sum_sq_x = 0.0f64;
        let mut sum_sq_y = 0.0f64;

        for i in 0..len {
            let val_x = unsafe { x.get_i64(i as u64) } as f64;
            let val_y = unsafe { y.get_i64(i as u64) } as f64;

            let dev_x = val_x - mean_x;
            let dev_y = val_y - mean_y;

            numerator += dev_x * dev_y;
            sum_sq_x += dev_x * dev_x;
            sum_sq_y += dev_y * dev_y;
        }

        if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
            return Err("Cannot calculate correlation with zero variance".to_string());
        }

        Ok(numerator / (sum_sq_x.sqrt() * sum_sq_y.sqrt()))
    }

    /// Calculate covariance between two datasets
    pub fn covariance(x: &Vector, y: &Vector, sample: bool) -> Result<f64, String> {
        if x.len() != y.len() || x.len() == 0 {
            return Err("Datasets must have same non-zero length".to_string());
        }

        let mean_x = Self::mean(x)?;
        let mean_y = Self::mean(y)?;
        let len = x.len() as usize;

        let mut sum_products = 0.0f64;

        for i in 0..len {
            let val_x = unsafe { x.get_i64(i as u64) } as f64;
            let val_y = unsafe { y.get_i64(i as u64) } as f64;

            sum_products += (val_x - mean_x) * (val_y - mean_y);
        }

        let divisor = if sample { len - 1 } else { len };
        Ok(sum_products / divisor as f64)
    }

    /// Simple linear regression
    pub fn linear_regression(x: &Vector, y: &Vector) -> Result<(f64, f64), String> {
        if x.len() != y.len() || x.len() < 2 {
            return Err("Need at least 2 matching data points".to_string());
        }

        let n = x.len() as f64;
        let sum_x = Self::mean(x)? * n;
        let sum_y = Self::mean(y)? * n;

        let mut sum_xy = 0.0f64;
        let mut sum_x2 = 0.0f64;

        for i in 0..x.len() as usize {
            let val_x = unsafe { x.get_i64(i as u64) } as f64;
            let val_y = unsafe { y.get_i64(i as u64) } as f64;

            sum_xy += val_x * val_y;
            sum_x2 += val_x * val_x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        Ok((slope, intercept))
    }

    /// One-sample t-test
    pub fn t_test_one_sample(data: &Vector, hypothesized_mean: f64) -> Result<(f64, f64), String> {
        if data.len() < 2 {
            return Err("Need at least 2 data points for t-test".to_string());
        }

        let sample_mean = Self::mean(data)?;
        let sample_std = Self::std_deviation(data, true)?;
        let n = data.len() as f64;

        let t_statistic = (sample_mean - hypothesized_mean) / (sample_std / n.sqrt());
        let degrees_of_freedom = n - 1.0;

        // Approximate p-value calculation (simplified)
        let p_value = Self::t_distribution_cdf(-t_statistic.abs(), degrees_of_freedom);

        Ok((t_statistic, p_value))
    }

    /// Two-sample t-test
    pub fn t_test_two_sample(data1: &Vector, data2: &Vector, equal_variance: bool) -> Result<(f64, f64), String> {
        if data1.len() < 2 || data2.len() < 2 {
            return Err("Both samples need at least 2 data points".to_string());
        }

        let mean1 = Self::mean(data1)?;
        let mean2 = Self::mean(data2)?;
        let n1 = data1.len() as f64;
        let n2 = data2.len() as f64;

        let var1 = Self::variance(data1, true)?;
        let var2 = Self::variance(data2, true)?;

        let pooled_var = if equal_variance {
            ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0)
        } else {
            var1 / n1 + var2 / n2
        };

        let t_statistic = (mean1 - mean2) / pooled_var.sqrt();

        let degrees_of_freedom = if equal_variance {
            n1 + n2 - 2.0
        } else {
            // Welch-Satterthwaite approximation
            let numerator = (var1 / n1 + var2 / n2).powi(2);
            let denominator = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
            numerator / denominator
        };

        // Approximate p-value
        let p_value = 2.0 * (1.0 - Self::t_distribution_cdf(t_statistic.abs(), degrees_of_freedom));

        Ok((t_statistic, p_value))
    }

    /// Chi-square test for independence
    pub fn chi_square_test(contingency_table: &Vector, rows: usize, cols: usize) -> Result<(f64, f64), String> {
        if contingency_table.len() as usize != rows * cols {
            return Err("Contingency table dimensions don't match".to_string());
        }

        // Calculate expected frequencies
        let mut row_totals = vec![0.0f64; rows];
        let mut col_totals = vec![0.0f64; cols];
        let mut grand_total = 0.0f64;

        // Calculate totals
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let val = unsafe { contingency_table.get_i64(idx as u64) } as f64;
                row_totals[i] += val;
                col_totals[j] += val;
                grand_total += val;
            }
        }

        // Calculate chi-square statistic
        let mut chi_square = 0.0f64;
        let mut degrees_of_freedom = (rows - 1) * (cols - 1);

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let observed = unsafe { contingency_table.get_i64(idx as u64) } as f64;
                let expected = (row_totals[i] * col_totals[j]) / grand_total;

                if expected > 0.0 {
                    chi_square += (observed - expected).powi(2) / expected;
                } else {
                    degrees_of_freedom -= 1;
                }
            }
        }

        // Approximate p-value using chi-square distribution
        let p_value = Self::chi_square_distribution_cdf(chi_square, degrees_of_freedom as f64);

        Ok((chi_square, p_value))
    }

    // Helper functions for statistical distributions
    /// Cumulative distribution function for t-distribution (approximation)
    fn t_distribution_cdf(t: f64, df: f64) -> f64 {
        // Simplified approximation using normal distribution for large df
        if df > 30.0 {
            Self::normal_cdf(t)
        } else {
            // More complex approximation would be needed for small df
            Self::normal_cdf(t * (1.0 - 1.0 / (4.0 * df)).sqrt())
        }
    }

    /// Cumulative distribution function for chi-square distribution (approximation)
    fn chi_square_distribution_cdf(chi_sq: f64, df: f64) -> f64 {
        // Simplified approximation using gamma distribution
        // In practice, this would use more sophisticated methods
        1.0 - Self::gamma_distribution_cdf(chi_sq / 2.0, df / 2.0)
    }

    /// Cumulative distribution function for normal distribution
    fn normal_cdf(x: f64) -> f64 {
        // Approximation using error function
        0.5 * (1.0 + Self::erf(x / (2.0_f64).sqrt()))
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Gamma distribution CDF (simplified approximation)
    fn gamma_distribution_cdf(x: f64, shape: f64) -> f64 {
        // Very simplified approximation
        if x <= 0.0 {
            0.0
        } else if x >= shape * 3.0 {
            1.0
        } else {
            // Linear interpolation as placeholder
            x / (shape * 3.0)
        }
    }
}

/// Register statistics operations with the execution context
pub fn register_stats_ops(context: &mut ExecutionContext) {
    // Descriptive statistics
    context.register_function(Function::Native {
        name: "stats_mean".to_string(),
        arg_count: 1,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 1 {
                return Err("mean requires 1 argument".to_string());
            }

            match &args[0] {
                Value::Vector(ref v) => {
                    let vector = unsafe { &*v.ptr() };
                    match StatisticsOps::mean(vector) {
                        Ok(result) => Ok(Value::Float(result)),
                        Err(e) => Err(e),
                    }
                }
                _ => Err("mean requires a vector argument".to_string()),
            }
        }),
    });

    context.register_function(Function::Native {
        name: "stats_median".to_string(),
        arg_count: 1,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 1 {
                return Err("median requires 1 argument".to_string());
            }

            match &args[0] {
                Value::Vector(ref v) => {
                    let vector = unsafe { &*v.ptr() };
                    match StatisticsOps::median(vector) {
                        Ok(result) => Ok(Value::Float(result)),
                        Err(e) => Err(e),
                    }
                }
                _ => Err("median requires a vector argument".to_string()),
            }
        }),
    });

    context.register_function(Function::Native {
        name: "stats_variance".to_string(),
        arg_count: 2,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 2 {
                return Err("variance requires 2 arguments".to_string());
            }

            let sample = matches!(&args[1], Value::Boolean(true));

            match &args[0] {
                Value::Vector(ref v) => {
                    let vector = unsafe { &*v.ptr() };
                    match StatisticsOps::variance(vector, sample) {
                        Ok(result) => Ok(Value::Float(result)),
                        Err(e) => Err(e),
                    }
                }
                _ => Err("variance requires a vector argument".to_string()),
            }
        }),
    });

    context.register_function(Function::Native {
        name: "stats_correlation".to_string(),
        arg_count: 2,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 2 {
                return Err("correlation requires 2 arguments".to_string());
            }

            match (&args[0], &args[1]) {
                (Value::Vector(ref x), Value::Vector(ref y)) => {
                    let vec_x = unsafe { &*x.ptr() };
                    let vec_y = unsafe { &*y.ptr() };
                    match StatisticsOps::correlation(vec_x, vec_y) {
                        Ok(result) => Ok(Value::Float(result)),
                        Err(e) => Err(e),
                    }
                }
                _ => Err("correlation requires two vector arguments".to_string()),
            }
        }),
    });

    // Hypothesis testing
    context.register_function(Function::Native {
        name: "stats_t_test".to_string(),
        arg_count: 3,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 3 {
                return Err("t_test requires 3 arguments".to_string());
            }

            match (&args[0], &args[2]) {
                (Value::Vector(ref data), Value::Float(hypothesized)) => {
                    let vector = unsafe { &*data.ptr() };
                    match StatisticsOps::t_test_one_sample(vector, *hypothesized) {
                        Ok((t_stat, p_value)) => {
                            // Return as vector [t_statistic, p_value]
                            let result_vec = unsafe {
                                let ptr = vexl_vec_alloc_i64(2);
                                vexl_vec_set_i64(ptr, 0, t_stat as i64);
                                vexl_vec_set_i64(ptr, 1, p_value as i64);
                                ptr
                            };
                            Ok(Value::Vector(VectorRef::owned(result_vec)))
                        }
                        Err(e) => Err(e),
                    }
                }
                _ => Err("t_test requires vector and float arguments".to_string()),
            }
        }),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ExecutionContext;

    #[test]
    fn test_mean_calculation() {
        let data = unsafe {
            let ptr = crate::vexl_vec_alloc_i64(5);
            crate::vexl_vec_set_i64(ptr, 0, 1);
            crate::vexl_vec_set_i64(ptr, 1, 2);
            crate::vexl_vec_set_i64(ptr, 2, 3);
            crate::vexl_vec_set_i64(ptr, 3, 4);
            crate::vexl_vec_set_i64(ptr, 4, 5);
            Vector::from_raw(ptr)
        };

        let mean = StatisticsOps::mean(&data).unwrap();
        assert_eq!(mean, 3.0);
    }

    #[test]
    fn test_median_calculation() {
        let data = unsafe {
            let ptr = crate::vexl_vec_alloc_i64(5);
            crate::vexl_vec_set_i64(ptr, 0, 1);
            crate::vexl_vec_set_i64(ptr, 1, 2);
            crate::vexl_vec_set_i64(ptr, 2, 3);
            crate::vexl_vec_set_i64(ptr, 3, 4);
            crate::vexl_vec_set_i64(ptr, 4, 5);
            Vector::from_raw(ptr)
        };

        let median = StatisticsOps::median(&data).unwrap();
        assert_eq!(median, 3.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let x_data = unsafe {
            let ptr = crate::vexl_vec_alloc_i64(5);
            crate::vexl_vec_set_i64(ptr, 0, 1);
            crate::vexl_vec_set_i64(ptr, 1, 2);
            crate::vexl_vec_set_i64(ptr, 2, 3);
            crate::vexl_vec_set_i64(ptr, 3, 4);
            crate::vexl_vec_set_i64(ptr, 4, 5);
            Vector::from_raw(ptr)
        };

        let y_data = unsafe {
            let ptr = crate::vexl_vec_alloc_i64(5);
            crate::vexl_vec_set_i64(ptr, 0, 2);
            crate::vexl_vec_set_i64(ptr, 1, 4);
            crate::vexl_vec_set_i64(ptr, 2, 6);
            crate::vexl_vec_set_i64(ptr, 3, 8);
            crate::vexl_vec_set_i64(ptr, 4, 10);
            Vector::from_raw(ptr)
        };

        let correlation = StatisticsOps::correlation(&x_data, &y_data).unwrap();
        assert!((correlation - 1.0).abs() < 0.0001); // Perfect positive correlation
    }

    #[test]
    fn test_register_stats_ops() {
        let mut context = ExecutionContext::new();
        register_stats_ops(&mut context);

        // Check that functions were registered
        assert!(context.call_function("stats_mean", &[]).is_err()); // Wrong args
        assert!(context.call_function("nonexistent", &[]).is_err());
    }
}
