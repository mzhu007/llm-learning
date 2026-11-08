
use std::time::Instant;

// Linear Congruential Generator (LCG)
struct Lcg {
    value: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { value: seed }
    }

    fn next(&mut self) -> u64 {
        const A: u64 = 1_664_525;
        const C: u64 = 1_013_904_223;
        const M: u64 = 1u64 << 32;

        // Use u128 for intermediate multiplication
        self.value = ((A as u128 * self.value as u128
            + C as u128) % M as u128) as u64;

        self.value
    }
}

// Generate random numbers and calculate maximum subarray sum
fn max_subarray_sum(
    n: usize,
    seed: u64,
    min_val: i64,
    max_val: i64,
) -> i128 {
    let mut lcg = Lcg::new(seed);

    let range = (max_val as i128 - min_val as i128 + 1) as u64;

    let mut max_sum = i128::MIN;
    let mut current_sum: i128 = 0;

    // Kadane's algorithm
    for _ in 0..n {
        let random_value = lcg.next();

        let value = (random_value % range) as i128
            + min_val as i128;

        current_sum = current_sum.max(0) + value;
        max_sum = max_sum.max(current_sum);
    }

    max_sum
}

// Calculate total maximum subarray sum over 20 runs
fn total_max_subarray_sum(
    n: usize,
    initial_seed: u64,
    min_val: i64,
    max_val: i64,
) -> i128 {
    let mut lcg = Lcg::new(initial_seed);
    let mut total_sum: i128 = 0;

    for _ in 0..20 {
        let seed = lcg.next();

        total_sum += max_subarray_sum(
            n,
            seed,
            min_val,
            max_val,
        );
    }

    total_sum
}

fn main() {
    // Parameters
    let n: usize = 10_000;
    let initial_seed: u64 = 42;
    let min_val: i64 = -10;
    let max_val: i64 = 10;

    // Timing
    let start_time = Instant::now();

    let result = total_max_subarray_sum(
        n,
        initial_seed,
        min_val,
        max_val,
    );

    let elapsed = start_time.elapsed();

    println!(
        "Total Maximum Subarray Sum (20 runs): {}",
        result
    );

    println!(
        "Execution Time: {:.6} seconds",
        elapsed.as_secs_f64()
    );
}