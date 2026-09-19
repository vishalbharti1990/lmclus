use std::time::Instant;
use rand::prelude::*;

use lmclus::distance::{distance_to_manifold, distance_to_manifold_parallel};
use lmclus::separation::find_separation;
use lmclus::sampling::{run_trials_serial, run_trials_parallel};

fn main() {
    println!("===================================================================================");
    println!("LMCLUS RUST COMPONENT BENCHMARK SUITE");
    println!("Rustc 1.98.1 | Optimized Release (LTO=fat, codegen-units=1)");
    println!("Rayon threads: {}", rayon::current_num_threads());
    println!("===================================================================================\n");

    let d = 50;
    let n = 20_000;
    let m = 2;

    println!("Setting up synthetic benchmark dataset: n = {n}, d = {d}, m = {m}...");
    let mut rng = StdRng::seed_from_u64(42);
    let mut x = vec![0.0; d * n];
    for val in x.iter_mut() {
        *val = rng.random::<f64>() * 2.0 - 1.0;
    }

    let mut origin = vec![0.0; d];
    for val in origin.iter_mut() {
        *val = rng.random::<f64>() * 2.0 - 1.0;
    }

    let mut basis = vec![0.0; d * m];
    for val in basis.iter_mut() {
        *val = rng.random::<f64>() * 2.0 - 1.0;
    }
    // Orthogonalize dummy basis
    lmclus::sampling::gram_schmidt(&mut basis, d, m);

    // =========================================================================
    // BENCHMARK 1: distance_to_manifold
    // =========================================================================
    println!("\n--- Benchmark 1: Distance to Manifold Kernel (n = {n}, d = {d}, m = {m}) ---");
    let mut distances_seq = vec![0.0; n];
    let mut distances_par = vec![0.0; n];

    // Warmup
    for _ in 0..10 {
        distance_to_manifold(&x, d, n, &origin, &basis, m, &mut distances_seq);
        distance_to_manifold_parallel(&x, d, n, &origin, &basis, m, &mut distances_par);
    }

    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        distance_to_manifold(&x, d, n, &origin, &basis, m, &mut distances_seq);
    }
    let elapsed_seq = t0.elapsed().as_secs_f64() / (iters as f64);

    let t0 = Instant::now();
    for _ in 0..iters {
        distance_to_manifold_parallel(&x, d, n, &origin, &basis, m, &mut distances_par);
    }
    let elapsed_par = t0.elapsed().as_secs_f64() / (iters as f64);

    println!("  Rust (Single-threaded):  {:>8.3} µs / call | Throughput: {:>8.2} M pts/sec",
             elapsed_seq * 1e6, (n as f64) / (elapsed_seq * 1e6));
    println!("  Rust (Rayon Parallel):   {:>8.3} µs / call | Throughput: {:>8.2} M pts/sec | Speedup: {:.2}x",
             elapsed_par * 1e6, (n as f64) / (elapsed_par * 1e6), elapsed_seq / elapsed_par);

    // =========================================================================
    // BENCHMARK 2: Histogram Binning & Kittler Separation
    // =========================================================================
    println!("\n--- Benchmark 2: Histogram Binning + Kittler Valley Search (n = {n}) ---");
    let mut scratch_sort = vec![0.0; n];
    let mut bins_buffer = vec![0.0; 256];

    // Warmup
    for _ in 0..10 {
        let _ = find_separation(&distances_seq, &mut scratch_sort, &mut bins_buffer, 0.1, 0, 7, 1e-5);
    }

    let iters_hist = 500;
    let t0 = Instant::now();
    for _ in 0..iters_hist {
        let _ = find_separation(&distances_seq, &mut scratch_sort, &mut bins_buffer, 0.1, 0, 7, 1e-5);
    }
    let elapsed_hist = t0.elapsed().as_secs_f64() / (iters_hist as f64);
    println!("  Rust Histogram + Kittler: {:>8.3} µs / call | Calls/sec: {:>8.0}",
             elapsed_hist * 1e6, 1.0 / elapsed_hist);

    // =========================================================================
    // BENCHMARK 3: End-to-End Candidate Plane Search (Q = 100 trials)
    // =========================================================================
    let q = 100;
    println!("\n--- Benchmark 3: Candidate Plane Search (Q = {q} trials, n = {n}, d = {d}, m = {m}) ---");

    // Warmup
    let _ = run_trials_serial(&x, d, n, m, 5, 42);
    let _ = run_trials_parallel(&x, d, n, m, 5, 42);

    let repeats = 5;
    let mut seq_times = Vec::new();
    for _ in 0..repeats {
        let t0 = Instant::now();
        let (sep, _, _) = run_trials_serial(&x, d, n, m, q, 42);
        seq_times.push(t0.elapsed().as_secs_f64());
        let _ = sep;
    }

    let mut par_times = Vec::new();
    for _ in 0..repeats {
        let t0 = Instant::now();
        let (sep, _, _) = run_trials_parallel(&x, d, n, m, q, 42);
        par_times.push(t0.elapsed().as_secs_f64());
        let _ = sep;
    }

    let min_seq = seq_times.iter().copied().fold(f64::INFINITY, f64::min);
    let min_par = par_times.iter().copied().fold(f64::INFINITY, f64::min);

    println!("  Rust Single-Threaded: Min: {:>7.4}s | Mean: {:>7.4}s",
             min_seq, seq_times.iter().sum::<f64>() / (repeats as f64));
    println!("  Rust Rayon Parallel:  Min: {:>7.4}s | Mean: {:>7.4}s | Multi-thread Speedup: {:.2}x",
             min_par, par_times.iter().sum::<f64>() / (repeats as f64), min_seq / min_par);

    println!("\n===================================================================================");
    println!("Rust micro-benchmarks completed successfully.");
    println!("===================================================================================");
}
