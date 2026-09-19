use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use lmclus::{lmclus, Parameters};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: lmclus_cli <csv_path> <max_dim> [k] [seed] [out_labels_bin]");
        std::process::exit(1);
    }

    let csv_path = &args[1];
    let max_dim: usize = args[2].parse().expect("max_dim must be positive int");
    let _k_nominal: usize = if args.len() >= 4 { args[3].parse().unwrap_or(4) } else { 4 };
    let seed: u64 = if args.len() >= 5 { args[4].parse().unwrap_or(42) } else { 42 };
    let out_bin = if args.len() >= 6 { Some(&args[5]) } else { None };

    if !Path::new(csv_path).exists() {
        eprintln!("Error: CSV file not found: {csv_path}");
        std::process::exit(1);
    }

    // 1. Read CSV: row-by-row
    let file = File::open(csv_path).expect("failed to open csv");
    let reader = BufReader::new(file);

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut ground_truth: Vec<i64> = Vec::new();

    for line_res in reader.lines() {
        let line = line_res.expect("failed to read line");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let label: i64 = parts.last().unwrap().trim().parse::<f64>().unwrap_or(0.0) as i64;
        let feats: Vec<f64> = parts[..parts.len() - 1]
            .iter()
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        ground_truth.push(label);
        rows.push(feats);
    }

    let n = rows.len();
    if n == 0 {
        eprintln!("Error: Empty CSV");
        std::process::exit(1);
    }
    let d = rows[0].len();

    // 2. Transpose to column-major (d x n)
    let mut x = vec![0.0; d * n];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            x[i * d + j] = val;
        }
    }

    // 3. Setup Parameters matching Julia and Python benchmark configurations
    let mut p = Parameters::new(max_dim);
    p.random_seed = seed;

    // 4. Warmup run
    let _ = lmclus(&x, d, n, &p);

    // 5. Timed runs
    let repeats = 5;
    let mut times = Vec::new();
    let mut last_res = None;

    for _ in 0..repeats {
        let t0 = Instant::now();
        let res = lmclus(&x, d, n, &p);
        let elapsed = t0.elapsed().as_secs_f64();
        times.push(elapsed);
        last_res = Some(res);
    }

    let res = last_res.unwrap();
    let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mean_time: f64 = times.iter().sum::<f64>() / (repeats as f64);
    let k_found = res.nclusters();
    let counts = res.counts();

    let assigns = res.assignments(n);
    let nmi = compute_nmi(&ground_truth, &assigns);

    // 6. Output assignments if requested
    if let Some(out_path) = out_bin {
        // Write as binary i64 array for exact parity with Julia/Python benchmark metrics
        let mut out_file = File::create(out_path).expect("failed to create labels binary");
        for &a in &assigns {
            let label_i64 = a as i64;
            out_file.write_all(&label_i64.to_le_bytes()).expect("write error");
        }
    }

    println!("{{");
    println!("  \"dataset\": \"{csv_path}\",");
    println!("  \"n\": {n},");
    println!("  \"d\": {d},");
    println!("  \"min_time\": {min_time:.6},");
    println!("  \"mean_time\": {mean_time:.6},");
    println!("  \"times\": {:?},", times);
    println!("  \"nclusters\": {k_found},");
    println!("  \"counts\": {:?},", counts);
    println!("  \"nmi\": {nmi:.4}");
    println!("}}");
}

fn compute_nmi(y_true: &[i64], y_pred: &[usize]) -> f64 {
    use std::collections::{HashMap, HashSet};
    let n = y_true.len() as f64;
    let mut class_map = HashMap::new();
    let mut cluster_map = HashMap::new();
    let mut contingency: HashMap<(usize, usize), f64> = HashMap::new();

    let mut classes = HashSet::new();
    for &y in y_true { classes.insert(y); }
    let mut class_list: Vec<i64> = classes.into_iter().collect();
    class_list.sort();
    for (i, &c) in class_list.iter().enumerate() { class_map.insert(c, i); }

    let mut clusters = HashSet::new();
    for &y in y_pred { clusters.insert(y); }
    let mut cluster_list: Vec<usize> = clusters.into_iter().collect();
    cluster_list.sort();
    for (i, &c) in cluster_list.iter().enumerate() { cluster_map.insert(c, i); }

    let num_classes = class_list.len();
    let num_clusters = cluster_list.len();

    let mut row_sums = vec![0.0; num_classes];
    let mut col_sums = vec![0.0; num_clusters];

    for i in 0..y_true.len() {
        let r = class_map[&y_true[i]];
        let c = cluster_map[&y_pred[i]];
        *contingency.entry((r, c)).or_insert(0.0) += 1.0;
        row_sums[r] += 1.0;
        col_sums[c] += 1.0;
    }

    let mut h_true = 0.0;
    for &rs in &row_sums {
        let p = rs / n;
        if p > 0.0 { h_true -= p * p.ln(); }
    }

    let mut h_pred = 0.0;
    for &cs in &col_sums {
        let p = cs / n;
        if p > 0.0 { h_pred -= p * p.ln(); }
    }

    if h_true + h_pred == 0.0 {
        return 1.0;
    }

    let mut mi = 0.0;
    for (&(r, c), &cnt) in &contingency {
        let p_joint = cnt / n;
        let p_r = row_sums[r] / n;
        let p_c = col_sums[c] / n;
        if p_joint > 0.0 {
            mi += p_joint * (p_joint / (p_r * p_c)).ln();
        }
    }

    (2.0 * mi / (h_true + h_pred)).clamp(0.0, 1.0)
}

