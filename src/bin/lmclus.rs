use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use lmclus::{lmclus, Parameters};

fn print_help() {
    println!(r#"Linear Manifold Clustering (LMCLUS) Engine

USAGE:
    lmclus [OPTIONS] <csv_path> <max_dim> [k_nominal] [seed] [out_labels_bin]
    lmclus --help | -h
    lmclus --version | -v

ARGUMENTS:
    <csv_path>
        Path to the input CSV file containing data points.
        Format: Each line represents a point with comma-separated feature values.
        By default, ALL columns are treated as feature dimensions.

    <max_dim>
        Maximum subspace manifold dimension to search (must be >= 1 and < feature dimension d).
        For example:
          1: search for 1D lines
          2: search for 1D lines and 2D planes
          3: search for 1D lines, 2D planes, and 3D hyperplanes

    [k_nominal]
        Nominal / expected number of clusters in the dataset (default: 4).
        Used by the stochastic sampling heuristic to size the number of random
        trials needed to find candidate manifolds with high probability.

    [seed]
        64-bit unsigned integer random seed (default: 42).
        Guarantees deterministic and reproducible trial generation across runs.

    [out_labels_bin]
        Optional output file path to write predicted point cluster assignments.
        Format: Raw little-endian binary array of 64-bit integers (i64).
        Points assigned to cluster 0 have label 0, cluster 1 have label 1, etc.
        Unclustered noise points have label -1.
        Can be read directly in Python via:
            np.fromfile("labels.bin", dtype=np.int64)
        Or in Julia via:
            reinterpret(Int64, read("labels.bin"))

OPTIONS:
    -l, --has-labels
        Specify that the last column in the CSV contains ground-truth integer labels.
        When set, the last column is excluded from features, and external validation
        metrics (NMI, ARI, and Purity) are calculated and reported in the output.
        Default: false (all columns are features).

    -h, --help
        Print this detailed help guide and exit.

    -v, --version
        Print version information and exit.

EXAMPLES:
    # Cluster unlabelled data (all CSV columns are features):
    lmclus data.csv 2

    # Cluster benchmark data where the last column has ground-truth labels:
    lmclus --has-labels benchmark.csv 4

    # Cluster expecting ~5 clusters, using random seed 1234:
    lmclus data.csv 3 5 1234

    # Cluster and export point assignments to binary file:
    lmclus data.csv 2 4 42 predicted_labels.bin
"#);
}

fn main() {
    let mut has_labels = false;
    let mut positional: Vec<String> = Vec::new();

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--version" | "-v" => {
                println!("lmclus {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            "--has-labels" | "--labels" | "-l" => {
                has_labels = true;
            }
            _ => {
                positional.push(arg);
            }
        }
    }

    if positional.len() < 2 {
        print_help();
        std::process::exit(1);
    }

    let csv_path = &positional[0];
    let max_dim: usize = match positional[1].parse() {
        Ok(m) if m > 0 => m,
        _ => {
            eprintln!("Error: <max_dim> must be a positive integer (got '{}')", positional[1]);
            std::process::exit(1);
        }
    };
    let k_nominal: usize = if positional.len() >= 3 {
        positional[2].parse().unwrap_or(4)
    } else {
        4
    };
    let seed: u64 = if positional.len() >= 4 {
        positional[3].parse().unwrap_or(42)
    } else {
        42
    };
    let out_bin = if positional.len() >= 5 { Some(&positional[4]) } else { None };

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
        if parts.is_empty() {
            continue;
        }

        if has_labels {
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
        } else {
            let feats: Vec<f64> = parts
                .iter()
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            rows.push(feats);
        }
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

    // 3. Setup Parameters
    let mut p = Parameters::new(max_dim);
    p.number_of_clusters = k_nominal;
    p.random_seed = seed;

    // 4. Execution run
    let t0 = Instant::now();
    let res = lmclus(&x, d, n, &p);
    let elapsed = t0.elapsed().as_secs_f64();
    let k_found = res.nclusters();
    let counts = res.counts();

    let assigns = res.assignments(n);

    // 5. Output assignments if requested
    if let Some(out_path) = out_bin {
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
    println!("  \"elapsed_time\": {elapsed:.6},");
    println!("  \"nclusters\": {k_found},");
    if has_labels {
        let m = compute_metrics(&ground_truth, &assigns);
        println!("  \"counts\": {:?},", counts);
        println!("  \"nmi\": {:.4},", m.nmi);
        println!("  \"ari\": {:.4},", m.ari);
        println!("  \"purity\": {:.4}", m.purity);
    } else {
        println!("  \"counts\": {:?}", counts);
    }
    println!("}}");
}

struct ClusterMetrics {
    nmi: f64,
    ari: f64,
    purity: f64,
}

fn compute_metrics(y_true: &[i64], y_pred: &[usize]) -> ClusterMetrics {
    use std::collections::{HashMap, HashSet};
    let n = y_true.len() as f64;
    if n == 0.0 {
        return ClusterMetrics { nmi: 0.0, ari: 0.0, purity: 0.0 };
    }

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

    // 1. Normalized Mutual Information (NMI)
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

    let nmi = if h_true + h_pred == 0.0 {
        1.0
    } else {
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
    };

    // 2. Adjusted Rand Index (ARI)
    let choose2 = |x: f64| -> f64 {
        if x < 2.0 { 0.0 } else { x * (x - 1.0) * 0.5 }
    };

    let mut sum_comb_contingency = 0.0;
    for &cnt in contingency.values() {
        sum_comb_contingency += choose2(cnt);
    }

    let mut sum_comb_rows = 0.0;
    for &rs in &row_sums {
        sum_comb_rows += choose2(rs);
    }

    let mut sum_comb_cols = 0.0;
    for &cs in &col_sums {
        sum_comb_cols += choose2(cs);
    }

    let total_comb = choose2(n);
    let expected_index = if total_comb > 0.0 {
        (sum_comb_rows * sum_comb_cols) / total_comb
    } else {
        0.0
    };
    let max_index = 0.5 * (sum_comb_rows + sum_comb_cols);
    let denom = max_index - expected_index;

    let ari = if denom.abs() < 1e-12 {
        if (sum_comb_contingency - expected_index).abs() < 1e-12 { 1.0 } else { 0.0 }
    } else {
        ((sum_comb_contingency - expected_index) / denom).clamp(-1.0, 1.0)
    };

    // 3. Purity: (sum_c max_r n_{rc}) / n
    let mut cluster_max_counts = vec![0.0; num_clusters];
    for (&(_r, c), &cnt) in &contingency {
        if cnt > cluster_max_counts[c] {
            cluster_max_counts[c] = cnt;
        }
    }
    let purity = (cluster_max_counts.iter().sum::<f64>() / n).clamp(0.0, 1.0);

    ClusterMetrics { nmi, ari, purity }
}

