use rand::prelude::*;
use rayon::prelude::*;

use crate::distance::distance_to_manifold;
use crate::separation::{find_separation, Separation};

/// Gram-Schmidt orthogonalization of vectors in V (d x m, column-major)
pub fn gram_schmidt(v: &mut [f64], d: usize, m: usize) {
    for j in 0..m {
        for i in 0..j {
            let mut dot = 0.0;
            for r in 0..d {
                dot += v[i * d + r] * v[j * d + r];
            }
            for r in 0..d {
                v[j * d + r] -= dot * v[i * d + r];
            }
        }
        let mut norm_sq = 0.0;
        for r in 0..d {
            let val = v[j * d + r];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for r in 0..d {
                v[j * d + r] *= inv;
            }
        }
    }
}

/// Form origin (d) and orthonormal basis (d x m) from m+1 sample indices
pub fn form_basis_from_sample(
    x: &[f64],
    d: usize,
    sample_indices: &[usize],
    origin: &mut [f64],
    basis: &mut [f64],
) {
    let m = sample_indices.len() - 1;
    let o_idx = sample_indices[0];

    // Origin = X[:, o_idx]
    origin.copy_from_slice(&x[o_idx * d..(o_idx + 1) * d]);

    // Basis columns: V[:, j] = X[:, sample_indices[j+1]] - origin
    for j in 0..m {
        let p_idx = sample_indices[j + 1];
        let p_col = &x[p_idx * d..(p_idx + 1) * d];
        for r in 0..d {
            basis[j * d + r] = p_col[r] - origin[r];
        }
    }

    gram_schmidt(basis, d, m);
}

/// Scratch buffers for a single trial worker thread (zero allocations inside loop)
pub struct TrialWorkspace {
    pub origin: Vec<f64>,
    pub basis: Vec<f64>,
    pub distances: Vec<f64>,
    pub scratch_sort: Vec<f64>,
    pub bins_buffer: Vec<f64>,
    pub sample_indices: Vec<usize>,
}

impl TrialWorkspace {
    pub fn new(d: usize, n: usize, m: usize) -> Self {
        Self {
            origin: vec![0.0; d],
            basis: vec![0.0; d * m],
            distances: vec![0.0; n],
            scratch_sort: vec![0.0; n],
            bins_buffer: vec![0.0; n.max(256)],
            sample_indices: vec![0; m + 1],
        }
    }
}

/// Execute Q trial manifolds sequentially (Single-threaded)
pub fn run_trials_serial(
    x: &[f64],
    d: usize,
    n: usize,
    m: usize,
    num_trials: usize,
    seed: u64,
) -> (Separation, Vec<f64>, Vec<f64>) {
    let mut ws = TrialWorkspace::new(d, n, m);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_sep = Separation::default();
    let mut best_origin = vec![0.0; d];
    let mut best_basis = vec![0.0; d * m];

    for _ in 0..num_trials {
        // Draw m+1 distinct points
        let mut count = 0;
        while count < m + 1 {
            let idx = rng.random_range(0..n);
            if !ws.sample_indices[..count].contains(&idx) {
                ws.sample_indices[count] = idx;
                count += 1;
            }
        }

        form_basis_from_sample(x, d, &ws.sample_indices, &mut ws.origin, &mut ws.basis);
        distance_to_manifold(x, d, n, &ws.origin, &ws.basis, m, &mut ws.distances);

        let sep = find_separation(
            &ws.distances,
            &mut ws.scratch_sort,
            &mut ws.bins_buffer,
            0.1,
            0,
            7,
            1e-5,
        );

        if sep.criteria > best_sep.criteria {
            best_sep = sep;
            best_origin.copy_from_slice(&ws.origin);
            best_basis.copy_from_slice(&ws.basis);
        }
    }

    (best_sep, best_origin, best_basis)
}

/// Execute Q trial manifolds in parallel using Rayon (Multi-threaded)
pub fn run_trials_parallel(
    x: &[f64],
    d: usize,
    n: usize,
    m: usize,
    num_trials: usize,
    base_seed: u64,
) -> (Separation, Vec<f64>, Vec<f64>) {
    // Parallel reduction over num_trials
    let (best_sep, best_origin, best_basis) = (0..num_trials)
        .into_par_iter()
        .map_init(
            || (TrialWorkspace::new(d, n, m), StdRng::seed_from_u64(base_seed)),
            |(ws, _), trial_idx| {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(trial_idx as u64 * 7919));
                let mut count = 0;
                while count < m + 1 {
                    let idx = rng.random_range(0..n);
                    if !ws.sample_indices[..count].contains(&idx) {
                        ws.sample_indices[count] = idx;
                        count += 1;
                    }
                }

                form_basis_from_sample(x, d, &ws.sample_indices, &mut ws.origin, &mut ws.basis);
                distance_to_manifold(x, d, n, &ws.origin, &ws.basis, m, &mut ws.distances);

                let sep = find_separation(
                    &ws.distances,
                    &mut ws.scratch_sort,
                    &mut ws.bins_buffer,
                    0.1,
                    0,
                    7,
                    1e-5,
                );

                (sep, ws.origin.clone(), ws.basis.clone())
            },
        )
        .reduce(
            || (Separation::default(), vec![0.0; d], vec![0.0; d * m]),
            |(best_s, best_o, best_b), (curr_s, curr_o, curr_b)| {
                if curr_s.criteria > best_s.criteria {
                    (curr_s, curr_o, curr_b)
                } else {
                    (best_s, best_o, best_b)
                }
            },
        );

    (best_sep, best_origin, best_basis)
}

/// Execute Q trial manifolds in parallel on an arbitrary subset of points `selected`
pub fn run_trials_subset(
    x: &[f64],
    d: usize,
    selected: &[usize],
    m: usize,
    num_trials: usize,
    base_seed: u64,
) -> (Separation, Vec<f64>, Vec<f64>) {
    let l = selected.len();
    if l <= m {
        return (Separation::default(), vec![0.0; d], vec![0.0; d * m]);
    }

    let (best_sep, best_origin, best_basis) = (0..num_trials)
        .into_par_iter()
        .map_init(
            || (TrialWorkspace::new(d, l, m), StdRng::seed_from_u64(base_seed)),
            |(ws, _), trial_idx| {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(trial_idx as u64 * 7919 + 1));
                let mut count = 0;
                while count < m + 1 {
                    let rand_pos = rng.random_range(0..l);
                    let idx = selected[rand_pos];
                    if !ws.sample_indices[..count].contains(&idx) {
                        ws.sample_indices[count] = idx;
                        count += 1;
                    }
                }

                form_basis_from_sample(x, d, &ws.sample_indices, &mut ws.origin, &mut ws.basis);

                // Compute distances for points in selected
                for (out_i, &pt_idx) in selected.iter().enumerate() {
                    let pt = &x[pt_idx * d..(pt_idx + 1) * d];
                    let mut d_norm_sq = 0.0;
                    let mut p_norm_sq = 0.0;

                    for k in 0..m {
                        let b_col = &ws.basis[k * d..(k + 1) * d];
                        let mut dot = 0.0;
                        for j in 0..d {
                            dot += (pt[j] - ws.origin[j]) * b_col[j];
                        }
                        p_norm_sq += dot * dot;
                    }

                    for j in 0..d {
                        let diff = pt[j] - ws.origin[j];
                        d_norm_sq += diff * diff;
                    }

                    let diff_sq = (d_norm_sq - p_norm_sq).abs();
                    ws.distances[out_i] = diff_sq.sqrt();
                }

                let sep = find_separation(
                    &ws.distances[..l],
                    &mut ws.scratch_sort[..l],
                    &mut ws.bins_buffer,
                    0.1,
                    0,
                    7,
                    1e-5,
                );

                (sep, ws.origin.clone(), ws.basis.clone())
            },
        )
        .reduce(
            || (Separation::default(), vec![0.0; d], vec![0.0; d * m]),
            |(best_s, best_o, best_b), (curr_s, curr_o, curr_b)| {
                if curr_s.criteria > best_s.criteria {
                    (curr_s, curr_o, curr_b)
                } else {
                    (best_s, best_o, best_b)
                }
            },
        );

    (best_sep, best_origin, best_basis)
}


