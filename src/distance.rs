use rayon::prelude::*;

/// Compute orthogonal distance from all n points in X (d x n, column-major)
/// to the linear manifold spanned by basis (d x m, column-major) translated to origin (d).
///
/// Output written directly into `distances` (length n) with ZERO heap allocations.
pub fn distance_to_manifold(
    x: &[f64],
    d: usize,
    n: usize,
    origin: &[f64],
    basis: &[f64],
    m: usize,
    distances: &mut [f64],
) {
    debug_assert_eq!(x.len(), d * n);
    debug_assert_eq!(origin.len(), d);
    debug_assert!(basis.len() >= d * m);
    debug_assert_eq!(distances.len(), n);

    for i in 0..n {
        let pt = &x[i * d..(i + 1) * d];
        let mut d_norm_sq = 0.0;
        let mut p_norm_sq = 0.0;

        // For each basis vector k in 0..m
        for k in 0..m {
            let b_col = &basis[k * d..(k + 1) * d];
            let mut dot = 0.0;
            for j in 0..d {
                dot += (pt[j] - origin[j]) * b_col[j];
            }
            p_norm_sq += dot * dot;
        }

        // Squared translated norm
        for j in 0..d {
            let diff = pt[j] - origin[j];
            d_norm_sq += diff * diff;
        }

        let diff_sq = (d_norm_sq - p_norm_sq).abs();
        distances[i] = diff_sq.sqrt();
    }
}

/// Multi-threaded version using Rayon chunking over points
pub fn distance_to_manifold_parallel(
    x: &[f64],
    d: usize,
    n: usize,
    origin: &[f64],
    basis: &[f64],
    m: usize,
    distances: &mut [f64],
) {
    debug_assert_eq!(x.len(), d * n);
    debug_assert_eq!(origin.len(), d);
    debug_assert!(basis.len() >= d * m);
    debug_assert_eq!(distances.len(), n);

    distances
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, dist_out)| {
            let pt = &x[i * d..(i + 1) * d];
            let mut d_norm_sq = 0.0;
            let mut p_norm_sq = 0.0;

            for k in 0..m {
                let b_col = &basis[k * d..(k + 1) * d];
                let mut dot = 0.0;
                for j in 0..d {
                    dot += (pt[j] - origin[j]) * b_col[j];
                }
                p_norm_sq += dot * dot;
            }

            for j in 0..d {
                let diff = pt[j] - origin[j];
                d_norm_sq += diff * diff;
            }

            let diff_sq = (d_norm_sq - p_norm_sq).abs();
            *dist_out = diff_sq.sqrt();
        });
}

/// Compute distance to manifold for an arbitrary subset of point indices
pub fn distance_to_manifold_indices(
    x: &[f64],
    d: usize,
    indices: &[usize],
    origin: &[f64],
    basis: &[f64],
    m: usize,
    distances: &mut [f64],
) {
    debug_assert_eq!(origin.len(), d);
    debug_assert!(basis.len() >= d * m);
    debug_assert_eq!(distances.len(), indices.len());

    for (out_i, &pt_idx) in indices.iter().enumerate() {
        let pt = &x[pt_idx * d..(pt_idx + 1) * d];
        let mut d_norm_sq = 0.0;
        let mut p_norm_sq = 0.0;

        for k in 0..m {
            let b_col = &basis[k * d..(k + 1) * d];
            let mut dot = 0.0;
            for j in 0..d {
                dot += (pt[j] - origin[j]) * b_col[j];
            }
            p_norm_sq += dot * dot;
        }

        for j in 0..d {
            let diff = pt[j] - origin[j];
            d_norm_sq += diff * diff;
        }

        let diff_sq = (d_norm_sq - p_norm_sq).abs();
        distances[out_i] = diff_sq.sqrt();
    }
}


