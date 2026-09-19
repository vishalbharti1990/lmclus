use crate::params::Parameters;
use crate::pca::adjust_basis;
use crate::sampling::run_trials_subset;
use crate::separation::Separation;
use crate::types::{LMCLUSResult, Manifold};

/// Compute number of trial samples to draw
pub fn sample_quantity(
    k: usize,
    data_size: usize,
    params: &Parameters,
    s_found: usize,
) -> usize {
    let s_max = params.number_of_clusters;
    if s_max <= 1 {
        return 1;
    }

    let p = 1.0 / (2.0f64.max((s_max.saturating_sub(s_found)) as f64));
    let big_p = p.powi(k as i32);
    if big_p >= 1.0 {
        return 1;
    }

    let log_1_minus_p = (1.0 - big_p).log10();
    let n_samples = if log_1_minus_p.abs() < 1e-12 {
        100_000.0
    } else {
        (params.error_bound.log10() / log_1_minus_p).abs()
    };

    let num_samples = match params.sampling_heuristic {
        1 => {
            if n_samples.is_infinite() {
                100_000
            } else {
                n_samples.round() as usize
            }
        }
        2 => {
            let nn = (data_size as f64) * params.sampling_factor;
            nn.round() as usize
        }
        3 => {
            let nn = ((data_size as f64) * params.sampling_factor).min(n_samples);
            nn.round() as usize
        }
        _ => 1,
    };

    num_samples.max(1)
}

/// Filter points into separated (inliers) and removed (outliers) relative to threshold
pub fn filter_separated(
    selected: &[usize],
    x: &[f64],
    d: usize,
    origin: &[f64],
    basis: &[f64],
    m: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut cluster_points = Vec::new();
    let mut removed_points = Vec::new();

    for &idx in selected {
        let pt = &x[idx * d..(idx + 1) * d];
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

        let dist = (d_norm_sq - p_norm_sq).abs().sqrt();
        if dist < threshold {
            cluster_points.push(idx);
        } else {
            removed_points.push(idx);
        }
    }

    (cluster_points, removed_points)
}

/// Find a single linear manifold cluster in active points subset
pub fn find_manifold(
    x: &[f64],
    d: usize,
    index: &[usize],
    params: &Parameters,
    found: usize,
) -> (Manifold, Separation, Vec<usize>) {
    let mut filtered: Vec<usize> = Vec::new();
    let mut selected: Vec<usize> = index.to_vec();

    let mut best_manifold = Manifold::new(
        params.min_dim,
        vec![0.0; d],
        vec![0.0; d * params.min_dim],
        index.to_vec(),
        0.0,
        0.0,
    );
    let mut best_separation = Separation::default();

    let mut sep_dim = params.min_dim;
    while sep_dim <= params.max_dim {
        let mut separations_count = 0;
        let mut state = 0; // 0: SEPARATION, 1: ALIGNMENT, 2: BOUND, 3: FINISHED
        let mut theta = f64::INFINITY;
        let mut sigma = f64::INFINITY;

        while state < 3 {
            let (sep, origin, basis) = if state == 0 {
                let num_trials = sample_quantity(sep_dim, selected.len(), params, found);
                let trial_seed = params.random_seed.wrapping_add(
                    (found as u64 * 100_000)
                        + (sep_dim as u64 * 10_000)
                        + (separations_count as u64 * 100)
                        + 1,
                );
                run_trials_subset(x, d, &selected, sep_dim, num_trials, trial_seed)
            } else if state == 1 && params.basis_alignment && best_manifold.size() > 0 {
                adjust_basis(
                    &mut best_manifold,
                    x,
                    d,
                    params.dim_adjustment,
                    params.dim_adjustment_ratio,
                );
                let origin = best_manifold.mu.clone();
                let basis = best_manifold.basis.clone();
                let mut dists = vec![0.0; best_manifold.size()];
                crate::distance::distance_to_manifold_indices(
                    x,
                    d,
                    &best_manifold.points,
                    &origin,
                    &basis,
                    best_manifold.d,
                    &mut dists,
                );
                let mut scratch_sort = dists.clone();
                let mut bins_buffer = Vec::new();
                let sep = crate::separation::find_separation(
                    &dists,
                    &mut scratch_sort,
                    &mut bins_buffer,
                    params.max_bin_portion,
                    params.hist_bin_size,
                    params.min_bin_num,
                    1e-5,
                );
                (sep, origin, basis)
            } else {
                break;
            };

            if sep.criteria < params.best_bound {
                let curr_thr = if state == 2 { sigma } else { theta };
                let thr = sep.max_dist;
                let new_thr = if curr_thr > 0.0 && thr > 0.0 {
                    curr_thr.min(thr)
                } else {
                    curr_thr
                };
                if state == 2 {
                    sigma = new_thr;
                    best_manifold.sigma = new_thr;
                } else {
                    theta = new_thr;
                    best_manifold.theta = new_thr;
                }

                state = match state {
                    0 => {
                        if params.basis_alignment {
                            1
                        } else if params.bounded_cluster {
                            2
                        } else {
                            3
                        }
                    }
                    1 => {
                        if params.bounded_cluster {
                            2
                        } else {
                            3
                        }
                    }
                    _ => 3,
                };
            } else {
                let thr = sep.threshold;
                if state == 2 {
                    sigma = thr;
                } else {
                    theta = thr;
                }

                let (separated, removed) =
                    filter_separated(&selected, x, d, &origin, &basis, sep_dim, thr);

                if separated.len() <= params.min_cluster_size {
                    state = 3;
                } else {
                    best_manifold = Manifold::new(
                        sep_dim,
                        origin,
                        basis,
                        separated.clone(),
                        theta,
                        sigma,
                    );
                    best_separation = sep;

                    filtered.extend(removed);
                    selected = separated;
                    state = 0; // Try separating again on refined subset
                    separations_count += 1;
                }
            }
        }

        if selected.len() <= params.min_cluster_size {
            break;
        }

        if !params.force_max_dim && separations_count > 0 {
            break;
        }

        sep_dim += 1;
    }

    let mut require_alignment = false;
    if best_separation.criteria < params.best_bound || best_manifold.size() == 0 || best_manifold.d == d {
        best_manifold.points = selected.clone();
        best_manifold.d = 1;

        // Compute max distance
        let mut max_dist = 0.0;
        let mut dist_buf = vec![0.0; selected.len()];
        crate::distance::distance_to_manifold_indices(x, d, &selected, &best_manifold.mu, &best_manifold.basis, 1.min(d), &mut dist_buf);
        for &d_val in &dist_buf {
            if d_val > max_dist {
                max_dist = d_val;
            }
        }
        best_manifold.theta = max_dist;
        require_alignment = true;
    }

    if params.basis_alignment || require_alignment {
        adjust_basis(
            &mut best_manifold,
            x,
            d,
            params.dim_adjustment,
            params.dim_adjustment_ratio,
        );
    }

    (best_manifold, best_separation, filtered)
}

/// Execute Linear Manifold Clustering (LMCLUS)
pub fn lmclus(
    x: &[f64],
    d: usize,
    n: usize,
    params: &Parameters,
) -> LMCLUSResult {
    let mut index: Vec<usize> = (0..n).collect();
    let mut manifolds: Vec<Manifold> = Vec::new();
    let mut separations: Vec<Separation> = Vec::new();

    let mut effective_params = params.clone();
    if d <= effective_params.max_dim {
        effective_params.max_dim = d.saturating_sub(1);
    }

    while index.len() > effective_params.min_cluster_size {
        let (best_manifold, best_sep, remains) = find_manifold(
            x,
            d,
            &index,
            &effective_params,
            manifolds.len(),
        );

        manifolds.push(best_manifold);
        separations.push(best_sep);

        if manifolds.len() >= effective_params.stop_after_cluster {
            break;
        }

        index = remains;
    }

    // Outlier points not belonging to any cluster
    if !index.is_empty() {
        let mut outliers = Manifold::new(
            0,
            vec![0.0; d],
            vec![0.0; d],
            index,
            0.0,
            0.0,
        );
        if effective_params.basis_alignment {
            adjust_basis(
                &mut outliers,
                x,
                d,
                effective_params.dim_adjustment,
                effective_params.dim_adjustment_ratio,
            );
        }
        manifolds.push(outliers);
        separations.push(Separation::default());
    }

    LMCLUSResult::new(manifolds, separations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lmclus_two_clusters() {
        let d = 5;
        let n = 200;
        let mut x = vec![0.0; d * n];

        // Cluster 1: 100 points along [1, 0, 0, 0, 0] at center [0, 0, 0, 0, 0]
        for i in 0..100 {
            let t = (i as f64) * 0.1;
            x[i * d + 0] = t;
        }

        // Cluster 2: 100 points along [0, 1, 0, 0, 0] at center [10, 10, 10, 10, 10]
        for i in 100..200 {
            let t = ((i - 100) as f64) * 0.1;
            x[i * d + 0] = 10.0;
            x[i * d + 1] = 10.0 + t;
            x[i * d + 2] = 10.0;
            x[i * d + 3] = 10.0;
            x[i * d + 4] = 10.0;
        }

        let mut p = Parameters::new(2);
        p.number_of_clusters = 2;
        p.sampling_heuristic = 2;
        p.sampling_factor = 0.5;
        p.random_seed = 42;
        p.min_cluster_size = 20;
        p.basis_alignment = true;

        let res = lmclus(&x, d, n, &p);
        println!("res: nclusters={}, counts={:?}", res.nclusters(), res.counts());
        for (i, m) in res.manifolds.iter().enumerate() {
            println!("  manifold {i}: dim={}, size={}, theta={}", m.d, m.size(), m.theta);
        }
        for (i, s) in res.separations.iter().enumerate() {
            println!("  separation {i}: criteria={}, depth={}, discrim={}, thr={}", s.criteria, s.depth, s.discriminability, s.threshold);
        }
        assert!(res.nclusters() >= 2);
    }
}

