use faer::Mat;
use crate::types::Manifold;

/// Adjust manifold basis and dimension using PCA (via Thin SVD).
///
/// Given full dataset X (d_ambient x n, column-major):
/// - Slices X[:, manifold.points]
/// - Computes centroid mu
/// - Centers data and computes SVD: (X_subset - mu) = U * S * V^T
/// - Aligns manifold.basis = U (d_ambient x d_ambient)
/// - If adjust_dim is true, adjusts manifold.d based on cumulative explained variance ratio
pub fn adjust_basis(
    manifold: &mut Manifold,
    x: &[f64],
    d: usize,
    adjust_dim: bool,
    adjust_dim_ratio: f64,
) {
    let pts = &manifold.points;
    let n_pts = pts.len();
    if n_pts == 0 {
        return;
    }

    // 1. Compute centroid mu = mean(X[:, pts])
    let mut mu = vec![0.0; d];
    for &idx in pts {
        let col = &x[idx * d..(idx + 1) * d];
        for j in 0..d {
            mu[j] += col[j];
        }
    }
    let inv_n = 1.0 / (n_pts as f64);
    for j in 0..d {
        mu[j] *= inv_n;
    }
    manifold.mu = mu.clone();

    // If there is only 1 point or less than 2, basis cannot be aligned with SVD
    if n_pts < 2 {
        return;
    }

    // 2. Build centered matrix in Faer (d x n_pts)
    let mut centered = Mat::<f64>::zeros(d, n_pts);
    for (c, &idx) in pts.iter().enumerate() {
        let col = &x[idx * d..(idx + 1) * d];
        for r in 0..d {
            centered[(r, c)] = col[r] - mu[r];
        }
    }

    // 3. Compute Thin SVD using faer solvers
    let svd = match faer::linalg::solvers::Svd::new_thin(centered.as_ref()) {
        Ok(s) => s,
        Err(_) => return,
    };
    let u_mat = svd.U();
    let s_diag = svd.S();

    // 4. Update basis (d x d, column-major)
    let u_cols = u_mat.ncols();
    let mut basis = vec![0.0; d * d];
    for c in 0..u_cols.min(d) {
        for r in 0..d {
            basis[c * d + r] = u_mat[(r, c)];
        }
    }

    // If u_cols < d, pad remaining columns with Gram-Schmidt basis from identity
    if u_cols < d {
        for c in u_cols..d {
            basis[c * d + c] = 1.0;
        }
        crate::sampling::gram_schmidt(&mut basis, d, d);
    }
    manifold.basis = basis;

    // 5. Automatic dimension adjustment
    if adjust_dim {
        let diag_len = s_diag.column_vector().nrows();
        let v: Vec<f64> = (0..diag_len)
            .map(|r| {
                let s = s_diag.column_vector()[r];
                (s * s) * inv_n
            })
            .collect();

        let total_var: f64 = v.iter().sum();
        if total_var > 0.0 {
            let threshold = total_var * adjust_dim_ratio;
            let mut cumsum = 0.0;
            let mut chosen_k = d;
            for (k_idx, &var_k) in v.iter().enumerate() {
                cumsum += var_k;
                if cumsum >= threshold {
                    chosen_k = k_idx + 1;
                    break;
                }
            }
            manifold.d = chosen_k.clamp(1, d - 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_basis() {
        // Points lying on a 1D line in 3D: x = [t, 2t, 0]
        let d = 3;
        let n = 100;
        let mut x = vec![0.0; d * n];
        let mut pts = Vec::new();
        for i in 0..n {
            let t = (i as f64) - 50.0;
            x[i * d + 0] = t;
            x[i * d + 1] = 2.0 * t;
            x[i * d + 2] = 0.0;
            pts.push(i);
        }

        let mut m = Manifold::new(1, vec![0.0; d], vec![0.0; d * d], pts, 0.0, 0.0);
        adjust_basis(&mut m, &x, d, true, 0.99);

        // Subspace dimension should adjust to 1
        assert_eq!(m.d, 1);
        // First basis vector should be proportional to [1, 2, 0] / sqrt(5)
        let b0 = m.basis[0];
        let b1 = m.basis[1];
        let b2 = m.basis[2];
        let ratio = b1 / b0;
        assert!((ratio - 2.0).abs() < 1e-4);
        assert!(b2.abs() < 1e-4);
    }
}
