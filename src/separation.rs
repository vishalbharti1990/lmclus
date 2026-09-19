/// Separation criteria result
#[derive(Debug, Clone, Copy)]
pub struct Separation {
    pub criteria: f64,
    pub depth: f64,
    pub discriminability: f64,
    pub threshold: f64,
    pub global_min_idx: usize,
    pub min_dist: f64,
    pub max_dist: f64,
    pub bins: usize,
}

impl Default for Separation {
    fn default() -> Self {
        Self {
            criteria: 0.0,
            depth: 0.0,
            discriminability: 0.0,
            threshold: 0.0,
            global_min_idx: 0,
            min_dist: 0.0,
            max_dist: 0.0,
            bins: 0,
        }
    }
}

/// Compute histogram bin count using Sturges or adaptive percentile bin width
pub fn get_histogram_bins(
    dists: &[f64],
    scratch_sort: &mut [f64],
    max_bin_portion: f64,
    hist_bin_size: usize,
    min_bin_num: usize,
) -> usize {
    if hist_bin_size > 0 {
        return hist_bin_size.max(min_bin_num);
    }

    let l = dists.len();
    if l == 0 {
        return min_bin_num;
    }

    scratch_sort[..l].copy_from_slice(dists);
    scratch_sort[..l].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let xmin = scratch_sort[0];
    let xmax = scratch_sort[l - 1];
    let xrng = xmax - xmin;

    if xrng <= 0.0 {
        return min_bin_num;
    }

    let mbp = ((l as f64) * max_bin_portion).round() as usize;
    let mbp = mbp.max(1);

    let mut binwidth = xmax;
    let mut prev_x = xmin;

    let mut i = mbp;
    while i < l {
        let diff = scratch_sort[i] - prev_x;
        if diff > 0.0 && diff < binwidth {
            binwidth = diff;
        }
        prev_x = scratch_sort[i];
        i += mbp;
    }

    let mut bns = (xrng / binwidth).round() as usize;
    if bns > l || bns == 0 {
        // Sturges formula fallback: ceil(log2(n) + 1)
        bns = ((l as f64).log2().ceil() as usize) + 1;
    }

    bns.max(min_bin_num)
}

/// Find global minimum and valley depth in criterion curve J
pub fn find_global_min(j: &[f64], tol: f64) -> Option<(f64, usize)> {
    let n = j.len();
    if n <= 1 {
        return None;
    }

    // Mark local minima
    let mut m = vec![false; n];
    let mut prev = j[1] - j[0];
    for i in 1..(n - 1) {
        let curr = j[i + 1] - j[i];
        m[i] = prev <= 0.0 && curr >= 0.0;
        prev = curr;
    }

    // Special case: flat minimum
    if n > 2 && m[1..(n - 1)].iter().all(|&b| b) {
        return Some((f64::INFINITY, 1));
    }

    // Collect minimum indices
    let min_indices: Vec<usize> = m.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect();
    if min_indices.is_empty() {
        return None;
    }

    let mut max_depth = 0.0;
    let mut global_min = 0;

    let mut idx = 0;
    while idx < min_indices.len() {
        let lmin = min_indices[idx];
        if lmin >= n - 1 {
            break;
        }

        let mut rmin = lmin;
        while rmin < n && m[rmin] {
            rmin += 1;
        }
        let loc_min = (lmin + rmin - 1) / 2;

        // Monotonically ascend to the left
        let mut lheight = loc_min;
        while lheight > 0 && j[lheight - 1] >= j[lheight] {
            lheight -= 1;
        }

        // Monotonically ascend to the right
        let mut rheight = loc_min;
        while rheight < n - 1 && j[rheight] <= j[rheight + 1] {
            rheight += 1;
        }

        let local_depth = j[lheight].min(j[rheight]) - j[loc_min];
        if local_depth > max_depth {
            max_depth = local_depth;
            global_min = loc_min;
        }

        while idx < min_indices.len() && min_indices[idx] < rmin {
            idx += 1;
        }
    }

    if max_depth < tol {
        None
    } else {
        Some((max_depth, global_min + 1)) // 1-based index
    }
}

/// Fast Kittler & Illingworth Minimum Error Thresholding on distance vector
pub fn find_separation(
    distances: &[f64],
    scratch_sort: &mut [f64],
    bins_buffer: &mut [f64],
    max_bin_portion: f64,
    hist_bin_size: usize,
    min_bin_num: usize,
    tol: f64,
) -> Separation {
    let n = distances.len();
    if n == 0 {
        return Separation::default();
    }

    let mut min_dist = distances[0];
    let mut max_dist = distances[0];
    for &d in &distances[1..] {
        if d < min_dist {
            min_dist = d;
        }
        if d > max_dist {
            max_dist = d;
        }
    }

    if (max_dist - min_dist).abs() < 1e-12 {
        return Separation::default();
    }

    let num_bins = get_histogram_bins(distances, scratch_sort, max_bin_portion, hist_bin_size, min_bin_num);
    let mut local_buf;
    let counts: &mut [f64] = if num_bins <= bins_buffer.len() {
        &mut bins_buffer[..num_bins]
    } else {
        local_buf = vec![0.0; num_bins];
        &mut local_buf[..]
    };
    counts.fill(0.0);

    let scale = (num_bins as f64) / (max_dist - min_dist);
    for &d in distances {
        let mut b = ((d - min_dist) * scale) as usize;
        if b >= num_bins {
            b = num_bins - 1;
        }
        counts[b] += 1.0;
    }

    // Normalized histogram
    let inv_total = 1.0 / (n as f64);
    for c in counts.iter_mut() {
        *c *= inv_total;
    }

    // Cumulative statistics
    let n_bins = num_bins;
    let eps = f64::EPSILON;

    let mut p1 = 0.0;
    let mut sum1 = 0.0;
    let mut sum_sq1 = 0.0;

    let mut total_mean = 0.0;
    let mut total_sq = 0.0;
    for (i, &w) in counts.iter().enumerate() {
        let x = (i + 1) as f64;
        total_mean += w * x;
        total_sq += w * x * x;
    }

    let mut j_values = vec![2.0 * eps.ln(); n_bins - 1];
    let mut stats_discrim = vec![0.0; n_bins - 1];

    for t in 0..(n_bins - 1) {
        let x = (t + 1) as f64;
        let w = counts[t];
        p1 += w;
        sum1 += w * x;
        sum_sq1 += w * x * x;

        let p2 = 1.0 - p1;
        if p1 <= 0.0 || p2 <= 0.0 {
            continue;
        }

        let mu1 = sum1 / p1;
        let var1 = (sum_sq1 / p1 - mu1 * mu1).max(0.0);

        let sum2 = total_mean - sum1;
        let sum_sq2 = total_sq - sum_sq1;
        let mu2 = sum2 / p2;
        let var2 = (sum_sq2 / p2 - mu2 * mu2).max(0.0);

        if var1 >= 0.0 && var2 >= 0.0 {
            let sig1 = var1.sqrt();
            let log_sig1 = if sig1 == 0.0 { eps.ln() } else { sig1.ln() };
            let sig2 = var2.sqrt();
            let log_sig2 = if sig2 == 0.0 { eps.ln() } else { sig2.ln() };

            let ses = p1 * log_sig1 + p2 * log_sig2;
            let se = p1 * p1.ln() + p2 * p2.ln();
            j_values[t] = 1.0 + 2.0 * ses - 2.0 * se;

            let denom = (var1 + var2).sqrt();
            stats_discrim[t] = if denom > 0.0 {
                (mu1 - mu2).abs() / denom
            } else {
                (mu1 - mu2).abs() / eps
            };
        }
    }

    let (depth, gmin_1based) = match find_global_min(&j_values, tol) {
        Some(res) => res,
        None => return Separation::default(),
    };

    let t_idx = gmin_1based - 1;
    let discriminability = stats_discrim[t_idx];
    let threshold = min_dist + (gmin_1based as f64) * (max_dist - min_dist) / (num_bins as f64);
    let criteria = discriminability * depth;

    Separation {
        criteria,
        depth,
        discriminability,
        threshold,
        global_min_idx: gmin_1based,
        min_dist,
        max_dist,
        bins: num_bins,
    }
}
