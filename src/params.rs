/// Configuration parameters for LMCLUS algorithm
#[derive(Debug, Clone)]
pub struct Parameters {
    /// Minimum dimension of the cluster manifold subspace
    pub min_dim: usize,
    /// Maximum dimension of the cluster manifold subspace
    pub max_dim: usize,
    /// Nominal / expected number of resulting clusters
    pub number_of_clusters: usize,
    /// Terminate algorithm after finding this number of clusters
    pub stop_after_cluster: usize,
    /// Force search in higher dimension subspaces even if separation is found
    pub force_max_dim: bool,
    /// Fixed number of bins for distance histograms (0 = calculate automatically)
    pub hist_bin_size: usize,
    /// Minimum number of bins for distance histogram
    pub min_bin_num: usize,
    /// Minimum cluster size; smaller collections are considered noise
    pub min_cluster_size: usize,
    /// Separation criteria bound threshold (depth * discriminability)
    pub best_bound: f64,
    /// Sampling error bound for sampling heuristic
    pub error_bound: f64,
    /// Portion of dataset used for estimating histogram bin width
    pub max_bin_portion: f64,
    /// RNG seed (0 = auto-generate from system entropy)
    pub random_seed: u64,
    /// Sampling heuristic (1: exponential, 2: linear, 3: min of 1 & 2)
    pub sampling_heuristic: usize,
    /// Sampling factor used in heuristics 2 & 3
    pub sampling_factor: f64,
    /// Enable alignment of cluster basis via PCA
    pub basis_alignment: bool,
    /// Enable automatic manifold dimension adjustment based on explained variance ratio
    pub dim_adjustment: bool,
    /// Explained variance ratio threshold for dimension adjustment
    pub dim_adjustment_ratio: f64,
    /// Enable bounded linear manifold clusters (evaluates orthogonal complement distance threshold σ)
    pub bounded_cluster: bool,
    /// Rayon thread pool concurrency (0 = use Rayon default)
    pub num_threads: usize,
}

impl Parameters {
    pub fn new(max_dim: usize) -> Self {
        Self {
            min_dim: 1,
            max_dim,
            number_of_clusters: 10,
            stop_after_cluster: 1000,
            force_max_dim: false,
            hist_bin_size: 0,
            min_bin_num: 7,
            min_cluster_size: 20,
            best_bound: 1.0,
            error_bound: 0.0001,
            max_bin_portion: 0.1,
            random_seed: 0,
            sampling_heuristic: 3,
            sampling_factor: 0.01,
            basis_alignment: false,
            dim_adjustment: false,
            dim_adjustment_ratio: 0.99,
            bounded_cluster: false,
            num_threads: 0,
        }
    }
}

