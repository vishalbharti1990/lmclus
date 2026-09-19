/// Linear Manifold representation
#[derive(Debug, Clone)]
pub struct Manifold {
    /// Dimension of the manifold
    pub d: usize,
    /// Origin translation vector (length d_ambient)
    pub mu: Vec<f64>,
    /// Orthonormal basis matrix (d_ambient x d_subspace, column-major)
    pub basis: Vec<f64>,
    /// Indices of points assigned to this manifold cluster
    pub points: Vec<usize>,
    /// Orthogonal complement distance threshold
    pub theta: f64,
    /// Subspace boundary threshold (if bounded_cluster enabled)
    pub sigma: f64,
}

impl Manifold {
    pub fn new(d: usize, mu: Vec<f64>, basis: Vec<f64>, points: Vec<usize>, theta: f64, sigma: f64) -> Self {
        Self {
            d,
            mu,
            basis,
            points,
            theta,
            sigma,
        }
    }

    pub fn empty(d_ambient: usize) -> Self {
        Self {
            d: 0,
            mu: vec![0.0; d_ambient],
            basis: Vec::new(),
            points: Vec::new(),
            theta: 0.0,
            sigma: 0.0,
        }
    }

    pub fn size(&self) -> usize {
        self.points.len()
    }
}

pub use crate::separation::Separation;

/// Container holding clustering results
#[derive(Debug, Clone)]
pub struct LMCLUSResult {
    pub manifolds: Vec<Manifold>,
    pub separations: Vec<Separation>,
}

impl LMCLUSResult {
    pub fn new(manifolds: Vec<Manifold>, separations: Vec<Separation>) -> Self {
        Self {
            manifolds,
            separations,
        }
    }

    pub fn nclusters(&self) -> usize {
        self.manifolds.len()
    }

    pub fn counts(&self) -> Vec<usize> {
        self.manifolds.iter().map(|m| m.size()).collect()
    }

    /// Point-to-cluster assignments matching Julia convention (0-indexed in Rust)
    pub fn assignments(&self, total_points: usize) -> Vec<usize> {
        let mut assign = vec![usize::MAX; total_points];
        for (cluster_id, m) in self.manifolds.iter().enumerate() {
            for &pt in &m.points {
                if pt < total_points {
                    assign[pt] = cluster_id;
                }
            }
        }
        assign
    }
}

