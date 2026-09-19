use lmclus::{lmclus, Parameters};
use lmclus::sampling::gram_schmidt;
use lmclus::distance::distance_to_manifold;

#[test]
fn test_gram_schmidt_orthogonalization() {
    let d = 3;
    let m = 3;
    // Columns: [1, 2, 2], [-1, 0, 2], [0, 0, 1]
    let mut basis = vec![
        1.0, 2.0, 2.0,
        -1.0, 0.0, 2.0,
        0.0, 0.0, 1.0,
    ];
    gram_schmidt(&mut basis, d, m);

    // Expected normalized columns (divided by 3):
    // Col 0: [1/3, 2/3, 2/3]
    // Col 1: [-2/3, -1/3, 2/3]
    // Col 2: [2/3, -2/3, 1/3]
    let expected = [
        1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
        -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0,
        2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0,
    ];

    for i in 0..9 {
        assert!((basis[i] - expected[i]).abs() < 1e-10, "mismatch at index {i}: {} vs {}", basis[i], expected[i]);
    }
}

#[test]
fn test_distance_to_manifold_kernel() {
    let d = 3;
    let n = 1;
    let m = 2;

    // Basis spanning xy-plane: col 0 = [1, 0, 0], col 1 = [0, 1, 0]
    let basis = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ];
    let origin = vec![0.0, 0.0, 0.0];
    let point = vec![1.0, 1.0, 1.0];

    let mut dists = vec![0.0; n];
    distance_to_manifold(&point, d, n, &origin, &basis, m, &mut dists);

    assert!((dists[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_synthetic_two_clusters() {
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
    assert!(res.nclusters() >= 2);
    let counts = res.counts();
    assert_eq!(counts.iter().sum::<usize>(), n);
}
