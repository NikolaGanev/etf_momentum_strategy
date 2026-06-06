import numpy as np
import pandas as pd
import gudhi


def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Convert correlation matrix to distance matrix:
      d_ij = sqrt(2 * (1 - corr_ij))

    Handles NaNs by setting corr_ij = -1 (max distance).
    Ensures diagonal = 0.
    """
    c = corr.copy()

    # Replace NaNs with -1 => distance = 2 (max)
    c = c.fillna(-1.0)

    # Clamp numerical noise outside [-1, 1]
    c = c.clip(lower=-1.0, upper=1.0)

    d = np.sqrt(2.0 * (1.0 - c.values))
    np.fill_diagonal(d, 0.0)

    return pd.DataFrame(d, index=c.index, columns=c.columns)


def persistence_summaries(diagrams, dims=(0, 1)) -> dict:
    """
    Convert persistence intervals into a small, interpretable feature set.
    diagrams: list of (dim, (birth, death)) from gudhi.simplex_tree.persistence()

    We compute per dimension:
    - count: number of finite intervals
    - total_persistence: sum(death - birth)
    - max_persistence: max(death - birth)
    - mean_persistence: mean(death - birth)
    """
    feats = {}
    for dim in dims:
        intervals = [(b, d) for (k, (b, d)) in diagrams if k == dim and np.isfinite(d)]
        lifetimes = np.array([d - b for (b, d) in intervals], dtype=float)

        feats[f"ph_dim{dim}_count"] = int(len(lifetimes))
        feats[f"ph_dim{dim}_total_persistence"] = float(lifetimes.sum()) if len(lifetimes) else 0.0
        feats[f"ph_dim{dim}_max_persistence"] = float(lifetimes.max()) if len(lifetimes) else 0.0
        feats[f"ph_dim{dim}_mean_persistence"] = float(lifetimes.mean()) if len(lifetimes) else 0.0

    return feats


def compute_persistent_homology_features(
    distance: pd.DataFrame,
    max_dim: int = 1,
    max_edge_length: float = 2.0
) -> dict:
    """
    Build Vietoris–Rips complex from a distance matrix and compute persistence.

    Returns a dict of summary features from H0 and H1 persistence.
    """
    dist_mat = distance.values.astype(float)

    # GUDHI RipsComplex expects either points or a distance matrix.
    rips = gudhi.RipsComplex(distance_matrix=dist_mat, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)

    # Compute persistence
    diagrams = st.persistence()

    feats = persistence_summaries(diagrams, dims=(0, 1))

    # A couple of extra global-ish descriptors
    feats["ph_simplex_tree_num_simplices"] = int(st.num_simplices())
    feats["ph_simplex_tree_num_vertices"] = int(st.num_vertices())

    return feats

def compute_ph_features_from_points(
    points: pd.DataFrame,
    max_dim: int = 1,
    max_edge_length: float = 2.0
) -> dict:
    """
    Compute persistent homology features from a point cloud (n x d).
    """
    X = points.values.astype(float)

    rips = gudhi.RipsComplex(points=X, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)

    diagrams = st.persistence()
    feats = persistence_summaries(diagrams, dims=(0, 1))
    feats["ph_simplex_tree_num_simplices"] = int(st.num_simplices())
    feats["ph_simplex_tree_num_vertices"] = int(st.num_vertices())
    return feats