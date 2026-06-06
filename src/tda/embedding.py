import numpy as np
import pandas as pd


def classical_mds(distance: pd.DataFrame, dim: int = 10) -> pd.DataFrame:
    """
    Classical MDS (Torgerson):
    Given distance matrix D, returns coordinates X in R^dim.

    Steps:
      B = -0.5 * J * D^2 * J
      eigendecompose B
      X = V * sqrt(Lambda)  (top dim positive eigs)

    Returns:
      DataFrame index=symbol, columns=mds_0..mds_{dim-1}
    """
    D = distance.values.astype(float)
    n = D.shape[0]

    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    # Symmetric eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep only positive eigenvalues
    pos = eigvals > 1e-12
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    k = min(dim, len(eigvals))
    if k == 0:
        X = np.zeros((n, dim))
    else:
        L = np.diag(np.sqrt(eigvals[:k]))
        V = eigvecs[:, :k]
        Xk = V @ L
        if k < dim:
            X = np.hstack([Xk, np.zeros((n, dim - k))])
        else:
            X = Xk

    cols = [f"mds_{i}" for i in range(dim)]
    return pd.DataFrame(X, index=distance.index, columns=cols)