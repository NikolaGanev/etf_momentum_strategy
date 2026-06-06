import numpy as np
import pandas as pd


def corr_to_adjacency(corr: pd.DataFrame, positive_only: bool = True) -> pd.DataFrame:
    """
    Convert correlation matrix to adjacency matrix A.

    - Optionally keep only positive correlations: A_ij = max(corr_ij, 0)
    - Set diagonal to 0 (no self-loops)
    """
    A = corr.copy()

    if positive_only:
        A = A.clip(lower=0.0)

    # Remove self edges
    np.fill_diagonal(A.values, 0.0)

    # Replace remaining NaNs with 0 weights
    A = A.fillna(0.0)

    return A


def adjacency_to_laplacian(A: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute:
    - Unnormalized Laplacian: L = D - A
    - Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    """
    degrees = A.sum(axis=1).values
    D = np.diag(degrees)

    L = D - A.values
    L_df = pd.DataFrame(L, index=A.index, columns=A.columns)

    # Normalized Laplacian
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = 1.0 / np.sqrt(degrees)

    inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt_deg)

    I = np.eye(A.shape[0])
    L_sym = I - D_inv_sqrt @ A.values @ D_inv_sqrt
    L_sym_df = pd.DataFrame(L_sym, index=A.index, columns=A.columns)

    return L_df, L_sym_df