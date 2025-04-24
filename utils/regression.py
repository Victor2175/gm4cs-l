import numpy as np
import time
from scipy.sparse.linalg import svds # For faster truncated SVD

def reduced_rank_regression(X, y, rank, lambda_, use_ols_only=False):
    """
    Performs Reduced Rank Regression (RRR) or simple OLS regression.

    X_all: (M*n, p) Combined input dataset from multiple simulations.
    Y_all: (M*n, q) Corresponding output responses.
    rank: Desired rank for dimensionality reduction.
    lambda_: Regularization parameter.
    use_ols_only: If True, skip SVD and use only OLS regression.
    
    Returns:
    - B_rrr: (p, q) Reduced-rank weight matrix or OLS weight matrix if use_ols_only is True.
    - B_ols: (p, q) Original OLS weight matrix.
    """

    # Fit OLS
    print("Fitting OLS...", flush = True)
    start_time = time.time()
    XtX = X.T @ X
    identity = np.eye(X.shape[1])
    # B_ols = np.linalg.inv(X.T @ X + lambda_ * identity) @ X.T @ y # Analytical solution of ridge regression (pseudo inverse)
    B_ols = np.linalg.solve(XtX + lambda_ * identity, X.T @ y) # Equivalent but faster implementation
    
    if use_ols_only:
        print("Skipping SVD as requested. Using Ridge solution directly.", flush=True)
        end_time = time.time()
        print(f"Ridge computation took {end_time - start_time:.4f} seconds.", flush = True)
        return B_ols, B_ols  # Return B_ols as both return values for consistent interface
    
    # Compute SVD
    print("Computing SVD...", flush = True)
    # _, _, Vt = np.linalg.svd(X @ B_ols, full_matrices=False)
    _, _, Vt = svds(X @ B_ols, k=rank) # Use svds for faster computation
    end_time = time.time()
    print(f"OLS and SVD computation took {end_time - start_time:.4f} seconds.", flush = True)
    
    # Truncate SVD to rank
    # U_r = U[:, :rank]
    # s_r = np.diag(s[:rank])
    Vt_r = Vt[:rank, :]

    # Compute B_rrr
    B_rrr = B_ols @ Vt_r.T @ Vt_r # Reduced-rank weight matrix
    print("RRR completed.", flush = True)
    return B_rrr, B_ols