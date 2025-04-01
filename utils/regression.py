import numpy as np

def reduced_rank_regression(X, y, rank, lambda_):
    """
    Performs Reduced Rank Regression (RRR).

    X_all: (M*n, p) Combinsed input dataset from multiple simulations.
    Y_all: (M*n, q) Corresponding output responses.
    rank: Desired rank for dimensionality reduction.
    
    Returns:
    - B_rrr: (p, q) Reduced-rank weight matrix.
    """

    # Fit OLS
    print("Fitting OLS...")
    identity = np.eye(X.shape[1])
    B_ols = np.linalg.pinv(X.T @ X + lambda_ * identity) @ X.T @ y # Analytical solution of ridge regression (pseudo inverse)
    # Compute SVD
    U, s, Vt = np.linalg.svd(X @ B_ols, full_matrices=False)
    

    # Truncate SVD to rank
    # U_r = U[:, :rank]
    # s_r = np.diag(s[:rank])
    Vt_r = Vt[:rank, :]

    # Compute B_rrr
    B_rrr = B_ols @ Vt_r.T @ Vt_r # Reduced-rank weight matrix
    print("RRR completed.")
    return B_rrr, B_ols