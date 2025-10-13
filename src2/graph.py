import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _gcn_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric GCN normalization with self-loops."""
    A = A.tocsr()
    A = A + sp.eye(A.shape[0], dtype=A.dtype, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    Dm = sp.diags(inv_sqrt)
    return (Dm @ A @ Dm).tocsr()


def _cosine_like_normalize(G: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(G).astype(np.float64), 1e-12, None)
    D = np.sqrt(np.outer(diag, diag))
    G_norm = (G / D).astype(np.float64)
    return np.clip(G_norm, -1.0, 1.0)


def _knn_from_X_cosine(X: np.ndarray, k: int, weighted: bool, symmetrize_mode: str) -> sp.csr_matrix:
    if k <= 0:
        # identity adjacency → self-loops only after normalization
        return sp.csr_matrix((X.shape[0], X.shape[0]), dtype=np.float32)
    # standardize columns for cosine
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xz)
    dists, neigh = nbrs.kneighbors(Xz)
    n = X.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        for j_idx, j in enumerate(neigh[i][1:]):
            rows.append(i)
            cols.append(j)
            if weighted:
                sim = 1.0 - float(dists[i][j_idx + 1])
                data.append(sim)
            else:
                data.append(1.0)
    A_dir = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A_sym = A_dir.minimum(A_dir.T) if symmetrize_mode == "mutual" else A_dir.maximum(A_dir.T)
    return A_sym.astype(np.float32)


def _knn_from_grm(GRM: np.ndarray, k: int, weighted: bool, symmetrize_mode: str) -> sp.csr_matrix:
    if k <= 0:
        return sp.csr_matrix((GRM.shape[0], GRM.shape[0]), dtype=np.float32)
    G_norm = _cosine_like_normalize(GRM)
    dist = 1.0 - G_norm
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(dist)
    _, neigh = nbrs.kneighbors(dist)
    n = GRM.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        for j in neigh[i][1:]:
            rows.append(i)
            cols.append(j)
            data.append(G_norm[i, j] if weighted else 1.0)
    A_dir = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A_sym = A_dir.minimum(A_dir.T) if symmetrize_mode == "mutual" else A_dir.maximum(A_dir.T)
    return A_sym.astype(np.float32)


def build_global_adjacency(
    X: np.ndarray,
    GRM: np.ndarray,
    cfg: dict,
) -> sp.csr_matrix:
    """
    Build ONE global adjacency for the entire dataset (transductive).
    cfg:
      - source: "snp" | "grm"
      - knn_k: int (0 => identity adjacency; model degenerates to MLP)
      - weighted_edges: bool
      - symmetrize_mode: "union" | "mutual"
      - normalize: bool
    """
    source = cfg.get("source", "snp")
    k = int(cfg.get("knn_k", 20))
    weighted = bool(cfg.get("weighted_edges", True))
    symm = cfg.get("symmetrize_mode", "union")
    normalize = bool(cfg.get("normalize", True))

    if source == "grm":
        assert GRM is not None, "GRM is required when source='grm'. Provide it in the NPZ as 'GRM'."
        A = _knn_from_grm(GRM, k, weighted, symm)
    else:
        A = _knn_from_X_cosine(X, k, weighted, symm)

    if normalize:
        A = _gcn_normalize(A)
    return A
