# src/graph.py
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors



def _cosine_like_normalize(G: np.ndarray) -> np.ndarray:
    # As in your notebook: divide by sqrt(diag outer diag), clip to [-1, 1]
    diag = np.clip(np.diag(G).astype(np.float64), 1e-12, None)
    D = np.sqrt(np.outer(diag, diag))
    G_norm = (G / D).astype(np.float64)
    G_norm = np.clip(G_norm, -1.0, 1.0)
    return G_norm


def build_knn_from_grm(GRM_df,
                       k: int = 5,
                       weighted_edges: bool = False,
                       symmetrize_mode: str = "union",
                       add_self_loops: bool = True) -> sp.csr_matrix:
    """
    Build adjacency from GRM via kNN on precomputed distance 1 - G_norm.
    - GRM_df: pandas DataFrame (square), index/cols are IDs (aligned to data).
    - k: neighbors
    - weighted_edges: if True, weights = normalized GRM similarities; else 1.0
    - symmetrize_mode: 'union' or 'mutual'
    - add_self_loops: add I
    returns: scipy.sparse CSR adjacency (symmetric)
    """
    ids = GRM_df.index.to_numpy()
    G = GRM_df.to_numpy().astype(np.float64)

    # Normalize as in the notebook
    G_norm = _cosine_like_normalize(G)  # :contentReference[oaicite:3]{index=3}

    # Precomputed distance = 1 - similarity
    dist = 1.0 - G_norm

    # kNN (include self, then drop it)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(dist)
    dists, neigh = nbrs.kneighbors(dist)

    # Build directed kNN edges
    n = len(ids)
    rows, cols, data = [], [], []
    for i in range(n):
        for j in neigh[i][1:]:  # skip self
            rows.append(i)
            cols.append(j)
            data.append(G_norm[i, j] if weighted_edges else 1.0)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Symmetrize
    if symmetrize_mode == "mutual":
        A_sym = A.minimum(A.T)  # keep only mutual neighbors
    else:
        A_sym = A.maximum(A.T)  # union (like your undirected NetworkX graph) :contentReference[oaicite:4]{index=4}

    # Optional self-loops
    if add_self_loops:
        A_sym = A_sym + sp.eye(n, dtype=A_sym.dtype, format="csr")

    return A_sym


def gcn_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    Assumes A already includes self-loops if desired.
    """
    A = A.tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def sample_subgraph_indices(train_idx: np.ndarray,
                            fraction: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Sample a subset of training indices according to the given fraction."""
    m = max(1, int(len(train_idx) * float(fraction)))
    return rng.choice(train_idx, size=m, replace=False)
