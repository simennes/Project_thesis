import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors


def _cosine_like_normalize(G: np.ndarray) -> np.ndarray:
    """Cosine-like normalization: scale by sqrt(diag_i * diag_j)."""
    diag = np.clip(np.diag(G).astype(np.float64), 1e-12, None)
    D = np.sqrt(np.outer(diag, diag))
    G_norm = (G / D).astype(np.float64)
    G_norm = np.clip(G_norm, -1.0, 1.0)
    return G_norm


def identity_csr(n: int) -> sp.csr_matrix:
    """Return identity matrix as CSR."""
    return sp.eye(n, format="csr", dtype=np.float32)


def gcn_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """Apply GCN normalization: D^{-1/2} A D^{-1/2}."""
    A = A.tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    deg = np.clip(deg, 1e-12, None)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_knn_from_grm(
    GRM_df,
    k: int = 5,
    weighted_edges: bool = False,
    symmetrize_mode: str = "union",
    add_self_loops: bool = False,
) -> sp.csr_matrix:
    """Build k-NN graph from genomic relationship matrix."""
    if k <= 0:
        n = GRM_df.shape[0]
        return identity_csr(n)

    G = GRM_df.to_numpy().astype(np.float64)
    G_norm = _cosine_like_normalize(G)
    dist = 1.0 - G_norm

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(dist)
    _, neigh = nbrs.kneighbors(dist)

    n = G.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        for j in neigh[i][1:]:
            rows.append(i)
            cols.append(j)
            data.append(G_norm[i, j] if weighted_edges else 1.0)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A_sym = A.minimum(A.T) if symmetrize_mode == "mutual" else A.maximum(A.T)

    if add_self_loops:
        A_sym = A_sym + sp.eye(n, dtype=A_sym.dtype, format="csr")

    return A_sym


def build_grm_cutoff_adjacency(
    GRM_df,
    cutoff: float = 0.5,
    grm_norm: str = "none",
    add_self_loops: bool = False,
) -> sp.csr_matrix:
    """Build adjacency by thresholding GRM entries at a cutoff.

    Parameters
    ----------
    GRM_df : pandas.DataFrame
        Square genomic relationship matrix.
    cutoff : float
        Threshold to keep edges >= cutoff.
    grm_norm : str
        One of {"none", "cosine", "gcn", "cosine_then_gcn"}.
    add_self_loops : bool
        Whether to add identity before GCN normalization.
    """
    assert GRM_df is not None, "GRM_df is None but graph_mode='cutoff' requested"
    G = GRM_df.to_numpy().astype(np.float64)

    norm = (grm_norm or "none").lower()
    if norm in ("cosine", "cosine_then_gcn", "cosine+gcn", "cosine_gcn"):
        M = _cosine_like_normalize(G)
    else:
        M = G

    # Threshold: keep entries >= cutoff
    M_thr = np.where(M >= float(cutoff), M, 0.0)
    A = sp.csr_matrix(M_thr)
    A.eliminate_zeros()

    if norm in ("gcn", "cosine_then_gcn", "cosine+gcn", "cosine_gcn"):
        if add_self_loops:
            n = A.shape[0]
            A = A + sp.eye(n, dtype=A.dtype, format="csr")
        return gcn_normalize(A)

    return A


def build_adjacency(
    X: np.ndarray,
    GRM_df,
    graph_cfg: dict,
    node_idx: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Unified adjacency builder.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    GRM_df : pandas.DataFrame or None
        Genomic relationship matrix aligned to X rows.
    graph_cfg : dict
        Configuration: {graph_mode, knn_k, weighted_edges, symmetrize_mode, 
                       cutoff, grm_norm, self_loops}.
    node_idx : np.ndarray | None
        Optional indices to subset nodes.

    Returns
    -------
    sp.csr_matrix
        CSR adjacency matrix.
    """
    if node_idx is None:
        n_nodes = X.shape[0]
        GRM_sub = GRM_df
    else:
        node_idx = np.asarray(node_idx, dtype=int)
        n_nodes = node_idx.size
        GRM_sub = GRM_df.iloc[node_idx, node_idx] if GRM_df is not None else None

    # Determine mode
    mode = graph_cfg.get("graph_mode")
    if mode is None:
        mode = "knn" if graph_cfg.get("graph_on", True) else "off"
    mode = str(mode).lower()

    if mode == "off":
        return identity_csr(n_nodes)

    if mode == "cutoff":
        return build_grm_cutoff_adjacency(
            GRM_sub,
            cutoff=float(graph_cfg.get("cutoff", 0.5)),
            grm_norm=graph_cfg.get("grm_norm", "none"),
            add_self_loops=graph_cfg.get("self_loops", False),
        )

    if mode == "knn":
        assert GRM_sub is not None, "GRM_df is required for graph_mode='knn'"
        A = build_knn_from_grm(
            GRM_sub,
            k=graph_cfg.get("knn_k", 5),
            weighted_edges=graph_cfg.get("weighted_edges", False),
            symmetrize_mode=graph_cfg.get("symmetrize_mode", "mutual"),
            add_self_loops=graph_cfg.get("self_loops", False),
        )
        return gcn_normalize(A)

    raise ValueError(f"Unsupported graph_mode '{mode}'. Expected 'off', 'knn', or 'cutoff'.")


def csr_to_edge_index(A: sp.csr_matrix, device: torch.device):
    """Convert a CSR adjacency matrix to edge_index and edge_weight tensors.
    
    Parameters
    ----------
    A : sp.csr_matrix
        Sparse adjacency matrix.
    device : torch.device
        Target device for tensors.
        
    Returns
    -------
    edge_index : torch.Tensor
        Shape (2, num_edges) tensor of edge indices.
    edge_weight : torch.Tensor
        Shape (num_edges,) tensor of edge weights.
    """
    coo = A.tocoo()
    if coo.nnz == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        idx_np = np.vstack([coo.row, coo.col])
        edge_index = torch.tensor(idx_np, dtype=torch.long, device=device)
        edge_weight = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return edge_index, edge_weight
