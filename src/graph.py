import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import laplacian as csgraph_laplacian


def _cosine_like_normalize(G: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(G).astype(np.float64), 1e-12, None)
    D = np.sqrt(np.outer(diag, diag))
    G_norm = (G / D).astype(np.float64)
    G_norm = np.clip(G_norm, -1.0, 1.0)
    return G_norm

def identity_csr(n: int) -> sp.csr_matrix:
    return sp.eye(n, format="csr", dtype=np.float32)

def build_knn_from_grm(
    GRM_df,
    k: int = 5,
    weighted_edges: bool = False,
    symmetrize_mode: str = "union",
    add_self_loops: bool = False,
) -> sp.csr_matrix:
    if k <= 0:
        n = GRM_df.shape[0]
        return identity_csr(n)
    ids = GRM_df.index.to_numpy()
    G = GRM_df.to_numpy().astype(np.float64)
    G_norm = _cosine_like_normalize(G)
    dist = 1.0 - G_norm
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(dist)
    _, neigh = nbrs.kneighbors(dist)
    n = len(ids)
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


def gcn_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocsr()
    # Degree can be negative if adjacency has negative weights; clip to small positive
    deg = np.array(A.sum(axis=1)).flatten()
    deg = np.clip(deg, 1e-12, None)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_grm_adjacency(
    GRM_df,
    grm_norm: str = "gcn",
    add_self_loops: bool = True,
) -> sp.csr_matrix:
    """Build adjacency directly from the GRM matrix with configurable normalization.

    Parameters
    ----------
    GRM_df : pandas.DataFrame
        Square genomic relationship matrix (rows/cols aligned to X rows).
    grm_norm : str
        One of {"none", "cosine", "gcn", "cosine_then_gcn"} controlling normalization:
        - none: use raw GRM values as adjacency entries
        - cosine: cosine-like normalization of GRM (scales by sqrt(diag_i diag_j))
        - gcn: apply GCN normalization D^{-1/2} A D^{-1/2} to raw GRM (with optional self-loops)
        - cosine_then_gcn: cosine-like normalization, then GCN normalization
    add_self_loops : bool
        Whether to add identity before GCN normalization (ignored for pure "none"/"cosine" outputs).

    Returns
    -------
    sp.csr_matrix
        CSR adjacency derived from GRM.
    """
    assert GRM_df is not None, "GRM_df is None but graph_mode='grm' requested"
    G = GRM_df.to_numpy().astype(np.float64)

    norm = (grm_norm or "gcn").lower()
    if norm == "none":
        A = sp.csr_matrix(G)
        return A
    if norm == "cosine":
        S = _cosine_like_normalize(G)
        return sp.csr_matrix(S)
    if norm == "gcn":
        # Clip negative weights to zero to ensure a valid non-negative adjacency for GCN normalization
        A = sp.csr_matrix(np.clip(G, a_min=0.0, a_max=None))
        if add_self_loops:
            n = A.shape[0]
            A = A + sp.eye(n, dtype=A.dtype, format="csr")
        return gcn_normalize(A)
    if norm in ("cosine_then_gcn", "cosine+gcn", "cosine_gcn"):
        S = _cosine_like_normalize(G)
        # Clip negative weights to zero prior to GCN normalization
        A = sp.csr_matrix(np.clip(S, a_min=0.0, a_max=None))
        if add_self_loops:
            n = A.shape[0]
            A = A + sp.eye(n, dtype=A.dtype, format="csr")
        return gcn_normalize(A)

    # Fallback to GCN if unknown
    A = sp.csr_matrix(G)
    if add_self_loops:
        n = A.shape[0]
        A = A + sp.eye(n, dtype=A.dtype, format="csr")
    return gcn_normalize(A)


def build_grm_cutoff_adjacency(
    GRM_df,
    cutoff: float = 0.5,
    grm_norm: str = "none",
    add_self_loops: bool = False,
) -> sp.csr_matrix:
    """Build adjacency by thresholding GRM entries at a cutoff, with optional normalization.

    Steps:
    - Start from raw GRM matrix G.
    - Optionally pre-normalize using cosine-like normalization if grm_norm in {"cosine", "cosine_then_gcn"}.
    - Zero out edges with value < cutoff (strictly less than cutoff are deleted).
    - If grm_norm is a GCN variant ("gcn" or "cosine_then_gcn"), add self-loops (if enabled) then apply GCN normalization.
    - If grm_norm is "none" or "cosine", return the thresholded matrix as-is (CSR).

    Parameters
    ----------
    cutoff : float
        Threshold in [0.2, 0.9] to keep edges >= cutoff.
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
    # none or cosine
    return A


def build_knn_from_snp(
    X: np.ndarray,
    k: int = 5,
    weighted_edges: bool = False,
    symmetrize_mode: str = "union",
    add_self_loops: bool = True,
    laplacian_smoothing: bool = True,
) -> sp.csr_matrix:
    n = X.shape[0]
    if k <= 0:
        return identity_csr(n)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    dists, neigh = nbrs.kneighbors(X)
    rows, cols, data = [], [], []
    for i in range(n):
        for j_idx in range(1, len(neigh[i])):
            j = neigh[i][j_idx]
            dist_val = dists[i][j_idx]
            rows.append(i)
            cols.append(j)
            data.append(dist_val if weighted_edges else 1.0)
    A_dir = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    if laplacian_smoothing:
        vals = A_dir.data
        if vals.size == 0:
            sim_matrix = sp.csr_matrix((n, n))
        else:
            max_val = vals.max()
            normalized = vals / max_val if max_val != 0 else np.zeros_like(vals)
            sim_vals = np.exp(-(normalized**2) / 2.0)
            sim_matrix = sp.coo_matrix((sim_vals, A_dir.nonzero()), shape=(n, n)).tocsr()
        sim_sym = sim_matrix.minimum(sim_matrix.T) if symmetrize_mode == "mutual" else sim_matrix.maximum(sim_matrix.T)
        L = csgraph_laplacian(sim_sym, normed=False)
        L = sp.csr_matrix(L)
        Lc = L.tocoo()
        if weighted_edges:
            max_w = np.max(np.abs(Lc.data)) if Lc.data.size > 0 else 1.0
            new_w = 1.0 - (np.abs(Lc.data) / max_w) if max_w != 0 else np.ones_like(Lc.data)
            return sp.coo_matrix((new_w, (Lc.row, Lc.col)), shape=(n, n)).tocsr()
        ones = np.ones_like(Lc.data, dtype=float)
        return sp.coo_matrix((ones, (Lc.row, Lc.col)), shape=(n, n)).tocsr()
    else:
        if weighted_edges:
            vals = A_dir.data
            max_val = vals.max() if vals.size > 0 else 1.0
            sim_vals = 1.0 - (vals / max_val) if max_val != 0 else np.ones_like(vals)
            A_dir = sp.coo_matrix((sim_vals, A_dir.nonzero()), shape=(n, n)).tocsr()
        A_sym = A_dir.minimum(A_dir.T) if symmetrize_mode == "mutual" else A_dir.maximum(A_dir.T)
        if add_self_loops:
            A_sym = A_sym + sp.eye(n, dtype=A_sym.dtype, format="csr")
        return A_sym


def build_global_adjacency(
    X: np.ndarray,
    GRM_df,
    graph_cfg: dict,
) -> sp.csr_matrix:
    """Build one global adjacency based on graph_cfg.

    Modes controlled by graph_cfg:
    - graph_mode == 'off': identity adjacency
    - graph_mode == 'cutoff': threshold GRM by cutoff, optional normalization (grm_norm)
    - graph_mode == 'knn': KNN graph derived from the GRM

    Backward compatibility: if graph_mode not provided, falls back to
    graph_on (True->'knn', False->'off').
    """
    # Determine mode with backward compat to graph_on
    mode = graph_cfg.get("graph_mode")
    if mode is None:
        mode = "knn" if graph_cfg.get("graph_on", True) else "off"
    mode = str(mode).lower()

    if mode == "off":
        n = X.shape[0] if X is not None else (0 if GRM_df is None else GRM_df.shape[0])
        return identity_csr(n)

    if mode == "cutoff":
        return build_grm_cutoff_adjacency(
            GRM_df,
            cutoff=float(graph_cfg.get("cutoff", 0.5)),
            grm_norm=graph_cfg.get("grm_norm", "none"),
            add_self_loops=graph_cfg.get("self_loops", False),
        )

    if mode == "knn":
        assert GRM_df is not None, "GRM_df is required for graph_mode='knn'"
        A = build_knn_from_grm(
            GRM_df,
            k=graph_cfg.get("knn_k", 5),
            weighted_edges=graph_cfg.get("weighted_edges", False),
            symmetrize_mode=graph_cfg.get("symmetrize_mode", "mutual"),
            add_self_loops=graph_cfg.get("self_loops", False),
        )
        return gcn_normalize(A)

    raise ValueError(f"Unsupported graph_mode '{mode}'. Expected 'off', 'knn', or 'cutoff'.")


def build_adjacency(
    X: np.ndarray,
    GRM_df,
    graph_cfg: dict,
    node_idx: np.ndarray | None = None,
) -> sp.csr_matrix:
    """
        Unified adjacency builder.
        - Controlled by graph_cfg["graph_mode"] in {"off", "knn", "cutoff"}.
            Backward compat: if "graph_mode" missing, use "graph_on" (False->off, True->knn).
    - If node_idx is provided, subset X (and GRM_df if present) before building.
    - Delegates actual construction to build_global_adjacency with the subset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    GRM_df : pandas.DataFrame or None
        Genomic relationship matrix aligned to X rows; if node_idx is given,
        both rows and columns will be subset by node_idx.
    graph_cfg : dict
        Configuration controlling graph construction: keys include
        {graph_mode, knn_k, weighted_edges, symmetrize_mode, cutoff, grm_norm, self_loops, graph_on}.
    node_idx : np.ndarray | None
        Optional integer indices selecting a subset of nodes.

    Returns
    -------
    sp.csr_matrix
        CSR adjacency for all nodes in X if node_idx is None, otherwise for the subset.
    """
    n_total = X.shape[0]
    if node_idx is None:
        n_nodes = n_total
        X_sub = X
        GRM_sub = GRM_df
    else:
        node_idx = np.asarray(node_idx, dtype=int)
        n_nodes = node_idx.size
        X_sub = X[node_idx]
        GRM_sub = GRM_df.iloc[node_idx, node_idx] if GRM_df is not None else None

    # Determine mode with backward compat to graph_on
    mode = graph_cfg.get("graph_mode")
    if mode is None:
        mode = "knn" if graph_cfg.get("graph_on", True) else "off"
    mode = str(mode).lower()
    if mode == "off":
        return identity_csr(n_nodes)

    # Pass through to global builder; ensure mode is set
    gcfg = dict(graph_cfg)
    gcfg["graph_mode"] = mode
    return build_global_adjacency(X_sub, GRM_sub, gcfg)


def induce_subgraph(A: sp.csr_matrix, nodes: np.ndarray) -> sp.csr_matrix:
    nodes = np.asarray(nodes)
    return A[nodes][:, nodes].tocsr()


def create_list_of_edges(
    num_nodes: int, edge_list: list[tuple[int, int]], bidirectional: bool = True
) -> list[list[int]]:
    edges_adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for (u, v) in edge_list:
        edges_adj[u].append(v)
        if bidirectional:
            edges_adj[v].append(u)
    return edges_adj


def naive_partition(
    edge_list: list[tuple[int, int]],
    size: int,
    bidirectional: bool = True,
    traversed: set[int] | None = None,
) -> list[int]:
    if traversed is None:
        traversed = set()
    nodes = set()
    for (u, v) in edge_list:
        nodes.add(u)
        nodes.add(v)
    if not nodes:
        return []
    max_node_id = max(nodes)
    adj = create_list_of_edges(max_node_id + 1, edge_list, bidirectional=bidirectional)
    remaining = nodes - traversed
    if not remaining:
        return []
    current = min(remaining)
    sub_nodes = [current]
    traversed |= {current}
    to_explore = adj[current]
    while len(sub_nodes) < size and len(traversed) < len(nodes) and to_explore:
        next_level: list[int] = []
        for neigh in to_explore:
            if neigh in traversed:
                continue
            traversed |= {neigh}
            sub_nodes.append(neigh)
            if len(sub_nodes) >= size:
                break
            next_level.extend(adj[neigh])
        if len(sub_nodes) >= size:
            break
        if not next_level and len(traversed) < len(nodes):
            current = min(nodes - traversed)
            traversed |= {current}
            sub_nodes.append(current)
            next_level.extend(adj[current])
        to_explore = next_level
    return sub_nodes


def partition_train_graph(A_train_csr: sp.csr_matrix, num_parts: int) -> list[list[int]]:
    """Split training CSR adjacency into `num_parts` disjoint connected subgraphs."""
    edge_list = list(zip(*A_train_csr.nonzero()))
    edge_list = [(u, v) for (u, v) in edge_list if u != v]  # drop self-loops
    traversed: set[int] = set()
    n = A_train_csr.shape[0]
    base = n // num_parts
    rem = n % num_parts
    sizes = [base + (1 if i < rem else 0) for i in range(num_parts)]
    parts: list[list[int]] = []
    for size in sizes:
        nodes = naive_partition(edge_list, size, bidirectional=True, traversed=set(traversed))
        traversed |= set(nodes)
        parts.append(nodes)
    return parts
