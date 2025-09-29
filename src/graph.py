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


def build_knn_from_grm(
    GRM_df,
    k: int = 5,
    weighted_edges: bool = False,
    symmetrize_mode: str = "union",
    add_self_loops: bool = True,
) -> sp.csr_matrix:
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
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_knn_from_snp(
    X: np.ndarray,
    k: int = 5,
    weighted_edges: bool = False,
    symmetrize_mode: str = "union",
    add_self_loops: bool = True,
    laplacian_smoothing: bool = True,
) -> sp.csr_matrix:
    n = X.shape[0]
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
    """Algorithm 2: build ONE global adjacency, then induce splits from it."""
    source = graph_cfg.get("source", "grm").lower()
    if source == "grm":
        A = build_knn_from_grm(
            GRM_df,
            k=graph_cfg.get("knn_k", 5),
            weighted_edges=graph_cfg.get("weighted_edges", False),
            symmetrize_mode=graph_cfg.get("symmetrize_mode", "mutual"),
            add_self_loops=graph_cfg.get("self_loops", True),
        )
        return gcn_normalize(A)
    # SNP source
    A = build_knn_from_snp(
        X,
        k=graph_cfg.get("knn_k", 5),
        weighted_edges=graph_cfg.get("weighted_edges", False),
        symmetrize_mode=graph_cfg.get("symmetrize_mode", "mutual"),
        add_self_loops=graph_cfg.get("self_loops", True),
        laplacian_smoothing=graph_cfg.get("laplacian_smoothing", True),
    )
    return A if graph_cfg.get("laplacian_smoothing", True) else gcn_normalize(A)


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
