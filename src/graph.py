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
    if symmetrize_mode == "mutual":
        A_sym = A.minimum(A.T)
    else:
        A_sym = A.maximum(A.T)
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
            sim_matrix = sp.coo_matrix(
                (sim_vals, A_dir.nonzero()), shape=(n, n)
            ).tocsr()
        sim_sym = (
            sim_matrix.minimum(sim_matrix.T)
            if symmetrize_mode == "mutual"
            else sim_matrix.maximum(sim_matrix.T)
        )
        L = csgraph_laplacian(sim_sym, normed=False)
        L = sp.csr_matrix(L)
        L_coo = L.tocoo()
        if weighted_edges:
            max_w = np.max(np.abs(L_coo.data)) if L_coo.data.size > 0 else 1.0
            new_weights = (
                1.0 - (np.abs(L_coo.data) / max_w)
                if max_w != 0
                else np.ones_like(L_coo.data)
            )
            A_result = sp.coo_matrix(
                (new_weights, (L_coo.row, L_coo.col)), shape=(n, n)
            ).tocsr()
        else:
            ones = np.ones_like(L_coo.data, dtype=float)
            A_result = sp.coo_matrix(
                (ones, (L_coo.row, L_coo.col)), shape=(n, n)
            ).tocsr()
        return A_result
    else:
        if weighted_edges:
            vals = A_dir.data
            max_val = vals.max() if vals.size > 0 else 1.0
            sim_vals = (
                1.0 - (vals / max_val) if max_val != 0 else np.ones_like(vals)
            )
            A_dir = sp.coo_matrix(
                (sim_vals, A_dir.nonzero()), shape=(n, n)
            ).tocsr()
        A_sym = (
            A_dir.minimum(A_dir.T)
            if symmetrize_mode == "mutual"
            else A_dir.maximum(A_dir.T)
        )
        if add_self_loops:
            A_sym = A_sym + sp.eye(n, dtype=A_sym.dtype, format="csr")
        return A_sym


def induce_subgraph(A: sp.csr_matrix, nodes: np.ndarray) -> sp.csr_matrix:
    """Induce a subgraph adjacency matrix from the given global adjacency and node indices."""
    nodes = np.array(nodes)
    mask = np.zeros(A.shape[0], dtype=bool)
    mask[nodes] = True
    A_sub = A[nodes][:, nodes]
    return A_sub.tocsr()


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
