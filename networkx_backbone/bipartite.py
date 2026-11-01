"""
Bipartite projection backbone methods.

These methods operate on bipartite networks and extract significant edges
in the one-mode projection among "agent" nodes.

Methods
-------
simple_projection
    Shared-neighbor weighted projection.
hyper_projection
    Hyperbolic weighted projection.
probs_projection
    ProbS random-walk weighted projection.
ycn_projection
    YCN stationary-flow weighted projection.
sdsm
    Stochastic Degree Sequence Model (Neal 2014).
fdsm
    Fixed Degree Sequence Model (Neal et al. 2021).
fixedfill, fixedrow, fixedcol
    Fixed fill / row / column null models (Neal et al. 2021).
"""

import warnings

import networkx as nx
from networkx.utils import not_implemented_for

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    # Bipartite projection methods
    "simple_projection",
    "hyper_projection",
    "probs_projection",
    "ycn_projection",
    "bipartite_projection",
    # Projection methods
    "sdsm",
    "fdsm",
    "fixedfill",
    "fixedrow",
    "fixedcol",
    # Utilities
    "bicm",
    "fastball",
    # High-level wrappers
    "backbone_from_projection",
    "backbone_from_weighted",
    "backbone_from_unweighted",
    "backbone",
]


# =====================================================================
# Validation and shared helpers
# =====================================================================


def _validate_bipartite(B, agent_nodes):
    """Validate that ``agent_nodes`` is one side of the bipartite graph ``B``.

    Returns ``agent_nodes`` materialised as a list, so that generators are
    consumed exactly once.
    """
    agent_nodes = list(agent_nodes)
    if not nx.is_bipartite(B):
        raise nx.NetworkXError("Input graph B is not bipartite.")
    try:
        agent_set = set(agent_nodes)
    except TypeError as err:
        raise nx.NetworkXError("agent_nodes must contain hashable nodes.") from err
    if not agent_set.issubset(B):
        raise nx.NetworkXError("agent_nodes contains nodes not in B.")
    for u, v in B.edges():
        u_in = u in agent_set
        if u_in == (v in agent_set):
            where = "inside" if u_in else "outside"
            raise nx.NetworkXError(
                "agent_nodes is not one side of the bipartition: edge "
                f"({u!r}, {v!r}) has both endpoints {where} agent_nodes."
            )
    return agent_nodes


def _as_generator(seed):
    """Return a :class:`numpy.random.Generator` for int/Generator/RandomState/None."""
    import numpy as np

    if isinstance(seed, np.random.RandomState):
        return np.random.Generator(seed._bit_generator)
    return np.random.default_rng(seed)


def _signed_pvalues(p_hi, p_lo):
    """Vectorized two-sided p-value and sign from upper/lower tails.

    ``sign`` is ``+1`` for a significantly strong (over-represented)
    co-occurrence and ``-1`` for a significantly weak (under-represented) one.
    """
    import numpy as np

    p_hi = np.clip(np.asarray(p_hi, dtype=float), 0.0, 1.0)
    p_lo = np.clip(np.asarray(p_lo, dtype=float), 0.0, 1.0)
    pval = np.minimum(1.0, 2.0 * np.minimum(p_hi, p_lo))
    sign = np.where(p_hi <= p_lo, 1, -1)
    return pval, sign


def _pair_graph(agents, p_hi, p_lo, attr, signed, alpha=None):
    """Build a projection graph from upper-triangle tail probabilities.

    ``p_hi`` / ``p_lo`` are flat arrays aligned with ``np.triu_indices(n, 1)``.
    If ``alpha`` is given only pairs with p-value ``< alpha`` are added.
    """
    import numpy as np

    G = nx.Graph()
    G.add_nodes_from(agents)
    n = len(agents)
    if n < 2:
        return G
    iu, ju = np.triu_indices(n, 1)
    if signed:
        pval, sign = _signed_pvalues(p_hi, p_lo)
    else:
        pval = np.clip(np.asarray(p_hi, dtype=float), 0.0, 1.0)
        sign = None
    keep = np.ones(len(iu), dtype=bool) if alpha is None else pval < alpha
    for idx in np.flatnonzero(keep):
        data = {attr: float(pval[idx])}
        if signed:
            data["sign"] = int(sign[idx])
        G.add_edge(agents[iu[idx]], agents[ju[idx]], **data)
    return G


def _bipartite_projection_matrix(B, agent_nodes):
    """Build the co-occurrence matrix for agent nodes.

    Returns
    -------
    agents : list
        Ordered list of agent nodes.
    artifacts : list
        Ordered list of artifact nodes.
    R : np.ndarray of shape (n_agents, n_artifacts)
        Binary incidence matrix (agents x artifacts).
    observed : np.ndarray of shape (n_agents, n_agents)
        Observed co-occurrence counts (symmetric).
    """
    import numpy as np

    agent_set = set(agent_nodes)
    agents = sorted(agent_set, key=str)
    artifacts = sorted(set(B.nodes()) - agent_set, key=str)

    a_idx = {v: i for i, v in enumerate(agents)}
    f_idx = {v: i for i, v in enumerate(artifacts)}

    na = len(agents)
    nf = len(artifacts)
    R = np.zeros((na, nf), dtype=int)

    for u, v in B.edges():
        if u in a_idx and v in f_idx:
            R[a_idx[u], f_idx[v]] = 1
        elif v in a_idx and u in f_idx:
            R[a_idx[v], f_idx[u]] = 1

    # Observed co-occurrence matrix
    observed = R @ R.T
    np.fill_diagonal(observed, 0)

    return agents, artifacts, R, observed


def _infer_agent_nodes(B):
    """Infer the agent partition of ``B``.

    Nodes with ``bipartite == 0`` are agents when that node attribute is set.
    Otherwise a two-coloring is used, which is only unique for connected
    graphs; disconnected graphs raise :exc:`networkx.AmbiguousSolution`.
    """
    attr_agents = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    if attr_agents:
        return sorted(attr_agents, key=str)
    if B.number_of_nodes() == 0:
        return []
    if not nx.is_bipartite(B):
        raise nx.NetworkXError("Input graph B is not bipartite.")
    if not nx.is_connected(B):
        raise nx.AmbiguousSolution(
            "B is disconnected, so its bipartition is ambiguous. Pass "
            "agent_nodes explicitly or set the 'bipartite' node attribute "
            "(0 for agents, 1 for artifacts)."
        )
    colors = nx.algorithms.bipartite.color(B)
    color0 = [n for n, c in colors.items() if c == 0]
    return sorted(color0, key=str)


# =====================================================================
# Poisson-binomial helpers
# =====================================================================


def _poisson_binomial_pmf(probs):
    """Exact Poisson-binomial pmf by sequential convolution.

    ``probs`` has shape ``(..., n)``; the result has shape ``(..., n + 1)``
    where ``pmf[..., k] = P(X = k)`` for ``X`` the sum of independent
    Bernoulli(``probs[..., t]``) variables.
    """
    import numpy as np

    q = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    n = q.shape[-1]
    pmf = np.zeros((*q.shape[:-1], n + 1), dtype=float)
    pmf[..., 0] = 1.0
    for t in range(n):
        p = q[..., t : t + 1]
        pmf[..., 1 : t + 2] = pmf[..., 1 : t + 2] * (1.0 - p) + pmf[..., 0 : t + 1] * p
        pmf[..., 0] *= 1.0 - p[..., 0]
    return pmf


def _pmf_tails(pmf, k):
    """Return ``(P(X >= k), P(X <= k))`` for 1-D ``pmf`` and integer array ``k``."""
    import numpy as np

    k = np.asarray(k, dtype=int)
    n = len(pmf) - 1
    sf = np.cumsum(pmf[::-1])[::-1]  # sf[j] = P(X >= j)
    cdf = np.cumsum(pmf)  # cdf[j] = P(X <= j)
    upper = np.where(k <= 0, 1.0, np.where(k > n, 0.0, sf[np.clip(k, 0, n)]))
    lower = np.where(k < 0, 0.0, np.where(k >= n, 1.0, cdf[np.clip(k, 0, n)]))
    return np.clip(upper, 0.0, 1.0), np.clip(lower, 0.0, 1.0)


def _poisson_binomial_sf(probs, k):
    """Exact Poisson-binomial upper tail ``P(X >= k)`` (via convolution)."""
    import numpy as np

    upper, _ = _pmf_tails(_poisson_binomial_pmf(np.ravel(probs)), k)
    return upper if np.ndim(k) else float(upper)


def _poisson_binomial_cdf(probs, k):
    """Exact Poisson-binomial lower tail ``P(X <= k)`` (via convolution)."""
    import numpy as np

    _, lower = _pmf_tails(_poisson_binomial_pmf(np.ravel(probs)), k)
    return lower if np.ndim(k) else float(lower)


def _rna_tails(mu, s2, s3, k):
    """Refined normal approximation (Hong 2013) to Poisson-binomial tails.

    Parameters are vectorized: ``mu = sum p``, ``s2 = sum p^2``,
    ``s3 = sum p^3`` for each test, and ``k`` the observed count.  Returns
    ``(P(X >= k), P(X <= k))`` where ``P(X <= k) ~ G((k + 0.5 - mu) / sigma)``
    with ``G(x) = Phi(x) + gamma (1 - x^2) phi(x) / 6``.
    """
    import numpy as np
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    k = np.asarray(k, dtype=float)
    var = np.asarray(mu - s2, dtype=float)
    skew = np.asarray(mu - 3.0 * s2 + 2.0 * s3, dtype=float)  # sum p(1-p)(1-2p)
    degenerate = var <= 1e-12
    sigma = np.sqrt(np.where(degenerate, 1.0, var))
    gamma = np.where(degenerate, 0.0, skew / sigma**3)

    x_hi = (k - 0.5 - mu) / sigma  # P(X >= k) = 1 - P(X <= k - 1)
    upper = norm.sf(x_hi) - gamma * (1.0 - x_hi**2) * norm.pdf(x_hi) / 6.0
    x_lo = (k + 0.5 - mu) / sigma
    lower = norm.cdf(x_lo) + gamma * (1.0 - x_lo**2) * norm.pdf(x_lo) / 6.0

    # Degenerate case: X is (numerically) the constant round(mu).
    m0 = np.round(mu)
    upper = np.where(degenerate, (k <= m0).astype(float), upper)
    lower = np.where(degenerate, (k >= m0).astype(float), lower)
    upper = np.where(k <= 0, 1.0, upper)
    lower = np.where(k < 0, 0.0, lower)
    return np.clip(upper, 0.0, 1.0), np.clip(lower, 0.0, 1.0)


def _pairwise_pb_tails(P, k, method="auto", need_lower=False):
    """Poisson-binomial tails for each agent pair under edge probabilities ``P``.

    For agents ``i < j`` (upper-triangle order) the co-occurrence count is a
    sum of independent Bernoulli(``P[i, t] * P[j, t]``) variables.
    """
    import numpy as np

    na, nf = P.shape
    iu, ju = np.triu_indices(na, 1)
    n_pairs = len(iu)
    if method == "auto":
        cheap = nf <= 200 and n_pairs * (nf + 1) * (nf + 1) <= 2e8
        method = "exact" if cheap else "rna"
    if method == "rna":
        mu = (P @ P.T)[iu, ju]
        P2 = P * P
        s2 = (P2 @ P2.T)[iu, ju]
        P3 = P2 * P
        s3 = (P3 @ P3.T)[iu, ju]
        upper, lower = _rna_tails(mu, s2, s3, k)
        return upper, (lower if need_lower else None)
    if method != "exact":
        raise ValueError("pvalue_method must be one of 'auto', 'exact', 'rna'.")

    upper = np.empty(n_pairs, dtype=float)
    lower = np.empty(n_pairs, dtype=float) if need_lower else None
    k = np.asarray(k, dtype=int)
    chunk = max(1, int(2_000_000 // (nf + 1)))
    for start in range(0, n_pairs, chunk):
        stop = min(start + chunk, n_pairs)
        pmf = _poisson_binomial_pmf(P[iu[start:stop]] * P[ju[start:stop]])
        kk = np.clip(k[start:stop], 0, nf)
        rows = np.arange(stop - start)
        sf = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
        upper[start:stop] = sf[rows, kk]
        if need_lower:
            lower[start:stop] = np.cumsum(pmf, axis=1)[rows, kk]
    upper = np.clip(upper, 0.0, 1.0)
    if need_lower:
        lower = np.clip(lower, 0.0, 1.0)
    return upper, lower


# =====================================================================
# Projection weightings
# =====================================================================


def _projection_weight_matrix(
    R, method="simple", symmetrize=True, max_iter=1000, tol=1e-12
):
    """Compute projection weights from a binary incidence matrix."""
    import numpy as np

    method_l = method.lower()
    if method_l == "hyperbolic":
        method_l = "hyper"

    if method_l == "simple":
        W = (R @ R.T).astype(float)
        np.fill_diagonal(W, 0.0)
        return W

    artifact_degree = R.sum(axis=0).astype(float)
    inv_artifact_degree = np.zeros_like(artifact_degree, dtype=float)
    nonzero_artifact = artifact_degree > 0
    inv_artifact_degree[nonzero_artifact] = 1.0 / artifact_degree[nonzero_artifact]
    H = (R * inv_artifact_degree) @ R.T

    if method_l == "hyper":
        W = H.astype(float)
        np.fill_diagonal(W, 0.0)
        return W

    agent_degree = R.sum(axis=1).astype(float)
    W_dir = np.divide(
        H,
        agent_degree[:, None],
        out=np.zeros_like(H, dtype=float),
        where=agent_degree[:, None] > 0,
    )

    if method_l == "probs":
        np.fill_diagonal(W_dir, 0.0)
        if symmetrize:
            W = 0.5 * (W_dir + W_dir.T)
            np.fill_diagonal(W, 0.0)
            return W
        return W_dir

    if method_l == "ycn":
        # Transition matrix for a random walk on the projected layer.
        T = W_dir.copy()
        zero_rows = np.where(agent_degree <= 0)[0]
        if zero_rows.size:
            T[zero_rows, :] = 0.0
            T[zero_rows, zero_rows] = 1.0

        n_agents = T.shape[0]
        if n_agents == 0:
            return np.zeros((0, 0), dtype=float)

        pi = np.full(n_agents, 1.0 / n_agents, dtype=float)
        for _ in range(max_iter):
            next_pi = pi @ T
            if np.linalg.norm(next_pi - pi, ord=1) <= tol:
                pi = next_pi
                break
            pi = next_pi

        total = pi.sum()
        if total > 0:
            pi = pi / total

        W_dir = pi[:, None] * T
        np.fill_diagonal(W_dir, 0.0)
        if symmetrize:
            W = 0.5 * (W_dir + W_dir.T)
            np.fill_diagonal(W, 0.0)
            return W
        return W_dir

    raise ValueError(
        "Unknown projection method. Choose one of: 'simple', 'hyper', 'probs', 'ycn'."
    )


def _projection_graph_from_matrix(agents, W, weight="weight", directed=False):
    """Create a weighted projected graph from a dense weight matrix."""
    projection = nx.DiGraph() if directed else nx.Graph()
    projection.add_nodes_from(agents)

    n_agents = len(agents)
    if directed:
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                wij = float(W[i, j])
                if wij > 0:
                    projection.add_edge(agents[i], agents[j], **{weight: wij})
        return projection

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            wij = float(W[i, j])
            if wij > 0:
                projection.add_edge(agents[i], agents[j], **{weight: wij})
    return projection


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def simple_projection(B, agent_nodes, weight="weight"):
    """Project a bipartite graph using shared-neighbor counts.

    For agent nodes ``u`` and ``v``, the weight is
    ``|Gamma(u) ∩ Gamma(v)|``, the number of artifacts they share.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.

    Returns
    -------
    projection : graph
        Undirected graph on the agent nodes; pairs with a positive weight are
        joined by an edge carrying the *weight* attribute.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    bipartite_projection, hyper_projection, probs_projection, ycn_projection

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import simple_projection
    >>> B = nx.Graph([(1, "a"), (1, "b"), (2, "a"), (2, "c"), (3, "b"), (3, "c")])
    >>> P = simple_projection(B, agent_nodes=[1, 2, 3])
    >>> sorted(P.edges(data="weight"))
    [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    """
    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(R, method="simple")
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=False)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def hyper_projection(B, agent_nodes, weight="weight"):
    """Project a bipartite graph using hyperbolic weighting.

    For agent nodes ``u`` and ``v``, the weight is
    ``sum_{z in Gamma(u) ∩ Gamma(v)} 1 / d(z)`` where ``d(z)`` is the
    artifact degree.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.

    Returns
    -------
    projection : graph
        Undirected weighted graph on the agent nodes.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    bipartite_projection, simple_projection

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import hyper_projection
    >>> B = nx.Graph([(1, "a"), (1, "b"), (2, "a"), (2, "c"), (3, "b"), (3, "c")])
    >>> P = hyper_projection(B, agent_nodes=[1, 2, 3])
    >>> sorted(P.edges(data="weight"))
    [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]
    """
    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(R, method="hyper")
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=False)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def probs_projection(B, agent_nodes, weight="weight", directed=False):
    """Project a bipartite graph using ProbS random-walk weights.

    The directed score from ``u`` to ``v`` is
    ``(1 / d(u)) * sum_{z in Gamma(u) ∩ Gamma(v)} 1 / d(z)``.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    directed : bool, optional (default=False)
        If ``True`` return the asymmetric scores as a :class:`~networkx.DiGraph`;
        otherwise the two directions are averaged.

    Returns
    -------
    projection : graph
        Weighted :class:`~networkx.Graph` (or :class:`~networkx.DiGraph`) on
        the agent nodes.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    bipartite_projection, ycn_projection

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import probs_projection
    >>> B = nx.Graph([("a", "x"), ("a", "y"), ("b", "x"), ("c", "y")])
    >>> P = probs_projection(B, agent_nodes=["a", "b", "c"], directed=True)
    >>> P["a"]["b"]["weight"], P["b"]["a"]["weight"]
    (0.25, 0.5)
    """
    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(R, method="probs", symmetrize=not directed)
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=directed)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def ycn_projection(
    B, agent_nodes, weight="weight", directed=False, max_iter=1000, tol=1e-12
):
    """Project a bipartite graph using YCN random-walk stationary flow.

    This method estimates a random-walk transition matrix on the projected
    layer and scores pairs by stationary flow ``pi_u * T_uv``.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    directed : bool, optional (default=False)
        If ``True`` return the asymmetric flows as a :class:`~networkx.DiGraph`;
        otherwise the two directions are averaged.
    max_iter : int, optional (default=1000)
        Maximum power iterations for the stationary distribution.
    tol : float, optional (default=1e-12)
        L1 convergence tolerance for the stationary distribution.

    Returns
    -------
    projection : graph
        Weighted :class:`~networkx.Graph` (or :class:`~networkx.DiGraph`) on
        the agent nodes.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    bipartite_projection, probs_projection

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. (2017). Network backboning with noisy
       data. arXiv:1906.09081.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import ycn_projection
    >>> B = nx.Graph([(1, "a"), (1, "b"), (2, "a"), (2, "c"), (3, "b"), (3, "c")])
    >>> P = ycn_projection(B, agent_nodes=[1, 2, 3])
    >>> round(P[1][2]["weight"], 4)
    0.0833
    """
    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, _, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    W = _projection_weight_matrix(
        R, method="ycn", symmetrize=not directed, max_iter=max_iter, tol=tol
    )
    return _projection_graph_from_matrix(agents, W, weight=weight, directed=directed)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def bipartite_projection(
    B,
    agent_nodes,
    method="simple",
    weight="weight",
    directed=False,
    max_iter=1000,
    tol=1e-12,
):
    """Dispatch weighted bipartite projections by method name.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    method : {"simple", "hyper", "hyperbolic", "probs", "ycn"}, optional
        Projection weighting (default ``"simple"``).
    weight : string, optional (default="weight")
        Edge attribute name for the projection weight.
    directed : bool, optional (default=False)
        Return directed scores; only supported for ``"probs"`` and ``"ycn"``.
    max_iter : int, optional (default=1000)
        Maximum iterations for ``method="ycn"``.
    tol : float, optional (default=1e-12)
        Convergence tolerance for ``method="ycn"``.

    Returns
    -------
    projection : graph
        Weighted projection on the agent nodes.

    Raises
    ------
    ValueError
        If *method* is unknown or ``directed=True`` is requested for a
        symmetric method.
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    simple_projection, hyper_projection, probs_projection, ycn_projection

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import bipartite_projection
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> P = bipartite_projection(B, women, method="hyper")
    >>> P.number_of_nodes()
    18
    """
    agent_nodes = _validate_bipartite(B, agent_nodes)
    method_l = method.lower()
    if method_l == "simple":
        if directed:
            raise ValueError("simple projection does not support directed=True")
        return simple_projection(B, agent_nodes=agent_nodes, weight=weight)
    if method_l in ("hyper", "hyperbolic"):
        if directed:
            raise ValueError("hyper projection does not support directed=True")
        return hyper_projection(B, agent_nodes=agent_nodes, weight=weight)
    if method_l == "probs":
        return probs_projection(
            B, agent_nodes=agent_nodes, weight=weight, directed=directed
        )
    if method_l == "ycn":
        return ycn_projection(
            B,
            agent_nodes=agent_nodes,
            weight=weight,
            directed=directed,
            max_iter=max_iter,
            tol=tol,
        )
    raise ValueError(
        "Unknown projection method. Choose one of: 'simple', 'hyper', 'probs', 'ycn'."
    )


def _apply_projection_weights(
    graph,
    B,
    agent_nodes,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Annotate graph edges with weights from a selected bipartite projection."""
    if projection_directed:
        raise ValueError(
            "projection_directed=True is not supported for undirected backbones"
        )

    projected = bipartite_projection(
        B,
        agent_nodes=agent_nodes,
        method=projection,
        weight=projection_weight,
        directed=False,
        max_iter=projection_max_iter,
        tol=projection_tol,
    )
    for u, v in graph.edges():
        if projected.has_edge(u, v):
            graph[u][v][projection_weight] = float(projected[u][v][projection_weight])
        else:
            graph[u][v][projection_weight] = 0.0

    return graph


# =====================================================================
# Bipartite Configuration Model
# =====================================================================


def _bicm_fit(R, required=None, prohibited=None, tol=1e-8, max_iter=10000):
    """Fit Bipartite Configuration Model probabilities to ``R``.

    Solves the maximum-entropy (Saracco et al. 2015) fixed point
    ``p_ik = x_i y_k / (1 + x_i y_k)`` with expected row and column sums equal
    to the observed ones.  Cells in the boolean masks ``required`` /
    ``prohibited`` are fixed at probability 1 / 0 and the model is fitted on
    the remaining (free) cells with correspondingly reduced margins
    (SDSM-EC, Neal & Neal 2023).  Rows or columns whose residual degree is 0
    or equal to their number of free cells are fixed at 0 / 1 outside the
    solve (iteratively, since fixing them changes the other margins).
    """
    import numpy as np

    R = (np.asarray(R) > 0).astype(int)
    n_rows, n_cols = R.shape
    P = np.zeros((n_rows, n_cols), dtype=float)
    free = np.ones((n_rows, n_cols), dtype=bool)
    if required is not None:
        P[required] = 1.0
        free &= ~required
    if prohibited is not None:
        free &= ~prohibited
    if n_rows == 0 or n_cols == 0:
        return P

    row_deg = R.sum(axis=1)
    col_deg = R.sum(axis=0)

    # Peel rows/columns whose free cells are all forced to 0 or 1.
    while True:
        fixed_mass = np.where(free, 0.0, P)
        r_res = row_deg - np.rint(fixed_mass.sum(axis=1)).astype(int)
        c_res = col_deg - np.rint(fixed_mass.sum(axis=0)).astype(int)
        r_free = free.sum(axis=1)
        c_free = free.sum(axis=0)
        if (
            (r_res < 0).any()
            or (r_res > r_free).any()
            or (c_res < 0).any()
            or (c_res > c_free).any()
        ):
            raise nx.NetworkXError(
                "Edge constraints are inconsistent with the observed degrees."
            )
        rows0 = (r_free > 0) & (r_res == 0)
        rows1 = (r_free > 0) & (r_res == r_free)
        cols0 = (c_free > 0) & (c_res == 0)
        cols1 = (c_free > 0) & (c_res == c_free)
        if not (rows0.any() or rows1.any() or cols0.any() or cols1.any()):
            break
        for mask_r, val in ((rows0, 0.0), (rows1, 1.0)):
            cells = free & mask_r[:, None]
            P[cells] = val
            free &= ~cells
        for mask_c, val in ((cols0, 0.0), (cols1, 1.0)):
            cells = free & mask_c[None, :]
            P[cells] = val
            free &= ~cells

    rows = np.flatnonzero(free.any(axis=1))
    cols = np.flatnonzero(free.any(axis=0))
    if rows.size == 0:
        return P

    r = r_res[rows].astype(float)
    c = c_res[cols].astype(float)
    F = free[np.ix_(rows, cols)]
    total = r.sum()

    if F.all():
        # No edge constraints: rows (columns) with equal degree share x (y),
        # so solve on the unique degrees weighted by multiplicity.
        ru, rinv, rn = np.unique(r, return_inverse=True, return_counts=True)
        cu, cinv, cn = np.unique(c, return_inverse=True, return_counts=True)
        x = ru / np.sqrt(total)
        y = cu / np.sqrt(total)
        err = np.inf
        for _ in range(max_iter):
            xy = x[:, None] * y[None, :]
            x = ru / ((cn * y)[None, :] / (1.0 + xy)).sum(axis=1)
            xy = x[:, None] * y[None, :]
            y = cu / ((rn * x)[:, None] / (1.0 + xy)).sum(axis=0)
            xy = x[:, None] * y[None, :]
            Pu = xy / (1.0 + xy)
            err = max(
                np.abs((Pu * cn).sum(axis=1) - ru).max(),
                np.abs((Pu * rn[:, None]).sum(axis=0) - cu).max(),
            )
            if err < tol:
                break
        sub = Pu[np.ix_(rinv.ravel(), cinv.ravel())]
    else:
        Ff = F.astype(float)
        x = r / np.sqrt(total)
        y = c / np.sqrt(total)
        err = np.inf
        for _ in range(max_iter):
            xy = x[:, None] * y[None, :]
            x = r / (Ff * y[None, :] / (1.0 + xy)).sum(axis=1)
            xy = x[:, None] * y[None, :]
            y = c / (Ff * x[:, None] / (1.0 + xy)).sum(axis=0)
            xy = x[:, None] * y[None, :]
            sub = Ff * xy / (1.0 + xy)
            err = max(
                np.abs(sub.sum(axis=1) - r).max(), np.abs(sub.sum(axis=0) - c).max()
            )
            if err < tol:
                break

    if not err < tol:
        warnings.warn(
            f"BiCM fit did not converge in {max_iter} iterations "
            f"(max margin error {err:.3g} > tol={tol:g}).",
            RuntimeWarning,
            stacklevel=3,
        )

    block = P[np.ix_(rows, cols)]
    block[F] = sub[F]
    P[np.ix_(rows, cols)] = block
    return P


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def bicm(B, agent_nodes, return_labels=False, *, tol=1e-8, max_iter=10000):
    """Estimate Bipartite Configuration Model edge probabilities.

    Fits the maximum-entropy BiCM [1]_: ``P[i, k] = x_i y_k / (1 + x_i y_k)``
    where the fitness parameters solve ``sum_k P[i, k] = r_i`` and
    ``sum_i P[i, k] = c_k`` for the observed agent degrees ``r`` and artifact
    degrees ``c``.  The fixed point
    ``x_i = r_i / sum_k y_k / (1 + x_i y_k)``,
    ``y_k = c_k / sum_i x_i / (1 + x_i y_k)`` is iterated to convergence.
    Degree-0 rows/columns get probability 0 and full-degree rows/columns get
    probability 1 (they are fixed before solving).

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition.
    return_labels : bool, optional (default=False)
        If ``True``, also return ``(agents, artifacts)`` labels.
    tol : float, optional (default=1e-8)
        Convergence tolerance on the maximum absolute error between expected
        and observed degrees.
    max_iter : int, optional (default=10000)
        Maximum fixed-point iterations.  A :class:`RuntimeWarning` is emitted
        if the fit has not converged.

    Returns
    -------
    P : np.ndarray
        Probability matrix of shape ``(n_agents, n_artifacts)`` where
        ``P[i, k]`` is the estimated probability of an edge between
        agent ``i`` and artifact ``k``.
    (P, agents, artifacts) : tuple
        Returned when ``return_labels=True``.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    Warns
    -----
    RuntimeWarning
        If the fixed point did not converge within *max_iter* iterations.

    See Also
    --------
    sdsm

    References
    ----------
    .. [1] Saracco, F., Di Clemente, R., Gabrielli, A., & Squartini, T. (2015).
       Randomizing bipartite networks: the case of the World Trade Web.
       *Scientific Reports*, 5, 10595.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from networkx_backbone import bicm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> P, agents, artifacts = bicm(B, women, return_labels=True)
    >>> degrees = np.array([B.degree(a) for a in agents])
    >>> bool(np.allclose(P.sum(axis=1), degrees, atol=1e-6))
    True
    """
    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, _ = _bipartite_projection_matrix(B, agent_nodes)
    P = _bicm_fit(R, tol=tol, max_iter=max_iter)

    if return_labels:
        return P, agents, artifacts
    return P


# =====================================================================
# Curveball / fastball sampling
# =====================================================================


def _curveball_trades(row_sets, n_swaps, rng):
    """Apply ``n_swaps`` curveball trades in place to a list of row sets."""
    n_rows = len(row_sets)
    if n_rows < 2 or n_swaps <= 0:
        return
    first = rng.integers(0, n_rows, size=n_swaps)
    second = rng.integers(0, n_rows - 1, size=n_swaps)
    second = second + (second >= first)
    for r1, r2 in zip(first.tolist(), second.tolist()):
        s1 = row_sets[r1]
        s2 = row_sets[r2]
        only1 = s1 - s2
        if not only1:
            continue
        only2 = s2 - s1
        if not only2:
            continue
        common = s1 & s2
        trade_pool = list(only1) + list(only2)
        rng.shuffle(trade_pool)
        keep1 = len(only1)
        row_sets[r1] = common | set(trade_pool[:keep1])
        row_sets[r2] = common | set(trade_pool[keep1:])


def _rows_to_matrix(row_sets, n_cols):
    import numpy as np

    out = np.zeros((len(row_sets), n_cols), dtype=int)
    for i, cols in enumerate(row_sets):
        if cols:
            out[i, list(cols)] = 1
    return out


def fastball(matrix, n_swaps=None, seed=None):
    """Randomize a binary matrix while preserving row and column sums.

    This is a Curveball-style randomizer commonly used for bipartite
    null-model generation.

    Parameters
    ----------
    matrix : array-like
        Binary matrix with shape ``(n_rows, n_cols)``.
    n_swaps : int or None, optional
        Number of row-pair trades. Defaults to ``5 * n_rows``.
    seed : integer, random_state, or None
        Random seed for reproducibility (int, :class:`numpy.random.Generator`,
        :class:`numpy.random.RandomState` or ``None``).

    Returns
    -------
    randomized : np.ndarray
        Randomized binary matrix with preserved row/column sums.

    Raises
    ------
    ValueError
        If *matrix* is not two-dimensional or *n_swaps* is negative.

    See Also
    --------
    fdsm

    References
    ----------
    .. [1] Godard, K., & Neal, Z. P. (2022). fastball: A fast algorithm to
       sample bipartite graphs with fixed degree sequences. *Journal of Complex
       Networks*, 10(6), cnac049.

    Examples
    --------
    >>> import numpy as np
    >>> from networkx_backbone import fastball
    >>> M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    >>> S = fastball(M, seed=1)
    >>> bool((S.sum(axis=0) == M.sum(axis=0)).all() and (S.sum(axis=1) == M.sum(axis=1)).all())
    True
    """
    import numpy as np

    M = np.asarray(matrix)
    if M.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    R = (M > 0).astype(int)
    n_rows, n_cols = R.shape
    if n_rows == 0 or n_cols == 0:
        return R.copy()

    if n_swaps is None:
        n_swaps = max(1, n_rows * 5)
    if n_swaps < 0:
        raise ValueError("n_swaps must be >= 0")

    rng = _as_generator(seed)
    row_sets = [set(np.flatnonzero(R[i]).tolist()) for i in range(n_rows)]
    _curveball_trades(row_sets, n_swaps, rng)
    return _rows_to_matrix(row_sets, n_cols)


def _fdsm_samples(R, trials, rng, burn_in=None, thin=None):
    """Yield ``trials`` matrices from the FDSM via a curveball Markov chain.

    The chain starts at the observed matrix ``R``, runs ``burn_in`` trades,
    then yields a sample every ``thin`` trades.  Every sample has exactly the
    row and column sums of ``R``.
    """
    import numpy as np

    n_rows, n_cols = R.shape
    if burn_in is None:
        burn_in = 10 * n_rows
    if thin is None:
        thin = 5 * n_rows
    if burn_in < 0 or thin < 0:
        raise ValueError("burn_in and thin must be >= 0")
    row_sets = [set(np.flatnonzero(R[i]).tolist()) for i in range(n_rows)]
    _curveball_trades(row_sets, burn_in, rng)
    for _ in range(trials):
        _curveball_trades(row_sets, thin, rng)
        yield _rows_to_matrix(row_sets, n_cols)


# =====================================================================
# Fixed fill / row / column models -- Neal et al. (2021)
# =====================================================================


def _fixedfill_pmf(n_agents, n_artifacts, n_filled):
    """Exact co-occurrence pmf for one agent pair under the fixed-fill model.

    The null is uniform over all ``n_agents x n_artifacts`` binary matrices
    with ``n_filled`` ones.  If the two agents' rows hold ``s`` ones (a
    hypergeometric count), the number of placements with exactly ``x`` shared
    columns is ``C(n, x) C(n - x, s - 2x) 2^(s - 2x)``, so

    ``P(X = x) = sum_s C(n, x) C(n - x, s - 2x) 2^(s - 2x) C(N - 2n, F - s) / C(N, F)``

    with ``n`` artifacts, ``N = n_agents * n`` cells and ``F`` filled cells.
    """
    import numpy as np
    from scipy.special import gammaln, logsumexp

    n = int(n_artifacts)
    N = int(n_agents) * n
    F = int(n_filled)
    K = 2 * n

    def lchoose(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

    pmf = np.zeros(n + 1, dtype=float)
    log_total = lchoose(N, F)
    for x in range(n + 1):
        lo = max(2 * x, F - (N - K))
        hi = min(K, F, n + x)
        if lo > hi:
            continue
        s = np.arange(lo, hi + 1, dtype=float)
        log_terms = (
            lchoose(n, x)
            + lchoose(n - x, s - 2 * x)
            + (s - 2 * x) * np.log(2.0)
            + lchoose(N - K, F - s)
            - log_total
        )
        pmf[x] = np.exp(logsumexp(log_terms))
    total = pmf.sum()
    if total > 0:
        pmf /= total
    return pmf


def _fixed_model_setup(B, agent_nodes):
    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, observed = _bipartite_projection_matrix(B, agent_nodes)
    return agent_nodes, agents, artifacts, R, observed


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def fixedfill(
    B,
    agent_nodes,
    alpha=0.05,
    signed=False,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Backbone from the fixed-fill null model for bipartite projections.

    Under the fixed fill model (FFM) [1]_ every bipartite matrix with the same
    number of agents, artifacts and edges (filled cells) is equally likely.
    The co-occurrence count of every agent pair then has the same exact
    distribution, obtained by conditioning on how many of the filled cells
    fall in the pair's two rows; the p-value is its upper tail at the
    observed count (and the lower tail too when ``signed=True``).

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    alpha : float, optional (default=0.05)
        Significance level; only pairs with p-value ``< alpha`` are kept.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the p-value is two-sided and a
        ``"sign"`` attribute marks significantly strong (``+1``) or weak
        (``-1``) co-occurrence.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.

    Returns
    -------
    backbone : graph
        Graph on the agent nodes whose edges are the significant pairs, each
        with a ``"fixedfill_pvalue"`` attribute and the projection weight.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *projection* is unknown or ``projection_directed=True``.

    See Also
    --------
    fixedrow, fixedcol, sdsm, fdsm, backbone_from_projection

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fixedfill
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> H = fixedfill(B, women, alpha=0.05)
    >>> all(d["fixedfill_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    import numpy as np

    agent_nodes, agents, _, R, observed = _fixed_model_setup(B, agent_nodes)
    na, nf = R.shape

    if na < 2 or nf == 0:
        backbone = nx.Graph()
        backbone.add_nodes_from(agents)
    else:
        pmf = _fixedfill_pmf(na, nf, int(R.sum()))
        k = observed[np.triu_indices(na, 1)]
        p_hi, p_lo = _pmf_tails(pmf, k)
        backbone = _pair_graph(agents, p_hi, p_lo, "fixedfill_pvalue", signed, alpha)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def fixedrow(
    B,
    agent_nodes,
    alpha=0.05,
    signed=False,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Backbone from the fixed-row null model for bipartite projections.

    Under the fixed row model (FRM) [1]_ each agent keeps its degree ``d_i``
    and its artifacts are a uniformly random ``d_i``-subset of all
    artifacts, so the co-occurrence of agents ``i`` and ``j`` is exactly
    hypergeometric with parameters ``(n_artifacts, d_i, d_j)``.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    alpha : float, optional (default=0.05)
        Significance level; only pairs with p-value ``< alpha`` are kept.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` attribute.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.

    Returns
    -------
    backbone : graph
        Graph on the agent nodes whose edges are the significant pairs, each
        with a ``"fixedrow_pvalue"`` attribute and the projection weight.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *projection* is unknown or ``projection_directed=True``.

    See Also
    --------
    fixedfill, fixedcol, sdsm, fdsm, backbone_from_projection

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fixedrow
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> H = fixedrow(B, women, alpha=0.05)
    >>> all(d["fixedrow_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    import numpy as np
    from scipy import stats as sp_stats

    agent_nodes, agents, _, R, observed = _fixed_model_setup(B, agent_nodes)
    na, nf = R.shape

    if na < 2 or nf == 0:
        backbone = nx.Graph()
        backbone.add_nodes_from(agents)
    else:
        iu, ju = np.triu_indices(na, 1)
        deg = R.sum(axis=1)
        k = observed[iu, ju]
        p_hi = sp_stats.hypergeom.sf(k - 1, nf, deg[iu], deg[ju])
        p_lo = sp_stats.hypergeom.cdf(k, nf, deg[iu], deg[ju])
        backbone = _pair_graph(agents, p_hi, p_lo, "fixedrow_pvalue", signed, alpha)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def fixedcol(
    B,
    agent_nodes,
    alpha=0.05,
    signed=False,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
):
    """Backbone from the fixed-column null model for bipartite projections.

    Under the fixed column model (FCM) [1]_ each artifact ``k`` keeps its
    degree ``c_k`` and its agents are a uniformly random ``c_k``-subset of the
    ``m`` agents, independently across artifacts.  A given pair co-occurs on
    artifact ``k`` with probability ``c_k (c_k - 1) / (m (m - 1))``, so the
    co-occurrence count is a Poisson-binomial variable with the same
    distribution for every pair; it is computed exactly once and its tails
    give the p-values.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    alpha : float, optional (default=0.05)
        Significance level; only pairs with p-value ``< alpha`` are kept.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` attribute.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.

    Returns
    -------
    backbone : graph
        Graph on the agent nodes whose edges are the significant pairs, each
        with a ``"fixedcol_pvalue"`` attribute and the projection weight.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *projection* is unknown or ``projection_directed=True``.

    See Also
    --------
    fixedfill, fixedrow, sdsm, fdsm, backbone_from_projection

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fixedcol
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> H = fixedcol(B, women, alpha=0.05)
    >>> all(d["fixedcol_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    import numpy as np

    agent_nodes, agents, _, R, observed = _fixed_model_setup(B, agent_nodes)
    na, nf = R.shape

    if na < 2 or nf == 0:
        backbone = nx.Graph()
        backbone.add_nodes_from(agents)
    else:
        col_sums = R.sum(axis=0).astype(float)
        pair_prob = np.clip(col_sums * (col_sums - 1) / (na * (na - 1)), 0.0, 1.0)
        pmf = _poisson_binomial_pmf(pair_prob)
        k = observed[np.triu_indices(na, 1)]
        p_hi, p_lo = _pmf_tails(pmf, k)
        backbone = _pair_graph(agents, p_hi, p_lo, "fixedcol_pvalue", signed, alpha)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


# =====================================================================
# High-level wrappers
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def backbone_from_projection(
    B,
    agent_nodes=None,
    method="sdsm",
    alpha=0.05,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
    mtc="none",
    **kwargs,
):
    """Extract a bipartite projection backbone using method-name dispatch.

    Mirrors ``backbone::backbone_from_projection`` in R: the selected null
    model scores every agent pair, the optional multiple-testing correction
    *mtc* is applied across all pairs, and the pairs with (adjusted) p-value
    ``< alpha`` are returned.  The result is thresholded the same way for
    every *method*.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable or None, optional (default=None)
        Nodes in the "agent" partition.  If ``None`` they are taken from the
        ``bipartite`` node attribute (value 0) when present, or from a
        two-coloring of a connected *B*.
    method : {"sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"}, optional
        Null model (default ``"sdsm"``).
    alpha : float, optional (default=0.05)
        Significance level for the returned backbone.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Not supported for the undirected backbone (raises if ``True``).
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    mtc : string, optional (default="none")
        Multiple-testing correction (see
        :func:`~networkx_backbone.adjust_pvalues`) applied to the p-values
        of all agent pairs before thresholding.
    **kwargs
        Passed to the selected model (e.g. ``signed``, ``trials``, ``seed``,
        ``prohibited``, ``required``).

    Returns
    -------
    backbone : graph
        Graph on the agent nodes containing the significant pairs, each with
        a ``"<method>_pvalue"`` attribute (unadjusted), the projection weight
        and, for ``signed=True``, a ``"sign"`` attribute.

    Raises
    ------
    ValueError
        If *method* is unknown.
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    AmbiguousSolution
        If *agent_nodes* is ``None`` and *B* is disconnected without
        ``bipartite`` node attributes.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.

    See Also
    --------
    sdsm, fdsm, fixedfill, fixedrow, fixedcol, backbone

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone_from_projection
    >>> B = nx.davis_southern_women_graph()
    >>> H = backbone_from_projection(B, method="sdsm", alpha=0.05)
    >>> all(d["sdsm_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    import numpy as np

    from networkx_backbone.filters import threshold_filter

    if agent_nodes is None:
        agent_nodes = _infer_agent_nodes(B)
    else:
        agent_nodes = list(agent_nodes)

    methods = {
        "sdsm": sdsm,
        "fdsm": fdsm,
        "fixedfill": fixedfill,
        "fixedrow": fixedrow,
        "fixedcol": fixedcol,
    }
    method_l = method.lower()
    if method_l not in methods:
        raise ValueError(
            "Unknown projection method. Choose one of: "
            "'sdsm', 'fdsm', 'fixedfill', 'fixedrow', 'fixedcol'."
        )

    # Score every pair (fixed* models only threshold internally, so ask them
    # for all pairs), then threshold uniformly so ``mtc`` sees every test.
    model_alpha = alpha if method_l in ("sdsm", "fdsm") else np.inf
    scored = methods[method_l](
        B,
        agent_nodes=agent_nodes,
        alpha=model_alpha,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
        **kwargs,
    )
    return threshold_filter(scored, f"{method_l}_pvalue", alpha, mode="below", mtc=mtc)


def backbone_from_weighted(
    G,
    method="disparity",
    weight="weight",
    alpha=0.05,
    signed=False,
    mtc="none",
    collapse_multiedges=True,
    edge_type_attr=None,
    **kwargs,
):
    """Extract a weighted-network backbone using method-name dispatch.

    Multi-edge inputs can be collapsed automatically before scoring.

    Parameters
    ----------
    G : graph
        A weighted NetworkX graph.
    method : {"disparity", "mlf", "lans", "global"}, optional
        Backbone method (aliases such as ``"disparity_filter"`` accepted).
    weight : string, optional (default="weight")
        Edge weight attribute.
    alpha : float, optional (default=0.05)
        Significance level for the statistical methods.
    signed : bool, optional (default=False)
        If ``True``, use a two-tailed test (for ``disparity``/``mlf``/``lans``)
        so the backbone retains significantly strong and significantly weak
        edges, each annotated with a ``"sign"`` attribute.
    mtc : string, optional (default="none")
        Multiple-testing correction applied before thresholding (see
        :func:`~networkx_backbone.adjust_pvalues`).
    collapse_multiedges : bool, optional (default=True)
        If ``True`` and ``G`` is a ``MultiGraph`` or ``MultiDiGraph``,
        collapse parallel edges using
        :func:`~networkx_backbone.multigraph_to_weighted`.
    edge_type_attr : string or None, optional (default=None)
        Edge attribute used when collapsing multi-edges. If provided, edge
        weights count distinct edge types per node pair; otherwise they count
        parallel edges.
    **kwargs
        ``threshold=...`` for ``method="global"``.

    Returns
    -------
    backbone : graph
        The filtered graph.

    Raises
    ------
    ValueError
        If *method* is unknown or ``method="global"`` lacks ``threshold``.

    See Also
    --------
    backbone, backbone_from_projection, backbone_from_unweighted

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone_from_weighted
    >>> G = nx.les_miserables_graph()
    >>> H = backbone_from_weighted(G, method="disparity", alpha=0.05)
    >>> H.number_of_edges() <= G.number_of_edges()
    True
    """
    from networkx_backbone.filters import (
        boolean_filter,
        multigraph_to_weighted,
        threshold_filter,
    )
    from networkx_backbone.statistical import (
        disparity_filter,
        lans_filter,
        marginal_likelihood_filter,
    )
    from networkx_backbone.structural import global_threshold_filter

    if collapse_multiedges and G.is_multigraph():
        G = multigraph_to_weighted(G, weight=weight, edge_type_attr=edge_type_attr)

    method_l = method.lower()

    if method_l in ("disparity", "disparity_filter"):
        scored = disparity_filter(G, weight=weight, signed=signed)
        return threshold_filter(
            scored, "disparity_pvalue", alpha, mode="below", mtc=mtc
        )

    if method_l in ("mlf", "marginal_likelihood", "marginal_likelihood_filter"):
        scored = marginal_likelihood_filter(G, weight=weight, signed=signed)
        return threshold_filter(scored, "ml_pvalue", alpha, mode="below", mtc=mtc)

    if method_l in ("lans", "lans_filter"):
        scored = lans_filter(G, weight=weight, signed=signed)
        return threshold_filter(scored, "lans_pvalue", alpha, mode="below", mtc=mtc)

    if method_l in ("global", "global_threshold", "global_threshold_filter"):
        threshold = kwargs.get("threshold")
        if threshold is None:
            raise ValueError("global threshold method requires `threshold=...`")
        scored = global_threshold_filter(G, threshold=threshold, weight=weight)
        return boolean_filter(scored, "global_threshold_keep")

    raise ValueError(
        "Unknown weighted method. Choose one of: 'disparity', 'mlf', 'lans', 'global'."
    )


def backbone_from_unweighted(G, method="sparsify", **kwargs):
    """Extract an unweighted-network backbone using method-name dispatch.

    Parameters
    ----------
    G : graph
        An unweighted NetworkX graph.
    method : {"sparsify", "lspar", "local_degree"}, optional
        Sparsification method (default ``"sparsify"``).
    **kwargs
        Passed to the selected method (e.g. ``s`` for ``"lspar"``).

    Returns
    -------
    backbone : graph
        Graph containing the retained edges.

    Raises
    ------
    ValueError
        If *method* is unknown.

    See Also
    --------
    backbone, backbone_from_weighted

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone_from_unweighted
    >>> G = nx.karate_club_graph()
    >>> H = backbone_from_unweighted(G, method="lspar", s=0.5)
    >>> H.number_of_nodes() == G.number_of_nodes()
    True
    """
    from networkx_backbone.filters import boolean_filter
    from networkx_backbone.unweighted import local_degree, lspar, sparsify

    method_l = method.lower()
    if method_l == "sparsify":
        scored = sparsify(G, **kwargs)
        return boolean_filter(scored, "sparsify_keep")
    if method_l == "lspar":
        scored = lspar(G, **kwargs)
        return boolean_filter(scored, "sparsify_keep")
    if method_l in ("local_degree", "localdegree"):
        scored = local_degree(G, **kwargs)
        return boolean_filter(scored, "sparsify_keep")

    raise ValueError(
        "Unknown unweighted method. Choose one of: 'sparsify', 'lspar', 'local_degree'."
    )


def backbone(G, method=None, **kwargs):
    """Unified dispatch wrapper across projection, weighted, and unweighted methods.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    method : string or None, optional (default=None)
        Any method name accepted by :func:`backbone_from_projection`,
        :func:`backbone_from_weighted` or :func:`backbone_from_unweighted`.
        If ``None``, the method family is chosen from *G*: undirected simple
        graphs with ``bipartite`` node attributes use the SDSM projection
        backbone, graphs with ``weight`` edge attributes use the disparity
        filter, and other graphs use ``sparsify``.
    **kwargs
        Passed to the selected wrapper (e.g. ``alpha``, ``agent_nodes``,
        ``mtc``, ``signed``).

    Returns
    -------
    backbone : graph
        The extracted backbone.

    Raises
    ------
    ValueError
        If *method* is not recognised.

    See Also
    --------
    backbone_from_projection, backbone_from_weighted, backbone_from_unweighted

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import backbone
    >>> B = nx.davis_southern_women_graph()
    >>> H = backbone(B, alpha=0.05)  # bipartite attributes -> SDSM
    >>> all(d["sdsm_pvalue"] < 0.05 for _, _, d in H.edges(data=True))
    True
    """
    method_l = method.lower() if isinstance(method, str) else None

    projection_methods = {"sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"}
    weighted_methods = {
        "disparity",
        "disparity_filter",
        "mlf",
        "marginal_likelihood",
        "marginal_likelihood_filter",
        "lans",
        "lans_filter",
        "global",
        "global_threshold",
        "global_threshold_filter",
    }
    unweighted_methods = {"sparsify", "lspar", "local_degree", "localdegree"}

    if method_l in projection_methods:
        return backbone_from_projection(G, method=method_l, **kwargs)
    if method_l in weighted_methods:
        return backbone_from_weighted(G, method=method_l, **kwargs)
    if method_l in unweighted_methods:
        return backbone_from_unweighted(G, method=method_l, **kwargs)
    if method is not None:
        raise ValueError(f"Unknown method {method!r}")

    has_bipartite_attr = any("bipartite" in data for _, data in G.nodes(data=True))
    if (
        has_bipartite_attr
        and not G.is_directed()
        and not G.is_multigraph()
        and nx.is_bipartite(G)
    ):
        return backbone_from_projection(G, **kwargs)

    weighted = any("weight" in data for _, _, data in G.edges(data=True))
    if weighted:
        return backbone_from_weighted(G, **kwargs)
    return backbone_from_unweighted(G, **kwargs)


# =====================================================================
# 1. SDSM -- Neal (2014)
# =====================================================================


def _constraint_mask(pairs, name, a_idx, f_idx, shape):
    """Validate ``(agent, artifact)`` constraint pairs into a boolean mask."""
    import numpy as np

    mask = np.zeros(shape, dtype=bool)
    if pairs is None:
        return mask
    for pair in pairs:
        try:
            a, f = pair
        except (TypeError, ValueError) as err:
            raise nx.NetworkXError(
                f"{name} must contain (agent, artifact) pairs; got {pair!r}."
            ) from err
        try:
            if a in a_idx and f in f_idx:
                mask[a_idx[a], f_idx[f]] = True
                continue
            reversed_pair = f in a_idx and a in f_idx
        except TypeError as err:
            raise nx.NetworkXError(
                f"{name} must contain hashable (agent, artifact) pairs; got {pair!r}."
            ) from err
        if reversed_pair:
            raise nx.NetworkXError(
                f"{name} pairs must be ordered (agent, artifact); got "
                f"(artifact, agent) pair {pair!r}."
            )
        raise nx.NetworkXError(
            f"{name} pair {pair!r} is not an (agent, artifact) pair of nodes in B."
        )
    return mask


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def sdsm(
    B,
    agent_nodes,
    alpha=0.05,
    weight=None,
    signed=False,
    prohibited=None,
    required=None,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
    pvalue_method="auto",
):
    """Score a bipartite projection using the SDSM.

    The Stochastic Degree Sequence Model [1]_ treats each bipartite edge as
    an independent Bernoulli variable whose probability ``P[i, k]`` comes from
    the Bipartite Configuration Model (BiCM, see :func:`bicm`), the
    maximum-entropy model whose *expected* row and column sums equal the
    observed agent and artifact degrees.  The co-occurrence count of agents
    ``i`` and ``j`` is then a Poisson-binomial variable with success
    probabilities ``P[i, k] * P[j, k]``, and the p-value is its upper tail at
    the observed count.  As in R ``backbone``, the tail is evaluated with the
    refined normal approximation (RNA) of Hong (2013) [3]_; when the exact
    distribution is cheap (at most 200 artifacts), it is computed exactly by
    convolution instead (see *pvalue_method*).

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    alpha : float, optional (default=0.05)
        Retained for API compatibility. This function returns the full
        projected graph with p-values on all edges; apply
        :func:`~networkx_backbone.threshold_filter` (or use
        :func:`backbone_from_projection`) to select significant edges.
    weight : None or string, optional (default=None)
        Not used for SDSM (bipartite is unweighted); reserved for API
        consistency.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the stored p-value becomes two-sided
        and a ``"sign"`` edge attribute marks significantly strong (``+1``)
        versus significantly weak (``-1``) co-occurrences.
    prohibited : iterable of (agent, artifact) pairs or None, optional
        SDSM-EC edge constraints (Neal & Neal, 2023) [2]_: cells fixed to
        probability 0 in the null model (edges that cannot occur).  Each pair
        must be an absent edge of *B*.
    required : iterable of (agent, artifact) pairs or None, optional
        SDSM-EC edge constraints: cells fixed to probability 1 in the null
        model (edges that must occur).  Each pair must be an edge of *B*.
        With constraints the BiCM is refitted on the unconstrained cells only,
        so expected degrees still match the observed ones.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Whether to use directed projection scores. Not supported for the
        undirected backbone output.
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    pvalue_method : {"auto", "exact", "rna"}, optional (default="auto")
        How to evaluate Poisson-binomial tails: ``"exact"`` (convolution,
        ``O(n_artifacts^2)`` per pair), ``"rna"`` (refined normal
        approximation, as in R ``backbone``) or ``"auto"`` (exact when there
        are at most 200 artifacts and the total cost is small, else RNA).

    Returns
    -------
    backbone : graph
        Full unipartite projection among agent nodes.  Each edge has a
        ``"sdsm_pvalue"`` attribute and projection weights.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite, *agent_nodes* is not one side of it, or a
        constraint is not an ``(agent, artifact)`` pair consistent with *B*.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If *pvalue_method* or *projection* is unknown.

    Warns
    -----
    RuntimeWarning
        If the BiCM fit does not converge.

    See Also
    --------
    bicm, fdsm, backbone_from_projection

    References
    ----------
    .. [1] Neal, Z. P. (2014). The backbone of bipartite projections:
       Inferring relationships from co-authorship, co-sponsorship,
       co-attendance and other co-behaviors. *Social Networks*, 39, 84-97.
    .. [2] Neal, Z. P., & Neal, J. W. (2023). Stochastic Degree Sequence Model
       with Edge Constraints (SDSM-EC) for Backbone Extraction. *Complex
       Networks 12*, 127-136.
    .. [3] Hong, Y. (2013). On computing the distribution function for the
       Poisson binomial distribution. *Computational Statistics & Data
       Analysis*, 59, 41-51.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import sdsm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> scored = sdsm(B, agent_nodes=women)
    >>> from networkx_backbone import threshold_filter
    >>> backbone = threshold_filter(scored, "sdsm_pvalue", 0.05, mode="below")
    >>> all("sdsm_pvalue" in data for _, _, data in backbone.edges(data=True))
    True
    """
    import numpy as np

    if pvalue_method not in ("auto", "exact", "rna"):
        raise ValueError("pvalue_method must be one of 'auto', 'exact', 'rna'.")

    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, artifacts, R, observed = _bipartite_projection_matrix(B, agent_nodes)
    na = len(agents)

    req_mask = pro_mask = None
    if required is not None or prohibited is not None:
        a_idx = {v: idx for idx, v in enumerate(agents)}
        f_idx = {v: idx for idx, v in enumerate(artifacts)}
        req_mask = _constraint_mask(required, "required", a_idx, f_idx, R.shape)
        pro_mask = _constraint_mask(prohibited, "prohibited", a_idx, f_idx, R.shape)
        if (req_mask & (R == 0)).any():
            raise nx.NetworkXError("required pairs must be edges of B.")
        if (pro_mask & (R == 1)).any():
            raise nx.NetworkXError("prohibited pairs must not be edges of B.")

    if na < 2 or R.sum() == 0:
        backbone = nx.Graph()
        backbone.add_nodes_from(agents)
    else:
        P = _bicm_fit(R, required=req_mask, prohibited=pro_mask)
        k = observed[np.triu_indices(na, 1)]
        p_hi, p_lo = _pairwise_pb_tails(P, k, method=pvalue_method, need_lower=signed)
        backbone = _pair_graph(agents, p_hi, p_lo, "sdsm_pvalue", signed)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


# =====================================================================
# 2. FDSM -- Neal et al. (2021)
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def fdsm(
    B,
    agent_nodes,
    alpha=0.05,
    trials=1000,
    seed=None,
    signed=False,
    projection="simple",
    projection_weight="weight",
    projection_directed=False,
    projection_max_iter=1000,
    projection_tol=1e-12,
    burn_in=None,
    thin=None,
):
    """Score a bipartite projection using the FDSM.

    The Fixed Degree Sequence Model [1]_ uses Monte Carlo simulation to
    estimate p-values.  Random bipartite graphs that *exactly* preserve both
    the row sums (agent degrees) and column sums (artifact degrees) are drawn
    with a curveball Markov chain (:func:`fastball`) started at the observed
    graph: after ``burn_in`` trades, one sample is taken every ``thin``
    trades.  The p-value is ``(count + 1) / (trials + 1)``, where ``count``
    is the number of samples whose co-occurrence is at least (for the lower
    tail: at most) the observed one, so it is never exactly zero.

    Parameters
    ----------
    B : graph
        An undirected bipartite NetworkX graph.
    agent_nodes : iterable
        Nodes in the "agent" partition (one full side of the bipartition).
    alpha : float, optional (default=0.05)
        Retained for API compatibility. This function returns the full
        projected graph with p-values on all edges; apply
        :func:`~networkx_backbone.threshold_filter` (or use
        :func:`backbone_from_projection`) to select significant edges.
    trials : int, optional (default=1000)
        Number of Monte Carlo samples.
    seed : integer, random_state, or None (default)
        Random seed for reproducibility (int, :class:`numpy.random.Generator`,
        :class:`numpy.random.RandomState` or ``None``).
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the stored p-value becomes two-sided
        and a ``"sign"`` edge attribute marks significantly strong (``+1``)
        versus significantly weak (``-1``) co-occurrences.
    projection : {"simple", "hyper", "probs", "ycn"}, optional
        Projection weighting assigned to each returned edge.
    projection_weight : str, optional (default="weight")
        Edge attribute name for the projection weight.
    projection_directed : bool, optional (default=False)
        Whether to use directed projection scores. Not supported for the
        undirected backbone output.
    projection_max_iter : int, optional (default=1000)
        Maximum iterations for ``projection="ycn"``.
    projection_tol : float, optional (default=1e-12)
        Convergence tolerance for ``projection="ycn"``.
    burn_in : int or None, optional (default=None)
        Curveball trades before the first sample (default ``10 * n_agents``).
    thin : int or None, optional (default=None)
        Curveball trades between consecutive samples (default
        ``5 * n_agents``).

    Returns
    -------
    backbone : graph
        Full unipartite projection among agent nodes.  Each edge has an
        ``"fdsm_pvalue"`` attribute and projection weights.

    Raises
    ------
    NetworkXError
        If *B* is not bipartite or *agent_nodes* is not one side of it.
    NetworkXNotImplemented
        If *B* is directed or a multigraph.
    ValueError
        If ``trials < 1`` or *burn_in* / *thin* is negative.

    See Also
    --------
    fastball, sdsm, backbone_from_projection

    References
    ----------
    .. [1] Neal, Z. P., Domagalski, R., & Sagan, B. (2021). Comparing
       alternatives to the fixed degree sequence model. *Scientific
       Reports*, 11, 23929.
    .. [2] Godard, K., & Neal, Z. P. (2022). fastball: A fast algorithm to
       sample bipartite graphs with fixed degree sequences. *Journal of
       Complex Networks*, 10(6), cnac049.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fdsm
    >>> B = nx.davis_southern_women_graph()
    >>> women = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    >>> scored = fdsm(B, agent_nodes=women, trials=100, seed=42)
    >>> from networkx_backbone import threshold_filter
    >>> backbone = threshold_filter(scored, "fdsm_pvalue", 0.05, mode="below")
    >>> all("fdsm_pvalue" in data for _, _, data in backbone.edges(data=True))
    True
    """
    import numpy as np

    if trials < 1:
        raise ValueError("trials must be >= 1")

    agent_nodes = _validate_bipartite(B, agent_nodes)
    agents, _, R, observed = _bipartite_projection_matrix(B, agent_nodes)

    na = len(agents)
    rng = _as_generator(seed)
    iu, ju = np.triu_indices(na, 1)
    obs = observed[iu, ju]

    # Count how often the random co-occurrence is >= observed (upper tail) and
    # <= observed (lower tail, for signed backbones).
    exceed_count = np.zeros(len(iu), dtype=np.int64)
    below_count = np.zeros(len(iu), dtype=np.int64)
    for R_rand in _fdsm_samples(R, trials, rng, burn_in=burn_in, thin=thin):
        co_rand = (R_rand @ R_rand.T)[iu, ju]
        exceed_count += co_rand >= obs
        if signed:
            below_count += co_rand <= obs

    p_hi = (exceed_count + 1) / (trials + 1)
    p_lo = (below_count + 1) / (trials + 1)
    backbone = _pair_graph(agents, p_hi, p_lo, "fdsm_pvalue", signed)

    return _apply_projection_weights(
        backbone,
        B,
        agent_nodes,
        projection=projection,
        projection_weight=projection_weight,
        projection_directed=projection_directed,
        projection_max_iter=projection_max_iter,
        projection_tol=projection_tol,
    )


_COMPLEXITY = {
    "simple_projection": {
        "time": "O(n_a^2 n_f + e_b)",
        "space": "O(n_a n_f + n_a^2)",
        "notes": "n_a=agents, n_f=artifacts, e_b=bipartite edges.",
    },
    "hyper_projection": {
        "time": "O(n_a^2 n_f + e_b)",
        "space": "O(n_a n_f + n_a^2)",
    },
    "probs_projection": {
        "time": "O(n_a^2 n_f + e_b)",
        "space": "O(n_a n_f + n_a^2)",
    },
    "ycn_projection": {
        "time": "O(n_a^2 n_f + I n_a^2 + e_b)",
        "space": "O(n_a n_f + n_a^2)",
        "notes": "I=max_iter for stationary-flow convergence.",
    },
    "bipartite_projection": {
        "time": "O(n_a^2 n_f + I n_a^2 + e_b)",
        "space": "O(n_a n_f + n_a^2)",
        "notes": "Worst-case over supported projection methods.",
    },
    "sdsm": {
        "time": "O(I n_a n_f + n_a^2 n_f^2 + e_b)",
        "space": "O(n_a n_f + n_a^2)",
        "notes": (
            "I=BiCM fixed-point iterations; the n_a^2 n_f^2 term is the exact "
            "Poisson-binomial path (RNA path: n_a^2 n_f)."
        ),
    },
    "fdsm": {
        "time": "O(T (S d + n_a^2 n_f) + e_b)",
        "space": "O(n_a n_f + n_a^2)",
        "notes": "T=trials, S=curveball trades per sample, d=max agent degree.",
    },
    "fixedfill": {
        "time": "O(n_a^2 + n_f^2 + n_a^2 n_f + e_b)",
        "space": "O(n_a n_f + n_a^2)",
    },
    "fixedrow": {
        "time": "O(n_a^2 n_f + e_b)",
        "space": "O(n_a n_f + n_a^2)",
    },
    "fixedcol": {
        "time": "O(n_f^2 + n_a^2 n_f + e_b)",
        "space": "O(n_a n_f + n_a^2)",
    },
    "bicm": {
        "time": "O(I n_a n_f + e_b)",
        "space": "O(n_a n_f)",
        "notes": "I=fixed-point iterations.",
    },
    "fastball": {
        "time": "O(r c + S c)",
        "space": "O(r c)",
        "notes": "r=rows, c=cols, S=n_swaps.",
    },
    "backbone_from_projection": {
        "time": "O(T n_a^2 n_f + e_b)",
        "space": "O(n_a n_f + n_a^2)",
        "notes": "Worst-case over {sdsm, fdsm, fixedfill, fixedrow, fixedcol}.",
    },
    "backbone_from_weighted": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Worst-case over {disparity, mlf, lans, global_threshold}.",
    },
    "backbone_from_unweighted": {
        "time": "O(mn)",
        "space": "O(n + m)",
        "notes": "Worst-case over {sparsify, lspar, local_degree}.",
    },
    "backbone": {
        "time": "O(n + m) dispatch checks + selected method cost",
        "space": "O(1) additional beyond selected method",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
