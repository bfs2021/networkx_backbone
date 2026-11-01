"""
Statistical backbone extraction methods.

These methods evaluate edge significance using hypothesis testing against
a null model. Each method computes a p-value (or score) for every edge,
which can then be filtered using functions in :mod:`networkx_backbone.filters`.

Methods
-------
disparity_filter
    Serrano et al. (2009) -- uniform distribution null model.
noise_corrected_filter
    Coscia & Neffke (2017) -- Bayesian binomial framework.
marginal_likelihood_filter
    Dianati (2016) -- binomial null considering both endpoints.
ecm_filter
    Gemmetto et al. (2017) -- enhanced configuration model.
lans_filter
    Foti et al. (2011) -- nonparametric, empirical CDF-based.
multiple_linkage_analysis
    Local linkage significance backbone extraction.
"""

import math

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "disparity_filter",
    "noise_corrected_filter",
    "marginal_likelihood_filter",
    "ecm_filter",
    "lans_filter",
    "multiple_linkage_analysis",
    # Short alias names
    "disparity",
    "mlf",
    "lans",
]


# =====================================================================
# Helper: validate positive weights
# =====================================================================


def _validate_weights(G, weight):
    """Check that all edges have a positive weight attribute."""
    for u, v, data in G.edges(data=True):
        w = data.get(weight)
        if w is None or w <= 0:
            raise nx.NetworkXError(
                f"Edge ({u}, {v}) has non-positive or missing weight."
            )


def _combine_tails(p_hi, p_lo):
    """Combine upper/lower-tail p-values into a two-sided p-value and a sign.

    Returns ``(two_sided_pvalue, sign)`` where ``sign`` is ``+1`` for a
    significantly strong (over-represented) edge and ``-1`` for a significantly
    weak (under-represented) edge.  Used by the signed backbone variants.
    """
    p_hi = min(max(p_hi, 0.0), 1.0)
    p_lo = min(max(p_lo, 0.0), 1.0)
    pval = min(1.0, 2.0 * min(p_hi, p_lo))
    sign = 1 if p_hi <= p_lo else -1
    return pval, sign


# =====================================================================
# 1. Disparity filter -- Serrano et al. (2009)
# =====================================================================


def disparity_filter(G, weight="weight", signed=False):
    r"""Compute disparity filter p-values for each edge.

    The disparity filter [1]_ tests whether an edge's weight is
    statistically significant compared to a null model where each node's
    total strength is uniformly distributed across its edges.

    For each node *u* with degree *k_u* and strength *s_u*, the normalised
    weight of edge (u, v) is ``p_uv = w_uv / s_u``.  Under the null the
    probability of observing a normalised weight >= p_uv is::

        alpha_uv = (1.0 - p_uv) ** (k_u - 1)

    For an undirected edge the p-value is the *minimum* of the values
    computed from each endpoint.  For a directed edge the p-value is
    computed from the source node only (out-strength, out-degree).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test: the stored p-value becomes two-sided
        and a ``"sign"`` edge attribute marks significantly strong (``+1``)
        versus significantly weak (``-1``) edges.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"disparity_pvalue"`` added as an
        edge attribute (and ``"sign"`` when ``signed=True``).

    Raises
    ------
    NetworkXError
        If any edge has a non-positive or missing weight.

    References
    ----------
    .. [1] Serrano, M. A., Boguna, M., & Vespignani, A. (2009).
       Extracting the multiscale backbone of complex weighted networks.
       *PNAS*, 106(16), 6483-6488.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import disparity_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = disparity_filter(G)
    >>> backbone = threshold_filter(H, "disparity_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    _validate_weights(H, weight)

    if G.is_directed():
        strength = dict(G.out_degree(weight=weight))
        degree = dict(G.out_degree())
    else:
        strength = dict(G.degree(weight=weight))
        degree = dict(G.degree())

    for u, v, data in H.edges(data=True):
        w = data[weight]
        if G.is_directed():
            p_hi, p_lo = _disparity_node_tails(w, strength[u], degree[u])
        else:
            hi_u, lo_u = _disparity_node_tails(w, strength[u], degree[u])
            hi_v, lo_v = _disparity_node_tails(w, strength[v], degree[v])
            # OR rule: strong/weak if significant from either endpoint.
            p_hi = min(hi_u, hi_v)
            p_lo = min(lo_u, lo_v)
        if signed:
            data["disparity_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["disparity_pvalue"] = p_hi

    return H


def _disparity_node_tails(w, s, k):
    """Upper- and lower-tail disparity p-values from one node's perspective."""
    if k <= 1:
        return 1.0, 1.0
    p = min(w / s, 1.0)
    try:
        p_hi = (1.0 - p) ** (k - 1)
    except (OverflowError, ValueError):
        p_hi = 0.0
    p_hi = max(p_hi, 0.0)
    return p_hi, max(1.0 - p_hi, 0.0)


# =====================================================================
# 2. Noise-corrected filter -- Coscia & Neffke (2017)
# =====================================================================


def noise_corrected_filter(G, weight="weight"):
    r"""Compute noise-corrected edge significance scores.

    Uses a Bayesian framework where each edge weight is modelled as the
    outcome of a binomial process [1]_.  The expected weight of edge
    (u, v) given a total network weight *W* is::

        E[w_uv] = (s_u * s_v) / W

    The score measures how many standard deviations the observed weight
    lies above the expectation (a z-score).  Higher scores indicate
    more significant edges.

    For directed graphs, out-strength of *u* and in-strength of *v*
    are used.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"nc_score"`` edge attribute
        (z-score; higher means more significant).

    Raises
    ------
    NetworkXError
        If any edge has a non-positive or missing weight.

    References
    ----------
    .. [1] Coscia, M. & Neffke, F. M. (2017). Network backboning with
       noisy data. *Proc. IEEE ICDE*, 425-436.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import noise_corrected_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = noise_corrected_filter(G)
    >>> backbone = threshold_filter(H, "nc_score", 2.0, mode="above")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    _validate_weights(H, weight)

    W = sum(d[weight] for _, _, d in G.edges(data=True))
    if W == 0:
        return H

    if G.is_directed():
        s_out = dict(G.out_degree(weight=weight))
        s_in = dict(G.in_degree(weight=weight))
    else:
        s_out = dict(G.degree(weight=weight))
        s_in = s_out

    for u, v, data in H.edges(data=True):
        w = data[weight]
        su = s_out[u]
        sv = s_in[v]

        p_ij = (su * sv) / (W * W)
        n = W
        expected = n * p_ij
        variance = n * p_ij * (1 - p_ij)

        if variance > 0:
            z = (w - expected) / math.sqrt(variance)
        else:
            z = 0.0

        data["nc_score"] = z

    return H


# =====================================================================
# 3. Marginal likelihood filter -- Dianati (2016)
# =====================================================================


def marginal_likelihood_filter(G, weight="weight", signed=False):
    r"""Compute marginal likelihood p-values for each edge.

    The marginal likelihood filter [1]_ considers edge weights as
    integer counts under a binomial null model that accounts for
    both endpoints' strengths.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.  Integer weights are recommended.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` edge attribute
        (``+1`` significantly strong, ``-1`` significantly weak).

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"ml_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive or missing weight.

    References
    ----------
    .. [1] Dianati, N. (2016). Unwinding the hairball graph: Pruning
       algorithms for weighted complex networks. *Physical Review E*,
       93, 012304.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import marginal_likelihood_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = marginal_likelihood_filter(G)
    >>> backbone = threshold_filter(H, "ml_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    from scipy import stats as sp_stats

    H = G.copy()
    _validate_weights(H, weight)

    W = sum(d[weight] for _, _, d in G.edges(data=True))
    if W == 0:
        return H

    if G.is_directed():
        s_out = dict(G.out_degree(weight=weight))
        s_in = dict(G.in_degree(weight=weight))
    else:
        s_out = dict(G.degree(weight=weight))
        s_in = s_out

    for u, v, data in H.edges(data=True):
        w = data[weight]
        su = s_out[u]
        sv = s_in[v]

        n_param = int(round(su))
        denom = W - su
        if denom > 0 and n_param > 0:
            p_param = min(sv / denom, 1.0)
            k = int(round(w))
            p_hi = sp_stats.binom.sf(k - 1, n_param, p_param)
            p_lo = sp_stats.binom.cdf(k, n_param, p_param)
        else:
            p_hi, p_lo = 1.0, 1.0

        if signed:
            data["ml_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["ml_pvalue"] = float(p_hi)

    return H


# =====================================================================
# 4. ECM filter -- Gemmetto et al. (2017)
# =====================================================================


def ecm_filter(G, weight="weight", max_iter=1000, tol=1e-6):
    r"""Compute ECM (Enhanced Configuration Model) p-values.

    The ECM [1]_ is a maximum-entropy null model for weighted networks
    that preserves the expected degree *and* strength sequence.  Lagrange
    multipliers are assigned to each node and solved iteratively.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    max_iter : int, optional (default=1000)
        Maximum number of iterations for the fixed-point solver.
    tol : float, optional (default=1e-6)
        Convergence tolerance for the Lagrange multipliers.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"ecm_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive or missing weight.

    References
    ----------
    .. [1] Gemmetto, V., Cardillo, A., & Garlaschelli, D. (2017).
       Irreducible network backbones: unbiased graph filtering via
       maximum entropy. arXiv:1706.00230.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import ecm_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = ecm_filter(G)
    >>> backbone = threshold_filter(H, "ecm_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    import numpy as np

    H = G.copy()
    _validate_weights(H, weight)

    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return H

    node_idx = {v: i for i, v in enumerate(nodes)}

    if G.is_directed():
        deg = np.array([G.out_degree(v) for v in nodes], dtype=float)
        strength = np.array(
            [G.out_degree(v, weight=weight) for v in nodes], dtype=float
        )
    else:
        deg = np.array([G.degree(v) for v in nodes], dtype=float)
        strength = np.array([G.degree(v, weight=weight) for v in nodes], dtype=float)

    # Initialise Lagrange multipliers
    deg_sum = deg.sum()
    str_sum = strength.sum()
    x = np.where(deg > 0, deg / max(deg_sum, 1), 1e-10)
    y = np.where(strength > 0, strength / max(str_sum, 1), 1e-10)
    y = np.clip(y, 1e-15, 1.0 - 1e-10)

    # Fixed-point iterations
    for _ in range(max_iter):
        x_new = np.zeros(n)
        y_new = np.zeros(n)

        for i in range(n):
            sum_x = 0.0
            sum_y = 0.0
            for j in range(n):
                if i == j:
                    continue
                yy = y[i] * y[j]
                if yy >= 1.0:
                    yy = 1.0 - 1e-15
                denom = 1.0 - yy
                xy = x[i] * x[j]
                p_ij = (xy * yy) / (denom + xy * yy)
                if denom > 0:
                    w_ij = (xy * yy) / (denom * denom + xy * yy * denom)
                else:
                    w_ij = 0
                sum_x += p_ij
                sum_y += w_ij

            x_new[i] = deg[i] / sum_x if sum_x > 0 else 1e-10
            y_new[i] = strength[i] / sum_y if sum_y > 0 else 1e-10

        y_new = np.clip(y_new, 1e-15, 1.0 - 1e-10)

        if np.max(np.abs(x_new - x)) < tol and np.max(np.abs(y_new - y)) < tol:
            x, y = x_new, y_new
            break
        x, y = x_new, y_new

    # Compute p-values using geometric distribution
    for u, v, data in H.edges(data=True):
        w = data[weight]
        i, j = node_idx[u], node_idx[v]
        yy = y[i] * y[j]
        if yy >= 1.0:
            yy = 1.0 - 1e-15

        q = yy
        if 0 < q < 1:
            pval = q ** int(round(w))
        elif q >= 1:
            pval = 0.0
        else:
            pval = 1.0

        data["ecm_pvalue"] = float(max(min(pval, 1.0), 0.0))

    return H


# =====================================================================
# 5. LANS -- Foti et al. (2011)
# =====================================================================


def lans_filter(G, weight="weight", signed=False):
    r"""Compute LANS (Locally Adaptive Network Sparsification) p-values.

    LANS [1]_ is a nonparametric method that makes no distributional
    assumptions.  For each edge (u, v), it computes the fraction of
    edges at node *u* (and node *v*) that have weight <= w_uv --- the
    empirical CDF value.  The edge p-value is one minus the maximum
    of the two eCDF values (so lower p means more significant).

    For directed graphs, the source node's out-edges and the target node's
    in-edges are both used, and the edge p-value is computed from the more
    significant of the two sides.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    signed : bool, optional (default=False)
        If ``True``, run a two-tailed test and add a ``"sign"`` edge attribute
        (``+1`` significantly strong, ``-1`` significantly weak).

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"lans_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive or missing weight.

    References
    ----------
    .. [1] Foti, N. J., Hughes, J. M., & Rockmore, D. N. (2011).
       Nonparametric sparsification of complex multiscale networks.
       *PLoS ONE*, 6(2), e16431.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import lans_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = lans_filter(G)
    >>> backbone = threshold_filter(H, "lans_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    _validate_weights(H, weight)

    # Pre-compute sorted weight lists per node
    if G.is_directed():
        out_weights = {}
        in_weights = {}
        for u in G.nodes():
            out_weights[u] = sorted(d[weight] for _, _, d in G.out_edges(u, data=True))
            in_weights[u] = sorted(d[weight] for _, _, d in G.in_edges(u, data=True))
    else:
        node_weights = {}
        for u in G.nodes():
            ws = sorted(G[u][v][weight] for v in G[u])
            node_weights[u] = ws

    for u, v, data in H.edges(data=True):
        w = data[weight]

        if G.is_directed():
            ecdf_u = _empirical_cdf(w, out_weights[u])
            ecdf_v = _empirical_cdf(w, in_weights[v])
            # OR rule: strong/weak if significant from either side of the edge.
            p_hi = 1.0 - max(ecdf_u, ecdf_v)
            p_lo = min(ecdf_u, ecdf_v)
        else:
            ecdf_u = _empirical_cdf(w, node_weights[u])
            ecdf_v = _empirical_cdf(w, node_weights[v])
            # OR rule: strong/weak if significant from either endpoint.
            p_hi = 1.0 - max(ecdf_u, ecdf_v)
            p_lo = min(ecdf_u, ecdf_v)

        if signed:
            data["lans_pvalue"], data["sign"] = _combine_tails(p_hi, p_lo)
        else:
            data["lans_pvalue"] = max(p_hi, 0.0)

    return H


def multiple_linkage_analysis(G, alpha=0.05, weight="weight"):
    r"""Extract a backbone using Multiple Linkage Analysis (MLA).

    MLA is a local-significance method that selects edges whose weights are
    unusually high relative to neighboring edges. This implementation uses
    edge-level empirical CDF p-values from :func:`lans_filter` and retains
    edges with p-value <= ``alpha``.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.
    alpha : float, optional (default=0.05)
        Significance threshold in [0, 1].
    weight : string, optional (default="weight")
        Edge attribute key for weights. All weights must be positive.

    Returns
    -------
    H : graph
        A copy of *G* with ``"lans_pvalue"``, ``"mla_pvalue"``, and boolean
        ``"mla_keep"`` edge attributes.

    Raises
    ------
    ValueError
        If ``alpha`` is outside [0, 1].
    NetworkXError
        If any edge has a non-positive or missing weight.

    References
    ----------
    .. [1] Van Nuffel, N., Heyndrickx, C., & Wets, G. (2010). Measuring
       hierarchy and reciprocity in networks.
    .. [2] Yassin, A., Haidar, A., Cherifi, H., Seba, H., & Togni, O. (2023).
       An evaluation tool for backbone extraction techniques in weighted
       complex networks. *Scientific Reports*, 13, 17000.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, multiple_linkage_analysis
    >>> G = nx.les_miserables_graph()
    >>> H = multiple_linkage_analysis(G, alpha=0.5)
    >>> backbone = boolean_filter(H, "mla_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    scored = lans_filter(G, weight=weight)
    for _, _, data in scored.edges(data=True):
        pvalue = data["lans_pvalue"]
        data["mla_pvalue"] = pvalue
        data["mla_keep"] = pvalue <= alpha
    return scored


def _empirical_cdf(w, sorted_weights):
    """Compute the empirical CDF value F(w) from a sorted list."""
    n = len(sorted_weights)
    if n == 0:
        return 0.0
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_weights[mid] <= w:
            lo = mid + 1
        else:
            hi = mid
    return lo / n


def disparity(G, weight="weight", signed=False):
    """Alias for :func:`disparity_filter`."""
    return disparity_filter(G, weight=weight, signed=signed)


def mlf(G, weight="weight", signed=False):
    """Alias for :func:`marginal_likelihood_filter`."""
    return marginal_likelihood_filter(G, weight=weight, signed=signed)


def lans(G, weight="weight", signed=False):
    """Alias for :func:`lans_filter`."""
    return lans_filter(G, weight=weight, signed=signed)


_COMPLEXITY = {
    "disparity_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "noise_corrected_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "marginal_likelihood_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "ecm_filter": {
        "time": "O(I * n^2 + m)",
        "space": "O(n + m)",
        "notes": "I=max_iter, n=|V|, m=|E|.",
    },
    "lans_filter": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Worst-case over node-local sorted edge-weight lookups.",
    },
    "multiple_linkage_analysis": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Dominated by lans_filter.",
    },
    "disparity": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "Alias for disparity_filter.",
    },
    "mlf": {
        "time": "O(n + m)",
        "space": "O(n + m)",
        "notes": "Alias for marginal_likelihood_filter.",
    },
    "lans": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Alias for lans_filter.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
