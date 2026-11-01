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
import warnings

import networkx as nx
from networkx.utils import not_implemented_for

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
    """Check that all edges have a positive, finite weight attribute."""
    for u, v, data in G.edges(data=True):
        w = data.get(weight)
        # ``not (0 < w < inf)`` also rejects NaN, which fails every comparison.
        if w is None or not (0 < w < math.inf):
            raise nx.NetworkXError(
                f"Edge ({u}, {v}) has a non-positive, non-finite, or missing "
                f"{weight!r} attribute."
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


@not_implemented_for("multigraph")
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
    computed from each endpoint.  For a directed edge (u, v) the test is run
    from the source's outgoing links (out-strength and out-degree of *u*) and
    from the target's incoming links (in-strength and in-degree of *v*), and
    the p-value is the minimum of the two, following [1]_.

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
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    See Also
    --------
    threshold_filter : Keep edges whose p-value is below a threshold.
    marginal_likelihood_filter : Null model that uses both endpoints jointly.

    Notes
    -----
    An endpoint of degree 1 cannot reject the null model, so it contributes a
    p-value of 1 and the edge's p-value comes from the other endpoint.

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
        s_out = dict(G.out_degree(weight=weight))
        k_out = dict(G.out_degree())
        s_in = dict(G.in_degree(weight=weight))
        k_in = dict(G.in_degree())
    else:
        strength = dict(G.degree(weight=weight))
        degree = dict(G.degree())

    for u, v, data in H.edges(data=True):
        w = data[weight]
        if G.is_directed():
            hi_u, lo_u = _disparity_node_tails(w, s_out[u], k_out[u])
            hi_v, lo_v = _disparity_node_tails(w, s_in[v], k_in[v])
            p_hi = min(hi_u, hi_v)
            p_lo = min(lo_u, lo_v)
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


@not_implemented_for("multigraph")
def noise_corrected_filter(G, weight="weight"):
    r"""Compute noise-corrected edge significance scores.

    Models each edge weight as the outcome of a binomial process in which
    the total network weight *W* is distributed according to the endpoint
    strengths, following the null model of Coscia & Neffke [1]_.  The
    expected weight of edge (u, v) is::

        E[w_uv] = (s_u * s_v) / W

    The score measures how many standard deviations the observed weight
    lies above the expectation (a z-score).  Higher scores indicate
    more significant edges.

    .. note::

       This is a simplified, frequentist version of the noise-corrected
       backbone.  It does not implement the Bayesian posterior variance
       (delta-method) estimate of the original paper, so scores differ from
       the reference implementation, especially for low-strength nodes.
       See the project roadmap.

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
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    See Also
    --------
    threshold_filter : Use ``mode="above"`` to keep high scores.

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


@not_implemented_for("multigraph")
def marginal_likelihood_filter(G, weight="weight", signed=False):
    r"""Compute marginal likelihood p-values for each edge.

    The marginal likelihood filter [1]_ treats an integer edge weight as a
    count of unit edges.  Under the null model each of the *T* unit edges
    picks its two endpoints independently with probability proportional to
    their strengths, so the weight of the pair (u, v) is binomial,
    ``B(T, p_uv)``, with

    .. math::

        T = \frac{1}{2}\sum_i s_i, \qquad p_{uv} = \frac{s_u s_v}{2 T^2}

    for undirected graphs, and :math:`T = \sum_i s^{out}_i`,
    :math:`p_{uv} = s^{out}_u s^{in}_v / T^2` for directed graphs.  The
    p-value is the upper tail :math:`P(X \geq w_{uv})`.  This null preserves
    the strength sequence in expectation and is symmetric in *u* and *v*.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.  Weights are rounded to the nearest integer, so the
        method is meant for count data.
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
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    See Also
    --------
    disparity_filter : Tests each endpoint separately.
    ecm_filter : Maximum-entropy null that also fixes degrees.

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

    if G.number_of_edges() == 0:
        return H

    # Strengths of the rounded (integer) weights.
    s_out, s_in = {}, {}
    T = 0
    for u, v, d in G.edges(data=True):
        w = int(round(d[weight]))
        T += w
        s_out[u] = s_out.get(u, 0) + w
        s_in[v] = s_in.get(v, 0) + w
    if not G.is_directed():
        # Each undirected edge adds its weight to both endpoints.
        for node, val in s_in.items():
            s_out[node] = s_out.get(node, 0) + val
        s_in = s_out

    norm = T * T if G.is_directed() else 2 * T * T

    for u, v, data in H.edges(data=True):
        k = int(round(data[weight]))
        if T > 0 and u != v:
            p_param = min(s_out[u] * s_in[v] / norm, 1.0)
            p_hi = float(sp_stats.binom.sf(k - 1, T, p_param))
            p_lo = float(sp_stats.binom.cdf(k, T, p_param))
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


@not_implemented_for("multigraph")
def ecm_filter(G, weight="weight", max_iter=1000, tol=1e-6):
    r"""Compute ECM (Enhanced Configuration Model) p-values.

    The ECM [1]_ is the maximum-entropy ensemble of integer-weighted graphs
    whose expected degree *and* strength sequences equal the observed ones.
    Each node gets two Lagrange multipliers, :math:`x_i > 0` and
    :math:`0 < y_i < 1`.  The probability that *i* and *j* are linked and
    the distribution of the weight of an existing link are

    .. math::

        p_{ij} = \frac{x_i x_j y_i y_j}{1 - y_i y_j + x_i x_j y_i y_j},
        \qquad
        P(w_{ij} \geq w) = p_{ij} (y_i y_j)^{w - 1}, \; w \geq 1,

    and the p-value of an observed link is :math:`P(w_{ij} \geq w)`.  The
    multipliers are found by maximizing the likelihood (L-BFGS-B with the
    analytic gradient), which is equivalent to solving
    :math:`\sum_j p_{ij} = k_i` and :math:`\sum_j \langle w_{ij}\rangle = s_i`.

    For directed graphs, each node has separate out- and in-multipliers that
    reproduce its out- and in-degree and strength.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A NetworkX graph.  Weights are rounded to the nearest integer, so the
        method is meant for count data.
    weight : string, optional (default="weight")
        Edge attribute key for weights.  All weights must be positive.
    max_iter : int, optional (default=1000)
        Maximum number of solver iterations.
    tol : float, optional (default=1e-6)
        Tolerance on the largest relative error of the reproduced degrees and
        strengths.  A :class:`RuntimeWarning` is raised if it is not met.

    Returns
    -------
    H : graph
        A copy of *G* (same type) with ``"ecm_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXError
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    Warns
    -----
    RuntimeWarning
        If the solver does not reproduce the degree and strength sequences to
        within ``tol``.

    See Also
    --------
    marginal_likelihood_filter : Binomial null that fixes only strengths.

    Notes
    -----
    Self-loops are ignored when fitting the null model and get a p-value
    of 1.  The fit costs :math:`O(n^2)` memory and time per iteration.

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
    >>> backbone.number_of_edges() < H.number_of_edges()
    True
    """
    import numpy as np

    H = G.copy()
    _validate_weights(H, weight)

    nodes = list(G)
    n = len(nodes)
    if n == 0 or G.number_of_edges() == 0:
        for _, _, data in H.edges(data=True):
            data["ecm_pvalue"] = 1.0
        return H

    idx = {v: i for i, v in enumerate(nodes)}
    k_out = np.zeros(n)
    k_in = np.zeros(n)
    s_out = np.zeros(n)
    s_in = np.zeros(n)
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = max(int(round(d[weight])), 1)
        i, j = idx[u], idx[v]
        k_out[i] += 1
        s_out[i] += w
        k_in[j] += 1
        s_in[j] += w
    if not G.is_directed():
        k_out = k_in = k_out + k_in
        s_out = s_in = s_out + s_in

    a_out, b_out, a_in, b_in, act_out, act_in, err = _fit_ecm(
        k_out, s_out, k_in, s_in, G.is_directed(), max_iter, tol
    )
    if err > tol:
        warnings.warn(
            f"ecm_filter: the ECM solver did not converge (largest relative "
            f"error {err:.2e} > tol={tol:g}); p-values may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    for u, v, data in H.edges(data=True):
        if u == v:
            data["ecm_pvalue"] = 1.0
            continue
        i, j = idx[u], idx[v]
        # x_i x_j y_i y_j = exp(-(a_i + a_j)); y_i y_j = exp(-(b_i + b_j)).
        # Work in log space: x can be huge when y is tiny.
        log_xxyy = -float(a_out[i] + a_in[j])
        log_yy = -float(b_out[i] + b_in[j])
        log_one_minus_yy = math.log(-math.expm1(log_yy))
        hi = max(log_xxyy, log_one_minus_yy)
        log_denom = hi + math.log(
            math.exp(log_xxyy - hi) + math.exp(log_one_minus_yy - hi)
        )
        w = max(int(round(data[weight])), 1)
        pval = math.exp(log_xxyy - log_denom + (w - 1) * log_yy)
        data["ecm_pvalue"] = float(min(max(pval, 0.0), 1.0))

    return H


# Largest parameter vector for which the dense Newton polish is used
# (2 parameters per node undirected, 4 directed).
_ECM_NEWTON_MAX_PARAMS = 4000


def _fit_ecm(k_out, s_out, k_in, s_in, directed, max_iter, tol):
    """Maximum-likelihood ECM multipliers.

    Works with ``x * y = exp(-c)`` and ``y = exp(-b)`` (``b > 0`` keeps
    ``y < 1``).  Parameterizing by ``x * y`` keeps the problem well
    conditioned for nodes whose links all have weight 1, where ``y -> 0``
    and ``x -> inf`` with finite product.  Nodes with zero degree are pinned
    to ``x = 0``.  Returns
    ``(x_out, y_out, x_in, y_in, max_relative_error)``.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = len(k_out)
    act_out = k_out > 0
    act_in = k_in > 0
    off_diag = ~np.eye(n, dtype=bool)

    def unpack(theta):
        if directed:
            a_out, b_out, a_in, b_in = np.split(theta, 4)
        else:
            a_out, b_out = np.split(theta, 2)
            a_in, b_in = a_out, b_out
        return a_out, b_out, a_in, b_in

    def expectations(theta):
        a_out, b_out, a_in, b_in = unpack(theta)
        # log of x_i x_j y_i y_j and of y_i y_j, for every ordered pair.
        log_yy = -(b_out[:, None] + b_in[None, :])
        log_xxyy = -(a_out[:, None] + a_in[None, :])
        one_minus_yy = -np.expm1(log_yy)
        # q_ij = x_i x_j y_i y_j / (1 - y_i y_j); log Z_ij = log(1 + q_ij)
        log_q = log_xxyy - np.log(one_minus_yy)
        mask = off_diag & act_out[:, None] & act_in[None, :]
        log_q = np.where(mask, log_q, -np.inf)
        log_z = np.logaddexp(0.0, log_q)
        p = np.exp(log_q - log_z)  # link probability
        w_mean = p / one_minus_yy  # expected weight
        return p, w_mean, log_z

    def objective(theta):
        a_out, b_out, a_in, b_in = unpack(theta)
        p, w_mean, log_z = expectations(theta)
        # With x*y = exp(-c), the sufficient statistic of b is s - k.
        if directed:
            nll = (
                a_out @ k_out
                + b_out @ (s_out - k_out)
                + a_in @ k_in
                + b_in @ (s_in - k_in)
                + log_z.sum()
            )
            grad = np.concatenate(
                [
                    k_out - p.sum(1),
                    (s_out - k_out) - (w_mean - p).sum(1),
                    k_in - p.sum(0),
                    (s_in - k_in) - (w_mean - p).sum(0),
                ]
            )
        else:
            # Symmetric matrices count each unordered pair twice.
            nll = a_out @ k_out + b_out @ (s_out - k_out) + 0.5 * log_z.sum()
            grad = np.concatenate(
                [k_out - p.sum(1), (s_out - k_out) - (w_mean - p).sum(1)]
            )
        return nll, grad

    def start(k, s):
        total = max(k.sum(), 1.0)
        # Chung-Lu-like guess for x*y, and mean link weight 1/(1-y) ~ s/k.
        a = -np.log(np.where(k > 0, k / np.sqrt(total), 1.0))
        ratio = np.where(k > 0, s / np.maximum(k, 1.0), 2.0)
        b = -np.log(np.clip(1.0 - 1.0 / np.maximum(ratio, 1.0 + 1e-6), 1e-12, 1.0))
        return a, b

    a0, b0 = start(k_out, s_out)
    if directed:
        a1, b1 = start(k_in, s_in)
        theta0 = np.concatenate([a0, b0, a1, b1])
    else:
        theta0 = np.concatenate([a0, b0])
    b_floor = 1e-10
    bounds = []
    blocks = 2 if directed else 1
    for _ in range(blocks):
        bounds += [(None, None)] * n + [(b_floor, None)] * n

    def rel_err(obs, fit):
        ok = obs > 0
        return float(np.max(np.abs(fit[ok] - obs[ok]) / obs[ok])) if ok.any() else 0.0

    def max_error(theta):
        p, w_mean, _ = expectations(theta)
        if directed:
            return max(
                rel_err(k_out, p.sum(1)),
                rel_err(s_out, w_mean.sum(1)),
                rel_err(k_in, p.sum(0)),
                rel_err(s_in, w_mean.sum(0)),
            )
        return max(rel_err(k_out, p.sum(1)), rel_err(s_out, w_mean.sum(1)))

    def hessian(theta):
        p, w_mean, _ = expectations(theta)
        m = w_mean - p
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(p > 0, m / np.where(p > 0, p, 1.0), 0.0)
        A = p * (1.0 - p)  # -dp/dt
        B = A * g  # -dp/dr  (= -dm/dt)
        C = A * g * g + p * g * (1.0 + g)  # -dm/dr
        if directed:
            Hm = np.zeros((4 * n, 4 * n))
            # parameter order: c_out, b_out, c_in, b_in
            pairs = {(0, 0): A, (0, 1): B, (1, 1): C}
            for (r, c), M in pairs.items():
                Hm[r * n : (r + 1) * n, c * n : (c + 1) * n] += np.diag(M.sum(1))
                Hm[(r + 2) * n : (r + 3) * n, (c + 2) * n : (c + 3) * n] += np.diag(
                    M.sum(0)
                )
                Hm[r * n : (r + 1) * n, (c + 2) * n : (c + 3) * n] += M
                Hm[(c + 2) * n : (c + 3) * n, r * n : (r + 1) * n] += M.T
                if r != c:
                    Hm[c * n : (c + 1) * n, r * n : (r + 1) * n] += np.diag(M.sum(1))
                    Hm[(c + 2) * n : (c + 3) * n, (r + 2) * n : (r + 3) * n] += np.diag(
                        M.sum(0)
                    )
                    Hm[c * n : (c + 1) * n, (r + 2) * n : (r + 3) * n] += M
                    Hm[(r + 2) * n : (r + 3) * n, c * n : (c + 1) * n] += M.T
            return Hm
        # Undirected: symmetric pair matrices, each unordered pair once per row.
        Hcc = np.diag(A.sum(1)) + A
        Hcb = np.diag(B.sum(1)) + B
        Hbb = np.diag(C.sum(1)) + C
        return np.block([[Hcc, Hcb], [Hcb.T, Hbb]])

    lower = np.array([lo if lo is not None else -np.inf for lo, _ in bounds])

    theta = theta0
    use_newton = len(theta0) <= _ECM_NEWTON_MAX_PARAMS
    # With a Newton polish available, L-BFGS-B only needs to get close.
    iters_left = min(max_iter, 200) if use_newton else max_iter
    # Restarting L-BFGS-B resets its curvature estimate, which helps it
    # finish ill-conditioned problems; a few restarts are plenty.
    for _ in range(1 if use_newton else 3):
        res = minimize(
            objective,
            theta,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": iters_left,
                "maxfun": 4 * max_iter,
                "gtol": 1e-12,
                "ftol": 1e-15,
            },
        )
        theta = res.x
        iters_left -= res.nit
        if max_error(theta) <= tol or iters_left <= 0:
            break

    # Damped Newton polish with the analytic Hessian.  The negative
    # log-likelihood is convex, so this converges quadratically; it needs a
    # dense solve, so it is only used for moderately sized graphs.
    if use_newton and max_error(theta) > tol:
        f, grad = objective(theta)
        for _ in range(min(100, max_iter)):
            Hm = hessian(theta)
            # The directed model has a gauge freedom (shift out-parameters
            # up and in-parameters down), so the Hessian is singular; a small
            # scaled ridge keeps the step in the identifiable subspace.
            Hm[np.diag_indices_from(Hm)] += 1e-9 * max(np.abs(Hm).max(), 1.0)
            try:
                step = np.linalg.solve(Hm, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(Hm, grad, rcond=None)[0]
            t = 1.0
            while t > 1e-8:
                cand = np.maximum(theta - t * step, lower)
                f_new, grad_new = objective(cand)
                if f_new <= f + 1e-4 * t * (grad @ (cand - theta)) or f_new < f:
                    break
                t *= 0.5
            else:
                break
            theta, f, grad = cand, f_new, grad_new
            if max_error(theta) <= tol:
                break

    err = max_error(theta)

    a_out, b_out, a_in, b_in = unpack(theta)
    return a_out, b_out, a_in, b_in, act_out, act_in, err


# =====================================================================
# 5. LANS -- Foti et al. (2011)
# =====================================================================


@not_implemented_for("multigraph")
def lans_filter(G, weight="weight", signed=False):
    r"""Compute LANS (Locally Adaptive Network Sparsification) p-values.

    LANS [1]_ is a nonparametric method that makes no distributional
    assumptions.  For each edge (u, v), it computes the fraction of
    edges at node *u* (and node *v*) that have weight <= w_uv --- the
    empirical CDF value.  The edge p-value is one minus the maximum
    of the two eCDF values (so lower p means more significant).

    For directed graphs, only the source node's out-edges are used.

    The p-value is ``1 - F(w)``, where ``F`` is the empirical CDF of the
    node's weights.  The strongest link of each node therefore gets a p-value
    of 0, and so does the only link of a degree-1 node.  LANS always keeps
    these links; this is how the method guarantees that no node becomes
    isolated.

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
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    See Also
    --------
    disparity_filter : Parametric counterpart with a uniform null model.

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
        node_weights = {}
        for u in G.nodes():
            ws = sorted(d[weight] for _, _, d in G.out_edges(u, data=True))
            node_weights[u] = ws
    else:
        node_weights = {}
        for u in G.nodes():
            ws = sorted(G[u][v][weight] for v in G[u])
            node_weights[u] = ws

    for u, v, data in H.edges(data=True):
        w = data[weight]

        if G.is_directed():
            ecdf_u = _empirical_cdf(w, node_weights[u])
            p_hi = 1.0 - ecdf_u
            p_lo = ecdf_u
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


@not_implemented_for("multigraph")
def multiple_linkage_analysis(G, alpha=0.05, weight="weight"):
    r"""Extract a backbone using Multiple Linkage Analysis (MLA).

    Selects edges whose weights are unusually high relative to the other
    edges of their endpoints, using the edge-level empirical-CDF p-values of
    :func:`lans_filter` and keeping edges with p-value <= ``alpha``.  This
    follows the netbone [2]_ implementation.

    .. note::

       The original multiple linkage analysis of Van Nuffel et al. [1]_
       instead compares each node's sorted link weights with a sequence of
       ideal "dominance" distributions and keeps the links of the best fit.
       That procedure is not implemented here yet; see the project roadmap.

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
        If any edge has a non-positive, non-finite, or missing weight.
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Van Nuffel, N., Saey, P., Derudder, B., Devriendt, L., & Witlox, F.
       (2010). Measuring hierarchical differentiation: connectivity and
       dominance in the European urban network. *Transportation Planning and
       Technology*, 33(4), 343-366. https://doi.org/10.1080/03081060.2010.494028
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
        "space": "O(n^2 + m)",
        "notes": "I=solver iterations; a dense O(n^3) Newton polish is used for n <= 2000.",
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
