"""Bayesian inference over the discovered simulator structure.

Win probability at tick t for machine i:
    logit = base[type] + gamma*signal + kappa*(M[t-lag] - m0) + sign(t)*w*e(t)
    e(t)  = +1 if win[t-a] != win[t-b] else -1, with (a, b) a machine-specific lag pair
    M[t]  = mean win flag across the episode's machines at tick t
sign(t) is constant ("stable"), or flips once: at one tick shared by the episode
("shared"), or at machine-specific ticks around an episode centre ("drift").

Payout sizes given a win come from one of two tables per machine type. The table is
chosen by (payout[t-L] >= threshold) XOR d. L is tied to the win pair through a learned
distribution; d is machine-specific. Size and win evidence are inferred jointly.

Only history strictly before the cutoff is read, and a tick whose lags are unobserved
contributes no evidence. Condition labels and true probabilities are never inputs to
inference; conditions are inferred from the histories.
"""
import json
import math
from pathlib import Path

import numpy as np

from slot_forecasting.schema import PAYOUTS
from .features import HistoryIndex

SIZES = PAYOUTS[1:]


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def sigmoid(x):
    return np.exp(log_sigmoid(x))


def logsum(values, axis=None):
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


DEFAULT_PARAMS = {
    "pairs": [[1, 3], [2, 5], [4, 7]],
    "pair_prior": [1 / 3, 1 / 3, 1 / 3],
    "size_lags": [1, 2, 4],
    "size_given_pair": [[0.96, 0.02, 0.02], [0.02, 0.96, 0.02], [0.02, 0.02, 0.96]],
    "size_direction_prior": [0.5, 0.5],
    "size_threshold": 2,
    "base": {"steady": 0.5, "balanced": -0.58, "jackpot": -1.78},
    "w": 2.3,
    "gamma": 0.3,
    "kappa": 0.8,
    "kappa_lag": 3,
    "m0": 0.45,
    "condition_prior": {"stable": 0.5, "shared": 0.25, "drift": 0.25},
    "tau_low": 0.45,
    "tau_high": 0.9,
    "drift_sd": 15.0,
    "tables": {
        "balanced": [[0.21406615, 0.54673659, 0.20502622, 0.03417104],
                     [0.49985311, 0.34792827, 0.1304731, 0.02174552]],
        "jackpot": [[0.12078673, 0.32392068, 0.41646944, 0.13882315],
                    [0.33514603, 0.2449462, 0.31493083, 0.10497694]],
        "steady": [[0.45512201, 0.43590239, 0.10679609, 0.00217951],
                   [0.75398939, 0.19680849, 0.04821808, 0.00098404]],
    },
}


class Episode:
    """Observed outcomes for one episode as a machines-by-ticks matrix; -1 means missing."""

    def __init__(self, queries, index):
        self.queries = sorted(queries, key=lambda q: q.machine_id)
        cutoffs = {q.cutoff for q in self.queries}
        if len(cutoffs) != 1:
            raise ValueError("An episode must share one cutoff")
        self.cutoff = cutoffs.pop()
        self.Y = np.full((len(self.queries), self.cutoff), -1, dtype=np.int64)
        for i, q in enumerate(self.queries):
            for tick, payout in index.before(q.episode_id, q.machine_id, q.cutoff):
                self.Y[i, tick] = payout
        self.observed = self.Y >= 0
        self.win = (self.Y > 0).astype(np.int8)
        count = self.observed.sum(axis=0)
        wins = (self.win * self.observed).sum(axis=0)
        self.M = np.divide(wins, count, out=np.full(self.cutoff, np.nan), where=count > 0)

    def mean_at(self, tick, default):
        if tick < 0 or tick >= self.cutoff or math.isnan(self.M[tick]):
            return default
        return float(self.M[tick])

    def mean_series(self, lag, default):
        """M[t-lag] for every t in 0..cutoff-1, with default where unavailable."""
        out = np.full(self.cutoff, float(default))
        known = self.M[:self.cutoff - lag]
        out[lag:] = np.where(np.isnan(known), default, known)
        return out


def machine_offsets(params, query):
    return params["base"][query.machine_type] + params["gamma"] * query.signal


def pair_terms(episode, i, pair):
    """Return availability mask and e(t) for all ticks, plus e at the cutoff (0 if unknown)."""
    a, b = pair
    T = episode.cutoff
    ok = np.zeros(T, dtype=bool)
    e = np.zeros(T)
    start = max(a, b)
    if T > start:
        y = episode.observed[i]
        ok[start:] = y[start:] & y[start - a:T - a] & y[start - b:T - b]
        parity = episode.win[i, start - a:T - a] ^ episode.win[i, start - b:T - b]
        e[start:] = 2.0 * parity - 1.0
    known = T >= max(a, b) and episode.observed[i, T - a] and episode.observed[i, T - b]
    final = 0.0
    if known:
        final = 2.0 * int(episode.win[i, T - a] ^ episode.win[i, T - b]) - 1.0
    return ok, e, final


def infer_win(episode, params, size_score=None):
    """Exact posterior over pair, final sign, flip tick and condition for one episode."""
    T = episode.cutoff
    if size_score is None:
        size_score = np.zeros((len(episode.queries), len(params["pairs"])))
    pairs = [tuple(p) for p in params["pairs"]]
    log_pair = np.log(np.asarray(params["pair_prior"], dtype=float))
    n, k = len(episode.queries), len(pairs)
    w, kappa, m0 = params["w"], params["kappa"], params["m0"]
    lag = params["kappa_lag"]
    shared_level = kappa * (episode.mean_series(lag, m0) - m0)
    final_level = kappa * (episode.mean_at(T - lag, m0) - m0)

    low, high = int(params["tau_low"] * T), int(params["tau_high"] * T)
    taus = np.arange(low, max(high, low + 1))
    delta = (taus[:, None] - taus[None, :]) / params["drift_sd"]
    log_kernel = -0.5 * delta ** 2
    log_kernel -= logsum(log_kernel, axis=1)[:, None]

    stable = np.zeros((n, k, 2))
    flip = np.zeros((n, k, 2, len(taus)))
    finals = np.zeros((n, k, 2))
    win = episode.win.astype(float)
    for i, query in enumerate(episode.queries):
        base = machine_offsets(params, query)
        for j, pair in enumerate(pairs):
            ok, e, final = pair_terms(episode, i, pair)
            per_sign = []
            for sign_index, sign in enumerate((1.0, -1.0)):
                eta = base + shared_level + sign * w * e
                ll = win[i] * log_sigmoid(eta) + (1 - win[i]) * log_sigmoid(-eta)
                per_sign.append(np.concatenate(([0.0], np.cumsum(np.where(ok, ll, 0.0)))))
                eta_final = base + final_level + sign * w * final
                finals[i, j, sign_index] = float(sigmoid(np.asarray(eta_final)))
            for sign_index in (0, 1):
                keep, before = per_sign[sign_index], per_sign[1 - sign_index]
                stable[i, j, sign_index] = keep[T]
                flip[i, j, sign_index] = before[taus] + keep[T] - keep[taus]

    log_machine_pair = log_pair[None, :] + size_score
    prior_stable = stable + log_machine_pair[:, :, None] + math.log(0.5)
    prior_flip = flip + log_machine_pair[:, :, None, None] + math.log(0.5)
    z_stable = logsum(prior_stable.reshape(n, -1), axis=1)

    z_shared = logsum(prior_flip.reshape(n, -1, len(taus)), axis=1)
    tau_shared = z_shared.sum(axis=0)
    log_shared = logsum(tau_shared) - math.log(len(taus))

    smooth = np.zeros_like(prior_flip)
    for i in range(n):
        for j in range(k):
            for s in (0, 1):
                smooth[i, j, s] = logsum(prior_flip[i, j, s][None, :] + log_kernel, axis=1)
    z_drift = logsum(smooth.reshape(n, -1, len(taus)), axis=1)
    tau_drift = z_drift.sum(axis=0)
    log_drift = logsum(tau_drift) - math.log(len(taus))

    cond = params["condition_prior"]
    log_cond = np.array([math.log(cond["stable"]) + z_stable.sum(),
                         math.log(cond["shared"]) + log_shared,
                         math.log(cond["drift"]) + log_drift])
    p_cond = np.exp(log_cond - logsum(log_cond))

    def tilt(table, totals, posterior):
        """Mix machine posteriors over the episode-level flip-time posterior."""
        weights = np.exp(table - totals[:, None, None, :]).transpose(0, 3, 1, 2)
        return np.einsum("t,itjs->ijs", posterior, weights)

    post_stable = np.exp(prior_stable - z_stable[:, None, None])
    post_shared = tilt(prior_flip, z_shared, np.exp(tau_shared - logsum(tau_shared)))
    post_drift = tilt(smooth, z_drift, np.exp(tau_drift - logsum(tau_drift)))
    posterior = p_cond[0] * post_stable + p_cond[1] * post_shared + p_cond[2] * post_drift
    p_win = (posterior * finals).sum(axis=(1, 2))
    p_flip_pair = posterior.sum(axis=2)
    return {"p_win": p_win, "conditions": p_cond, "posterior": posterior,
            "pair_posterior": p_flip_pair, "finals": finals}


def size_evidence(episode, params):
    """Per machine and win pair: log evidence from payout sizes, and P(table 1 at cutoff).

    Each pair carries a learned distribution over size lags, so payout sizes also
    inform which lag pair the machine uses.
    """
    T = episode.cutoff
    lags = params["size_lags"]
    threshold = params["size_threshold"]
    given = np.log(np.asarray(params["size_given_pair"], dtype=float))
    log_d = np.log(np.asarray(params["size_direction_prior"], dtype=float))
    n, k = len(episode.queries), given.shape[0]
    evidence, table = np.zeros((n, k)), np.zeros((n, k))
    index = {value: position for position, value in enumerate(SIZES)}
    for i, query in enumerate(episode.queries):
        tables = np.log(np.asarray(params["tables"][query.machine_type], dtype=float))
        y = episode.Y[i]
        wins = np.nonzero(y > 0)[0]
        codes = np.array([index[int(v)] for v in y[wins]], dtype=int)
        delta = tables[1, codes] - tables[0, codes]
        # A common reference keeps the evidence comparable between lags: ticks
        # without an observed lag contribute the average of the two tables.
        scores = np.zeros((len(lags), 2))
        active = np.zeros((len(lags), 2))
        for position, lag in enumerate(lags):
            usable = np.zeros(len(wins), dtype=bool)
            usable[wins >= lag] = True
            usable[usable] = y[wins[usable] - lag] >= 0
            flag = np.zeros(len(wins), dtype=bool)
            flag[usable] = y[wins[usable] - lag] >= threshold
            last = y[T - lag] if T >= lag else -1
            for d in (0, 1):
                on = flag != bool(d)
                scores[position, d] = (delta[usable & on].sum()
                                       + 0.5 * delta[~usable].sum()) + log_d[d]
                active[position, d] = 0.5 if last < 0 else float((last >= threshold) != bool(d))
        for j in range(k):
            joint = given[j][:, None] + scores
            total = logsum(joint.reshape(-1))
            evidence[i, j] = total
            table[i, j] = float((np.exp(joint - total) * active).sum())
    return evidence, table


STACK_WINDOWS = (8, 16, 32, 64)
STACK_NAMES = (["steady", "balanced", "jackpot", "signal", "short", "own_all"]
               + ["own%d" % k for k in STACK_WINDOWS]
               + ["other%d" % k for k in STACK_WINDOWS]
               + ["mean8", "mean16", "mean32"])


def map_paths(episode, win, params):
    """Most probable sign path per machine: fitted logits and usable-tick mask."""
    T = episode.cutoff
    n = len(episode.queries)
    w, kappa, m0 = params["w"], params["kappa"], params["m0"]
    level = kappa * (episode.mean_series(params["kappa_lag"], m0) - m0)
    eta, usable = np.zeros((n, T)), np.zeros((n, T), dtype=bool)
    signed = np.zeros((n, T))
    ticks = np.arange(T)
    for i, query in enumerate(episode.queries):
        posterior = win["posterior"][i]
        j, s = np.unravel_index(int(posterior.argmax()), posterior.shape)
        sign = 1.0 if s == 0 else -1.0
        ok, e, _ = pair_terms(episode, i, params["pairs"][j])
        base = machine_offsets(params, query) + level
        flag = episode.win[i].astype(float)

        def loglik(direction):
            value = base + direction * w * e
            return np.where(ok, flag * log_sigmoid(value) + (1 - flag) * log_sigmoid(-value), 0.0)

        keep = np.concatenate(([0.0], np.cumsum(loglik(sign))))
        other = np.concatenate(([0.0], np.cumsum(loglik(-sign))))
        candidate = other + keep[-1] - keep
        tau = int(candidate.argmax())
        if win["conditions"][0] >= 0.5 or candidate[tau] < keep[-1] + 3.0:
            tau = 0
        signed[i] = np.where(ticks < tau, -sign, sign) * e
        eta[i] = base + w * signed[i]
        usable[i] = ok
    return eta, usable, signed


def stack_features(episode, win, params):
    """Residual summaries, computed from pre-cutoff outcomes under the fitted paths."""
    eta, usable, _ = map_paths(episode, win, params)
    p = sigmoid(eta)
    residual = (episode.win - p) * usable
    variance = p * (1 - p) * usable
    T, n = episode.cutoff, len(episode.queries)
    rows = np.zeros((n, len(STACK_NAMES)))
    short = float(episode.observed.sum() < episode.observed.size)

    def recent_mean(k):
        values = episode.M[max(0, T - k):T]
        values = values[~np.isnan(values)]
        return float(values.mean()) - params["m0"] if len(values) else 0.0

    means = [recent_mean(8), recent_mean(16), recent_mean(32)]
    for i, query in enumerate(episode.queries):
        others = np.arange(n) != i
        row = [float(query.machine_type == name) for name in ("steady", "balanced", "jackpot")]
        row += [float(query.signal), short, residual[i].sum() / (variance[i].sum() + 1)]
        for k in STACK_WINDOWS:
            low = max(0, T - k)
            row.append(residual[i, low:T].sum() / (variance[i, low:T].sum() + 1))
        for k in STACK_WINDOWS:
            low = max(0, T - k)
            row.append(residual[others, low:T].sum() / (variance[others, low:T].sum() + 1))
        rows[i] = row + means
    return rows


def corrected_win(win, features, params):
    """Apply the fitted logit correction to every hypothesis, then re-mix."""
    stack = params.get("stack")
    if not stack:
        return win["p_win"]
    delta = features @ np.asarray(stack["coef"], dtype=float)
    finals = np.clip(win["finals"], 1e-9, 1 - 1e-9)
    logits = np.log(finals / (1 - finals)) + delta[:, None, None]
    return (win["posterior"] * sigmoid(logits)).sum(axis=(1, 2))


def predict_episode(episode, params):
    evidence, tables_by_pair = size_evidence(episode, params)
    win = infer_win(episode, params, evidence)
    table = (win["pair_posterior"] * tables_by_pair).sum(axis=1)
    if params.get("stack"):
        win["p_win"] = corrected_win(win, stack_features(episode, win, params), params)
    output = {}
    for i, query in enumerate(episode.queries):
        tables = np.asarray(params["tables"][query.machine_type], dtype=float)
        sizes = (1 - table[i]) * tables[0] + table[i] * tables[1]
        p = float(np.clip(win["p_win"][i], 1e-6, 1 - 1e-6))
        values = [1 - p] + [p * float(v) for v in sizes]
        total = math.fsum(values)
        output[query.query_id] = tuple(v / total for v in values)
    return output, win, table


def group_episodes(queries, history):
    index = HistoryIndex(history)
    groups = {}
    for q in queries:
        groups.setdefault(q.episode_id, []).append(q)
    return [Episode(groups[key], index) for key in sorted(groups)]


def predict(queries, history, params):
    output = {}
    for episode in group_episodes(queries, history):
        output.update(predict_episode(episode, params)[0])
    return output


def save_params(path, params):
    Path(path).write_text(json.dumps(params, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_params(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
