"""Learn every structural parameter from a round's public data, then forecast.

Works unchanged on main or adaptation data:

    python -m solution.fit_structural --data public --out runs/structural_01

Learned from histories: lag-pair pool, size threshold, size lags and their link to
pairs, shared lag, base logits, parity strength and signal effect.
Learned from validation true probabilities: payout tables and the logit correction.
Condition labels are used only to select stable episodes for pool discovery; inference
never reads them. Episode-grouped cross-validation gives the out-of-sample score.
"""
import argparse
import copy
import hashlib
import itertools
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from slot_forecasting.io import load_inputs, read_rows, read_targets, write_json, write_submission
from slot_forecasting.rounds import score_round
from slot_forecasting.schema import PAYOUTS
from . import structural as st

ALL_PAIRS = list(itertools.combinations(range(1, 9), 2))
THRESHOLDS = (1, 2, 5, 20)
TYPES = ("steady", "balanced", "jackpot")
MIN_POOL_SHARE = 0.03
SEED_ALPHA = 30.0
FOLDS = 6


def fit_tables(targets, queries):
    """Each machine type has two exact size tables; split on the largest gap in p(1|win)."""
    machine_type = {q.query_id: q.machine_type for q in queries}
    grouped = {}
    for target in targets:
        p = np.asarray(target.probabilities, dtype=float)
        grouped.setdefault(machine_type[target.query_id], []).append(p[1:] / p[1:].sum())
    tables, spread = {}, {}
    for name, rows in grouped.items():
        rows = np.asarray(rows)
        order = np.sort(rows[:, 0])
        cut = order[int(np.argmax(np.diff(order)))] + 1e-9
        low, high = rows[rows[:, 0] <= cut], rows[rows[:, 0] > cut]
        tables[name] = [low.mean(axis=0).tolist(), high.mean(axis=0).tolist()]
        # Near zero when two tables explain every target; large means the family changed.
        spread[name] = float(max(low.std(axis=0).max(), high.std(axis=0).max()))
    return tables, spread


def parity_likelihood(episode, i, pair):
    """Log likelihood of wins under a free two-cell parity table."""
    ok, e, _ = st.pair_terms(episode, i, pair)
    flag, parity = episode.win[i][ok], e[ok] > 0
    total = 0.0
    for cell in (flag[parity], flag[~parity]):
        n, k = len(cell), int(cell.sum())
        p = (k + 0.5) / (n + 1)
        total += k * np.log(p) + (n - k) * np.log(1 - p)
    return total


def discover_pairs(episodes):
    """Find which lag pairs recur, using long stable histories."""
    best = Counter()
    for episode in episodes:
        for i in range(len(episode.queries)):
            scores = sorted(((parity_likelihood(episode, i, pair), pair) for pair in ALL_PAIRS),
                            reverse=True)
            if scores[0][0] - scores[1][0] > 4:
                best[scores[0][1]] += 1
    total = sum(best.values())
    kept = sorted(pair for pair, count in best.items() if count >= MIN_POOL_SHARE * total)
    if not kept:
        raise ValueError("No recurring lag pair found in stable episodes")
    counts = np.array([best[pair] for pair in kept], dtype=float)
    return [list(pair) for pair in kept], (counts / counts.sum()).tolist(), dict(
        (str(k), v) for k, v in best.most_common(12))


def size_scores(episode, i, tables, threshold):
    """Signed size evidence for each candidate lag 1..8."""
    y = episode.Y[i]
    wins = np.nonzero(y > 0)[0]
    if len(wins) == 0:
        return np.zeros(8)
    table = np.log(np.asarray(tables, dtype=float))
    codes = np.array([PAYOUTS.index(int(v)) - 1 for v in y[wins]], dtype=int)
    delta = table[1, codes] - table[0, codes]
    out = np.zeros(8)
    for lag in range(1, 9):
        usable = wins >= lag
        usable[usable] = y[wins[usable] - lag] >= 0
        flag = y[wins[usable] - lag] >= threshold
        out[lag - 1] = (delta[usable] * (2.0 * flag - 1)).sum()
    return out


def discover_sizes(episodes, params, confident=14.0):
    """Find the size threshold, then which lags each win pair uses."""
    pairs = params["pairs"]
    totals = {}
    for threshold in THRESHOLDS:
        value = 0.0
        for episode in episodes:
            for i, query in enumerate(episode.queries):
                value += np.abs(size_scores(episode, i, params["tables"][query.machine_type],
                                            threshold)).max()
        totals[threshold] = value
    threshold = max(totals, key=totals.get)
    counts = np.zeros((len(pairs), 8))
    direction = np.zeros(2)
    for episode in episodes:
        win = st.infer(episode, params)
        for i, query in enumerate(episode.queries):
            scores = size_scores(episode, i, params["tables"][query.machine_type], threshold)
            lag = int(np.abs(scores).argmax())
            if abs(scores[lag]) > confident:
                j = int(win["pair_posterior"][i].argmax())
                counts[j, lag] += 1
                direction[0 if scores[lag] > 0 else 1] += 1
    share = counts.sum(axis=0) / max(counts.sum(), 1)
    lags = [lag for lag in range(8) if share[lag] >= MIN_POOL_SHARE]
    if not lags:
        raise ValueError("No recurring size lag found")
    table = counts[:, lags] + 0.5
    table /= table.sum(axis=1, keepdims=True)
    # Off-diagonal mass is capped at a small floor so sampling noise is not read as signal.
    table = np.maximum(table, 0.02)
    table /= table.sum(axis=1, keepdims=True)
    params["size_threshold"] = int(threshold)
    params["size_lags"] = [lag + 1 for lag in lags]
    params["size_given_pair"] = table.tolist()
    params["size_direction_prior"] = ((direction + 1) / (direction.sum() + 2)).tolist()
    return {"threshold_scores": {str(k): v for k, v in totals.items()}, "lag_counts": counts.tolist()}


def logistic(X, y, offset=None, steps=30, ridge=1e-3):
    """Newton logistic regression; deterministic and numpy-only."""
    beta = np.zeros(X.shape[1])
    offset = np.zeros(len(y)) if offset is None else offset
    for _ in range(steps):
        p = st.sigmoid(X @ beta + offset)
        weight = np.maximum(p * (1 - p), 1e-9)
        hessian = (X * weight[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, X.T @ (y - p) - ridge * beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def discover_shared_lag(episodes, params):
    """Choose the lag at which episode-mean wins best explain residual wins."""
    m0 = params["m0"]
    gain = np.zeros(8)
    trial = copy.deepcopy(params)
    trial["kappa"] = 0.0
    for episode in episodes:
        win = st.infer(episode, trial)
        eta, usable, _, _ = st.map_paths(episode, win, trial)
        residual = ((episode.win - st.sigmoid(eta)) * usable).sum(axis=0)
        for lag in range(1, 9):
            gain[lag - 1] += float(residual @ (episode.mean_series(lag, m0) - m0))
    params["kappa_lag"] = int(gain.argmax()) + 1
    return gain.tolist()


def fit_logit(episodes, params, rounds=3):
    """Alternate between MAP rule paths and a pooled logistic fit of the win logit."""
    trace = []
    for _ in range(rounds):
        rows, labels = [], []
        for episode in episodes:
            win = st.infer(episode, params)
            _, usable, signed, regime = st.map_paths(episode, win, params)
            level = episode.mean_series(params["kappa_lag"], params["m0"]) - params["m0"]
            for i, query in enumerate(episode.queries):
                ticks = np.nonzero(usable[i])[0]
                block = np.zeros((len(ticks), 11))
                position = TYPES.index(query.machine_type)
                block[:, position] = 1.0
                block[:, 3 + position] = signed[i, ticks]
                block[:, 6 + position] = regime[i, ticks]
                block[:, 9] = query.signal
                block[:, 10] = level[ticks]
                rows.append(block)
                labels.append(episode.win[i, ticks].astype(float))
        beta = logistic(np.concatenate(rows), np.concatenate(labels))
        params["base"] = dict(zip(TYPES, map(float, beta[:3])))
        params["w"] = dict(zip(TYPES, map(float, beta[3:6])))
        params["nu"] = dict(zip(TYPES, map(float, beta[6:9])))
        params["gamma"], params["kappa"] = float(beta[9]), float(beta[10])
        trace.append([float(v) for v in beta])
    return trace


def flip_tick(episode, i, win, params):
    """Most likely flip tick for machine i under its MAP rule, or None if no clear flip."""
    *_, candidate, keep = st.flip_scan(episode, i, win, params)
    tau = int(candidate.argmax())
    return tau if candidate[tau] > keep + 6.0 else None


def discover_flips(episodes, conditions, params):
    """Learn the flip-time window and drift spread from labelled changing episodes."""
    fractions, spreads = [], []
    trial = dict(params)
    trial["condition_prior"] = {"stable": 1e-6, "shared": 0.5, "drift": 0.5}
    trial["tau_low"], trial["tau_high"] = 0.2, 0.97
    for episode in episodes:
        condition = conditions[episode.queries[0].query_id]
        if condition not in ("drift", "shared_change") or not episode.observed.all():
            continue
        win = st.infer(episode, trial)
        taus = [flip_tick(episode, i, win, params) for i in range(len(episode.queries))]
        taus = [tau for tau in taus if tau is not None]
        if taus:
            fractions.append(float(np.median(taus)) / episode.cutoff)
            if condition == "drift" and len(taus) > 2:
                spreads.append(float(np.std(taus)))
    if len(fractions) < 8:
        return {"episodes": len(fractions), "kept_defaults": True}
    params["tau_low"] = max(0.05, float(np.min(fractions)) - 0.08)
    params["tau_high"] = min(0.98, float(np.max(fractions)) + 0.12)
    if len(spreads) >= 4:
        params["drift_sd"] = max(4.0, float(np.median(spreads)))
    return {"episodes": len(fractions), "fraction_range": [float(np.min(fractions)), float(np.max(fractions))],
            "drift_spread_median": float(np.median(spreads)) if spreads else None}


def discover_short_flips(episodes, params):
    """Decide from short-history evidence whether those episodes also flip."""
    short = [episode for episode in episodes if episode.short]
    evidence = np.zeros(3)
    for episode in short:
        evidence += st.infer(episode, params)["log_evidence"]
    if len(short) and evidence[0] > evidence[1:].max():
        params["condition_prior_short"] = {"stable": 0.98, "shared": 0.01, "drift": 0.01}
    return {"episodes": len(short), "log_evidence": evidence.tolist(),
            "prior": params["condition_prior_short"]}


def stack_data(episodes, targets, params):
    frame = {"X": [], "y": [], "group": [], "use": []}
    cached = []
    for k, episode in enumerate(episodes):
        win = st.infer(episode, params)
        features = st.stack_features(episode, win, params)
        cached.append((episode, win, features))
        for i, query in enumerate(episode.queries):
            posterior = win["posterior"][i]
            index = np.unravel_index(int(posterior.argmax()), posterior.shape)
            p0 = float(targets[query.query_id].probabilities[0])
            fitted = float(np.clip(win["finals"][i][index], 1e-9, 1 - 1e-9))
            frame["X"].append(features[i])
            frame["y"].append(np.log((1 - p0) / p0) - np.log(fitted / (1 - fitted)))
            frame["group"].append(k)
            frame["use"].append(posterior.max() > 0.9)
    return {key: np.asarray(value) for key, value in frame.items()}, cached


def ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def forecasts(cached, coefficients, params):
    output = {}
    for k, (episode, win, features) in enumerate(cached):
        p_win = st.corrected_win(win, features, coefficients[k])
        for i, query in enumerate(episode.queries):
            output[query.query_id] = st.probabilities(query, p_win[i], win["table"][i], params)
    return output


def cross_validated(data, cached, alpha):
    """Out-of-sample correction: every episode's coefficients exclude that episode."""
    groups = np.arange(len(cached))
    rng = np.random.default_rng(20261107)
    rng.shuffle(groups)
    coefficients = np.zeros((len(cached), data["X"].shape[1]))
    for held in np.array_split(groups, FOLDS):
        fit = ~np.isin(data["group"], held) & data["use"]
        coefficients[held] = ridge(data["X"][fit], data["y"][fit], alpha)
    return coefficients


def condition_map(path):
    return {row["query_id"]: row["condition"]
            for row in read_rows(path, ("query_id", "condition"))}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(predictions, public, out, name):
    path = out / name
    write_submission(path, predictions)
    result = score_round(path, public)
    return {"technical_points": result["technical_points"], "mean_loss_ratio": result["mean_loss_ratio"],
            "conditions": {key: value["loss_ratio"] for key, value in result["by_condition"].items()}}


def fit(data, out, alpha=SEED_ALPHA):
    data = Path(data)
    queries = {}
    history = {}
    conditions = {}
    for split in ("train", "validation"):
        queries[split], history[split] = load_inputs(data / split)
        conditions.update(condition_map(data / split / "conditions.csv"))
    targets_list = read_targets(data / "validation/targets.csv")
    targets = {target.query_id: target for target in targets_list}

    params = copy.deepcopy(st.DEFAULT_PARAMS)
    params["stack"] = None
    params["tables"], table_spread = fit_tables(targets_list, queries["validation"])
    episodes = (st.group_episodes(queries["train"], history["train"])
                + st.group_episodes(queries["validation"], history["validation"]))
    stable = [e for e in episodes if conditions[e.queries[0].query_id] == "stable"]
    long_history = [e for e in episodes if e.observed.all()]

    report = {"table_spread": table_spread}
    params["pairs"], params["pair_prior"], report["pair_counts"] = discover_pairs(stable)
    # Bootstrap with no size link; discover_sizes replaces both of these.
    params["size_lags"] = [1]
    params["size_given_pair"] = [[1.0] for _ in params["pairs"]]
    report["size"] = discover_sizes(long_history, params)
    report["shared_lag_gain"] = discover_shared_lag(stable, params)
    report["logit_trace"] = fit_logit(episodes, params)
    report["flips"] = discover_flips(episodes, conditions, params)
    report["logit_refit"] = fit_logit(episodes, params, rounds=1)
    report["size_refit"] = discover_sizes(long_history, params)
    report["logit_with_regime"] = fit_logit(episodes, params)
    report["short_flips"] = discover_short_flips(episodes, params)

    stack, cached = stack_data(st.group_episodes(queries["validation"], history["validation"]),
                               targets, params)
    n_features = stack["X"].shape[1]
    report["validation_no_correction"] = score(
        forecasts(cached, np.zeros((len(cached), n_features)), params), data, out, "validation_uncorrected.csv")
    report["validation_cross_validated"] = score(
        forecasts(cached, cross_validated(stack, cached, alpha), params), data, out, "validation_cv.csv")
    use = stack["use"]
    coefficient = ridge(stack["X"][use], stack["y"][use], alpha)
    params["stack"] = {"coef": coefficient.tolist(), "names": list(st.STACK_NAMES), "alpha": alpha}
    return params, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("public"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=SEED_ALPHA)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Choose a fresh output folder")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    params, report = fit(args.data, args.out, args.alpha)
    st.save_params(args.out / "params.json", params)

    loaded = st.load_params(args.out / "params.json")
    for split, name in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
        queries, history = load_inputs(args.data / split)
        write_submission(args.out / name, st.predict(queries, history, loaded))
    report["validation_in_sample_correction"] = score_round(args.out / "validation_predictions.csv", args.data)[
        "technical_points"]
    report["metadata"] = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    report["input_sha256"] = {str(path): sha256(path) for split in ("train", "validation", "test")
                              for path in sorted((args.data / split).glob("*.csv"))}
    report["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "platform": platform.platform()}
    report["seconds"] = time.perf_counter() - start
    write_json(args.out / "report.json", report)
    summary = {key: report[key] for key in ("pair_counts", "validation_no_correction",
                                            "validation_cross_validated", "seconds")}
    summary["params"] = {key: params[key] for key in ("pairs", "pair_prior", "size_threshold", "size_lags",
                                                      "size_given_pair", "base", "w", "nu", "gamma", "kappa", "kappa_lag",
                                                      "tau_low", "tau_high", "drift_sd")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
