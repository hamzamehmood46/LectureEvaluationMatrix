"""
Evaluation harness for the paper's comparison of three input conditions
(transcript, transcript plus on-screen text, transcript plus video) against
expert annotations under rubric v2.

Agreement measures
  exact_agreement      share of criterion scores identical to the expert's
  weighted_kappa       quadratic-weighted Cohen's kappa on the 0/1/2 scale
  kendall_tau_b        rank agreement across lectures for one criterion
  evidence_precision   share of cited quotes that Stage 2's checker verified
Comparison
  paired_bootstrap     difference between two conditions, resampling lectures,
                       with a 95% interval
Planning
  power_curve          simulation of how many lectures are needed to detect
                       a given gain from adding video
"""
import numpy as np

LEVELS = (0, 1, 2)


def exact_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean())


def weighted_kappa(a, b, levels=LEVELS):
    a, b = np.asarray(a), np.asarray(b)
    k = len(levels)
    idx = {v: i for i, v in enumerate(levels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom else 1.0


def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    d = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / d) if d else float("nan")


def evidence_precision(stage2_results):
    """stage2_results: list of stage2_judge.py output dicts."""
    cited = verified = 0
    for res in stage2_results:
        for dim in res["dimensions"]:
            for s in dim["scores"]:
                cited += s["evidence_items"]
                verified += s["evidence_verified"]
    return verified / cited if cited else float("nan")


def paired_bootstrap(expert, cond_a, cond_b, metric=exact_agreement, n_boot=5000, seed=0):
    """
    expert, cond_a, cond_b: arrays shaped (lectures, criteria).
    Returns metric(B) - metric(A) with a 95% interval from resampling lectures.
    """
    rng = np.random.default_rng(seed)
    expert, cond_a, cond_b = map(np.asarray, (expert, cond_a, cond_b))
    n = len(expert)
    point = metric(expert.ravel(), cond_b.ravel()) - metric(expert.ravel(), cond_a.ravel())
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        e = expert[i].ravel()
        diffs.append(metric(e, cond_b[i].ravel()) - metric(e, cond_a[i].ravel()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(point), "ci95": [float(lo), float(hi)]}


# ------------------------------------------------------------------ planning

def _simulate_judge(expert, p, rng):
    """Matches the expert with probability p (per lecture), otherwise one level off."""
    out = expert.copy()
    miss = rng.random(expert.shape) > p[:, None]
    step = np.where(rng.random(expert.shape) < 0.5, -1, 1)
    alt = expert + step
    alt = np.where((alt < 0) | (alt > 2), expert - step, alt)
    out[miss] = alt[miss]
    return out


def power_curve(n_lectures, p_transcript=0.55, gain=0.15, visual_criteria=4,
                lecture_sd=0.10, sims=1000, n_boot=400, seed=1):
    """
    Share of simulated studies in which the 95% interval for the video gain on
    visual criteria excludes zero. Lectures differ in difficulty (lecture_sd),
    and that difficulty affects both conditions alike, as it would in a paired design.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        expert = rng.integers(0, 3, (n_lectures, visual_criteria))
        shift = rng.normal(0, lecture_sd, n_lectures)
        pa = np.clip(p_transcript + shift, 0.05, 0.98)
        pb = np.clip(p_transcript + gain + shift, 0.05, 0.98)
        a, b = _simulate_judge(expert, pa, rng), _simulate_judge(expert, pb, rng)
        per_a = (a == expert).mean(1)
        per_b = (b == expert).mean(1)
        d = per_b - per_a
        boots = d[rng.integers(0, n_lectures, (n_boot, n_lectures))].mean(1)
        if np.percentile(boots, 2.5) > 0:
            hits += 1
    return hits / sims
