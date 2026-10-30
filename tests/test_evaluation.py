import sys, numpy as np
sys.path.insert(0, "pipeline")
import evaluation as ev
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau
rng = np.random.default_rng(3)
for _ in range(200):
    a = rng.integers(0, 3, 40); b = np.where(rng.random(40) < 0.6, a, rng.integers(0, 3, 40))
    if len(set(a)) < 2 and len(set(b)) < 2: continue
    assert abs(ev.weighted_kappa(a, b) - cohen_kappa_score(a, b, weights="quadratic", labels=[0,1,2])) < 1e-9
    x = rng.integers(0, 4, 12); y = rng.integers(0, 4, 12)
    t = kendalltau(x, y).statistic
    if not np.isnan(t): assert abs(ev.kendall_tau_b(x, y) - t) < 1e-9
assert ev.exact_agreement([0,1,2,2],[0,1,1,2]) == 0.75
e = rng.integers(0,3,(20,4)); r = ev.paired_bootstrap(e, (e+1)%3, e)
assert r["difference"] == 1.0 and r["ci95"][0] == 1.0
s2 = [{"dimensions":[{"scores":[{"evidence_items":2,"evidence_verified":1},{"evidence_items":2,"evidence_verified":2}]}]}]
assert ev.evidence_precision(s2) == 0.75
print("evaluation metrics match sklearn/scipy on 200 random cases; all checks passed")
