import sys, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "pipeline")
import evaluation as ev
NS = [10, 15, 20, 30, 40, 60]
GAINS = [0.10, 0.15, 0.20]
def job(args):
    n, g = args
    return n, g, ev.power_curve(n, gain=g, sims=1000, n_boot=400, seed=n * 100 + int(g * 100))
with ProcessPoolExecutor(2) as ex:
    res = list(ex.map(job, [(n, g) for g in GAINS for n in NS]))
table = {f"{g:.2f}": {str(n): p for n, gg, p in res if gg == g} for g in GAINS}
print(json.dumps(table, indent=1))
json.dump({"assumptions": {"p_transcript": 0.55, "visual_criteria": 4, "lecture_sd": 0.10,
           "sims": 1000, "n_boot": 400, "test": "95% bootstrap interval for the paired per-lecture gain lies above 0"},
           "power": table}, open("results/power_2026-10-29.json", "w"), indent=1)
