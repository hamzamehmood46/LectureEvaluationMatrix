import csv, json, sys, collections
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "pipeline")
import screen_router as sr
split = sys.argv[1]
rows = list(csv.DictReader(open(f"tests/data/screens/{split}_labels.csv")))
def run(r):
    lab, send, f = sr.classify_image(r["path"])
    return {**r, "pred": lab, "send": send, **f}
with ProcessPoolExecutor(4) as ex:
    res = list(ex.map(run, rows))
classes = ["text", "code", "terminal", "diagram"]
conf = {t: collections.Counter() for t in classes}
for r in res: conf[r["label"]][r["pred"]] += 1
acc = sum(r["label"] == r["pred"] for r in res) / len(res)
print(split, "accuracy", round(acc, 3), "n", len(res))
print("true\\pred", classes)
for t in classes: print(t, [conf[t][p] for p in classes])
need = [r for r in res if r["label"] != "text"]
print("visual screens routed to VLM:", sum(r["send"] for r in need), "/", len(need))
txt = [r for r in res if r["label"] == "text"]
print("text slides skipped:", sum(not r["send"] for r in txt), "/", len(txt))
json.dump({"split": split, "accuracy": acc, "confusion": {t: dict(conf[t]) for t in classes},
           "visual_routed": [sum(r["send"] for r in need), len(need)],
           "text_skipped": [sum(not r["send"] for r in txt), len(txt)],
           "errors": [{k: r[k] for k in ("path","label","pred","lines","prompt_lines","code_lines","numbered_lines","words","nontext_ink")} for r in res if r["label"] != r["pred"]]},
          open(f"results/router_{split}.json", "w"), indent=1)
