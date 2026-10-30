"""
Scores Stage 1 against the synthetic lecture's ground truth.
Change detection: precision/recall with a +/-2 s tolerance, plus how many
code-typing builds were wrongly treated as new segments.
OCR: share of ground-truth words recovered from each segment's keyframe.
Alignment: share of mock spoken words attached to the right segment.
"""
import json
import re
import sys

sys.path.insert(0, sys.argv[3])
import stage1_extract as s1

video, truth_path = sys.argv[1], sys.argv[2]
truth = json.load(open(truth_path))
TOL = 2.0
MASK = (0.78, 0.70, 1.0, 1.0)


def match(detected, gt):
    used, tp = set(), 0
    for g in gt:
        hit = next((i for i, d in enumerate(detected) if i not in used and abs(d - g) <= TOL), None)
        if hit is not None:
            used.add(hit)
            tp += 1
    return tp


def words(s):
    return [w for w in re.findall(r"[a-z0-9_]+", s.lower()) if len(w) > 1]


report = {"runs": []}
for label, mask in [("no mask", None), ("webcam masked", MASK)]:
    for thr in [0.20, 0.35, 0.50]:
        states, _ = s1.detect_changes(video, threshold=thr, mask=mask,
                                      keyframe_dir=None if thr != 0.35 else
                                      f"tests/data/kf_{label.replace(' ', '_')}")
        det = [st["start"] for st in states[1:]]
        tp = match(det, truth["changes"])
        builds = sum(1 for d in det if any(abs(d - b) <= TOL for b in truth["build_events"])
                     and not any(abs(d - g) <= TOL for g in truth["changes"]))
        run = {"condition": label, "threshold": thr, "detected": len(det),
               "true_changes": len(truth["changes"]), "hits": tp,
               "precision": round(tp / len(det), 2) if det else None,
               "recall": round(tp / len(truth["changes"]), 2),
               "builds_split_off": builds, "detected_times": det}
        if thr == 0.35:
            ocr_rows = []
            for st in states:
                mid = (st["start"] + st["end"]) / 2
                gt = next(s for s in truth["slides"] if s["start"] <= mid < s["end"])
                got = set(words(s1.ocr_image(st["keyframe"])))
                want = words(gt["text"])
                rec = sum(1 for w in want if w in got) / max(len(want), 1)
                ocr_rows.append({"segment_start": st["start"], "slide": gt["kind"],
                                 "ocr_word_recall": round(rec, 2)})
            run["ocr"] = ocr_rows
            # alignment: one mock word every 0.5 s, labelled with the true slide index
            mock = [{"start": t / 2, "end": t / 2 + 0.3, "word": f"w{t}"} for t in range(0, 399)]
            segs = s1.align([dict(st) for st in states], mock)
            ok = tot = 0
            for st in segs:
                for w in st["words"]:
                    mid = (w["start"] + w["end"]) / 2
                    true_slide = next(i for i, s in enumerate(truth["slides"]) if s["start"] <= mid < s["end"])
                    seg_mid = (st["start"] + st["end"]) / 2
                    seg_slide = next(i for i, s in enumerate(truth["slides"]) if s["start"] <= seg_mid < s["end"])
                    ok += true_slide == seg_slide
                    tot += 1
            run["alignment_accuracy"] = round(ok / tot, 3)
        report["runs"].append(run)
        print(label, thr, {k: run[k] for k in ("detected", "hits", "precision", "recall", "builds_split_off")})

json.dump(report, open("results/stage1_synthetic_eval.json", "w"), indent=2)
for r in report["runs"]:
    if "ocr" in r:
        print(r["condition"], "alignment", r["alignment_accuracy"])
        for o in r["ocr"]:
            print("   ", o)
