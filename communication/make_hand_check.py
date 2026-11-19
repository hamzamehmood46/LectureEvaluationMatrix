"""Select the 12 hand-check lectures and cut their windows (preregistration, HAND CHECK).

Two lectures per course, chosen with random seed 20261114 from the course's sorted manifest
ids. For each, the 1,000-word window at 50 percent of the transcript is written to a
text file with an anonymous code, no title, no course and no institution. The code-to-
lecture key is a separate file. No metric is computed here.

    python communication/make_hand_check.py --data-dir C:/Users/hamza/lecture_data
"""
import argparse
import csv
import json
import random
from pathlib import Path

import score_comm as sc

SEED = 20261114
PER_COURSE = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(Path(__file__).with_name("corpus_manifest.csv")))
    p.add_argument("--data-dir", required=True)
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    rng = random.Random(SEED)
    chosen = []
    for subject in dict.fromkeys(r["subject"] for r in rows):  # manifest order
        ids = sorted(r["id"] for r in rows if r["subject"] == subject)
        chosen += rng.sample(ids, PER_COURSE)
    order = list(chosen)
    random.Random(SEED + 1).shuffle(order)  # so the rater cannot infer course from position
    by_id = {r["id"]: r for r in rows}

    out = Path(a.data_dir) / "hand_check"
    out.mkdir(parents=True, exist_ok=True)
    blocks, key = [], []
    for k, lid in enumerate(order, 1):
        t = sc.load(Path(a.data_dir) / "transcripts" / f"{lid}.json")
        window = sc.windows(t["text"])[2]
        code = f"H{k:02d}"
        blocks.append(f"==================== {code} ====================\n\n{window}\n")
        key.append({"code": code, "id": lid, "subject": by_id[lid]["subject"], "words": len(window.split())})
    (out / "hand_check_windows.txt").write_text("\n\n".join(blocks), encoding="utf-8")
    with open(out / "key_DO_NOT_OPEN_UNTIL_SCORED.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["code", "id", "subject", "words"])
        w.writeheader()
        w.writerows(key)
    (Path(__file__).with_name("hand_check_selection.json")).write_text(
        json.dumps({"seed": SEED, "per_course": PER_COURSE, "ids": sorted(chosen)}, indent=1), encoding="utf-8")
    print("selected", len(chosen), "lectures;", len(order), "windows written to", out)


if __name__ == "__main__":
    main()
