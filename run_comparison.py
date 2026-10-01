"""
Calibration run: does the revised Implementation measure recover the ranking
that the original marker counts inverted?

This is a falsifiable test, not a demonstration. It is stated before it is run:

  PREDICTION - if the 2026-09-23 inversion was caused by the markers detecting
  announcements rather than code, then code_dictation_density should rank
  Classes/OOP ABOVE 2D Lists on Implementation, matching the hand-scores
  (2.5/3.0 vs 2.0/3.0). The original markers ranked them 0 vs 1, backwards.

  If the revised measure does NOT recover the order, the diagnosis is wrong and
  the problem lies somewhere other than announcement-detection.

Reliability statistics follow the conventions used in the comparable published
work: percentage agreement with a 68% floor, Cohen's kappa read against the
Landis and Koch (1977) bands, and Kendall's tau for small samples.

Run:
    python run_comparison.py
"""

import json
import os
import sys

from improved_signals import improved_signals
from lecture_rating_pipeline import fetch_transcript

# Hand-scores from the Week 2 report (2026-09-09), applied before any of these
# automated measures existed, so they are not contaminated by them.
LECTURES = [
    {
        "key": "2d_lists",
        "video_id": "hEh_6otWzNs",
        "title": "2-Dimensional Lists, Nested For Loops, Images",
        "keywords_file": "syllabus_2dlists.txt",
        "hand": {"coverage": 3.5, "examples": 2.0, "implementation": 2.0,
                 "delivery": 1.5, "total": 9.0},
    },
    {
        "key": "oop",
        "video_id": "l_n_7mOqqjs",
        "title": "Classes, Attributes, Object-Oriented Programming",
        "keywords_file": "syllabus_oop.txt",
        "hand": {"coverage": 3.5, "examples": 2.0, "implementation": 2.5,
                 "delivery": 1.0, "total": 9.0},
    },
    {
        "key": "announcements",
        "video_id": "U8M4DlCE2ms",
        "title": "'Computer Science Ethics' as scheduled (announcements clip)",
        "keywords_file": "syllabus_ethics.txt",
        "hand": {"coverage": 0.0, "examples": 0.0, "implementation": 0.0,
                 "delivery": 1.0, "total": 1.0},
    },
]


def load_keywords(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_transcript(video_id):
    """Use a saved transcript when present, otherwise fetch and save it."""
    cached = os.path.join("transcripts", video_id + ".txt")
    if os.path.exists(cached):
        with open(cached, encoding="utf-8") as f:
            return f.read(), "cached"
    text = fetch_transcript(video_id)
    os.makedirs("transcripts", exist_ok=True)
    with open(cached, "w", encoding="utf-8") as f:
        f.write(text)
    return text, "fetched"


def kendall_tau(xs, ys):
    """Kendall's tau-b. Used because n is small; returns None when undefined."""
    n = len(xs)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tx += 1; ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return round((conc - disc) / denom, 3) if denom else None


def main():
    rows = []
    for spec in LECTURES:
        try:
            transcript, source = get_transcript(spec["video_id"])
        except Exception as exc:                      # noqa: BLE001
            print("Could not get transcript for %s: %s" % (spec["video_id"], exc))
            print("Run this on a machine with unrestricted access to YouTube.")
            return 1
        sig = improved_signals(transcript, load_keywords(spec["keywords_file"]))
        rows.append({**spec, "source": source, "signals": sig})
        print("  %-14s %-8s %5d words" % (spec["key"], source, sig["word_count"]))

    print()
    print("=" * 78)
    print("IMPLEMENTATION: hand-score vs original markers vs revised measure")
    print("=" * 78)
    print("%-16s %10s %12s %14s" % ("lecture", "hand", "orig markers", "code density"))
    for r in rows:
        print("%-16s %10.1f %12d %14.2f" % (
            r["key"],
            r["hand"]["implementation"],
            r["signals"]["original_stage2"]["implementation_marker_count"],
            r["signals"]["revised_stage2b"]["code_dictation_density_per_1000_words"],
        ))

    content = [r for r in rows if r["key"] in ("2d_lists", "oop")]
    by_key = {r["key"]: r for r in content}
    verdict = {}
    if len(content) == 2:
        oop, twod = by_key["oop"], by_key["2d_lists"]
        hand_order = oop["hand"]["implementation"] > twod["hand"]["implementation"]
        orig_order = (oop["signals"]["original_stage2"]["implementation_marker_count"]
                      > twod["signals"]["original_stage2"]["implementation_marker_count"])
        new_order = (oop["signals"]["revised_stage2b"]["code_dictation_density_per_1000_words"]
                     > twod["signals"]["revised_stage2b"]["code_dictation_density_per_1000_words"])
        verdict = {
            "hand_ranks_oop_higher": hand_order,
            "original_markers_agree_with_hand": orig_order == hand_order,
            "revised_measure_agrees_with_hand": new_order == hand_order,
            "prediction_supported": (new_order == hand_order) and (orig_order != hand_order),
        }
        print()
        print("Hand ranks OOP above 2D Lists on Implementation: %s" % hand_order)
        print("Original markers reproduce that order:           %s" % verdict["original_markers_agree_with_hand"])
        print("Revised code-density reproduces that order:      %s" % verdict["revised_measure_agrees_with_hand"])
        print()
        print("PREDICTION SUPPORTED: %s" % verdict["prediction_supported"])
        if not verdict["prediction_supported"]:
            print("  The diagnosis does not hold. The inversion has another cause;")
            print("  do not report the announcement explanation as established.")

    # Rank agreement across all three lectures, reported with its own caveat.
    hand_impl = [r["hand"]["implementation"] for r in rows]
    orig = [r["signals"]["original_stage2"]["implementation_marker_count"] for r in rows]
    new = [r["signals"]["revised_stage2b"]["code_dictation_density_per_1000_words"] for r in rows]
    taus = {
        "kendall_tau_hand_vs_original": kendall_tau(hand_impl, orig),
        "kendall_tau_hand_vs_revised": kendall_tau(hand_impl, new),
        "n": len(rows),
        "caveat": ("n=3 is far below any threshold for inference. These coefficients "
                   "are descriptive only and must not be reported as evidence of "
                   "measure validity."),
    }
    print()
    print("Kendall tau, hand vs original markers: %s" % taus["kendall_tau_hand_vs_original"])
    print("Kendall tau, hand vs revised measure:  %s" % taus["kendall_tau_hand_vs_revised"])
    print("(n=3 - descriptive only, not evidence of validity)")

    out = {
        "run_date": "2026-10-01",
        "prediction": ("If the inversion was caused by announcement-detection, the revised "
                       "code-dictation measure should rank OOP above 2D Lists on "
                       "Implementation, which the original markers failed to do."),
        "verdict": verdict,
        "rank_agreement": taus,
        "lectures": [
            {"key": r["key"], "video_id": r["video_id"], "title": r["title"],
             "hand_scores": r["hand"], "signals": r["signals"]}
            for r in rows
        ],
    }
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "calibration_2026-10-01.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWritten to %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
