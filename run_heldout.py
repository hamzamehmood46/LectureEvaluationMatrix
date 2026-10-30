"""Held-out validation run. Hand-scores were locked before this executed."""
import json, io, os
from improved_signals import improved_signals

T = "/home/claude/week6/transcripts"
LECT = [
    ("L3_iteration",  "lzYrMcI92kw", "Lecture 3: Iteration",            3.0, 10.0),
    ("L10_lists",     "Z4bm7xzYpKM", "Lecture 10: Lists and Mutability", 3.0, 10.0),
    ("L17_classes",   "tsMrUdynbQQ", "Lecture 17: Python Classes",       2.0,  9.0),
]

rows = []
for key, vid, title, hand_impl, hand_total in LECT:
    text = io.open(os.path.join(T, vid + ".txt"), encoding="utf-8").read()
    sig = improved_signals(text)
    rows.append({"key": key, "video_id": vid, "title": title,
                 "hand_implementation": hand_impl, "hand_total": hand_total,
                 "signals": sig})

print("=" * 86)
print("HELD-OUT SET: MIT 6.100L (Ana Bell) — measures run AFTER hand-scores were locked")
print("=" * 86)
print("%-14s %7s %8s %9s %10s %11s" % ("lecture", "words", "hand", "orig mk", "code ev", "density/1k"))
for r in rows:
    s = r["signals"]
    print("%-14s %7d %8.1f %9d %10d %11.2f" % (
        r["key"], s["word_count"], r["hand_implementation"],
        s["original_stage2"]["implementation_marker_count"],
        s["revised_stage2b"]["code_evidence_total"],
        s["revised_stage2b"]["code_dictation_density_per_1000_words"]))

print()
print("P1 — does the measure reproduce the hand ranking?")
print("    Hand gives one ordered pair: L3 and L10 (3.0) must both outrank L17 (2.0).")
d = {r["key"]: r["signals"]["revised_stage2b"]["code_dictation_density_per_1000_words"] for r in rows}
o = {r["key"]: r["signals"]["original_stage2"]["implementation_marker_count"] for r in rows}
pairs = [("L3_iteration", "L17_classes"), ("L10_lists", "L17_classes")]
rev_ok = all(d[a] > d[b] for a, b in pairs)
org_ok = all(o[a] > o[b] for a, b in pairs)
for a, b in pairs:
    print("      %-14s vs %-12s  revised %7.2f vs %7.2f  -> %s" % (
        a, b, d[a], d[b], "correct" if d[a] > d[b] else "WRONG"))
print("    revised measure reproduces hand ranking : %s" % rev_ok)
print("    original markers reproduce hand ranking : %s" % org_ok)
print()
print("    P1 SUPPORTED: %s" % rev_ok)

print()
print("P3 — polysemy on unseen text (ambiguous keywords are excluded from the score)")
for r in rows:
    s = r["signals"]["revised_stage2b"]
    print("      %-14s excluded ambiguous keywords=%3d  scored evidence=%3d (punct %d + literals %d)" % (
        r["key"], s["ambiguous_keywords_excluded"], s["code_evidence_total"],
        s["code_punctuation_words"], s["code_literals"]))

print()
print("P5 — does the original marker weakness generalise beyond Howard?")
print("      original markers per lecture: %s" % {k: v for k, v in o.items()})

out = {"run_date": "2026-10-07", "set": "MIT 6.100L held-out",
       "hand_scores_locked_before_run": True,
       "P1_revised_reproduces_hand_ranking": rev_ok,
       "P1_original_reproduces_hand_ranking": org_ok,
       "lectures": rows}
os.makedirs("/home/claude/week6/results", exist_ok=True)
with open("/home/claude/week6/results/heldout_2026-10-07.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwritten: results/heldout_2026-10-07.json")
