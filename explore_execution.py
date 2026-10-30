"""EXPLORATORY — generated after seeing P1 fail. Not a test; needs a new held-out set.

Hypothesis from the failure: the rubric's Implementation dimension rewards code
being RUN and its output observed (IM1/IM2). Code dictation density measures code
being SPOKEN. Lecture 17 is dense with spoken code but never executes any, which
is precisely why it hand-scored lower while scoring high on density.

If execution leaves its own surface trace - an instructor reporting what appeared
on screen - then a separate 'execution evidence' signal should separate them.
"""
import re, io, os

EXECUTION_EVIDENCE = [
    r"\bit gave me\b", r"\bgives me\b", r"\band it does\b", r"\bit does\b",
    r"\bwe run it\b", r"\bif (?:i|we) run\b", r"\blet'?s run\b", r"\brun it\b",
    r"\bso i (?:just )?ran\b", r"\bit print(?:s|ed)\b", r"\bthis print(?:s|ed)\b",
    r"\bwill print\b", r"\bit (?:would )?show(?:s|ed)? me\b", r"\bin the console\b",
    r"\bthe output is\b", r"\bit('?s| is) going to print\b", r"\bwe get back\b",
    r"\blooks? (?:much )?better\b", r"\bperfect\b", r"\bstill works\b",
    r"\bit works\b", r"\bthat worked\b",
]

T = "/home/claude/week6/transcripts"
LECT = [("L3_iteration","lzYrMcI92kw",3.0), ("L10_lists","Z4bm7xzYpKM",3.0),
        ("L17_classes","tsMrUdynbQQ",2.0)]

print("EXPLORATORY: execution-evidence signal (hypothesis-generating, NOT a test)")
print("-" * 78)
print("%-14s %7s %6s %12s %12s" % ("lecture","words","hand","exec hits","per 1k"))
vals = {}
for key, vid, hand in LECT:
    t = io.open(os.path.join(T, vid + ".txt"), encoding="utf-8").read()
    tl = t.lower(); w = len(t.split())
    n = sum(len(re.findall(p, tl)) for p in EXECUTION_EVIDENCE)
    vals[key] = n / w * 1000
    print("%-14s %7d %6.1f %12d %12.2f" % (key, w, hand, n, vals[key]))

print()
pairs = [("L3_iteration","L17_classes"), ("L10_lists","L17_classes")]
ok = all(vals[a] > vals[b] for a,b in pairs)
for a,b in pairs:
    print("  %-14s vs %-12s  %6.2f vs %6.2f -> %s" % (
        a,b,vals[a],vals[b], "correct" if vals[a]>vals[b] else "WRONG"))
print()
print("  separates the hand ranking on THIS set: %s" % ok)
print()
print("  This set was used to form the hypothesis, so the result above is not")
print("  evidence. It only says the idea is worth a proper held-out test.")
