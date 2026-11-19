"""Deterministic communication metrics for one lecture transcript.

Input is a transcript JSON from transcribe.py (segments with word timestamps).
Every word list and threshold below is part of the frozen measure definition
(see preregistration/PREREGISTRATION_2026-11-14.txt and its 2026-11-19
addendum). Change them only through a new dated addendum.

    python communication/score_comm.py L001 --data-dir C:/Users/hamza/lecture_data

Per the preregistration, do NOT run this on the real transcripts until the
hand-scores are frozen. tests/test_score_comm.py checks it on synthetic text.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

LONG_PAUSE_S = 1.0
WINDOW_WORDS = 1000
WINDOW_AT = (0.10, 0.30, 0.50, 0.70, 0.90)

FILLER_TOKENS = {"um", "uh", "er", "ah", "hmm", "umm", "uhh"}
FILLER_PHRASES = ["you know", "i mean"]

# Organisation: explicit signposting of structure.
SIGNPOST = ["today we", "last time", "to recap", "let's recap", "to summarize", "in summary",
            "in conclusion", "we're going to", "we are going to", "the plan", "first of all",
            "next,", "finally,", "second,", "third,", "so far"]
# Examples: cues that an example or analogy follows.
EXAMPLE_CUES = ["for example", "for instance", "suppose", "imagine", "let's say", "say we have",
                "consider the"]
# Visual dependency: speech that only makes sense with the screen or board in view.
DEIXIS = ["as you can see", "you can see here", "over here", "right here", "look at this",
          "look at that", "this slide", "this graph", "this picture", "this equation",
          "this diagram", "this figure", "on the board", "up here", "down here", "this one here",
          "you see here", "see this"]
# Writing or drawing in front of the class.
BOARD_CUES = ["let me write", "i'll write", "i will write", "write down", "let me draw", "i'll draw",
              "i will draw", "on the board", "writing this", "let me put", "sketch"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def words_of(t):
    return [(s, e, w.strip()) for seg in t["segments"] for s, e, w in seg["words"] if w.strip()]


def norm_tokens(text):
    return re.findall(r"[a-z']+", text.lower())


def count_phrases(text, phrases):
    """Non-overlapping whole-word matches, longest phrase first, so one stretch of speech counts once."""
    pat = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    low = re.sub(r"\s+", " ", text.lower())
    return len(re.findall(rf"(?<![a-z'])(?:{pat})(?![a-z'])", low))


def per_1000(n, words):
    return 1000.0 * n / words if words else 0.0


def speech_seconds(t):
    return sum(max(0.0, seg["end"] - seg["start"]) for seg in t["segments"])


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def mtld(tokens, ttr=0.72):
    """Measure of textual lexical diversity (McCarthy and Jarvis), mean of both directions."""
    def one(seq):
        factors, seen, count = 0.0, set(), 0
        for w in seq:
            count += 1
            seen.add(w)
            if len(seen) / count <= ttr:
                factors += 1
                seen, count = set(), 0
        if count:
            cur = len(seen) / count
            factors += (1 - cur) / (1 - ttr)
        return len(seq) / factors if factors else float(len(seq))
    return (one(tokens) + one(tokens[::-1])) / 2 if len(tokens) >= 50 else None


def repetition_count(tokens):
    n = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    n += sum(1 for i in range(len(tokens) - 3)
             if tokens[i:i + 2] == tokens[i + 2:i + 4] and tokens[i] != tokens[i + 1])
    return n


def pace_variability(words, window_s=60.0, min_words=20):
    if not words:
        return None
    counts = {}
    for s, _, _ in words:
        counts[int(s // window_s)] = counts.get(int(s // window_s), 0) + 1
    vals = [c for c in counts.values() if c >= min_words]
    return statistics.pstdev(vals) if len(vals) >= 3 else None


def windows(text, size=WINDOW_WORDS, at=WINDOW_AT):
    """Fixed excerpts for the judge: `size` words starting at each fraction of the lecture."""
    toks = text.split()
    out = []
    for f in at:
        start = min(int(f * len(toks)), max(0, len(toks) - size))
        out.append(" ".join(toks[start:start + size]))
    return out


def score(t):
    text = t["text"]
    ws = words_of(t)
    n = len(ws)
    toks = norm_tokens(text)
    sp = speech_seconds(t)
    dur = t["duration_s"]
    gaps = [b[0] - a[1] for a, b in zip(ws, ws[1:])]
    sents = sentences(text)
    fillers = sum(1 for x in toks if x in FILLER_TOKENS) + count_phrases(text, FILLER_PHRASES)
    return {
        "id": t["id"],
        "words": n,
        "duration_min": round(dur / 60, 1),
        "speech_fraction": round(sp / dur, 3) if dur else None,
        "m1_words_per_min": round(n / (sp / 60), 1) if sp else None,
        "m2_fillers_per_1000": round(per_1000(fillers, n), 2),
        "m3_mean_sentence_words": round(n / len(sents), 1) if sents else None,
        "m4_long_pauses_per_10min": round(sum(g >= LONG_PAUSE_S for g in gaps) / (dur / 600), 2) if dur else None,
        "m6_repetitions_per_1000": round(per_1000(repetition_count(toks), n), 2),
        "m7_signposts_per_1000": round(per_1000(count_phrases(text, SIGNPOST), n), 2),
        "m8_mtld": round(mtld(toks), 1) if mtld(toks) else None,
        "m9_pace_sd_words_per_min": round(pace_variability(ws), 1) if pace_variability(ws) else None,
        "q_questions_per_10min": round(sum(s.endswith("?") for s in sents) / (dur / 600), 2) if dur else None,
        "x_example_cues_per_1000": round(per_1000(count_phrases(text, EXAMPLE_CUES), n), 2),
        "v_deixis_per_1000": round(per_1000(count_phrases(text, DEIXIS), n), 2),
        "v_board_cues_per_1000": round(per_1000(count_phrases(text, BOARD_CUES), n), 2),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ids", nargs="+")
    p.add_argument("--data-dir", required=True)
    a = p.parse_args()
    for i in a.ids:
        print(json.dumps(score(load(Path(a.data_dir) / "transcripts" / f"{i}.json")), indent=1))
