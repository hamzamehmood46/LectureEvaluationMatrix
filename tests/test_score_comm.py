"""Checks score_comm.py on a synthetic transcript with known counts.

    python tests/test_score_comm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "communication"))
import score_comm as sc  # noqa: E402


def make(sentences_words, gap=0.0, start_gap_after=None):
    """Build a transcript: one segment per sentence, words 0.5 s long."""
    segs, text, t = [], [], 0.0
    for k, sent in enumerate(sentences_words):
        words, seg_start = [], t
        for w in sent.split():
            words.append([round(t, 2), round(t + 0.5, 2), " " + w])
            t += 0.5
        segs.append({"start": seg_start, "end": round(t, 2), "text": sent, "words": words})
        text.append(sent)
        t += gap if start_gap_after is None or k == start_gap_after else 0.0
    return {"id": "T1", "duration_s": round(t, 1), "segments": segs, "text": " ".join(text)}


def test_counts():
    t = make(["So um today we will look at this equation on the board.",
              "You know the the answer is simple, for example two plus two?",
              "I mean we we can write down a proof."], gap=2.0)
    r = sc.score(t)
    assert r["words"] == 33, r["words"]
    assert r["m2_fillers_per_1000"] == round(1000 * 3 / 33, 2), r      # um, you know, i mean
    assert r["m6_repetitions_per_1000"] == round(1000 * 2 / 33, 2), r  # the the, we we
    assert r["m7_signposts_per_1000"] == round(1000 * 1 / 33, 2), r    # "today we"
    assert r["x_example_cues_per_1000"] == round(1000 * 1 / 33, 2), r  # "for example"
    assert r["v_deixis_per_1000"] == round(1000 * 2 / 33, 2), r        # "this equation", "on the board"
    assert r["v_board_cues_per_1000"] > 0, r                           # "on the board", "write down"
    assert r["m3_mean_sentence_words"] == round(33 / 3, 1), r
    assert abs(r["m4_long_pauses_per_10min"] - 2 / (t["duration_s"] / 600)) < 0.01, r  # two 2 s gaps


def test_no_pauses_and_rate():
    t = make(["alpha beta gamma delta epsilon zeta eta theta iota kappa."] * 6)
    r = sc.score(t)
    assert r["m4_long_pauses_per_10min"] == 0, r
    assert r["m1_words_per_min"] == 120.0, r  # 0.5 s per word


def test_windows():
    text = " ".join(f"w{i}" for i in range(10000))
    ws = sc.windows(text)
    assert len(ws) == 5 and all(len(w.split()) == 1000 for w in ws)
    assert ws[0].startswith("w1000 ") and ws[2].startswith("w5000 ")
    short = sc.windows(" ".join(f"w{i}" for i in range(300)))
    assert all(len(w.split()) == 300 for w in short)


def test_mtld_orders_diversity():
    repetitive = ("the cat sat on the mat " * 40).split()
    diverse = [f"word{i}" for i in range(240)]
    assert sc.mtld(repetitive) < sc.mtld(diverse)
    assert sc.mtld(["a"] * 10) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

