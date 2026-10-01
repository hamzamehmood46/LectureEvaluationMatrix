"""
Stage 2b - revised signal extraction, with one measure kept and one abandoned.

Background
----------
The 2026-09-23 run showed the original Stage 2 proxies inverting the ranking of
the two Howard content lectures on Implementation:

                        hand Impl.   impl. markers
  2D Lists / loops        2.0/3.0          1
  Classes / OOP           2.5/3.0          0      <- higher by hand, zero markers

Diagnosis: the marker lists detect ANNOUNCEMENTS ("let's write this"), not
content. An instructor who dictates code directly - "self dot name equals name"
- announces nothing and scores zero.

Two replacements were attempted. One works. One does not, and the reason it
does not is itself a result.

KEPT: code_dictation_density
----------------------------
Code has a distinctive surface form when spoken aloud: "dot", "equals",
"bracket", "underscore", plus language keywords and literal tokens that survive
into captions. This fires on dictation regardless of whether it was announced,
which is precisely the failure mode above.

ABANDONED: lexical example-span estimation
------------------------------------------
The intent was to measure how much of a transcript sits INSIDE a developed
example, rather than how often one is announced, so that a single sustained
example would score correctly.

Three variants were implemented and tested (fixed word cap; discourse-boundary
termination; anchor-term recurrence with windows of 6/8/12/25). All three failed
the same discrimination test: a sustained example and a passage of repetitive
filler both saturate at a span ratio of 1.000, and in the anchor-term variant
the filler scored HIGHER than the genuine example at every window size.

The reason is not a tuning problem. An example is a semantic relation - a
concrete instantiation of an abstract concept - and not a surface form. Lexical
recurrence cannot separate "the seating chart recurs because it is being
developed" from "this phrase recurs because the speaker is repetitive." No
choice of window or threshold reaches a distinction that is not present in the
lexical signal.

This is consistent with the published literature. Whitehill et al. (arXiv
2404.02444) report that language models match human rater agreement on discrete,
LOW-INFERENCE variables while efficacy diminishes on high-inference practices -
and example quality is high-inference. Conversely, the two-step GPT-4 method of
the explanatory-sequences study (J. Math. Teacher Educ., 2026) reached 92.4%
agreement with an expert coding team on segmenting explanatory sequences, the
same task attempted lexically here.

Conclusion carried into the design: example detection belongs in the Stage 3
judge, not in a lexical pre-pass. The function below is retained only so the
negative result stays reproducible, and is reported under "abandoned".

Usage
-----
    python improved_signals.py <video_id_or_url> [syllabus_keywords.txt]
    python improved_signals.py --transcript transcripts/foo.txt [keywords.txt]
    python improved_signals.py --self-test
"""

import json
import os
import re
import sys

try:
    from lecture_rating_pipeline import (
        fetch_transcript,
        extract_video_id,
        EXAMPLE_MARKERS,
        IMPLEMENTATION_MARKERS,
        _count_markers,
        topic_coverage_signal,
    )
except ImportError:  # pragma: no cover
    raise SystemExit("Run this from the repository root, beside lecture_rating_pipeline.py")


# --------------------------------------------------------------------------
# KEPT MEASURE: code dictation density
# --------------------------------------------------------------------------

CODE_PUNCTUATION_WORDS = [
    r"\bdot\b", r"\bequals\b", r"\bopen paren(?:thesis)?\b", r"\bclose paren(?:thesis)?\b",
    r"\bbracket\b", r"\bbrackets\b", r"\bunderscore\b", r"\bcolon\b", r"\bsemicolon\b",
    r"\bquote\b", r"\bindent(?:ation|ed)?\b", r"\bparentheses\b",
]

CODE_KEYWORDS_AMBIGUOUS = [
    r"\bdef\b", r"\breturn\b", r"\bprint\b", r"\bimport\b", r"\bself\b",
    r"\bclass\b", r"\brange\b", r"\blen\b", r"\bappend\b", r"\binit\b",
    r"\bfor loop\b", r"\bwhile loop\b", r"\bif statement\b",
    r"\bfunction call\b", r"\bargument\b", r"\bparameter\b", r"\bvariable\b",
    r"\bstring\b", r"\binteger\b", r"\bboolean\b", r"\bdictionary\b",
]

CODE_LITERAL_PATTERNS = [
    r"[A-Za-z_][A-Za-z0-9_]*\s*\[\s*\d+\s*\]",
    r"[A-Za-z_][A-Za-z0-9_]{1,}\.[A-Za-z_][A-Za-z0-9_]{1,}",
    r"[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\)",
    r"==|!=|\+=|->",
]


def code_dictation_signal(transcript):
    """Density of spoken-code register per 1000 words.

    Reported component-wise so a reader can see which part carries the signal
    rather than trusting one opaque number.
    """
    text_l = transcript.lower()
    words = len(transcript.split())

    punct = sum(len(re.findall(p, text_l)) for p in CODE_PUNCTUATION_WORDS)
    kw = sum(len(re.findall(p, text_l)) for p in CODE_KEYWORDS_AMBIGUOUS)
    literal = sum(len(re.findall(p, transcript)) for p in CODE_LITERAL_PATTERNS)

    # Headline metric uses HIGH-SPECIFICITY evidence only: spoken punctuation and
    # literal code tokens. Language keywords are reported but excluded, because
    # the 2026-10-01 run showed them producing false positives through polysemy -
    # "class" fired three times in an announcements clip in the sense of a course,
    # ranking a no-content clip above a genuine lecture. Words like class, list,
    # string and object are ordinary English in an educational setting, which is
    # exactly the domain this runs in.
    total = punct + literal

    return {
        "code_punctuation_words": punct,
        "code_literals": literal,
        "code_evidence_total": total,
        "code_dictation_density_per_1000_words": round(total / words * 1000, 2) if words else 0.0,
        "ambiguous_keywords_excluded": kw,
        "code_tokens_total": total,
    }


# --------------------------------------------------------------------------
# ABANDONED MEASURE: retained so the negative result stays reproducible
# --------------------------------------------------------------------------

DISCOURSE_BOUNDARIES = [
    r"\bokay,? so\b", r"\ball right,? so\b", r"\bnow,? ", r"\bmoving on\b",
    r"\bnext,? ", r"\blet'?s move\b", r"\bso that'?s\b", r"\bany questions\b",
    r"\bto summar(?:ise|ize)\b", r"\bin summary\b",
]

STOPWORDS = set("""a an the and or but if then than that this these those is are was were be been being
have has had do does did will would can could should may might must of in on at to for with from by as
it its we you they he she i me my our your their them us so now just like get got go going one two
there here what when where which who how why not no yes very really much many more most some any all
each every into out up down over under again back new well also too only own same other another thing
things way ways make makes made take takes took see sees saw know knows knew think thinks let lets""".split())


def _tokens(text):
    return [(m.group(0), m.start()) for m in re.finditer(r"\S+", text)]


def _norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def example_span_signal_ABANDONED(transcript, cap=200, anchor_window=25, probe=25, min_words=25):
    """Lexical example-span estimation. DOES NOT DISCRIMINATE - see module docstring.

    Kept for reproducibility of the negative result only. Not used in scoring.
    """
    toks = _tokens(transcript)
    n = len(toks)
    if n == 0:
        return {"example_spans": 0, "example_span_words": 0, "example_span_ratio": 0.0,
                "mean_span_words": 0.0, "status": "abandoned - does not discriminate"}

    text_l = transcript.lower()
    starts = [pos for _, pos in toks]

    def widx(cpos):
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] < cpos:
                lo = mid + 1
            else:
                hi = mid
        return lo

    markers = sorted({widx(m.start()) for p in EXAMPLE_MARKERS for m in re.finditer(p, text_l)})
    boundaries = sorted({widx(m.start()) for p in DISCOURSE_BOUNDARIES for m in re.finditer(p, text_l)})

    spans = []
    for start in markers:
        anchors = {_norm(w) for w, _ in toks[start:start + anchor_window]
                   if _norm(w) and _norm(w) not in STOPWORDS and len(_norm(w)) > 2}
        if not anchors:
            continue
        end = min(start + anchor_window, n)
        while end < n and (end - start) < cap:
            nxt = min(end + probe, n)
            if any(b for b in boundaries if end <= b < nxt):
                break
            if not ({_norm(w) for w, _ in toks[end:nxt]} & anchors):
                break
            end = nxt
        if end - start >= min_words:
            spans.append((start, end))

    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    span_words = sum(e - s for s, e in merged)
    return {
        "example_spans": len(merged),
        "example_span_words": span_words,
        "example_span_ratio": round(span_words / n, 3),
        "mean_span_words": round(span_words / len(merged), 1) if merged else 0.0,
        "status": "abandoned - does not discriminate, see module docstring",
    }


# --------------------------------------------------------------------------

def improved_signals(transcript, syllabus_keywords=None):
    words = len(transcript.split())
    ex = _count_markers(transcript, EXAMPLE_MARKERS)
    im = _count_markers(transcript, IMPLEMENTATION_MARKERS)

    result = {
        "word_count": words,
        "original_stage2": {
            "example_marker_count": ex,
            "example_markers_per_1000_words": round(ex / words * 1000, 2) if words else 0,
            "implementation_marker_count": im,
            "implementation_markers_per_1000_words": round(im / words * 1000, 2) if words else 0,
        },
        "revised_stage2b": code_dictation_signal(transcript),
        "abandoned": example_span_signal_ABANDONED(transcript),
    }
    if syllabus_keywords:
        result["topic_coverage"] = topic_coverage_signal(transcript, syllabus_keywords)
    return result


# --------------------------------------------------------------------------
# Self-test. The 2026-09-23 report flagged that the pattern list had never been
# exercised against text containing every marker, which is how the e.g. bug
# survived. This fixture covers all of them, plus the discrimination test that
# the abandoned measure fails.
# --------------------------------------------------------------------------

FIXTURE_EXAMPLE = (
    "For example this. Let's say that. Suppose so. Imagine this. "
    "Consider this case. Consider the case here. As an example, take it. "
    "We use lists, e.g. a seating chart. Take this case now. "
    "Here's an example to finish."
)

FIXTURE_IMPLEMENTATION = (
    "Let's write it. Let's code it. Let's type it. Let's run it. Let's implement it. "
    "Run this now. On the screen you see it. On your screen too. The output is five. "
    "This prints nothing. Let's open the file. In vscode we start. In the terminal next. "
    "In jupyter also. Debugging now. We were debug. A syntax error appears. A traceback follows."
)


def self_test():
    failures, notes = [], []

    for label, patterns, text in (
        ("EXAMPLE", EXAMPLE_MARKERS, FIXTURE_EXAMPLE),
        ("IMPLEMENTATION", IMPLEMENTATION_MARKERS, FIXTURE_IMPLEMENTATION),
    ):
        for pat in patterns:
            if not re.search(pat, text.lower()):
                failures.append("%s pattern never fires: %s" % (label, pat))

    if not re.search(r"\be\.g\.", "we use lists, e.g. a seating chart"):
        failures.append("regression: e.g. marker does not match in ordinary prose")

    # The substantive claim: the measure detects dictated code that the original
    # markers miss entirely. The previous threshold of 3 was arbitrary and was
    # written for the old composite metric; the claim being tested is detection
    # where the markers return nothing, not a particular count.
    dictated = "self dot name equals name and then def take damage with an argument"
    tokens = code_dictation_signal(dictated)["code_evidence_total"]
    markers = _count_markers(dictated, IMPLEMENTATION_MARKERS)
    if tokens <= 0:
        failures.append("code dictation measure missed plainly dictated code")
    if markers != 0:
        failures.append("fixture assumption broken: dictated code should announce nothing")
    notes.append("dictated code -> %d code evidence, %d original markers (the inversion case)"
                 % (tokens, markers))

    # Literal attribute access must be caught, and must not false-match on
    # ordinary abbreviations. "p.m" produced a false literal match on 2026-10-01.
    with_literal = "we can read woody.strength and woody.health directly"
    if code_dictation_signal(with_literal)["code_literals"] < 2:
        failures.append("literal attribute access not detected")
    if code_dictation_signal("the talk is at 5 p.m in the reading room")["code_literals"] != 0:
        failures.append("regression: abbreviation false-matches as attribute access")

    # Polysemy regression. On 2026-10-01 the word "class" in the sense of a
    # course caused an announcements clip with no code to out-rank a genuine
    # lecture. Course-sense text must score zero evidence.
    course_sense = ("if you are in both classes the grade counts for our class "
                    "and my class will also use this project")
    ev = code_dictation_signal(course_sense)["code_evidence_total"]
    if ev != 0:
        failures.append("regression: course-sense text scores %d code evidence, "
                        "expected 0 (polysemy false positive has returned)" % ev)
    notes.append("course-sense 'class' text -> %d code evidence (must be 0)" % ev)

    # The measure that does not work, asserted as a KNOWN failure so that a
    # future fix is detected rather than silently assumed.
    sustained = ("Let's say we have a seating chart in the classroom. " +
                 "Each row of the seating chart is a list of seats. " * 20)
    scattered = ("For example a seating chart. " +
                 "Completely unrelated filler material follows. " * 12 +
                 "Suppose a different thing. " +
                 "More unrelated filler material follows here. " * 12)
    s = example_span_signal_ABANDONED(sustained)["example_span_ratio"]
    c = example_span_signal_ABANDONED(scattered)["example_span_ratio"]
    notes.append("abandoned span measure: sustained=%.3f scattered=%.3f" % (s, c))
    if s > c + 0.15:
        failures.append("span measure now discriminates (%.3f vs %.3f) - revisit the "
                        "abandonment decision and update the module docstring" % (s, c))

    for n in notes:
        print(" ", n)
    if failures:
        print("\nFAIL (%d)" % len(failures))
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll checks passed (including the asserted known failure).")
    return 0


# --------------------------------------------------------------------------

def main(argv):
    if "--self-test" in argv:
        return self_test()
    if not argv:
        print("usage: python improved_signals.py <video_id|--transcript FILE> [keywords.txt]")
        return 1

    if argv[0] == "--transcript":
        transcript = open(argv[1], encoding="utf-8").read()
        rest = argv[2:]
    else:
        transcript = fetch_transcript(argv[0])
        rest = argv[1:]
        try:
            os.makedirs("transcripts", exist_ok=True)
            path = os.path.join("transcripts", extract_video_id(argv[0]) + ".txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print("# transcript saved to %s" % path, file=sys.stderr)
        except OSError:
            pass

    keywords = None
    if rest:
        with open(rest[0], encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]

    print(json.dumps(improved_signals(transcript, keywords), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
