"""
Stage 3 front end: classify each Stage 1 keyframe as a text slide, code,
terminal/output or diagram, and decide which ones go to the vision-language model.

Why: a 75-minute lecture yields roughly 60-150 screen states. Sending all of them
to a VLM is slow on local hardware, and for plain text slides the OCR text already
carries the content. Code, program output and diagrams are where the VLM is needed
(the October 23 test showed OCR recovers only a quarter of a diagram's words).

The classifier is rule-based and uses no learned weights, so it runs on a laptop
and its decisions can be explained line by line.
  - terminal: shell or REPL prompts, notebook In/Out markers, tracebacks
  - code: lines with programming syntax (keywords, brackets, assignments, line numbers)
  - diagram: much of the foreground ink lies outside any word OCR found
  - text: everything else
"""
import re

import cv2
import numpy as np

PROMPT = re.compile(r"^\s*(\$\s|>>>|\.\.\.\s|[A-Za-z]:\\[^>]*>|In \[\d+\]|Out\[\d+\]|PS [A-Z]:\\)")
TRACE = re.compile(r"(Traceback|Error:|Exception|line \d+, in)")
CODE = re.compile(
    r"(\bdef\b|\bclass\b|\breturn\b|\bimport\b|\bwhile\b|\bfor\b.*\bin\b|\bif\b.*:|\belif\b|\belse:|"
    r"[A-Za-z_]\w*\s*=\s*\S|\w+\([^)]*\)|[{};]\s*$|#include|\bpublic\b|\bint\b|//|\+=|==|<=|>=)")
LINE_NO = re.compile(r"^\s*\d{1,3}\s+\S")


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def ocr(img):
    """Returns (text, word boxes with confidence > 40) using Tesseract."""
    import pytesseract
    g = _gray(img)
    if np.median(g) < 110:
        g = 255 - g
    data = pytesseract.image_to_data(g, config="--psm 6", output_type=pytesseract.Output.DICT)
    lines, boxes = {}, []
    for i, w in enumerate(data["text"]):
        if not w.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(w)
        if float(data["conf"][i]) > 40:
            boxes.append((data["left"][i], data["top"][i], data["width"][i], data["height"][i]))
    text = "\n".join(" ".join(v) for _, v in sorted(lines.items()))
    return text, boxes


def features(img, text=None, boxes=None):
    g = _gray(img)
    if text is None:
        text, boxes = ocr(img)
    bg = int(np.median(g))
    ink = np.abs(g.astype(np.int16) - bg) > 60
    covered = np.zeros_like(ink)
    for x, y, w, h in boxes or []:
        covered[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4] = True
    ink_total = int(ink.sum())
    lines = [l for l in text.splitlines() if len(l.strip()) > 1]
    return {
        "dark": bg < 110,
        "lines": len(lines),
        "prompt_lines": sum(bool(PROMPT.search(l)) for l in lines),
        "trace_lines": sum(bool(TRACE.search(l)) for l in lines),
        "code_lines": sum(bool(CODE.search(l)) for l in lines),
        "numbered_lines": sum(bool(LINE_NO.search(l)) for l in lines),
        "words": len(boxes or []),
        "nontext_ink": round(float((ink & ~covered).sum()) / max(ink_total, 1), 3),
    }


def classify(f):
    """Rules, applied in order. Thresholds were set on the dev split only."""
    if f["prompt_lines"] >= 2 or f["trace_lines"] >= 2 or \
            (f["prompt_lines"] >= 1 and f["prompt_lines"] / max(f["lines"], 1) >= 0.3):
        return "terminal"
    if f["lines"] and (f["code_lines"] >= 3 and f["code_lines"] / f["lines"] >= 0.4
                       or f["numbered_lines"] >= 4):
        return "code"
    if f["nontext_ink"] >= 0.45 or (f["words"] <= 10 and f["nontext_ink"] >= 0.10):
        return "diagram"
    return "text"


def route(label, f):
    """True if the VLM should see this keyframe."""
    return label in ("code", "terminal", "diagram") or f["words"] < 5


def apply_mask(img, mask):
    """Blank a region given as fractions (x0, y0, x1, y1), e.g. a webcam inset."""
    if not mask:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = mask
    out = img.copy()
    out[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = np.median(img.reshape(-1, 3), axis=0)
    return out


def classify_image(path, mask=None):
    img = apply_mask(cv2.imread(path), mask)
    f = features(img)
    label = classify(f)
    return label, route(label, f), f
