"""
Builds a 200-second synthetic lecture video with known ground truth, to test
Stage 1 change detection, OCR and alignment without needing a real recording.

Includes the cases that break naive detectors:
  - code typed in line by line (builds that should NOT start new segments)
  - a moving laser-pointer dot (should be ignored)
  - a webcam inset with a moving 'instructor' (tests the --mask option)
  - a dark terminal slide (tests OCR inversion)
  - two slides with the same title and layout but different bullets (hard change)
  - a return to an earlier slide (should count as a change)
  - sensor noise plus H.264 compression at a low bitrate
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DUR = 1280, 720, 10, 200
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
f_title = ImageFont.truetype(BOLD, 46)
f_body = ImageFont.truetype(SANS, 32)
f_code = ImageFont.truetype(MONO, 28)

CODE = [
    "def total_marks(scores):",
    "    total = 0",
    "    for s in scores:",
    "        if s < 0:",
    "            continue",
    "        total += s",
    "    return total",
    "",
    "marks = [78, 91, -1, 64]",
    "print(total_marks(marks))",
]
TERMINAL = ["$ python marks.py", "233", "$ python -c \"print(sum([78, 91, 64]))\"", "233",
            "$ python marks.py --debug", "skipping invalid score -1", "233"]
BULLETS_A = ["Loops repeat a block of code", "A for loop visits each list element",
             "The loop variable changes every pass", "Accumulators collect a running total"]
BULLETS_B = ["Lists are mutable sequences", "append adds an element at the end",
             "Index assignment replaces one element", "Aliasing shares one list object"]

# (start, end, kind, payload); ground-truth changes are every start after 0
SLIDES = [
    (0, 25, "title", ("Lecture 7: Loops and Lists", "Introduction to Programming")),
    (25, 55, "bullets", ("Iteration", BULLETS_A)),
    (55, 95, "code", None),
    (95, 125, "terminal", None),
    (125, 150, "diagram", None),
    (150, 170, "bullets", ("Iteration", BULLETS_B)),
    (170, 200, "code_full", None),
]


def slide_text(kind, payload, n_lines=len(CODE)):
    if kind == "title":
        return " ".join(payload)
    if kind == "bullets":
        return payload[0] + " " + " ".join(payload[1])
    if kind in ("code", "code_full"):
        return "Worked example " + " ".join(CODE[:n_lines])
    if kind == "terminal":
        return " ".join(TERMINAL)
    if kind == "diagram":
        return "Memory model marks list 78 91 -1 64 total"


def render(kind, payload, t, t0):
    dark = kind == "terminal"
    img = Image.new("RGB", (W, H), (24, 24, 28) if dark else (250, 250, 247))
    d = ImageDraw.Draw(img)
    ink = (225, 225, 225) if dark else (20, 20, 20)
    if kind == "title":
        d.text((120, 260), payload[0], font=f_title, fill=ink)
        d.text((120, 340), payload[1], font=f_body, fill=(70, 70, 70))
    elif kind == "bullets":
        d.text((80, 60), payload[0], font=f_title, fill=ink)
        for i, b in enumerate(payload[1]):
            d.text((110, 170 + i * 70), "• " + b, font=f_body, fill=ink)
        if kind == "bullets" and payload[1] is BULLETS_A:      # laser pointer
            x = 300 + int(250 * np.sin((t - t0) * 0.8))
            d.ellipse((x, 250, x + 14, 264), fill=(230, 30, 30))
    elif kind in ("code", "code_full"):
        d.text((80, 50), "Worked example", font=f_title, fill=ink)
        n = len(CODE) if kind == "code_full" else min(len(CODE), 1 + int((t - t0) / 4))
        for i, line in enumerate(CODE[:n]):
            d.text((100, 140 + i * 40), line, font=f_code, fill=ink)
    elif kind == "terminal":
        for i, line in enumerate(TERMINAL):
            d.text((60, 60 + i * 44), line, font=f_code, fill=ink)
    elif kind == "diagram":
        d.text((80, 50), "Memory model", font=f_title, fill=ink)
        d.rectangle((100, 200, 260, 260), outline=ink, width=3)
        d.text((125, 212), "marks", font=f_body, fill=ink)
        for i, v in enumerate(["78", "91", "-1", "64"]):
            x = 420 + i * 120
            d.rectangle((x, 200, x + 100, 260), outline=ink, width=3)
            d.text((x + 25, 212), v, font=f_body, fill=ink)
        d.line((260, 230, 420, 230), fill=ink, width=3)
        d.polygon([(420, 230), (405, 222), (405, 238)], fill=ink)
        d.rectangle((100, 380, 260, 440), outline=ink, width=3)
        d.text((125, 392), "total", font=f_body, fill=ink)
    return img


def add_webcam(img, t):
    d = ImageDraw.Draw(img)
    x0, y0 = int(W * 0.78), int(H * 0.72)
    d.rectangle((x0, y0, W - 10, H - 10), fill=(90, 110, 130))
    cx = x0 + 70 + int(40 * np.sin(t * 1.3))
    d.ellipse((cx, y0 + 25, cx + 70, y0 + 95), fill=(200, 170, 140))      # head
    d.rectangle((cx - 15, y0 + 100, cx + 85, H - 10), fill=(40, 60, 90))  # body
    return img


def main(out_mp4, out_truth):
    rng = np.random.default_rng(7)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "32",
         "-pix_fmt", "yuv420p", out_mp4], stdin=subprocess.PIPE)
    for k in range(DUR * FPS):
        t = k / FPS
        s = next(s for s in SLIDES if s[0] <= t < s[1])
        img = add_webcam(render(s[2], s[3], t, s[0]), t)
        arr = np.asarray(img).astype(np.int16)
        arr = np.clip(arr + rng.normal(0, 4, arr.shape), 0, 255).astype(np.uint8)
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()

    truth = {"changes": [s[0] for s in SLIDES[1:]],
             "build_events": [55 + 4 * i for i in range(1, len(CODE))],
             "slides": [{"start": s[0], "end": s[1], "kind": s[2],
                         "text": slide_text(s[2], s[3])} for s in SLIDES]}
    json.dump(truth, open(out_truth, "w"), indent=2)
    print("wrote", out_mp4, out_truth)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
