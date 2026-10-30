"""
Generates labelled lecture screen images in four classes for testing the
Stage 3 screen classifier: text slide, code, terminal/output, diagram.

Two splits are built so the classifier is tuned on one and judged on the other:
  dev   - DejaVu fonts, mild noise
  test  - unseen font families (Liberation, FreeFont, Carlito/Caladea), unseen
          content templates mixed in, and harsher degradation (downscaled to
          480p and back, stronger noise, JPEG compression, slight blur)
Usage: python make_screen_set.py OUT_DIR SPLIT N
"""
import csv
import io
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
F = "/usr/share/fonts/truetype/"
FONTS = {
    "dev": {"sans": [F + "dejavu/DejaVuSans.ttf", F + "dejavu/DejaVuSerif.ttf"],
            "bold": [F + "dejavu/DejaVuSans-Bold.ttf"],
            "mono": [F + "dejavu/DejaVuSansMono.ttf"]},
    "test": {"sans": [F + "liberation/LiberationSans-Regular.ttf", F + "freefont/FreeSerif.ttf",
                      F + "crosextra/Carlito-Regular.ttf", F + "crosextra/Caladea-Regular.ttf"],
             "bold": [F + "liberation/LiberationSans-Bold.ttf", F + "crosextra/Carlito-Bold.ttf"],
             "mono": [F + "liberation/LiberationMono-Regular.ttf", F + "freefont/FreeMono.ttf"]},
}

TOPICS = ["Recursion", "Hash tables", "Binary search", "Loops", "Classes and objects",
          "Sorting", "Graphs", "Dynamic programming", "File input and output", "Exceptions",
          "Stacks and queues", "Big-O notation", "Linked lists", "Dictionaries", "Testing"]
BULLETS = ["A function can call itself", "Every call needs a base case", "Keys map to values",
           "Lookups take constant time on average", "Split the search range in half",
           "Objects bundle data and behaviour", "Each pass moves one element into place",
           "Nodes are connected by edges", "Store answers to subproblems", "Always close the file",
           "Catch only the errors you expect", "The last item in is the first out",
           "Count the steps as input grows", "Each node points to the next one",
           "Write the test before the fix", "Iteration repeats a block of code",
           "Mutable objects can change in place", "Inheritance reuses an existing class"]
CODE_PY = [
    ["def factorial(n):", "    if n <= 1:", "        return 1", "    return n * factorial(n - 1)", "",
     "print(factorial(5))"],
    ["class Stack:", "    def __init__(self):", "        self.items = []", "    def push(self, x):",
     "        self.items.append(x)", "    def pop(self):", "        return self.items.pop()"],
    ["def binary_search(a, target):", "    lo, hi = 0, len(a) - 1", "    while lo <= hi:",
     "        mid = (lo + hi) // 2", "        if a[mid] == target:", "            return mid",
     "        elif a[mid] < target:", "            lo = mid + 1", "        else:", "            hi = mid - 1",
     "    return -1"],
    ["counts = {}", "for word in text.split():", "    counts[word] = counts.get(word, 0) + 1",
     "top = sorted(counts.items(), key=lambda kv: -kv[1])", "print(top[:5])"],
]
CODE_OTHER = [   # unseen languages, test split only
    ["public int sum(int[] xs) {", "    int total = 0;", "    for (int x : xs) {", "        total += x;",
     "    }", "    return total;", "}"],
    ["#include <stdio.h>", "int main(void) {", "    for (int i = 0; i < 3; i++) {",
     "        printf(\"%d\\n\", i);", "    }", "    return 0;", "}"],
]
TERM = [
    ["$ python factorial.py", "120", "$ python factorial.py 10", "3628800"],
    [">>> s = Stack()", ">>> s.push(4)", ">>> s.push(7)", ">>> s.pop()", "7", ">>> len(s.items)", "1"],
    ["$ python search.py", "Searching for 42 ...", "found at index 17", "$ python search.py 99",
     "Searching for 99 ...", "not found (-1)"],
    ["$ python wordcount.py notes.txt", "[('the', 41), ('a', 23), ('loop', 9), ('list', 8), ('is', 7)]"],
    ["Traceback (most recent call last):", "  File \"main.py\", line 4, in <module>",
     "    print(items[10])", "IndexError: list index out of range"],
]
TERM_OTHER = [   # unseen templates, test split only
    ["C:\\Users\\student> javac Sum.java", "C:\\Users\\student> java Sum", "Total: 15"],
    ["In [3]: total_marks([78, 91, 64])", "Out[3]: 233", "In [4]: len(marks)", "Out[4]: 4"],
]


def font(split, kind, size):
    return ImageFont.truetype(random.choice(FONTS[split][kind]), size)


def text_slide(split):
    bg = random.choice([(250, 250, 247), (255, 255, 255), (235, 240, 248), (30, 40, 70)])
    ink = (240, 240, 240) if sum(bg) < 300 else (25, 25, 25)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.text((80, 60), random.choice(TOPICS), font=font(split, "bold", random.randint(40, 56)), fill=ink)
    f = font(split, "sans", random.randint(26, 36))
    for i, b in enumerate(random.sample(BULLETS, random.randint(3, 6))):
        d.text((110, 170 + i * random.randint(60, 75)), "\u2022 " + b, font=f, fill=ink)
    return img


def code_slide(split):
    dark = random.random() < 0.5
    bg = (30, 30, 36) if dark else (252, 252, 250)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    pool = CODE_PY + (CODE_OTHER if split == "test" else [])
    lines = random.choice(pool)
    f = font(split, "mono", random.randint(22, 30))
    colors = [(220, 220, 220), (86, 156, 214), (206, 145, 120), (181, 206, 168)] if dark else \
             [(20, 20, 20), (0, 0, 180), (160, 30, 30), (0, 120, 0)]
    if random.random() < 0.6:   # editor chrome: line numbers and a file tab
        d.rectangle((0, 0, W, 40), fill=(45, 45, 52) if dark else (230, 230, 230))
        d.text((20, 8), random.choice(["main.py", "stack.py", "search.py", "lecture7.py"]),
               font=font(split, "sans", 18), fill=(200, 200, 200) if dark else (60, 60, 60))
    for i, line in enumerate(lines):
        y = 70 + i * 40
        d.text((30, y), f"{i + 1:>2}", font=f, fill=(120, 120, 120))
        d.text((100, y), line, font=f, fill=random.choice(colors))
    return img


def terminal_slide(split):
    dark = random.random() < 0.7
    bg = random.choice([(12, 12, 12), (1, 36, 86), (40, 40, 40)]) if dark else (255, 255, 255)
    ink = (230, 230, 230) if dark else (20, 20, 20)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    pool = TERM + (TERM_OTHER if split == "test" else [])
    f = font(split, "mono", random.randint(22, 30))
    y = 40
    for block in random.sample(pool, random.randint(1, 2)):
        for line in block:
            d.text((40, y), line, font=f, fill=ink)
            y += 42
        y += 20
    return img


def diagram_slide(split):
    bg = random.choice([(250, 250, 247), (255, 255, 255), (240, 244, 250)])
    ink = (25, 25, 25)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    if random.random() < 0.7:
        d.text((80, 40), random.choice(TOPICS), font=font(split, "bold", 44), fill=ink)
    f = font(split, "sans", 26)
    kind = random.choice(["tree", "boxes", "plot", "graph"])
    if kind == "tree":
        nodes = [(640, 180), (420, 340), (860, 340), (320, 500), (520, 500), (760, 500), (960, 500)]
        for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
            d.line((*nodes[a], *nodes[b]), fill=ink, width=3)
        for i, (x, y) in enumerate(nodes):
            d.ellipse((x - 35, y - 35, x + 35, y + 35), fill=bg, outline=ink, width=3)
            d.text((x - 12, y - 15), str(random.randint(1, 99)), font=f, fill=ink)
    elif kind == "boxes":
        n = random.randint(3, 6)
        for i in range(n):
            x = 120 + i * 180
            d.rectangle((x, 300, x + 120, 380), outline=ink, width=3)
            d.text((x + 35, 322), str(random.randint(0, 99)), font=f, fill=ink)
            if i < n - 1:
                d.line((x + 120, 340, x + 180, 340), fill=ink, width=3)
                d.polygon([(x + 180, 340), (x + 166, 332), (x + 166, 348)], fill=ink)
    elif kind == "plot":
        d.line((150, 600, 1150, 600), fill=ink, width=3)
        d.line((150, 600, 150, 150), fill=ink, width=3)
        pts = [(150 + i * 50, 600 - int(450 * (i / 20) ** random.choice([1, 2, 0.5]))) for i in range(21)]
        d.line(pts, fill=(200, 40, 40), width=4)
        d.text((600, 620), "n", font=f, fill=ink)
        d.text((90, 360), "time", font=f, fill=ink)
    else:
        pts = [(random.randint(200, 1080), random.randint(180, 620)) for _ in range(6)]
        for _ in range(8):
            a, b = random.sample(pts, 2)
            d.line((*a, *b), fill=ink, width=3)
        for i, (x, y) in enumerate(pts):
            d.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(255, 230, 160), outline=ink, width=3)
            d.text((x - 9, y - 15), "ABCDEF"[i], font=f, fill=ink)
    return img


MAKERS = {"text": text_slide, "code": code_slide, "terminal": terminal_slide, "diagram": diagram_slide}


def degrade(img, split, rng):
    if split == "test":
        img = img.resize((854, 480), Image.BILINEAR).resize((W, H), Image.BILINEAR)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(0.8))
    arr = np.asarray(img).astype(np.int16)
    arr = np.clip(arr + rng.normal(0, 6 if split == "test" else 3, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=45 if split == "test" else 80)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def main(out, split, n):
    random.seed({"dev": 1, "test": 2}[split])
    rng = np.random.default_rng({"dev": 1, "test": 2}[split])
    os.makedirs(os.path.join(out, split), exist_ok=True)
    rows = []
    classes = list(MAKERS)
    for i in range(n):
        c = classes[i % len(classes)]
        path = os.path.join(out, split, f"{c}_{i:04d}.png")
        degrade(MAKERS[c](split), split, rng).save(path)
        rows.append({"path": path, "label": c})
    with open(os.path.join(out, f"{split}_labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows(rows)
    print(split, n, "images")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
