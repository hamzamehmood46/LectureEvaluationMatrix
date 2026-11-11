"""Build and validate the lecture corpus manifest for the communication study.

Reads only public YouTube listing metadata through yt-dlp (titles, ids, durations).
Nothing is downloaded. Lectures shorter than --min-minutes are dropped; if a course
has more than --cap qualifying lectures, an evenly spaced, deterministic subset is
kept so the sample spans the whole course.

    python communication/build_manifest.py build  --yt-dlp <path-to-yt-dlp>
    python communication/build_manifest.py validate
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "corpus_manifest.csv"

# subject, stem flag, university, course, official uploader, playlist id
COURSES = [
    ("Machine Learning", True, "Stanford", "CS229 Machine Learning (Autumn 2018)", "Stanford Online",
     "PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU"),
    ("Quantum Physics", True, "MIT", "8.04 Quantum Physics I (Spring 2013)", "MIT OpenCourseWare",
     "PLUl4u3cNGP61-9PEhRognw5vryrSEVLPr"),
    ("Probability", True, "Harvard", "Statistics 110 Probability", "Harvard University",
     "PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo"),
    ("Philosophy", False, "Harvard", "Justice with Michael Sandel", "Harvard University",
     "PL30C13C91CFFEFEA6"),
    ("Economics", False, "MIT", "14.01 Principles of Microeconomics (Fall 2018)", "MIT OpenCourseWare",
     "PLUl4u3cNGP62oJSoqb4Rf-vZMGUBe59G-"),
    ("Behavioral Science", False, "Stanford", "Human Behavioral Biology", "Stanford",
     "PL848F2368C90DDC3D"),
]

FIELDS = ["id", "subject", "stem", "university", "course", "channel", "lecture_title",
          "video_id", "url", "duration_min", "playlist_url", "listing_status"]


def list_playlist(yt_dlp, playlist_id):
    out = subprocess.run(
        [yt_dlp, "--flat-playlist", "-J", f"https://www.youtube.com/playlist?list={playlist_id}"],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["entries"]


def evenly(items, cap):
    if len(items) <= cap:
        return items
    return [items[round(i * (len(items) - 1) / (cap - 1))] for i in range(cap)]


def build(args):
    rows = []
    for subject, stem, uni, course, channel, pid in COURSES:
        entries = [e for e in list_playlist(args.yt_dlp, pid)
                   if e.get("duration") and e["duration"] >= args.min_minutes * 60]
        for e in evenly(entries, args.cap):
            rows.append({
                "id": f"L{len(rows) + 1:03d}", "subject": subject, "stem": "STEM" if stem else "non-STEM",
                "university": uni, "course": course, "channel": channel,
                "lecture_title": e["title"], "video_id": e["id"],
                "url": f"https://www.youtube.com/watch?v={e['id']}",
                "duration_min": round(e["duration"] / 60, 1),
                "playlist_url": f"https://www.youtube.com/playlist?list={pid}",
                "listing_status": "listed_on_channel_playlist",
            })
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {MANIFEST}")
    validate(args)


def validate(args):
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    problems = []
    ids = Counter(r["video_id"] for r in rows)
    problems += [f"duplicate video {v}" for v, n in ids.items() if n > 1]
    problems += [f"{r['id']} is {r['duration_min']} min" for r in rows
                 if float(r["duration_min"]) < args.min_minutes]
    unis = {r["university"] for r in rows}
    stem = Counter(r["stem"] for r in rows)
    print("rows:", len(rows), "| universities:", sorted(unis), "| by stem:", dict(stem))
    for subj, n in sorted(Counter(r["subject"] for r in rows).items()):
        hrs = sum(float(r["duration_min"]) for r in rows if r["subject"] == subj) / 60
        print(f"  {subj:20s} {n:3d} lectures  {hrs:5.1f} h")
    if len(unis) < 3:
        problems.append("fewer than 3 universities")
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ")
        sys.exit(1)
    print("manifest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "validate"])
    p.add_argument("--yt-dlp", default="yt-dlp")
    p.add_argument("--min-minutes", type=float, default=45)
    p.add_argument("--cap", type=int, default=17, help="max lectures kept per course")
    a = p.parse_args()
    build(a) if a.command == "build" else validate(a)
