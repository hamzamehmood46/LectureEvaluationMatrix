"""Download audio for each lecture in the manifest and transcribe it with faster-whisper.

Resumable: a lecture whose transcript JSON already exists is skipped. Audio and
transcripts are third-party course content and stay on this machine (they are
git-ignored); only the manifest is published.

    python communication/transcribe.py --data-dir C:/Users/hamza/lecture_data --model large-v3
    python communication/transcribe.py --ids L001 L002        # just these
"""
import argparse
import csv
import json
import os
import site
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def add_cuda_dlls():
    """Make the pip-installed cuBLAS/cuDNN DLLs visible to ctranslate2 on Windows."""
    for sp in site.getsitepackages():
        for sub in ("cublas", "cudnn"):
            d = Path(sp) / "nvidia" / sub / "bin"
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


class BotBlocked(Exception):
    """YouTube refused the download; stop instead of hammering it."""


def download_audio(url, out_stem):
    for ext in ("m4a", "webm", "mp4", "opus", "mp3"):
        p = Path(f"{out_stem}.{ext}")
        if p.exists():
            return p
    cmd = [sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "--no-playlist",
           "-o", f"{out_stem}.%(ext)s", url]
    deno = Path(sys.executable).with_name("deno.exe")
    if deno.exists():
        cmd[3:3] = ["--js-runtimes", f"deno:{deno}"]
    if os.environ.get("YT_COOKIES_BROWSER"):  # e.g. "edge" or "edge:Profile 1"
        cmd[3:3] = ["--cookies-from-browser", os.environ["YT_COOKIES_BROWSER"]]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        if "not a bot" in res.stderr:
            raise BotBlocked(res.stderr.strip().splitlines()[-1])
        raise RuntimeError(res.stderr.strip().splitlines()[-1])
    return next(Path(out_stem).parent.glob(Path(out_stem).name + ".*"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(HERE / "corpus_manifest.csv"))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--keep-audio", action="store_true")
    p.add_argument("--pause", type=float, default=60, help="seconds to wait after each lecture")
    a = p.parse_args()

    add_cuda_dlls()
    from faster_whisper import WhisperModel

    data = Path(a.data_dir)
    (data / "audio").mkdir(parents=True, exist_ok=True)
    (data / "transcripts").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    if a.ids:
        rows = [r for r in rows if r["id"] in a.ids]
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute_type)

    for r in rows:
        out = data / "transcripts" / f"{r['id']}.json"
        if out.exists():
            continue
        t0 = time.time()
        try:
            audio = download_audio(r["url"], str(data / "audio" / r["id"]))
            segs, info = model.transcribe(str(audio), language="en", word_timestamps=True,
                                          vad_filter=True, condition_on_previous_text=False)
            segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                         "words": [[round(w.start, 2), round(w.end, 2), w.word] for w in (s.words or [])]}
                        for s in segs]
        except BotBlocked as e:
            print(f"STOPPED at {r['id']}: YouTube bot check ({e}). Wait a few hours and rerun.", flush=True)
            return
        except Exception as e:  # keep going; failures are listed at the end
            print(f"FAILED {r['id']}: {e}", flush=True)
            with open(data / "failed.txt", "a") as f:
                f.write(f"{r['id']}\t{e}\n")
            continue
        out.write_text(json.dumps({
            "id": r["id"], "video_id": r["video_id"], "model": a.model,
            "duration_s": round(info.duration, 1), "segments": segments,
            "text": " ".join(s["text"] for s in segments)}, ensure_ascii=False), encoding="utf-8")
        if not a.keep_audio:
            audio.unlink(missing_ok=True)
        wall = time.time() - t0
        time.sleep(a.pause)  # pace downloads so YouTube does not flag the run
        print(f"{r['id']} {r['subject']:18s} {info.duration/60:5.1f} min audio in {wall/60:4.1f} min "
              f"({info.duration/wall:4.1f}x realtime)", flush=True)


if __name__ == "__main__":
    main()
