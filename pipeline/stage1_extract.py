"""
Stage 1 of the multimodal lecture pipeline.

Takes a lecture video and produces one JSON file of time-aligned segments:
  - a timestamped transcript (faster-whisper, word-level timestamps)
  - optional speaker labels (pyannote, needs a Hugging Face token)
  - on-screen change detection, with one keyframe saved per screen state
  - OCR text for each keyframe (Tesseract)
Each segment covers one screen state and carries the words spoken while it was visible.
That file is the input for the transcript judge (Stage 2) and the VLM (Stage 3).

Usage
  python stage1_extract.py lecture.mp4 --out out_dir
  python stage1_extract.py lecture.mp4 --out out_dir --mask 0.78,0.70,1.0,1.0   # ignore a webcam inset
  python stage1_extract.py lecture.mp4 --out out_dir --no-asr                  # visual side only
  python stage1_extract.py lecture.mp4 --out out_dir --diarize --hf-token XXX

Everything runs locally. Models are downloaded once, then cached.
"""
import argparse
import json
import os

import cv2
import numpy as np

# ---------------------------------------------------------------- visual changes

def _prep(frame, mask):
    """Grayscale, downscale, blur; blank out the masked region (e.g. a webcam inset)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (640, 360), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    if mask is not None:
        x0, y0, x1, y1 = mask
        g[int(y0 * 360):int(y1 * 360), int(x0 * 640):int(x1 * 640)] = int(np.median(g))
    return g


def _ink(g):
    """Foreground pixels: anything far from the frame's dominant background shade."""
    bg = int(np.median(g))
    return np.abs(g.astype(np.int16) - bg) > 40


def change_score(a, b):
    """
    Share of foreground ('ink') that differs between two prepared frames.
    Normalising by ink instead of by frame area means a new slide scores near 1.0,
    while adding one line of code to a ten-line slide scores around 0.1,
    regardless of how much empty background the slide has.
    """
    diff = cv2.absdiff(a, b) > 40
    ink = _ink(a) | _ink(b)
    denom = max(int(ink.sum()), 1)
    return float((diff & ink).sum()) / denom


def detect_changes(video, sample_fps=1.0, threshold=0.35, min_gap=3.0, mask=None,
                   keyframe_dir=None):
    """
    Returns a list of screen states: {start, end, keyframe, score}.
    A new state begins when the current frame differs from the last state's keyframe
    by more than `threshold` and at least `min_gap` seconds have passed.
    Small incremental edits (typing, a build animation) are absorbed into the
    current state, but the keyframe is refreshed so it shows the most complete version.
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    step = max(int(round(fps / sample_fps)), 1)

    states, ref, ref_frame = [], None, None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps
            g = _prep(frame, mask)
            if ref is None:
                states.append({"start": 0.0, "score": 1.0})
                ref, ref_frame = g, frame
            else:
                s = change_score(ref, g)
                if s > threshold and t - states[-1]["start"] >= min_gap:
                    states[-1]["_frame"] = ref_frame
                    states.append({"start": round(t, 2), "score": round(s, 3)})
                    ref, ref_frame = g, frame
                elif s > 0.02:
                    # incremental edit: keep the state, refresh its keyframe
                    ref, ref_frame = g, frame
        idx += 1
    cap.release()
    if states:
        states[-1]["_frame"] = ref_frame

    for i, st in enumerate(states):
        st["end"] = states[i + 1]["start"] if i + 1 < len(states) else round(duration, 2)
        if keyframe_dir:
            os.makedirs(keyframe_dir, exist_ok=True)
            path = os.path.join(keyframe_dir, f"state_{i:03d}_{int(st['start']):05d}s.png")
            cv2.imwrite(path, st["_frame"])
            st["keyframe"] = path
        st.pop("_frame", None)
    return states, duration


# ---------------------------------------------------------------- OCR

def ocr_image(path):
    import pytesseract
    img = cv2.imread(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.median(g) < 128:          # dark terminal / IDE theme: invert for Tesseract
        g = 255 - g
    g = cv2.resize(g, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(g, config="--psm 6").strip()


# ---------------------------------------------------------------- speech

def transcribe(video, model_size="large-v3", device="auto", compute_type="auto"):
    """Word-level timestamps with faster-whisper. Returns [{start, end, word}]."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(video, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"start": round(w.start, 2), "end": round(w.end, 2),
                          "word": w.word.strip()})
    return words


def diarize(video, hf_token):
    """Speaker turns with pyannote. Returns [{start, end, speaker}]."""
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    out = pipe(video)
    return [{"start": round(t.start, 2), "end": round(t.end, 2), "speaker": spk}
            for t, _, spk in out.itertracks(yield_label=True)]


def label_speakers(words, turns):
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        w["speaker"] = next((t["speaker"] for t in turns if t["start"] <= mid < t["end"]), None)
    return words


# ---------------------------------------------------------------- alignment

def align(states, words):
    """Attach each word to the screen state visible at the word's midpoint."""
    for st in states:
        st["words"] = []
    j = 0
    for w in sorted(words, key=lambda x: x["start"]):
        mid = (w["start"] + w["end"]) / 2
        while j + 1 < len(states) and mid >= states[j]["end"]:
            j += 1
        states[j]["words"].append(w)
    for st in states:
        st["transcript"] = " ".join(w["word"] for w in st["words"])
    return states


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min-gap", type=float, default=3.0)
    ap.add_argument("--mask", default=None, help="x0,y0,x1,y1 as fractions of the frame")
    a = ap.parse_args()

    mask = tuple(float(v) for v in a.mask.split(",")) if a.mask else None
    os.makedirs(a.out, exist_ok=True)

    states, duration = detect_changes(a.video, threshold=a.threshold, min_gap=a.min_gap,
                                      mask=mask, keyframe_dir=os.path.join(a.out, "keyframes"))
    print(f"{len(states)} screen states over {duration/60:.1f} min")
    for st in states:
        st["ocr"] = ocr_image(st["keyframe"])

    words = []
    if not a.no_asr:
        words = transcribe(a.video, a.model)
        if a.diarize:
            if not a.hf_token:
                raise SystemExit("--diarize needs a Hugging Face token (--hf-token or HF_TOKEN)")
            words = label_speakers(words, diarize(a.video, a.hf_token))
        print(f"{len(words)} words transcribed")
    states = align(states, words)

    result = {"video": os.path.basename(a.video), "duration_s": round(duration, 1),
              "settings": {"threshold": a.threshold, "min_gap": a.min_gap, "mask": mask,
                           "asr_model": None if a.no_asr else a.model, "diarized": a.diarize},
              "segments": states}
    path = os.path.join(a.out, "stage1_segments.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
