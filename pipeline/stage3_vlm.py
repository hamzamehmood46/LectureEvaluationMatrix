"""
Stage 3: vision-language model over the keyframes that need it.

1. Each Stage 1 keyframe is classified by screen_router (text / code / terminal / diagram).
2. Only code, terminal and diagram states (and states where OCR found almost nothing)
   go to the VLM. Plain text slides keep their OCR text.
3. The VLM answers a fixed set of yes/no questions about the image and writes a
   one-sentence description, returned as JSON.
4. A simple temporal rule links states: a code state followed within two minutes
   by a state showing program output is marked as a candidate for "code executed"
   (rubric v2, IM2). The judge still decides the score; this only supplies evidence.

Runs against a local Ollama server with an open VLM, for example qwen2.5vl:7b.
  python stage3_vlm.py out_L03/stage1_segments.json --model qwen2.5vl:7b
  python stage3_vlm.py out_L03/stage1_segments.json --route-only
"""
import argparse
import base64
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_router  # noqa: E402

QUESTIONS = {
    "shows_code": "Is source code visible?",
    "code_being_edited": "Does the code look like it is being typed or edited live (cursor, partial line, editor open)?",
    "shows_program_output": "Is the output of running a program visible (console, terminal, notebook output cell)?",
    "shows_error": "Is an error message or traceback visible?",
    "shows_diagram": "Is a diagram, plot, tree, table or drawing visible?",
    "shows_worked_steps": "Does the screen show a worked example with intermediate steps?",
    "handwriting": "Is there handwriting (board, tablet or paper)?",
}


def vlm_prompt():
    qs = "\n".join(f'- "{k}": {v}' for k, v in QUESTIONS.items())
    return ("This is a single frame from a university lecture recording. Answer each question "
            "with true or false, judging only what is visible in the image.\n" + qs +
            '\nAlso give "description": one plain sentence saying what is on screen.\n'
            "Return JSON only, with exactly these keys.")


def ask_vlm(image_path, model, host):
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": vlm_prompt(), "images": [img]}]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def validate(ans):
    """Keep only expected keys; anything not a clear boolean becomes None."""
    out = {k: ans.get(k) if isinstance(ans.get(k), bool) else None for k in QUESTIONS}
    out["description"] = str(ans.get("description", ""))[:300]
    return out


def link_execution(segments, window=120.0):
    """Candidate IM2 evidence: code state followed by an output state within `window` seconds."""
    links = []
    for i, s in enumerate(segments):
        v = s.get("vlm") or {}
        is_code = v.get("shows_code") or s.get("screen_type") == "code"
        if not is_code:
            continue
        for t in segments[i + 1:]:
            if t["start"] - s["end"] > window:
                break
            w = t.get("vlm") or {}
            if w.get("shows_program_output") or t.get("screen_type") == "terminal":
                links.append({"code_at": s["start"], "output_at": t["start"],
                              "error": bool(w.get("shows_error"))})
                break
    return links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--route-only", action="store_true", help="classify and route, no VLM calls")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = json.load(open(a.stage1, encoding="utf-8"))
    segs = data["segments"]
    sent = 0
    mask = (data.get("settings") or {}).get("mask")   # same webcam mask Stage 1 used
    for s in segs:
        label, send, feats = screen_router.classify_image(s["keyframe"], mask)
        s["screen_type"], s["sent_to_vlm"] = label, send
        if send and not a.route_only:
            s["vlm"] = validate(ask_vlm(s["keyframe"], a.model, a.host))
        sent += send
    data["execution_links"] = link_execution(segs)
    data["stage3"] = {"model": None if a.route_only else a.model, "states": len(segs), "sent_to_vlm": sent}
    out = a.out or a.stage1.replace("stage1_segments", "stage3_segments")
    json.dump(data, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{sent}/{len(segs)} states sent to VLM; {len(data['execution_links'])} code-to-output links; wrote {out}")


if __name__ == "__main__":
    main()
