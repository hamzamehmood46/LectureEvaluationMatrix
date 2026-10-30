"""
Stage 2: rubric judge over the Stage 1 transcript, using a local open-weight model.

Two input conditions from the study design:
  transcript       - timestamped transcript only
  transcript_ocr   - transcript plus the OCR text of each on-screen state
(The third condition, transcript plus video, adds the Stage 3 VLM.)

The model must return JSON: for each criterion a level (0/1/2) and evidence
items {time, quote}. Every quote is then checked against the transcript or OCR text of the segment
it cites, so fabricated or misplaced evidence is caught automatically. This feeds
the "correctness of supporting evidence" measure in the evaluation.

Runs against Ollama (http://localhost:11434) by default, so nothing leaves the machine.
  python stage2_judge.py out/stage1_segments.json --rubric rubric_v2.json --model qwen2.5:14b
  python stage2_judge.py out/stage1_segments.json --rubric rubric_v2.json --dry-run
"""
import argparse
import json
import re
import urllib.request


def fmt_time(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def parse_time(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def lecture_text(stage1, condition):
    out = []
    for seg in stage1["segments"]:
        line = f"[{fmt_time(seg['start'])}-{fmt_time(seg['end'])}] {seg.get('transcript', '')}"
        if condition == "transcript_ocr" and seg.get("ocr"):
            line += f"\n  (on screen: {' '.join(seg['ocr'].split())})"
        out.append(line)
    return "\n".join(out)


def build_prompt(dim, stage1, condition, objectives):
    crit = "\n".join(
        f"{c['id']} {c['name']}\n" + "\n".join(f"  {k}: {v}" for k, v in c["levels"].items())
        for c in dim["criteria"])
    return f"""You are scoring one university lecture against part of a teaching rubric.
Dimension: {dim['name']}
Stated learning objectives: {'; '.join(objectives) if objectives else 'not provided'}

Criteria and level definitions:
{crit}

Rules:
- Give the level whose definition is fully met. If unsure between two levels, give the lower.
- Support every level with 1 to 3 evidence items. Each item has the segment start time
  exactly as written in the lecture (mm:ss) and a short quote copied word for word from that segment.
- If there is no evidence for a criterion, give level 0 and an empty evidence list.
- Return JSON only, in this form:
{{"scores": [{{"id": "<criterion id>", "level": 0, "evidence": [{{"time": "mm:ss", "quote": "..."}}], "note": "one sentence"}}]}}

Lecture (segments are screen states, with what was said while each was visible):
{lecture_text(stage1, condition)}
"""


def ask_ollama(prompt, model, host):
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       # Ollama's default context is only a few thousand tokens; a full
                       # 75-minute lecture needs roughly 15-20k, so raise it explicitly.
                       "options": {"temperature": 0, "num_ctx": 32768},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def check(response, dim, stage1, condition):
    """Validate structure and verify every quote against the segment it cites."""
    by_start = {fmt_time(s["start"]): s for s in stage1["segments"]}
    want = {c["id"] for c in dim["criteria"]}
    rows, problems = [], []
    got = {s.get("id"): s for s in response.get("scores", [])}
    for cid in sorted(want):
        s = got.get(cid)
        if s is None:
            problems.append(f"{cid}: missing")
            continue
        if s.get("level") not in (0, 1, 2):
            problems.append(f"{cid}: invalid level {s.get('level')!r}")
        ev_ok = 0
        for ev in s.get("evidence", []):
            seg = by_start.get(ev.get("time", ""))
            src = (seg.get("transcript", "") + (" " + seg.get("ocr", "") if condition == "transcript_ocr" else "")) if seg else ""
            if seg is None:
                problems.append(f"{cid}: cites {ev.get('time')}, which is not a segment start")
            elif norm(ev.get("quote", "")) and norm(ev["quote"]) in norm(src):
                ev_ok += 1
            else:
                problems.append(f"{cid}: quote not found in segment {ev.get('time')}: {ev.get('quote', '')[:60]!r}")
        rows.append({"id": cid, "level": s.get("level"), "evidence_items": len(s.get("evidence", [])),
                     "evidence_verified": ev_ok, "note": s.get("note", "")})
    return rows, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1")
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--condition", choices=["transcript", "transcript_ocr"], default="transcript")
    ap.add_argument("--objectives", default="", help="semicolon-separated learning objectives")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="stage2_scores.json")
    a = ap.parse_args()

    stage1 = json.load(open(a.stage1, encoding="utf-8"))
    rubric = json.load(open(a.rubric, encoding="utf-8"))
    objectives = [o.strip() for o in a.objectives.split(";") if o.strip()]
    results = {"lecture": stage1["video"], "condition": a.condition, "model": a.model,
               "rubric": rubric["version"], "dimensions": []}
    for dim in rubric["dimensions"]:
        prompt = build_prompt(dim, stage1, a.condition, objectives)
        if a.dry_run:
            print(f"===== {dim['id']} ({len(prompt.split())} words) =====\n{prompt[:1500]}\n")
            continue
        resp = ask_ollama(prompt, a.model, a.host)
        rows, problems = check(resp, dim, stage1, a.condition)
        results["dimensions"].append({"id": dim["id"], "scores": rows, "problems": problems})
        print(dim["id"], [(r["id"], r["level"]) for r in rows], f"{len(problems)} problems")
    if not a.dry_run:
        json.dump(results, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
