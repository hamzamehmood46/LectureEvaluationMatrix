import json, sys
sys.path.insert(0, "pipeline")
import stage2_judge as j
stage1 = {"video": "synthetic", "segments": [
 {"start": 0, "end": 25, "transcript": "Today we look at loops and lists. By the end you should be able to write a for loop.", "ocr": "Lecture 7: Loops and Lists"},
 {"start": 55, "end": 95, "transcript": "Let me type this out. If the score is negative we skip it with continue.", "ocr": "def total_marks(scores):"},
 {"start": 95, "end": 125, "transcript": "I expect 233. Let's run it. And there it is, 233, matching what we predicted.", "ocr": "$ python marks.py 233"}]}
rubric = json.load(open("pipeline/rubric_v2.json"))
im = next(d for d in rubric["dimensions"] if d["id"] == "IM")
resp = {"scores": [
 {"id": "IM1", "level": 2, "evidence": [{"time": "00:55", "quote": "Let me type this out"}]},          # real
 {"id": "IM2", "level": 2, "evidence": [{"time": "01:35", "quote": "233, matching what we predicted"},  # real
                                        {"time": "01:35", "quote": "the output was correct on the first try"}]},  # fabricated
 {"id": "IM3", "level": 3, "evidence": [{"time": "02:10", "quote": "this line implements the accumulator"}]}]}  # bad level, bad time
rows, problems = j.check(resp, im, stage1, "transcript")
for r in rows: print(r)
for p in problems: print("PROBLEM:", p)
assert rows[0]["evidence_verified"] == 1 and rows[1]["evidence_verified"] == 1
assert any("not found" in p for p in problems) and any("invalid level" in p for p in problems) and any("not a segment start" in p for p in problems)
# OCR-only quote is accepted only in the transcript_ocr condition
r2 = {"scores": [{"id": "IM1", "level": 1, "evidence": [{"time": "00:55", "quote": "def total_marks(scores)"}]},
                 {"id": "IM2", "level": 0, "evidence": []}, {"id": "IM3", "level": 0, "evidence": []}]}
assert j.check(r2, im, stage1, "transcript")[0][0]["evidence_verified"] == 0
assert j.check(r2, im, stage1, "transcript_ocr")[0][0]["evidence_verified"] == 1
p = j.build_prompt(im, stage1, "transcript_ocr", ["write a for loop"])
assert "(on screen:" in p and "[01:35-02:05]" in p
print("all stage 2 checks passed")
