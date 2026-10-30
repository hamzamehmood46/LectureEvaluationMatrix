import sys
sys.path.insert(0, "pipeline")
import stage3_vlm as s3
v = s3.validate({"shows_code": True, "shows_program_output": "yes", "extra": 1, "description": "An editor with Python code."})
assert v["shows_code"] is True and v["shows_program_output"] is None and "extra" not in v and set(s3.QUESTIONS) <= set(v)
segs = [
  {"start": 0, "end": 40, "screen_type": "text"},
  {"start": 40, "end": 100, "screen_type": "code", "vlm": {"shows_code": True}},
  {"start": 100, "end": 130, "screen_type": "text"},
  {"start": 130, "end": 160, "screen_type": "terminal", "vlm": {"shows_program_output": True, "shows_error": True}},
  {"start": 400, "end": 460, "screen_type": "code", "vlm": {"shows_code": True}},
  {"start": 700, "end": 720, "screen_type": "terminal", "vlm": {"shows_program_output": True}},  # too late
]
links = s3.link_execution(segs)
assert links == [{"code_at": 40, "output_at": 130, "error": True}], links
print("stage 3 checks passed")
