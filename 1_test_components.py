"""
Run: python 1_test_components.py
Tests all components before launching the app.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 55)
print("TEST 1 — Whisper")
print("=" * 55)
from whisper_transcriber import WhisperTranscriber
t = WhisperTranscriber()
print("✅ Whisper loaded\n")

print("=" * 55)
print("TEST 2 — ASL Gloss Mapper (vocabulary check)")
print("=" * 55)
from asl_gloss_mapper import ASLGlossMapper
mapper = ASLGlossMapper()

test_sentences = [
    "Hello thank you please",
    "Where is the hospital",
    "I am deaf please help me",
    "Call the police I am lost",
    "I need water and food",
]
for s in test_sentences:
    glosses = mapper.text_to_gloss(s)
    print(f"  '{s}'")
    print(f"   → {glosses}")
print("✅ Mapper working\n")

print("=" * 55)
print("TEST 3 — Keypoints shape check")
print("=" * 55)
for sign in ["HELLO", "THANK YOU", "YES", "NO", "HELP", "A", "B", "C"]:
    kp = mapper.get_keypoints(sign)
    assert kp.shape == (21, 2), f"Bad shape for {sign}: {kp.shape}"
    print(f"  {sign:12s} → shape {kp.shape} ✓")
print("✅ All keypoints valid\n")

print("=" * 55)
print("TEST 4 — Avatar Renderer")
print("=" * 55)
from avatar_renderer import AvatarRenderer
renderer = AvatarRenderer()
kp    = mapper.get_keypoints("HELLO")
frame = renderer.render_frame(kp, "HELLO", "Hello world", progress=0.5)
print(f"  Frame shape : {frame.shape}")
assert frame.shape == (480, 640, 3)
print("✅ Renderer working\n")

print("=" * 55)
print("TEST 5 — Full pipeline (no audio)")
print("=" * 55)
from pose_generator import PoseGenerator
gen = PoseGenerator()
os.makedirs("./output", exist_ok=True)
path, glosses, n = gen.text_to_video(
    "HELLO THANK YOU PLEASE HELP",
    output_path="./output/test_output.gif"
)
print(f"✅ Output: {path}  ({n} frames, {len(glosses)} signs)\n")

print("=" * 55)
print("TEST 6 — LM Studio (optional)")
print("=" * 55)
from text_simplifier import TextSimplifier
s = TextSimplifier()
if s.is_available():
    out = s.simplify("Can you please help me find the nearest hospital?")
    print(f"  Original  : Can you please help me find the nearest hospital?")
    print(f"  Simplified: {out}")
    print("✅ LM Studio working")
else:
    out = s._rule_based_simplify("Can you please help me find the nearest hospital?")
    print(f"  LM Studio not running — rule-based result: {out}")
    print("⚠️  Start LM Studio for AI simplification (optional)")

print("\n✅ ALL TESTS PASSED — Run python 2_app.py")