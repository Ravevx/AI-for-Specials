"""
Run this first: python diagnose_wlasl.py
Tells you exactly what's wrong with the WLASL setup.
"""
import os, json, sys

# ── paste your path here ──────────────────────────────────
WLASL_VIDEO_DIR       = r"J:\Agent My Learning\agent 5\data\wlasl_videos"
WLASL_JSON_FILE       = "WLASL_v0.3.json"
WLASL_VIDEOS_SUBFOLDER = "videos"
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("WLASL DIAGNOSTIC")
print("=" * 60)

# 1. Base folder
print(f"\n1. Base folder exists?  ", end="")
if os.path.isdir(WLASL_VIDEO_DIR):
    print("✅ YES")
else:
    print(f"❌ NO — not found at:\n   {WLASL_VIDEO_DIR}")
    sys.exit(1)

# 2. JSON file
json_path = os.path.join(WLASL_VIDEO_DIR, WLASL_JSON_FILE)
print(f"2. JSON file exists?    ", end="")
if os.path.isfile(json_path):
    print(f"✅ YES  ({os.path.getsize(json_path)//1024} KB)")
else:
    print(f"❌ NO — expected at:\n   {json_path}")
    print("   Files actually in base folder:")
    for f in os.listdir(WLASL_VIDEO_DIR):
        print(f"     {f}")
    sys.exit(1)

# 3. Videos subfolder
videos_dir = os.path.join(WLASL_VIDEO_DIR, WLASL_VIDEOS_SUBFOLDER)
print(f"3. Videos subfolder?   ", end="")
if os.path.isdir(videos_dir):
    all_vids = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
    print(f"✅ YES  ({len(all_vids)} mp4 files)")
else:
    print(f"❌ NO — expected at:\n   {videos_dir}")
    print("   Subfolders actually in base folder:")
    for f in os.listdir(WLASL_VIDEO_DIR):
        full = os.path.join(WLASL_VIDEO_DIR, f)
        tag  = "[DIR] " if os.path.isdir(full) else "[FILE]"
        print(f"     {tag} {f}")
    sys.exit(1)

# 4. Parse JSON
print(f"4. Parsing JSON...     ", end="")
with open(json_path) as f:
    data = json.load(f)
print(f"✅  {len(data)} gloss entries")

# 5. Build index (same logic as pose_generator)
print(f"5. Building index...   ", end="")
index   = {}
indexed = 0
missing = 0

for entry in data:
    word = entry.get("gloss", "").upper().strip()
    for inst in entry.get("instances", []):
        vid_id = str(inst.get("video_id", "")).zfill(5)
        path   = os.path.join(videos_dir, f"{vid_id}.mp4")
        if os.path.isfile(path):
            index.setdefault(word, []).append(path)
            indexed += 1
        else:
            missing += 1

print(f"✅  {len(index)} words, {indexed} clips found, {missing} missing")

# 6. Check specific words that were fingerspelled
print(f"\n6. Checking problem words:")
check = ["NICE", "VERY", "MUCH", "DOCTOR", "MEET", "HAPPY", "BAD", "HELP"]
for w in check:
    clips = index.get(w, [])
    if clips:
        print(f"   {w:15s} → ✅ {len(clips)} clips  (first: {os.path.basename(clips[0])})")
    else:
        print(f"   {w:15s} → ❌ NOT in index")

# 7. Show sample of what IS indexed
print(f"\n7. Sample of indexed words (first 40):")
sample = sorted(index.keys())[:40]
for i in range(0, len(sample), 8):
    print("  ", "  ".join(f"{w:12s}" for w in sample[i:i+8]))

print("\n" + "=" * 60)
print("If step 5 shows 0 clips: your video filenames don't match")
print("the IDs in the JSON. Show me the output and I'll fix it.")
print("=" * 60)