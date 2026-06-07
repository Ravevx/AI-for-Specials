# ── Whisper ────────────────────────────────────────────────
WHISPER_MODEL    = "base"       # tiny/base/small/medium/large
WHISPER_LANGUAGE = "en"

# ── LM Studio ──────────────────────────────────────────────
LM_STUDIO_URL    = "http://localhost:1234/v1/chat/completions"

# ── Avatar Rendering ───────────────────────────────────────
AVATAR_WIDTH      = 640
AVATAR_HEIGHT     = 480
AVATAR_FPS        = 12
SIGN_HOLD_FRAMES  = 10          # frames each sign is held (keypoint mode)
TRANSITION_FRAMES = 4           # blend frames between keypoint signs
BG_COLOR          = (20, 20, 20)
HAND_COLOR        = (0, 220, 180)
BONE_COLOR        = (0, 160, 120)
BODY_COLOR        = (180, 180, 180)
FACE_COLOR        = (200, 180, 140)

# ── WLASL Dataset ───────────────────────────────────────────
#
# Your folder structure:
#   J:\Agent My Learning\agent 5\data\wlasl_videos\
#     video\          ← 11,980 mp4 files (numeric IDs)
#     WLASL_v0.3.json ← full 2000-word index  ← we use this one
#     nslt_100.json
#     nslt_300.json
#     nslt_1000.json
#     nslt_2000.json
#     wlasl_class_list.txt
#     missing.txt
#
WLASL_VIDEO_DIR  = r"J:\Agent My Learning\agent 5\data\wlasl_videos"

# Which JSON to use:
#   WLASL_v0.3.json  → full 2000 words  (recommended)
#   nslt_2000.json   → 2000 words subset
#   nslt_1000.json   → 1000 words subset (faster loading)
#   nslt_300.json    → 300 words  (lightest)
#   nslt_100.json    → 100 words  (fastest, least coverage)
WLASL_JSON_FILE  = "WLASL_v0.3.json"

# Sub-folder inside WLASL_VIDEO_DIR that holds the actual mp4 files
# In risangbaskoro/wlasl-processed this is "video" (singular)
WLASL_VIDEOS_SUBFOLDER = "videos"

# ── Output ──────────────────────────────────────────────────
OUTPUT_DIR   = "./output"
OUTPUT_VIDEO = "./output/asl_output.mp4"