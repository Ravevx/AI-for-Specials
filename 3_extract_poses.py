"""
Step 1 — Pose Extraction (FAST parallel version)
-------------------------------------------------
Uses 4 CPU workers in parallel + frame skipping for ~4x speedup.
Already-extracted clips are skipped so you can resume safely.

Run:
  python 3_extract_poses.py

Expected time: 30-45 minutes for all 11,980 videos on 4 cores.
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from multiprocessing import Pool, current_process
import time

# ── Config ─────────────────────────────────────────────────
WLASL_VIDEO_DIR  = r"J:\Agent My Learning\agent 5\data\wlasl_videos"
WLASL_JSON_FILE  = "WLASL_v0.3.json"
VIDEOS_SUBFOLDER = "videos"
OUTPUT_DIR       = r"J:\Agent My Learning\agent 5\data\poses"

VECTOR_SIZE      = 63    # 21 landmarks x 3 (x,y,z)
NUM_WORKERS      = 4     # one per CPU core
FRAME_SKIP       = 2     # process every 2nd frame (2x faster, no quality loss)
# ──────────────────────────────────────────────────────────


def process_word(args):
    """
    Worker function — runs in its own process with its own MediaPipe instance.
    Each worker gets a chunk of words to process.
    """
    word_entries, videos_dir, output_dir = args

    # Each worker creates its own MediaPipe instance (can't share across processes)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    done = skipped = failed = missing = 0

    for entry in word_entries:
        word = entry.get("gloss", "").upper().strip()
        if not word:
            continue

        word_dir = os.path.join(output_dir, word)
        os.makedirs(word_dir, exist_ok=True)

        for inst in entry.get("instances", []):
            vid_id      = str(inst.get("video_id", "")).zfill(5)
            frame_start = max(0, inst.get("frame_start", 1) - 1)
            frame_end   = inst.get("frame_end", -1)

            out_path = os.path.join(word_dir, f"{vid_id}.npy")
            vid_path = os.path.join(videos_dir, f"{vid_id}.mp4")

            # Already done — skip
            if os.path.exists(out_path):
                skipped += 1
                continue

            # Video missing from dataset
            if not os.path.isfile(vid_path):
                missing += 1
                continue

            # Extract keypoints
            seq = _extract_video(vid_path, frame_start, frame_end, hands)

            if seq is not None:
                np.save(out_path, seq)
                done += 1
            else:
                failed += 1

    hands.close()
    return done, skipped, failed, missing


def _extract_video(video_path, frame_start, frame_end, hands):
    """Extract keypoint sequence from one video, sampling every FRAME_SKIP frames."""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx < frame_start:
            idx += 1
            continue
        if frame_end != -1 and idx > frame_end:
            break

        # Frame skipping — process every Nth frame
        if (idx - frame_start) % FRAME_SKIP == 0:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            vec = np.zeros(VECTOR_SIZE, dtype=np.float32)
            if result.multi_hand_landmarks:
                for i, lm in enumerate(result.multi_hand_landmarks[0].landmark):
                    vec[i*3]     = lm.x
                    vec[i*3 + 1] = lm.y
                    vec[i*3 + 2] = lm.z

            frames.append(vec)

        idx += 1

    cap.release()
    return np.array(frames, dtype=np.float32) if len(frames) >= 3 else None


def main():
    json_path  = os.path.join(WLASL_VIDEO_DIR, WLASL_JSON_FILE)
    videos_dir = os.path.join(WLASL_VIDEO_DIR, VIDEOS_SUBFOLDER)

    print(f"Loading {WLASL_JSON_FILE}...")
    with open(json_path) as f:
        data = json.load(f)

    total_words = len(data)
    total_clips = sum(len(e.get("instances", [])) for e in data)
    print(f"  {total_words} words, {total_clips} clip instances")
    print(f"  {NUM_WORKERS} parallel workers, every {FRAME_SKIP} frames sampled")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split words evenly across workers
    chunk_size = total_words // NUM_WORKERS
    chunks = []
    for i in range(NUM_WORKERS):
        start = i * chunk_size
        end   = start + chunk_size if i < NUM_WORKERS - 1 else total_words
        chunks.append((data[start:end], videos_dir, OUTPUT_DIR))

    print(f"\nStarting {NUM_WORKERS} workers...\n")
    t_start = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_word, chunks)

    # Aggregate results
    total_done    = sum(r[0] for r in results)
    total_skipped = sum(r[1] for r in results)
    total_failed  = sum(r[2] for r in results)
    total_missing = sum(r[3] for r in results)

    elapsed = (time.time() - t_start) / 60

    print(f"\n{'='*55}")
    print(f"EXTRACTION COMPLETE  ({elapsed:.1f} minutes)")
    print(f"  Extracted : {total_done}")
    print(f"  Skipped   : {total_skipped}  (already existed)")
    print(f"  Missing   : {total_missing}  (not in dataset)")
    print(f"  Failed    : {total_failed}   (too short/unreadable)")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*55}")
    print(f"\nNext: python 4_train_model.py")


if __name__ == "__main__":
    main()