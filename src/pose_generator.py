"""
Pose Generator
--------------
Sign lookup priority per gloss token:

  1. Trained model   -- neural net generates smooth pose sequence
  2. WLASL video     -- real signer clip (fallback before model is trained)
  3. Keypoint        -- built-in handshape (~100 common signs)
  4. Fingerspell     -- letter by letter (last resort)

WLASL dataset: kaggle.com/datasets/risangbaskoro/wlasl-processed
Expected layout:
  <WLASL_VIDEO_DIR>/
    WLASL_v0.3.json
    videos/
      00001.mp4  00002.mp4  ...
"""

import os
import json
import cv2
import numpy as np
import config

from asl_gloss_mapper import ASLGlossMapper
from avatar_renderer  import AvatarRenderer
from sign_inf  import SignInferencer


class PoseGenerator:
    def __init__(self):
        self.mapper     = ASLGlossMapper()
        self.renderer   = AvatarRenderer()
        self.inferencer = SignInferencer()   # trained pose model (Step 2)

        # word (UPPER) -> list of {path, frame_start, frame_end}
        self._wlasl_index = {}
        self._wlasl_ready = False
        self._init_wlasl()

    # ── WLASL init ────────────────────────────────────────

    def _init_wlasl(self):
        base = config.WLASL_VIDEO_DIR

        print(f"  WLASL base  : {base}")

        if not os.path.isdir(base):
            print(f"  WLASL not found -- keypoint + fingerspell mode.")
            return

        json_path  = os.path.join(base, config.WLASL_JSON_FILE)
        videos_dir = os.path.join(base, config.WLASL_VIDEOS_SUBFOLDER)

        print(f"  WLASL json  : {'OK' if os.path.isfile(json_path) else 'NOT FOUND'}")
        print(f"  WLASL videos: {'OK' if os.path.isdir(videos_dir) else 'NOT FOUND'}")

        if os.path.isfile(json_path) and os.path.isdir(videos_dir):
            self._load_from_json(json_path, videos_dir)
        else:
            print("  Falling back to folder scan...")
            self._load_from_folders(base)

    def _load_from_json(self, json_path: str, videos_dir: str):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WLASL JSON unreadable: {e}")
            return

        indexed = 0
        missing = 0

        for entry in data:
            word = entry.get("gloss", "").upper().strip()
            if not word:
                continue
            for inst in entry.get("instances", []):
                vid_id      = str(inst.get("video_id", "")).zfill(5)
                frame_start = max(0, inst.get("frame_start", 1) - 1)
                frame_end   = inst.get("frame_end", -1)
                path        = os.path.join(videos_dir, f"{vid_id}.mp4")

                if not os.path.isfile(path):
                    missing += 1
                    continue

                self._wlasl_index.setdefault(word, []).append({
                    "path":        path,
                    "frame_start": frame_start,
                    "frame_end":   frame_end,
                })
                indexed += 1

        if indexed:
            print(f"  WLASL ready: {len(self._wlasl_index)} words, "
                  f"{indexed} clips ({missing} files not found).")
            self._wlasl_ready = True
        else:
            print(f"  WLASL JSON parsed but no video files matched.")
            print(f"  Expected: {videos_dir}/XXXXX.mp4")

    def _load_from_folders(self, base: str):
        """Fallback for other Kaggle variants with word-named sub-folders."""
        import glob
        roots = [base,
                 os.path.join(base, "train"), os.path.join(base, "val"),
                 os.path.join(base, "test"),  os.path.join(base, "videos")]
        found = 0
        for root in roots:
            if not os.path.isdir(root):
                continue
            for entry in os.scandir(root):
                if not entry.is_dir():
                    continue
                word  = entry.name.upper().strip()
                clips = (glob.glob(os.path.join(entry.path, "*.mp4")) +
                         glob.glob(os.path.join(entry.path, "*.avi")))
                for clip in clips:
                    self._wlasl_index.setdefault(word, []).append(
                        {"path": clip, "frame_start": 0, "frame_end": -1}
                    )
                    found += 1
        if found:
            print(f"  WLASL ready (folder mode): {len(self._wlasl_index)} words.")
            self._wlasl_ready = True

    # ── Public ────────────────────────────────────────────

    def text_to_video(self, text: str, output_path: str = None) -> tuple:
        """Convert text to signing GIF. Returns (path, glosses, n_frames)."""
        if output_path is None:
            output_path = config.OUTPUT_VIDEO

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        glosses = self.mapper.text_to_gloss(text)
        if not glosses:
            glosses = ["DEFAULT"]

        print(f"\nText    : {text}")
        print(f"Glosses : {' -> '.join(glosses)}\n")

        all_frames = []

        for i, gloss in enumerate(glosses):
            frames, source = self._sign_frames(gloss, i, glosses, text)
            print(f"  [{source:11s}] {gloss:20s} -> {len(frames)} frames")
            all_frames.extend(frames)

        # End frame
        end = np.full((config.AVATAR_HEIGHT, config.AVATAR_WIDTH, 3),
                      config.BG_COLOR, dtype=np.uint8)
        cv2.putText(end, "Done",
                    (config.AVATAR_WIDTH // 2 - 38, config.AVATAR_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 100), 2, cv2.LINE_AA)
        for _ in range(config.AVATAR_FPS):
            all_frames.append(end)

        print(f"\nTotal frames : {len(all_frames)}")

        gif_path = _ensure_gif_ext(output_path)
        saved    = self.renderer.frames_to_gif(all_frames, gif_path)

        if os.path.exists(saved):
            kb = os.path.getsize(saved) // 1024
            print(f"Saved : {os.path.abspath(saved)} ({kb} KB)")
        else:
            print(f"NOT saved at {saved}")

        return saved, glosses, len(all_frames)

    def wlasl_coverage(self) -> dict:
        return {
            "ready":      self._wlasl_ready,
            "word_count": len(self._wlasl_index),
            "words":      sorted(self._wlasl_index.keys()),
        }

    # ── Core priority resolver ────────────────────────────

    def _sign_frames(self, gloss: str, idx: int,
                     all_glosses: list, transcript: str):
        """
        Returns (frames, source_label) for one gloss token.

        Priority:
          1. Trained model   -- generated pose sequence (most fluid)
          2. WLASL video     -- real signer clip (accurate, no model needed)
          3. Keypoint        -- built-in handshape (instant fallback)
          4. Fingerspell     -- letter by letter (last resort)
        """
        prog = (idx + 1) / len(all_glosses)

        # 1. Trained pose model
        if self.inferencer.ready and self.inferencer.can_sign(gloss):
            seq    = self.inferencer.generate(gloss)
            frames = self._render_model_seq(seq, gloss, transcript, prog)
            return frames, "model"

        # 2. WLASL real video
        if self._wlasl_ready:
            wlasl = self._load_wlasl_clip(gloss)
            if wlasl:
                frames = self._overlay_label(wlasl, gloss, transcript, prog)
                return frames, "WLASL"

        # 3. Keypoint skeleton
        if self.mapper.has_sign(gloss):
            frames = self._render_keypoint(gloss, idx, all_glosses, transcript)
            return frames, "keypoint"

        # 4. Fingerspell
        frames = self._fingerspell(gloss, transcript)
        return frames, "fingerspell"

    # ── Private rendering methods ─────────────────────────

    def _render_model_seq(self, seq: np.ndarray, gloss: str,
                          transcript: str, prog: float) -> list:
        """Render a model-generated pose sequence as avatar frames."""
        frames = []
        for frame_vec in seq:
            hand_xy = frame_vec[:63].reshape(21, 3)[:, :2]
            hand_xy = np.clip(hand_xy, 0, 1)
            frames.append(
                self.renderer.render_frame(hand_xy, gloss, transcript, prog)
            )
        return frames

    def _load_wlasl_clip(self, word: str) -> list:
        entries = self._wlasl_index.get(word.upper(), [])
        if not entries:
            return []
        e = entries[0]
        return self._extract_frames(e["path"], e["frame_start"], e["frame_end"])

    def _extract_frames(self, path: str,
                        frame_start: int = 0,
                        frame_end:   int = -1) -> list:
        cap    = cv2.VideoCapture(path)
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
            frames.append(
                cv2.resize(frame, (config.AVATAR_WIDTH, config.AVATAR_HEIGHT))
            )
            idx += 1
        cap.release()
        return frames

    def _render_keypoint(self, gloss: str, idx: int,
                         all_glosses: list, transcript: str) -> list:
        kp     = self.mapper.get_keypoints(gloss)
        prog   = (idx + 1) / len(all_glosses)
        frames = []

        # Smooth blend from previous keypoint sign
        if idx > 0 and self.mapper.has_sign(all_glosses[idx - 1]):
            prev_kp = self.mapper.get_keypoints(all_glosses[idx - 1])
            for t in range(config.TRANSITION_FRAMES):
                alpha  = (t + 1) / (config.TRANSITION_FRAMES + 1)
                interp = prev_kp * (1 - alpha) + kp * alpha
                frames.append(
                    self.renderer.render_frame(interp, gloss, transcript, prog)
                )

        for _ in range(config.SIGN_HOLD_FRAMES):
            frames.append(
                self.renderer.render_frame(kp, gloss, transcript, prog)
            )
        return frames

    def _overlay_label(self, frames: list, gloss: str,
                       transcript: str, prog: float) -> list:
        result = []
        H, W   = config.AVATAR_HEIGHT, config.AVATAR_WIDTH
        for frame in frames:
            f = frame.copy()
            label = f"SIGN: {gloss}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(f, (14, 12), (14 + tw + 12, 12 + th + 10),
                          (0, 0, 0), -1)
            cv2.putText(f, label, (20, 12 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 220, 180), 2, cv2.LINE_AA)
            if transcript:
                cv2.putText(f, transcript[:55], (16, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1, cv2.LINE_AA)
            bar_w  = W - 40
            bar_y  = H - 8
            cv2.rectangle(f, (20, bar_y), (20 + bar_w, bar_y + 5),
                          (50, 50, 50), -1)
            cv2.rectangle(f, (20, bar_y),
                          (20 + int(bar_w * prog), bar_y + 5),
                          (0, 200, 150), -1)
            result.append(f)
        return result

    def _fingerspell(self, word: str, transcript: str) -> list:
        frames  = []
        letters = [c for c in word.upper() if c.isalpha()]
        total   = max(len(letters), 1)
        for i, letter in enumerate(letters):
            kp   = self.mapper.get_keypoints(letter)
            prog = (i + 1) / total
            for _ in range(config.SIGN_HOLD_FRAMES):
                frames.append(
                    self.renderer.render_frame(
                        kp, f"{letter} (spell)", transcript, prog)
                )
        return frames


def _ensure_gif_ext(path: str) -> str:
    for ext in (".mp4", ".avi", ".mov"):
        path = path.replace(ext, ".gif")
    return path if path.endswith(".gif") else path + ".gif"