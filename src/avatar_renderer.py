"""
Avatar Renderer
---------------
Renders ASL signing as a clean skeleton avatar using OpenCV.
Draws:
  - Body silhouette (head + torso + arms) for spatial context
  - Dominant (right) hand skeleton with MediaPipe 21-point landmarks
  - Sign label + transcript overlay
"""

import cv2
import numpy as np
import config

# MediaPipe hand bone connections (21 landmarks)
HAND_CONNECTIONS = [
    (0,  1), (1, 2),  (2,  3),  (3,  4),   # Thumb
    (0,  5), (5, 6),  (6,  7),  (7,  8),   # Index
    (0,  9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13,14), (14, 15), (15, 16),  # Ring
    (0, 17), (17,18), (18, 19), (19, 20),  # Pinky
    (5,  9), (9, 13), (13, 17),            # Palm
]

# Finger-tip landmark indices (for highlighting)
FINGERTIPS = [4, 8, 12, 16, 20]


class AvatarRenderer:
    def __init__(self):
        self.W = config.AVATAR_WIDTH
        self.H = config.AVATAR_HEIGHT
        self._precompute_body()

    def _precompute_body(self):
        """
        Pre-compute static body silhouette coordinates.
        Body is centred, head near top, hand signing area in lower-centre.
        """
        cx = self.W // 2
        # Head
        self.head_center = (cx, 80)
        self.head_radius = 42
        # Neck
        self.neck_top    = (cx, 122)
        self.neck_bot    = (cx, 155)
        # Shoulders
        self.shoulder_l  = (cx - 95, 155)
        self.shoulder_r  = (cx + 95, 155)
        # Torso bottom
        self.torso_bot_l = (cx - 70, 310)
        self.torso_bot_r = (cx + 70, 310)
        # Arms
        self.elbow_r     = (cx + 140, 240)
        self.wrist_r     = (cx + 150, 330)
        self.elbow_l     = (cx - 140, 240)
        self.wrist_l     = (cx - 150, 330)

    # ── Public API ────────────────────────────────────────

    def render_frame(self, keypoints: np.ndarray,
                     label: str = "",
                     transcript: str = "",
                     progress: float = 0.0) -> np.ndarray:
        """
        Render one frame.
        keypoints : 21×2 normalized hand landmarks
        label     : sign name to display
        transcript: full sentence shown at bottom
        progress  : 0.0–1.0 sign sequence progress (for progress bar)
        """
        frame = np.full((self.H, self.W, 3), config.BG_COLOR, dtype=np.uint8)
        self._draw_body(frame)
        self._draw_hand(frame, keypoints)
        self._draw_hud(frame, label, transcript, progress)
        return frame

    def render_sign_sequence(self, glosses: list,
                              mapper,
                              transcript: str = "") -> list:
        """
        Render full sign sequence as a list of frames.
        Includes smooth interpolation between signs.
        """
        frames = []
        hold   = config.SIGN_HOLD_FRAMES
        trans  = config.TRANSITION_FRAMES
        total  = len(glosses)

        for i, gloss in enumerate(glosses):
            kp      = mapper.get_keypoints(gloss)
            progress = (i + 1) / total

            # Transition from previous sign
            if i > 0:
                prev_kp = mapper.get_keypoints(glosses[i - 1])
                for t in range(trans):
                    alpha = (t + 1) / (trans + 1)
                    interp = prev_kp * (1 - alpha) + kp * alpha
                    frames.append(
                        self.render_frame(interp, gloss, transcript, progress)
                    )

            # Hold current sign
            for _ in range(hold):
                frames.append(
                    self.render_frame(kp, gloss, transcript, progress)
                )

        # End frame
        end = np.full((self.H, self.W, 3), config.BG_COLOR, dtype=np.uint8)
        self._draw_body(end)
        cv2.putText(end, "Done", (self.W // 2 - 38, self.H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 100), 2, cv2.LINE_AA)
        for _ in range(config.AVATAR_FPS):   # 1 second hold
            frames.append(end)

        return frames

    def frames_to_video(self, frames: list, output_path: str) -> str:
        """Save frames to MP4. Falls back to AVI if H264 unavailable."""
        import os
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else "./output",
            exist_ok=True
        )

        # Try H264 first (browser-compatible)
        for fourcc_str, ext in [("avc1", ".mp4"), ("mp4v", ".mp4"), ("XVID", ".avi")]:
            path = output_path if output_path.endswith(ext) else \
                   output_path.rsplit(".", 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out    = cv2.VideoWriter(path, fourcc, config.AVATAR_FPS, (self.W, self.H))
            if out.isOpened():
                for f in frames:
                    out.write(f)
                out.release()
                return path
        raise RuntimeError("Could not open any VideoWriter codec.")

    def frames_to_gif(self, frames: list, output_path: str) -> str:
        """Save frames as animated GIF (always works, no codec needed)."""
        import os
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required for GIF: pip install Pillow")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else "./output",
            exist_ok=True
        )
        pil_frames = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        ms_per_frame = int(1000 / config.AVATAR_FPS)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=ms_per_frame,
            loop=0,
            optimize=False,
        )
        return output_path

    # ── Private drawing helpers ───────────────────────────

    def _draw_body(self, frame: np.ndarray):
        """Draw a simple body silhouette for spatial context."""
        c = (80, 80, 80)   # subtle grey

        # Head
        cv2.circle(frame, self.head_center, self.head_radius, c, 2, cv2.LINE_AA)

        # Neck
        cv2.line(frame, self.neck_top, self.neck_bot, c, 2, cv2.LINE_AA)

        # Shoulders
        cv2.line(frame, self.shoulder_l, self.shoulder_r, c, 2, cv2.LINE_AA)

        # Torso
        cv2.line(frame, self.shoulder_l, self.torso_bot_l, c, 2, cv2.LINE_AA)
        cv2.line(frame, self.shoulder_r, self.torso_bot_r, c, 2, cv2.LINE_AA)
        cv2.line(frame, self.torso_bot_l, self.torso_bot_r, c, 2, cv2.LINE_AA)

        # Arms
        cv2.line(frame, self.shoulder_r, self.elbow_r, c, 2, cv2.LINE_AA)
        cv2.line(frame, self.elbow_r, self.wrist_r, c, 2, cv2.LINE_AA)
        cv2.line(frame, self.shoulder_l, self.elbow_l, c, 2, cv2.LINE_AA)
        cv2.line(frame, self.elbow_l, self.wrist_l, c, 2, cv2.LINE_AA)

    def _keypoints_to_pixels(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Map normalized 0–1 keypoints into the signing region.
        Hand occupies the centre-right of the frame, below the torso.
        Region: x 220–540, y 160–440
        """
        px = keypoints.copy()
        # x: map [0.1, 0.9] → [220, 540]
        px[:, 0] = (keypoints[:, 0] - 0.1) / 0.8 * 320 + 220
        # y: map [0.15, 0.95] → [160, 440]
        px[:, 1] = (keypoints[:, 1] - 0.15) / 0.80 * 280 + 160
        return px.astype(int)

    def _draw_hand(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw hand skeleton with bones and joints."""
        px = self._keypoints_to_pixels(keypoints)

        # Bones
        for (a, b) in HAND_CONNECTIONS:
            pt_a = tuple(np.clip(px[a], [0, 0], [self.W - 1, self.H - 1]))
            pt_b = tuple(np.clip(px[b], [0, 0], [self.W - 1, self.H - 1]))
            cv2.line(frame, pt_a, pt_b, config.BONE_COLOR, 2, cv2.LINE_AA)

        # Joints
        for i, (x, y) in enumerate(px):
            x = int(np.clip(x, 0, self.W - 1))
            y = int(np.clip(y, 0, self.H - 1))
            if i == 0:
                # Wrist — larger
                cv2.circle(frame, (x, y), 7, config.HAND_COLOR, -1, cv2.LINE_AA)
            elif i in FINGERTIPS:
                # Finger tips — accent colour
                cv2.circle(frame, (x, y), 6, (255, 255, 100), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 6, config.HAND_COLOR, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 4, config.HAND_COLOR, -1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray,
                  label: str, transcript: str, progress: float):
        """Draw sign label, transcript, and progress bar."""

        # ── Sign label ──
        if label:
            text  = f"SIGN: {label}"
            scale = 0.8
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            # Background pill
            cv2.rectangle(frame, (14, 12), (14 + tw + 12, 12 + th + 10),
                          (40, 40, 40), -1, cv2.LINE_AA)
            cv2.putText(frame, text, (20, 12 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 180), thick, cv2.LINE_AA)

        # ── Transcript ──
        if transcript:
            words = transcript.split()
            line, lines, max_chars = "", [], 42
            for w in words:
                if len(line) + len(w) + 1 <= max_chars:
                    line += w + " "
                else:
                    lines.append(line.strip())
                    line = w + " "
            lines.append(line.strip())
            display = lines[-2:]  # last 2 lines

            y_start = self.H - 16 - len(display) * 26
            for i, ln in enumerate(display):
                cv2.putText(frame, ln, (16, y_start + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                            (160, 160, 160), 1, cv2.LINE_AA)

        # ── Progress bar ──
        if progress > 0:
            bar_w  = self.W - 40
            bar_h  = 5
            bar_y  = self.H - 8
            filled = int(bar_w * progress)
            cv2.rectangle(frame, (20, bar_y), (20 + bar_w, bar_y + bar_h),
                          (50, 50, 50), -1)
            cv2.rectangle(frame, (20, bar_y), (20 + filled, bar_y + bar_h),
                          (0, 200, 150), -1)