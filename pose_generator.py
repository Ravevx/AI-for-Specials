"""
        Returns (frames, source_label) for one gloss token.

        Priority:
          1. Trained model   -- generated pose sequence (most fluid)
          2. WLASL video     -- real signer clip (most accurate when model not ready)
          3. Keypoint        -- built-in handshape (instant fallback)
          4. Fingerspell     -- letter by letter (last resort)
        """
        prog = (idx + 1) / len(all_glosses)

        # 1. Trained pose model
        if self.inferencer.ready and self.inferencer.can_sign(gloss):
            seq    = self.inferencer.generate(gloss)   # (60, 75)
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

    # ── Private ───────────────────────────────────────────

    def _render_model_seq(self, seq: "np.ndarray", gloss: str,
                           transcript: str, prog: float) -> list:
        """Render a model-generated pose sequence as avatar frames."""
        import numpy as np
        frames = []
        for i, frame_vec in enumerate(seq):
            hand_xy = frame_vec[:63].reshape(21, 3)[:, :2]
            hand_xy = np.clip(hand_xy, 0, 1)
            frame_prog = prog
            frames.append(
                self.renderer.render_frame(hand_xy, gloss, transcript, frame_prog)
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