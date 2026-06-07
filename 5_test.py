"""
Step 4 — Test the trained model
--------------------------------
Run: python 5_test_model.py

Generates signing sequences for test words and saves
them as GIFs so you can visually verify the output.
"""

import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from sign_inf import SignInferencer
from src.avatar_renderer import AvatarRenderer
import config

OUTPUT_DIR = "./output/model_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_WORDS = [
    "DOCTOR", "WHERE", "HELLO", "HELP",
    "WATER", "POLICE", "THANK", "YES", "NO"
]

print("Loading model...")
inferencer = SignInferencer()
renderer   = AvatarRenderer()

if not inferencer.ready:
    print("❌ Model not trained yet. Run 4_train_model.py first.")
    sys.exit(1)

print(f"\nGenerating test signs...\n")

for word in TEST_WORDS:
    if not inferencer.can_sign(word):
        print(f"  {word:15s} → ❌ not in model vocab")
        continue

    # Generate full sequence
    seq = inferencer.generate(word)   # (60, 75)
    print(f"  {word:15s} → sequence shape: {seq.shape}")

    # Render each frame
    frames = []
    for i, frame_vec in enumerate(seq):
        # Extract hand xy from frame vector
        hand_xy = frame_vec[:63].reshape(21, 3)[:, :2]

        # Normalize to 0-1 range for renderer
        hand_xy = np.clip(hand_xy, 0, 1)

        progress = (i + 1) / len(seq)
        frame    = renderer.render_frame(hand_xy, word, f"Model generated: {word}", progress)
        frames.append(frame)

    # Save as GIF
    out_path = os.path.join(OUTPUT_DIR, f"{word.lower()}.gif")
    saved    = renderer.frames_to_gif(frames, out_path)
    kb       = os.path.getsize(saved) // 1024
    print(f"  {'':15s}   saved: {saved}  ({kb} KB)")

print(f"\n✅ Done. Check {OUTPUT_DIR} for generated GIFs.")
print(f"\nIf the hand shapes look roughly correct for each sign,")
print(f"the model is working. Run python 2_app.py to use it in the app.")