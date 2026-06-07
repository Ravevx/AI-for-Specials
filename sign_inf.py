"""
Sign Inferencer — CVAE version
--------------------------------
Loads the trained CVAE model and generates pose sequences at inference time.
At inference: label + random noise from N(0,1) -> unique signing sequence.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

# Must match 4_train_model.py exactly
VECTOR_SIZE  = 63
MAX_SEQ_LEN  = 60
EMBED_DIM    = 128
HIDDEN_DIM   = 256
LATENT_DIM   = 64
NUM_LAYERS   = 2

MODEL_DIR  = r"J:\Agent My Learning\agent 5\data\model"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model definition (must match training) ──────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm   = nn.LSTM(VECTOR_SIZE + EMBED_DIM, HIDDEN_DIM,
                              NUM_LAYERS, batch_first=True, dropout=0.1)
        self.mu     = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, seq, label):
        emb = self.embed(label).unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1)
        x   = torch.cat([seq, emb], dim=-1)
        _, (h, _) = self.lstm(x)
        return self.mu(h[-1]), self.logvar(h[-1])


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm  = nn.LSTM(EMBED_DIM + LATENT_DIM, HIDDEN_DIM,
                             NUM_LAYERS, batch_first=True, dropout=0.1)
        self.out   = nn.Linear(HIDDEN_DIM, VECTOR_SIZE)

    def forward(self, label, z):
        emb = self.embed(label)
        z_e = torch.cat([emb, z], dim=-1)
        x   = z_e.unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1)
        out, _ = self.lstm(x)
        return self.out(out)


class SignCVAE(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def generate(self, label_idx, n_samples=1):
        self.eval()
        with torch.no_grad():
            label = torch.tensor([label_idx] * n_samples,
                                  dtype=torch.long, device=DEVICE)
            z     = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
            seq   = self.decoder(label, z)
        return seq.cpu().numpy()


# ── Inferencer ───────────────────────────────────────────────

class SignInferencer:
    def __init__(self):
        self.model    = None
        self.vocab    = {}
        self.idx2word = {}
        self.ready    = False
        self._load()

    def _load(self):
        if not os.path.isfile(MODEL_PATH):
            print(f"  No trained model at {MODEL_PATH}")
            print(f"  Run 4_train_model.py first.")
            return
        if not os.path.isfile(VOCAB_PATH):
            print(f"  No vocab at {VOCAB_PATH}")
            return

        with open(VOCAB_PATH, "rb") as f:
            v = pickle.load(f)
            self.vocab    = v["vocab"]
            self.idx2word = v["idx2word"]

        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        arch       = ckpt.get("arch", "lstm")

        if arch != "cvae":
            print("  Old LSTM model detected — please retrain with 4_train_model.py")
            return

        vocab_size = ckpt["vocab_size"]
        self.model = SignCVAE(vocab_size).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.ready = True

        print(f"  Sign model ready: {len(self.vocab)} signs  "
              f"(loss: {ckpt['loss']:.4f})")

    def can_sign(self, word: str) -> bool:
        return self.ready and word.upper() in self.vocab

    def generate(self, word: str) -> np.ndarray:
        """
        Generate a pose sequence for a word.
        Returns (MAX_SEQ_LEN, 63) array or None.
        """
        if not self.ready:
            return None
        word = word.upper()
        if word not in self.vocab:
            return None
        idx = self.vocab[word]
        seq = self.model.generate(idx)   # (1, 60, 63)
        return seq[0]                    # (60, 63)

    def generate_keypoints(self, word: str) -> np.ndarray:
        """
        Return peak-frame 21x2 hand keypoints for avatar renderer.
        """
        seq = self.generate(word)
        if seq is None:
            return None
        # Peak = frame with highest hand movement variance
        peak_idx = seq.var(axis=1).sum(axis=-1).argmax()  # wrong shape fix
        peak     = seq[peak_idx]                           # (63,)
        hand_xy  = peak.reshape(21, 3)[:, :2]             # (21, 2)
        return np.clip(hand_xy, 0, 1)