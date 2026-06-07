"""
Step 2 — Train Conditional VAE Sign Model
------------------------------------------
Architecture: Conditional Variational Autoencoder (CVAE)

Why CVAE instead of plain LSTM:
  - Plain LSTM gets label -> outputs average of all examples = blurry motion
  - CVAE gets label + latent noise -> outputs diverse, distinct motion per sign
  - Encoder compresses real sequences to latent space during training
  - Decoder generates sequences from label + sampled noise at inference
  - This forces the model to learn WHAT makes each sign distinct

Expected loss: 0.05-0.15 (vs 0.45+ with plain LSTM)

Run:
  python 4_train_model.py

Training time: ~20-40 min on RTX 2060
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle

# ── Config ──────────────────────────────────────────────────
POSES_DIR    = r"J:\Agent My Learning\agent 5\data\poses"
MODEL_DIR    = r"J:\Agent My Learning\agent 5\data\model"
MODEL_PATH   = os.path.join(MODEL_DIR, "sign_model.pt")
VOCAB_PATH   = os.path.join(MODEL_DIR, "vocab.pkl")

VECTOR_SIZE  = 63     # 21 hand landmarks x 3
MAX_SEQ_LEN  = 60     # all sequences normalised to this length
EMBED_DIM    = 128    # sign label embedding
HIDDEN_DIM   = 256    # LSTM hidden size
LATENT_DIM   = 64     # VAE latent space size
NUM_LAYERS   = 2      # LSTM layers
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 3e-4   # Adam sweet spot for VAEs
KL_WEIGHT    = 0.001  # weight of KL loss (start small, model learns structure first)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────


# ── Dataset ──────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, poses_dir):
        self.samples  = []
        self.vocab    = {}
        self.idx2word = {}

        print("Loading pose sequences...")
        for word_dir in sorted(Path(poses_dir).iterdir()):
            if not word_dir.is_dir():
                continue
            word  = word_dir.name.upper()
            files = list(word_dir.glob("*.npy"))
            if not files:
                continue

            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word]   = idx
                self.idx2word[idx] = word

            label_idx = self.vocab[word]
            for f in files:
                try:
                    seq = np.load(str(f)).astype(np.float32)
                    if seq.shape[1] != VECTOR_SIZE:
                        continue
                    seq = self._normalise(seq)
                    seq = self._resample(seq)
                    self.samples.append((label_idx, seq))
                except Exception:
                    continue

        print(f"  {len(self.vocab)} words, {len(self.samples)} sequences loaded.")

    def _normalise(self, seq):
        # Per-sequence normalisation: zero mean, unit std
        mean = seq.mean(axis=0, keepdims=True)
        std  = seq.std(axis=0,  keepdims=True) + 1e-8
        return (seq - mean) / std

    def _resample(self, seq):
        n = len(seq)
        if n == MAX_SEQ_LEN:
            return seq
        indices = np.linspace(0, n - 1, MAX_SEQ_LEN).astype(int)
        return seq[indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, seq = self.samples[idx]
        return (torch.tensor(label, dtype=torch.long),
                torch.tensor(seq,   dtype=torch.float32))


# ── Model ────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Encodes a real pose sequence + label into a latent distribution (mu, logvar).
    Used only during TRAINING to give the decoder good latent codes to learn from.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm   = nn.LSTM(VECTOR_SIZE + EMBED_DIM, HIDDEN_DIM,
                              NUM_LAYERS, batch_first=True, dropout=0.1)
        self.mu     = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, seq, label):
        emb   = self.embed(label).unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1)
        x     = torch.cat([seq, emb], dim=-1)          # (B, T, 63+128)
        _, (h, _) = self.lstm(x)
        h     = h[-1]                                   # last layer hidden (B, 256)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """
    Generates a pose sequence from label + latent code.
    At inference: label + random noise -> new signing sequence.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm  = nn.LSTM(EMBED_DIM + LATENT_DIM, HIDDEN_DIM,
                             NUM_LAYERS, batch_first=True, dropout=0.1)
        self.out   = nn.Linear(HIDDEN_DIM, VECTOR_SIZE)

    def forward(self, label, z):
        emb = self.embed(label)                              # (B, 128)
        z_e = torch.cat([emb, z], dim=-1)                   # (B, 128+64)
        x   = z_e.unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1) # (B, T, 192)
        out, _ = self.lstm(x)
        return self.out(out)                                 # (B, T, 63)


class SignCVAE(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def reparameterise(self, mu, logvar):
        """Sample z = mu + eps * std (differentiable)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, seq, label):
        mu, logvar = self.encoder(seq, label)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(label, z)
        return recon, mu, logvar

    def generate(self, label_idx, n_samples=1):
        """Inference: sample from prior N(0,1) and decode."""
        self.eval()
        with torch.no_grad():
            label = torch.tensor([label_idx] * n_samples,
                                  dtype=torch.long, device=DEVICE)
            z     = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
            seq   = self.decoder(label, z)
        return seq.cpu().numpy()   # (n_samples, T, 63)


# ── Loss ────────────────────────────────────────────────────

def cvae_loss(recon, target, mu, logvar, kl_weight):
    # Reconstruction: MSE on positions
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # Velocity: frame-to-frame differences should also match
    pred_vel   = recon[:, 1:, :] - recon[:, :-1, :]
    targ_vel   = target[:, 1:, :] - target[:, :-1, :]
    vel_loss   = F.mse_loss(pred_vel, targ_vel, reduction="mean")

    # KL divergence: keep latent space close to N(0,1)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + 0.5 * vel_loss + kl_weight * kl_loss, recon_loss


# ── Training ────────────────────────────────────────────────

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")

    dataset = SignDataset(POSES_DIR)
    if not dataset.samples:
        print("No pose data found. Run 3_extract_poses.py first.")
        return

    loader     = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, pin_memory=True)
    vocab_size = len(dataset.vocab)

    print(f"Vocab  : {vocab_size} signs")
    print(f"Samples: {len(dataset)}")

    # Save vocab
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"vocab": dataset.vocab, "idx2word": dataset.idx2word}, f)

    model     = SignCVAE(vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}  (~{n_params*4//1024} KB)\n")

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_recon = 0.0

        # Anneal KL weight: start near 0, reach full weight at epoch 30
        # This lets the model learn reconstruction first before worrying about KL
        kl_w = min(KL_WEIGHT, KL_WEIGHT * epoch / 30)

        for label_idx, target_seq in loader:
            label_idx  = label_idx.to(DEVICE)
            target_seq = target_seq.to(DEVICE)

            recon, mu, logvar = model(target_seq, label_idx)
            loss, recon_loss  = cvae_loss(recon, target_seq, mu, logvar, kl_w)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_recon += recon_loss.item()

        scheduler.step()
        avg_loss  = total_loss  / len(loader)
        avg_recon = total_recon / len(loader)

        if avg_recon < best_loss:
            best_loss = avg_recon
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "vocab_size":  vocab_size,
                "loss":        best_loss,
                "arch":        "cvae",
            }, MODEL_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss: {avg_loss:.4f}  "
                  f"recon: {avg_recon:.4f}  "
                  f"best: {best_loss:.4f}  "
                  f"lr: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\nTraining complete. Best recon loss: {best_loss:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"\nNext: python 5_test_model.py")


if __name__ == "__main__":
    train()