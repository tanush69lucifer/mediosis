"""BiLSTM-CRF tagger for prescription lines: DRUG / STRENGTH / FORM / FREQ / INSTR / DUR.

Input features per token = fastText subword vector (OOV-safe, so garbled OCR words still get a
meaningful vector) + 4 shape features. Spatial dropout is applied over the feature channels.
"""
import re

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score
from torchcrf import CRF

from .rx_data import synth_dataset

LABELS = ["O"] + [f"{p}-{t}" for t in ["DRUG", "STRENGTH", "FORM", "FREQ", "INSTR", "DUR"] for p in "BI"]
L2I = {l: i for i, l in enumerate(LABELS)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shape(tok):
    return [float(tok[0].isdigit()),
            float(bool(re.search(r"(mg|ml|mcg|g|iu)$", tok)) and tok[0].isdigit()),
            float(bool(re.fullmatch(r"\d-\d-\d", tok))),
            min(len(tok), 15) / 15]


def featurize(ft, toks):
    return np.stack([np.concatenate([ft.wv.get_vector(t), shape(t)]) for t in toks]).astype(np.float32)


class BiLSTMCRF(nn.Module):
    def __init__(self, in_dim, hidden=128, n_tags=len(LABELS), dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.spatial_dropout = nn.Dropout1d(dropout)  # drops whole feature channels (SpatialDropout1D)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, n_tags)
        self.crf = CRF(n_tags, batch_first=True)

    def emissions(self, x):
        h = torch.relu(self.proj(x))                       # (B,T,H)
        h = self.spatial_dropout(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.out(h)

    def loss(self, x, y, mask):
        return -self.crf(self.emissions(x), y, mask=mask, reduction="mean")

    def decode(self, x, mask):
        return self.crf.decode(self.emissions(x), mask=mask)


def _batch(ft, data):
    T = max(len(t) for t, _ in data)
    D = 100 + 4
    X = np.zeros((len(data), T, D), np.float32)
    Y = np.zeros((len(data), T), np.int64)
    M = np.zeros((len(data), T), bool)
    for i, (toks, tags) in enumerate(data):
        X[i, :len(toks)] = featurize(ft, toks)
        if tags:
            Y[i, :len(tags)] = [L2I[t] for t in tags]
        M[i, :len(toks)] = True
    return (torch.tensor(X, device=DEVICE), torch.tensor(Y, device=DEVICE),
            torch.tensor(M, device=DEVICE))


def train_ner(ft, n_train=8000, epochs=6, bs=64, seed=0, log=print):
    torch.manual_seed(seed)
    data = synth_dataset(n_train, seed=seed)
    model = BiLSTMCRF(104).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        model.train()
        perm = np.random.RandomState(ep).permutation(len(data))
        tot = 0
        for b in range(0, len(data), bs):
            X, Y, M = _batch(ft, [data[i] for i in perm[b:b + bs]])
            loss = model.loss(X, Y, M)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        log(f"  NER epoch {ep + 1}/{epochs}  loss={tot / (len(data) / bs):.3f}")
    return model


@torch.no_grad()
def predict(model, ft, token_lists):
    model.eval()
    X, _, M = _batch(ft, [(t, None) for t in token_lists])
    return [[LABELS[i] for i in seq] for seq in model.decode(X, M)]


def evaluate_ner(model, ft, n=2000, seed=999, noise_p=0.6):
    """Held-out set: different seed + heavier OCR noise than training."""
    data = synth_dataset(n, seed=seed, noise_p=noise_p)
    pred = []
    for b in range(0, n, 256):
        pred += predict(model, ft, [t for t, _ in data[b:b + 256]])
    gold = [g for _, g in data]
    return f1_score(gold, pred), classification_report(gold, pred, digits=3)


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path):
    m = BiLSTMCRF(104).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m
