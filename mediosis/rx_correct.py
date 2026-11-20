"""fastText subword drug-name corrector.

Idea: OCR breaks drug names ("Azithrornycin"). fastText builds a word vector from its
character n-grams, so a misspelt word still lands near the real one. We train fastText on
(a) synthetic prescription lines and (b) 'alias sentences' where each canonical drug
co-occurs with simulated OCR corruptions of itself, then rank candidates by an ensemble of
fastText cosine similarity and edit-distance similarity.
"""
import random

import numpy as np
from gensim.models import FastText
from rapidfuzz import fuzz

from .lexicon import CANONICAL
from .rx_data import ocr_noise, synth_dataset

DIM = 100


def build_corpus(seed=0, n_lines=20000, n_alias=40):
    rng = random.Random(seed)
    corpus = [toks for toks, _ in synth_dataset(n_lines, seed=seed)]
    for d in CANONICAL:
        for _ in range(n_alias):
            sent = [d] + [ocr_noise(d, rng=rng) for _ in range(4)] + [d]
            rng.shuffle(sent)
            corpus.append(sent)
    return corpus


def train_fasttext(seed=0, epochs=10):
    corpus = build_corpus(seed)
    ft = FastText(vector_size=DIM, window=4, min_count=1, sg=1, min_n=2, max_n=5,
                  workers=4, seed=seed, bucket=200000)
    ft.build_vocab(corpus_iterable=corpus)
    ft.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=epochs)
    return ft


class DrugCorrector:
    def __init__(self, ft, alpha=0.5, threshold=0.55):
        self.ft, self.alpha, self.threshold = ft, alpha, threshold
        M = np.stack([ft.wv.get_vector(c) for c in CANONICAL])
        self.M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

    def scores(self, word, mode="ensemble"):
        word = word.lower()
        v = self.ft.wv.get_vector(word)  # works for OOV via n-grams
        cos = self.M @ (v / (np.linalg.norm(v) + 1e-9))
        lev = np.array([fuzz.ratio(word, c) / 100 for c in CANONICAL])
        if mode == "fasttext":
            return cos
        if mode == "fuzzy":
            return lev
        return self.alpha * cos + (1 - self.alpha) * lev

    def correct(self, word, mode="ensemble"):
        s = self.scores(word, mode)
        i = int(np.argmax(s))
        conf = float(s[i])
        return (CANONICAL[i] if conf >= self.threshold else None), conf


def evaluate(corrector, n_per_drug=20, seed=123, n_edits=None, tests=None):
    """Top-1 accuracy on held-out OCR corruptions (different seed than training)."""
    rng = random.Random(seed)
    tests = tests or [(ocr_noise(d, n_edits=n_edits, rng=rng), d) for d in CANONICAL for _ in range(n_per_drug)]
    out = {}
    for mode in ["fuzzy", "fasttext", "ensemble"]:
        hit = sum(CANONICAL[int(np.argmax(corrector.scores(w, mode)))] == d for w, d in tests)
        out[mode] = hit / len(tests)
    return out, len(tests)
