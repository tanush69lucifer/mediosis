"""Tokenizer, OCR-noise simulator and synthetic prescription generator (BIO-tagged)."""
import random
import re

from .lexicon import CANONICAL

TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?(?:mg|ml|mcg|gm|g|iu)\b"   # 650mg
    r"|\d-\d-\d"                                # 1-0-1
    r"|[a-z0-9]*[a-z][a-z0-9]*"                 # words (OCR may put digits inside: 0mez)
    r"|\d+(?:\.\d+)?"                           # numbers
)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# Typical OCR confusions (both directions)
_CONF = [("o", "0"), ("l", "1"), ("i", "l"), ("m", "rn"), ("c", "e"), ("a", "o"),
         ("s", "5"), ("b", "6"), ("g", "9"), ("z", "2"), ("d", "cl"), ("n", "h"),
         ("u", "v"), ("e", "c"), ("t", "f"), ("r", "n")]


def ocr_noise(word, n_edits=None, rng=random):
    """Corrupt a word the way OCR / sloppy handwriting does."""
    w = word
    n_edits = n_edits or rng.choice([1, 1, 2])
    for _ in range(n_edits):
        op = rng.random()
        if op < 0.5:  # confusion substitution
            pairs = [(a, b) for a, b in _CONF if a in w] + [(b, a) for a, b in _CONF if b in w]
            if pairs:
                a, b = rng.choice(pairs)
                idx = [m.start() for m in re.finditer(re.escape(a), w)]
                i = rng.choice(idx)
                w = w[:i] + b + w[i + len(a):]
                continue
        if len(w) > 3 and op < 0.7:  # deletion
            i = rng.randrange(1, len(w))
            w = w[:i] + w[i + 1:]
        elif op < 0.85:  # insertion
            i = rng.randrange(len(w) + 1)
            w = w[:i] + rng.choice("aeilmnorst") + w[i:]
        elif len(w) > 2:  # adjacent swap
            i = rng.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w if w != word else word + rng.choice("aeil")


FORMS = ["tab", "cap", "syp", "inj", "tabs", "susp"]
STRENGTHS = ["250mg", "500mg", "650mg", "40mg", "20mg", "10mg", "5mg", "1g", "100mg", "200mg", "75mg", "5ml", "10ml"]
STRENGTHS_SPLIT = [("500", "mg"), ("650", "mg"), ("40", "mg"), ("20", "mg"), ("5", "ml")]
FREQS = [[f] for f in ["od", "bd", "bid", "tds", "tid", "qid", "hs", "sos", "stat",
                        "1-0-1", "1-1-1", "0-0-1", "1-0-0", "0-1-0", "1-1-0"]] + [
    ["once", "daily"], ["twice", "daily"], ["thrice", "daily"], ["daily"], ["at", "night"],
    ["three", "times", "a", "day"], ["twice", "a", "day"], ["when", "required"]]
INSTRS = [["after", "food"], ["before", "food"], ["ac"], ["pc"], ["empty", "stomach"], ["with", "milk"],
          ["after", "meals"], ["before", "breakfast"], ["with", "water"]]
DURS = [["x", "{n}", "days"], ["for", "{n}", "days"], ["{n}", "days"], ["x", "{n}", "d"],
        ["for", "{n}", "weeks"], ["x", "{n}", "wk"]]
FILLERS = ["rx", "dr", "advice", "take"] + [str(i) for i in range(1, 16)]
TRAILERS = [["if", "fever"], ["then", "stop"], ["do", "not", "skip"], ["review", "after"]]


def synth_line(rng=random, noise_p=0.35):
    """Return (tokens, tags) for one synthetic prescription line."""
    toks, tags = [], []

    def add(ts, label):
        for i, t in enumerate(ts):
            toks.append(t)
            tags.append(label if label == "O" else ("B-" if i == 0 else "I-") + label)

    if rng.random() < 0.4:
        add([rng.choice(FILLERS)], "O")
    if rng.random() < 0.8:
        add([rng.choice(FORMS)], "FORM")
    drug = rng.choice(CANONICAL)
    if rng.random() < noise_p:
        drug = ocr_noise(drug, rng=rng)
    add([drug], "DRUG")
    r = rng.random()
    if r < 0.6:
        add([rng.choice(STRENGTHS)], "STRENGTH")
    elif r < 0.8:
        add(list(rng.choice(STRENGTHS_SPLIT)), "STRENGTH")
    parts = []
    parts.append(("FREQ", list(rng.choice(FREQS))))
    if rng.random() < 0.6:
        parts.append(("INSTR", list(rng.choice(INSTRS))))
    if rng.random() < 0.75:
        n = str(rng.choice([3, 5, 7, 10, 14, 1, 2, 30]))
        parts.append(("DUR", [t.replace("{n}", n) for t in rng.choice(DURS)]))
    rng.shuffle(parts)
    for lab, ts in parts:
        add(ts, lab)
    if rng.random() < 0.15:
        add(rng.choice(TRAILERS), "O")
    return toks, tags


def synth_dataset(n, seed=0, noise_p=0.35):
    rng = random.Random(seed)
    return [synth_line(rng, noise_p) for _ in range(n)]
