"""Radiology-report NLP: find each pathology mention and decide if the report says it is
PRESENT, ABSENT or UNCERTAIN.

Two assertion engines:
  * NegEx-style rules (baseline)
  * Keras BiGRU + self-attention classifier (Embedding -> SpatialDropout1D -> BiGRU with
    recurrent_dropout -> MultiHeadAttention self-attention -> pooling -> softmax), with our own
    vocabulary hash table and PRE-padding.
"""
import json
import os
import random
import re

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import numpy as np

CLASSES = ["present", "absent", "uncertain"]
MAXLEN = 40

SYN = {
    "Atelectasis": ["atelectasis", "atelectatic", "collapse"],
    "Consolidation": ["consolidation", "consolidations", "airspace disease"],
    "Infiltration": ["infiltrate", "infiltrates", "infiltration"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces"],
    "Edema": ["pulmonary edema", "edema", "oedema", "vascular congestion"],
    "Emphysema": ["emphysema", "hyperinflation", "hyperinflated lungs"],
    "Fibrosis": ["fibrosis", "fibrotic changes", "scarring"],
    "Effusion": ["pleural effusions", "pleural effusion", "effusions", "effusion", "costophrenic blunting"],
    "Pneumonia": ["pneumonia", "pneumonic"],
    "Pleural_Thickening": ["pleural thickening"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement", "heart is enlarged",
                     "heart size is enlarged", "enlarged cardiac silhouette"],
    "Nodule": ["nodules", "nodule", "nodular opacity"],
    "Mass": ["mass", "masses"],
    "Lung Lesion": ["lesion", "lesions"],
    "Fracture": ["fractures", "fracture"],
    "Lung Opacity": ["opacities", "opacity", "opacification"],
    "Enlarged Cardiomediastinum": ["widened mediastinum", "mediastinal widening", "enlarged cardiomediastinal silhouette"],
    "Hernia": ["hiatal hernia", "hernia"],
}
PHRASES = sorted([(s, lab) for lab, ss in SYN.items() for s in ss], key=lambda x: -len(x[0]))

NORMAL_PATTERNS = [
    (r"heart size (is )?(normal|within normal limits)|normal heart size|heart is normal in size|"
     r"(cardiac|cardiomediastinal) silhouette (is )?(normal|within normal limits|unremarkable)|normal cardiac silhouette",
     ["Cardiomegaly", "Enlarged Cardiomediastinum"]),
    (r"lungs are clear|clear lungs|lungs are well expanded and clear",
     ["Consolidation", "Lung Opacity", "Pneumonia", "Edema", "Infiltration", "Atelectasis"]),
    (r"no acute cardiopulmonary (process|abnormality|disease|findings?)",
     ["Consolidation", "Lung Opacity", "Pneumonia", "Edema", "Effusion", "Pneumothorax", "Cardiomegaly"]),
]

PRE_NEG = ["no", "without", "negative for", "free of", "no evidence of", "absence of", "resolution of",
           "no signs of", "not", "nor", "clear of", "ruled out"]
POST_NEG = ["is not seen", "are not seen", "not identified", "is absent", "are absent", "has resolved",
            "have resolved", "is ruled out", "not present", "not demonstrated", "no longer visualized", "no longer seen"]
UNCERTAIN = ["possible", "possibly", "may represent", "may be", "cannot exclude", "can not exclude",
             "cannot be excluded", "not excluded", "not entirely excluded", "suspicious for", "questionable",
             "likely", "suggestive of", "concerning for", "probable", "could represent", "differential", "versus",
             "?", "borderline", "equivocal"]
SCOPE_BREAK = r"\b(but|however|although|though|with interval|which)\b"


def sentences(text):
    return [s.strip() for s in re.split(r"[.;\n]+", text.lower()) if s.strip()]


def mentions(sentence):
    """Find pathology mentions (longest phrase first, no overlaps). Returns [(label, start, end)]."""
    taken, out = [], []
    for ph, lab in PHRASES:
        for m in re.finditer(r"\b" + re.escape(ph) + r"\b", sentence):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            out.append((lab, m.start(), m.end()))
    return sorted(out, key=lambda x: x[1])


def mask_sentence(sentence, target, all_mentions):
    """Replace target mention with 'tgt' and other findings with 'oth'."""
    s, offset = sentence, 0
    for lab, a, b in all_mentions:
        rep = " tgt " if (a, b) == target else " oth "
        s = s[:a + offset] + rep + s[b + offset:]
        offset += len(rep) - (b - a)
    return s


def words(s):
    return re.findall(r"[a-z]+|\?", s)


# ---------------- rule baseline (NegEx-style) ----------------
def rule_assert(masked):
    clause = [c for c in re.split(SCOPE_BREAK, masked) if "tgt" in c]
    c = " " + (clause[0] if clause else masked) + " "
    ws = words(c)
    if "tgt" not in ws:
        return "present"
    i = ws.index("tgt")
    pre = " " + " ".join(ws[max(0, i - 6):i]) + " "
    post = " " + " ".join(ws[i + 1:i + 7]) + " "
    if any(f" {u} " in f" {c} " or (u == "?" and "?" in c) for u in UNCERTAIN):
        return "uncertain"
    if any(f" {n} " in pre for n in PRE_NEG) or any(f" {n} " in post for n in POST_NEG):
        return "absent"
    return "present"


# ---------------- synthetic training data ----------------
TRAIN_T = {
    "present": ["there is {mod} {f}", "{f} is seen in the {loc}", "{f} noted", "findings consistent with {f}",
                "interval increase in {f}", "{loc} {f} is present", "{mod} {loc} {f}", "new {f} in the {loc}",
                "persistent {f}", "{f} is again demonstrated", "redemonstrated {f}", "{f} has developed"],
    "absent": ["no {f}", "no evidence of {f}", "there is no {f}", "negative for {f}", "{f} is not seen",
               "without {f}", "{f} has resolved", "free of {f}", "no {loc} {f}", "no signs of {f}",
               "resolution of previously seen {f}", "{f} not identified"],
    "uncertain": ["possible {f}", "cannot exclude {f}", "may represent {f}", "suspicious for {f}",
                  "questionable {loc} {f}", "{f} is not entirely excluded", "likely {f}", "concerning for {f}",
                  "probable {mod} {f}", "could represent {f}", "{loc} opacity ? {f}", "suggestive of {f}"],
}
TEST_T = {  # held-out wording, never seen in training
    "present": ["the study demonstrates {mod} {f}", "{f} is identified at the {loc}", "evidence of {f} is noted",
                "{mod} {f} remains"],
    "absent": ["there is no evidence of {f}", "no definite {f} is seen", "{f} are absent", "no new {f}"],
    "uncertain": ["{f} cannot be excluded", "findings may be due to {f}", "equivocal {f}", "differential includes {f}"],
}
MODS = ["small", "moderate", "large", "mild", "minimal", "trace", "subtle", "extensive", "patchy"]
LOCS = ["right base", "left base", "right upper lobe", "left lower lobe", "bilateral", "right", "left",
        "retrocardiac", "perihilar", "lingular", "apical"]
JOINERS = [" but ", ", ", " and ", " with ", ". however, "]


def _one(lab, templates, rng):
    f = rng.choice(PHRASES)[0]
    s = rng.choice(templates[lab]).format(f="TGT_" + f, mod=rng.choice(MODS), loc=rng.choice(LOCS))
    return s


FILL = ["the", "is", "are", "seen", "at", "in", "of", "a", "there", "again", "on", "study", "today", "this",
        "compared", "prior", "exam", "chest", "radiograph", "findings", "lungs", "lobe", "zone", "which",
        "stable", "unchanged", "within", "region", "view", "frontal", "lateral", "portable", "patient"]
PRESENT_CUES = ["is seen", "is noted", "demonstrates", "remains", "is identified", "is evident", "shows",
                "persists", "has increased", "has developed", "with", "and", "consistent with", "evidence of",
                "compatible with", "represents", "is present", "is again seen"]
POST_UNC = ["cannot be excluded", "not excluded", "not entirely excluded", "is questioned", "is possible",
            "is suspected", "is likely", "is not ruled out"]


def _lexicon_sentence(lab, rng):
    """Compose sentences straight from the trigger lexicons + random filler, so the network learns
    the triggers themselves rather than memorising whole templates."""
    f = "TGT_" + rng.choice(PHRASES)[0]
    fill = lambda k: " ".join(rng.choice(FILL) for _ in range(rng.randint(0, k)))
    mod = rng.choice(MODS + LOCS + [""])
    if lab == "absent":
        core = (f"{rng.choice(PRE_NEG)} {fill(2)} {mod} {f}" if rng.random() < 0.6
                else f"{mod} {f} {rng.choice(POST_NEG)}")
    elif lab == "uncertain":
        u = [x for x in UNCERTAIN if x != "?"]
        core = (f"{rng.choice(u)} {fill(2)} {mod} {f}" if rng.random() < 0.6
                else f"{mod} {f} {rng.choice(POST_UNC)}")
    else:
        core = (f"{mod} {f} {rng.choice(PRESENT_CUES)} {fill(2)}" if rng.random() < 0.5
                else f"{fill(2)} {rng.choice(PRESENT_CUES)} {mod} {f}")
    return re.sub(r"\s+", " ", f"{fill(3)} {core} {fill(2)}").strip()


def synth_assertions(n, templates, seed, lexicon_p=0.0):
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        lab = rng.choice(CLASSES)
        s = _lexicon_sentence(lab, rng) if rng.random() < lexicon_p else _one(lab, templates, rng)
        if rng.random() < 0.45:  # add a distractor clause about another finding
            other = rng.choice(CLASSES)
            o = (_lexicon_sentence(other, rng) if rng.random() < lexicon_p
                 else _one(other, templates, rng)).replace("TGT_", "")
            s = (s + rng.choice(JOINERS) + o) if rng.random() < 0.5 else (o + rng.choice(JOINERS) + s)
        s = s.lower()
        # mask: target -> tgt, any other finding -> oth
        s_clean = s.replace("tgt_", "@@")
        ms = mentions(s_clean.replace("@@", ""))
        # find target position = mention whose start matches the @@ marker
        pos = s_clean.index("@@")
        plain = s_clean.replace("@@", "")
        target = next(((a, b) for _, a, b in ms if a == pos), None)
        if target is None:
            continue
        data.append((mask_sentence(plain, target, ms), lab))
    return data


# hand-written realistic report sentences (sentence, pathology, gold)
REAL_SET = [
    ("No pleural effusion or pneumothorax.", "Effusion", "absent"),
    ("No pleural effusion or pneumothorax.", "Pneumothorax", "absent"),
    ("Mild cardiomegaly is again noted.", "Cardiomegaly", "present"),
    ("Patchy opacity at the left base may represent atelectasis or pneumonia.", "Pneumonia", "uncertain"),
    ("Patchy opacity at the left base may represent atelectasis or pneumonia.", "Atelectasis", "uncertain"),
    ("Small right pleural effusion has increased in size.", "Effusion", "present"),
    ("There is no focal consolidation.", "Consolidation", "absent"),
    ("Pulmonary edema has resolved.", "Edema", "absent"),
    ("Moderate pulmonary edema with bilateral effusions.", "Edema", "present"),
    ("Moderate pulmonary edema with bilateral effusions.", "Effusion", "present"),
    ("Cannot exclude a small left apical pneumothorax.", "Pneumothorax", "uncertain"),
    ("A 1 cm nodule is seen in the right upper lobe.", "Nodule", "present"),
    ("No displaced rib fracture is identified.", "Fracture", "absent"),
    ("Hyperinflated lungs consistent with emphysema.", "Emphysema", "present"),
    ("Right lower lobe consolidation concerning for pneumonia.", "Consolidation", "present"),
    ("Right lower lobe consolidation concerning for pneumonia.", "Pneumonia", "uncertain"),
    ("Small hiatal hernia.", "Hernia", "present"),
    ("Left basilar atelectasis without effusion.", "Atelectasis", "present"),
    ("Left basilar atelectasis without effusion.", "Effusion", "absent"),
    ("No new opacity to suggest pneumonia.", "Pneumonia", "absent"),
    ("No new opacity to suggest pneumonia.", "Lung Opacity", "absent"),
    ("Possible small right effusion.", "Effusion", "uncertain"),
    ("Biapical pleural thickening and scarring.", "Pleural_Thickening", "present"),
    ("Biapical pleural thickening and scarring.", "Fibrosis", "present"),
    ("There is no pneumothorax but a small effusion persists.", "Pneumothorax", "absent"),
    ("There is no pneumothorax but a small effusion persists.", "Effusion", "present"),
    ("The previously seen mass is no longer visualized.", "Mass", "absent"),
    ("Widened mediastinum, aortic injury not excluded.", "Enlarged Cardiomediastinum", "present"),
    ("Interstitial infiltrates are likely infectious.", "Infiltration", "present"),
    ("Retrocardiac opacity, atelectasis versus consolidation.", "Atelectasis", "uncertain"),
]


def real_examples():
    out = []
    for sent, lab, gold in REAL_SET:
        s = sentences(sent)[0]
        ms = mentions(s)
        tgt = next(((a, b) for l, a, b in ms if l == lab), None)
        if tgt:
            out.append((mask_sentence(s, tgt, ms), gold))
    return out


# ---------------- Keras model ----------------
class Vocab:
    """'Active vocabulary' hash table: word -> id (0 = PAD, 1 = OOV)."""

    def __init__(self, itos=None):
        self.itos = itos or ["<pad>", "<oov>"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def fit(self, texts, min_freq=1):
        from collections import Counter
        c = Counter(w for t in texts for w in words(t))
        for w, n in c.most_common():
            if n >= min_freq and w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)
        return self

    def encode(self, texts):
        from keras.utils import pad_sequences
        seqs = [[self.stoi.get(w, 1) for w in words(t)] for t in texts]
        return pad_sequences(seqs, maxlen=MAXLEN, padding="pre", truncating="pre")  # PRE-padding


def build_model(vocab_size):
    import keras
    from keras import layers
    inp = keras.Input((MAXLEN,), dtype="int32")
    x = layers.Embedding(vocab_size, 64)(inp)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, recurrent_dropout=0.1))(x)
    att = layers.MultiHeadAttention(num_heads=2, key_dim=32, name="self_attention")(x, x)  # Q=K=V=x
    x = layers.LayerNormalization()(x + att)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(3, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train_assertion(n=20000, epochs=6, seed=0, verbose=2):
    import keras
    keras.utils.set_random_seed(seed)
    train = synth_assertions(n, TRAIN_T, seed, lexicon_p=0.6)
    vocab = Vocab().fit([s for s, _ in train])
    X = vocab.encode([s for s, _ in train])
    y = np.array([CLASSES.index(l) for _, l in train])
    model = build_model(len(vocab.itos))
    model.fit(X, y, epochs=epochs, batch_size=128, validation_split=0.1, verbose=verbose)
    return model, vocab


class Asserter:
    def __init__(self, model, vocab, min_conf=0.7):
        self.model, self.vocab, self.min_conf = model, vocab, min_conf

    def predict(self, masked_list):
        if not masked_list:
            return []
        P = self.model.predict(self.vocab.encode(masked_list), verbose=0)
        out = []
        for s, p in zip(masked_list, P):
            k = int(p.argmax())
            if p[k] >= self.min_conf:
                out.append((CLASSES[k], float(p[k]), "model"))
            else:
                out.append((rule_assert(s), float(p[k]), "rules (low model conf)"))
        return out

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        self.model.save(os.path.join(d, "assertion.keras"))
        json.dump(self.vocab.itos, open(os.path.join(d, "vocab.json"), "w"))

    @classmethod
    def load(cls, d):
        import keras
        return cls(keras.models.load_model(os.path.join(d, "assertion.keras")),
                   Vocab(json.load(open(os.path.join(d, "vocab.json")))))


def evaluate_assertion(asserter, seed=7):
    from sklearn.metrics import accuracy_score, f1_score
    res = {}
    sets = {"synthetic held-out templates": synth_assertions(1500, TEST_T, seed),
            "hand-written real-style (n=%d)" % len(REAL_SET): real_examples()}
    for name, data in sets.items():
        gold = [l for _, l in data]
        rules = [rule_assert(s) for s, _ in data]
        model = [p[0] for p in asserter.predict([s for s, _ in data])]
        res[name] = {"NegEx rules": (accuracy_score(gold, rules), f1_score(gold, rules, average="macro")),
                     "BiGRU+self-attn (+rule fallback)": (accuracy_score(gold, model), f1_score(gold, model, average="macro"))}
    return res


def analyse_report(text, asserter):
    """Returns {pathology: (status, evidence_sentence, engine)} for everything the report talks about."""
    found = {}
    rank = {"present": 0, "uncertain": 1, "absent": 2}
    jobs = []
    for s in sentences(text):
        ms = mentions(s)
        for lab, a, b in ms:
            jobs.append((lab, s, mask_sentence(s, (a, b), ms)))
    preds = asserter.predict([j[2] for j in jobs])
    for (lab, s, _), (status, conf, eng) in zip(jobs, preds):
        if lab not in found or rank[status] < rank[found[lab][0]]:
            found[lab] = (status, s, f"{eng} {conf:.2f}")
    low = text.lower()
    for pat, labs in NORMAL_PATTERNS:
        m = re.search(pat, low)
        if m:
            for lab in labs:
                found.setdefault(lab, ("absent", m.group(0), "normal-phrase rule"))
    return found
