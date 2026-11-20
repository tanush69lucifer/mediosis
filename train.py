"""Train + evaluate every Mediosis model and write artifacts/ and results/metrics.md.

    python train.py            # ~2-4 min on Colab T4, ~3 min on CPU
"""
import json
import os
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from mediosis import rx_correct, rx_ner, xr_report
from mediosis.rx_realtest import REAL_DRUG_GOLD, real_test_set

ART, RES = "artifacts", "results"
os.makedirs(ART, exist_ok=True)
os.makedirs(RES, exist_ok=True)
t0 = time.time()
M = {}

print("1/3  fastText subword model + drug-name corrector")
ft = rx_correct.train_fasttext()
ft.save(f"{ART}/fasttext.model")
corr = rx_correct.DrugCorrector(ft)
M["corrector_synthetic_1-2_edits"], n = rx_correct.evaluate(corr)
M["corrector_synthetic_3_edits"], _ = rx_correct.evaluate(corr, n_edits=3, seed=321)
M["corrector_real_style"], _ = rx_correct.evaluate(corr, tests=[(k, v) for k, v in REAL_DRUG_GOLD.items()])
print("   ", {k: v for k, v in M.items()})

print("2/3  BiLSTM-CRF prescription NER")
ner = rx_ner.train_ner(ft)
rx_ner.save(ner, f"{ART}/ner.pt")
f1_syn, rep_syn = rx_ner.evaluate_ner(ner, ft)
real = real_test_set()
from seqeval.metrics import classification_report, f1_score
pred = rx_ner.predict(ner, ft, [t for t, _ in real])
f1_real = f1_score([g for _, g in real], pred)
rep_real = classification_report([g for _, g in real], pred, digits=3, zero_division=0)
M["ner_f1_synthetic_heavy_noise"], M["ner_f1_real_style"] = f1_syn, f1_real
print(f"    F1 synthetic={f1_syn:.3f}  real-style={f1_real:.3f}")

print("3/3  Report assertion classifier (BiGRU + self-attention)")
model, vocab = xr_report.train_assertion(verbose=2)
asserter = xr_report.Asserter(model, vocab)
asserter.save(ART)
M["assertion"] = xr_report.evaluate_assertion(asserter)

json.dump(M, open(f"{RES}/metrics.json", "w"), indent=2, default=float)

# ---------- markdown table ----------
L = ["# Mediosis — evaluation results", "",
     "## RxDecode · drug-name correction (top-1 accuracy)", "",
     "| Test set | Edit distance only | fastText only | **Ensemble** |", "|---|---|---|---|"]
for k, name in [("corrector_synthetic_1-2_edits", f"Synthetic OCR noise, 1–2 edits (n={n})"),
                ("corrector_synthetic_3_edits", "Synthetic OCR noise, 3 edits (harder)"),
                ("corrector_real_style", f"Hand-written real-style misreads (n={len(REAL_DRUG_GOLD)})")]:
    r = M[k]
    L.append(f"| {name} | {r['fuzzy']:.3f} | {r['fasttext']:.3f} | **{r['ensemble']:.3f}** |")
L += ["", "## RxDecode · BiLSTM-CRF NER (entity-level F1)", "",
      "| Test set | F1 |", "|---|---|",
      f"| Synthetic, unseen seed, heavier OCR noise (n=2000 lines) | {f1_syn:.3f} |",
      f"| Hand-written real-style lines, unseen formats (n={len(real)} lines) | {f1_real:.3f} |",
      "", "```", rep_real.strip(), "```",
      "", "## XCheck · report assertion (present / absent / uncertain)", "",
      "| Test set | NegEx rules acc / macro-F1 | BiGRU+self-attn acc / macro-F1 |", "|---|---|---|"]
for name, r in M["assertion"].items():
    a, b = r["NegEx rules"], r["BiGRU+self-attn (+rule fallback)"]
    L.append(f"| {name} | {a[0]:.3f} / {a[1]:.3f} | {b[0]:.3f} / {b[1]:.3f} |")
L += ["", "## XCheck · image model", "",
      "Pretrained TorchXRayVision DenseNet-121 (`densenet121-res224-all`), published AUROCs in "
      "Cohen et al., *TorchXRayVision* (MIDL 2022). Not re-trained here.", "",
      "## Honest caveats", "",
      "- Synthetic test sets come from generators we wrote; real-world accuracy will be lower.",
      "- Real-style sets are small (tens of examples) — treat as sanity checks, not clinical validation.",
      "- Not a medical device. Not clinically validated. For education/research demos only.",
      "", f"_Training time: {time.time() - t0:.0f}s_"]
open(f"{RES}/metrics.md", "w").write("\n".join(L))
print("\n".join(L))
