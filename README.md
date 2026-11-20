# 🩺 Mediosis

**Prescription decoding + radiology second-reader, built with classic NLP and deep learning.**
A Genzis project · [Open in Colab](#run-in-colab)

> ⚠️ Research / education demo. Not a medical device and not clinically validated. Don't change medication or act on a scan because of this tool.

Mediosis has two modules:

| Module | What it does | Core tech |
|---|---|---|
| 💊 **RxDecode** | Photo of a prescription → reads every medicine, fixes OCR-garbled drug names, explains dose/frequency/duration in **English or Hindi**, flags **drug interactions and duplicate ingredients** (e.g. Crocin + Combiflam = double paracetamol) and gives the generic name to ask for at Jan Aushadhi | EasyOCR (+ optional TrOCR for handwriting) · **fastText subword** embeddings · **BiLSTM-CRF** NER · rule-based interaction engine |
| 🫁 **XCheck** | Chest X-ray + the radiologist's report → finds 18 conditions in the image, reads the report (including negation like *"no effusion"*), and **flags contradictions and possible missed findings**, with Grad-CAM heatmaps | TorchXRayVision DenseNet-121 · test-time-augmentation uncertainty · **Grad-CAM** · **BiGRU + self-attention** assertion classifier vs a NegEx baseline |

## Why it's different

- **OCR-noise-aware drug matching.** OCR turns *Azithromycin* into *Azithrornycin* (`m`→`rn`). fastText builds word vectors from character n-grams, so a misspelt word still lands close to the real one. We train it on prescription text plus simulated OCR corruptions, then combine its cosine similarity with edit distance.
- **Second reader, not a diagnosis.** XCheck doesn't try to replace the radiologist. It flags places where the image model and the written report *disagree*. That's the kind of safety net hospitals actually use.
- **Uncertainty built in.** The image is scored under 8 small augmentations. If the predictions disagree, the result is marked *unstable* instead of giving a confident wrong answer.

## Architecture

```
RxDecode   image ─► EasyOCR ─► line grouping ─► OCR clean-up ─► tokenizer
                                                                   │
                     fastText subword vectors + shape features ◄───┘
                                     │
                         BiLSTM ─► CRF  (DRUG/STRENGTH/FORM/FREQ/INSTR/DUR)
                                     │
             drug corrector (fastText cos ⊕ edit distance) ─► brand → generic
                                     │
              interaction + duplicate rules ─► EN / हिन्दी explanation

XCheck     X-ray ─► DenseNet-121 (8× TTA) ─► probs ± std ─► Grad-CAM
           report ─► sentence split ─► finding mentions ─► mask target (tgt/oth)
                     ─► pre-pad ─► Embedding ─► SpatialDropout1D ─► BiGRU(recurrent_dropout)
                     ─► Multi-head self-attention ─► softmax {present, absent, uncertain}
           image status × report status ─► 🔴 contradiction / 🟠 missed / ✅ consistent
```

## Results

See [`results/metrics.md`](results/metrics.md). It's regenerated every time you run `train.py`.

| | Metric | Score |
|---|---|---|
| Drug-name correction, synthetic OCR noise (n=1780) | top-1 acc (ensemble) | ~0.99 |
| Drug-name correction, heavy noise (3 edits) | top-1 acc (ensemble) | ~0.93 |
| Rx NER, hand-written real-style lines (unseen formats) | entity F1 | ~0.93 |
| Report assertion, hand-written real-style sentences (n=30) | accuracy | 0.90 (NegEx baseline 0.90) |

**Honest caveats:** the synthetic test sets come from our own generators, and the real-style sets are small. These are sanity checks, not clinical validation. The image model is pretrained ([Cohen et al., TorchXRayVision, MIDL 2022](https://github.com/mlmed/torchxrayvision)) and was not re-trained here.

## Run in Colab

1. Open `Mediosis_Colab.ipynb` in Colab → *Runtime → Change runtime type → T4 GPU*.
2. *Runtime → Run all*. It installs packages, trains everything (~2–3 min), evaluates, and prints a public **gradio.live** link.

## Run locally

```bash
pip install -r requirements.txt
python make_samples.py
python train.py          # writes artifacts/ and results/metrics.md
python app.py --share    # --share gives a public link
```

## Project layout

```
mediosis/
  lexicon.py       drug lexicon, interaction rules, Rx shorthand (OD/BD/TDS/1-0-1…)
  rx_data.py       tokenizer, OCR-noise simulator, synthetic BIO-tagged prescriptions
  rx_correct.py    fastText subword drug-name corrector + eval
  rx_ner.py        BiLSTM-CRF tagger
  rx_ocr.py        EasyOCR + optional TrOCR handwriting
  rx_pipeline.py   end-to-end prescription decoding + EN/HI explanation
  rx_realtest.py   hand-written real-style test lines
  xr_vision.py     DenseNet-121, TTA uncertainty, Grad-CAM
  xr_report.py     report NLP: mention finder, NegEx rules, BiGRU+self-attention classifier
  xr_check.py      image ↔ report cross-check logic
train.py · app.py · make_samples.py
```

## Roadmap

- Real prescription dataset with doctor annotations; fine-tune TrOCR on Indian handwriting
- Replace the rule-based interaction list with DDInter / DrugBank
- Validate XCheck on MIMIC-CXR report–image pairs (needs credentialed access)
- Generate radiology reports with a CNN encoder + attention decoder (IU X-Ray)

## Credits

Built by **Tanush Bhardwaj** ([@tanush69lucifer](https://github.com/tanush69lucifer)) · Genzis — *Generations Ahead*.
Sample X-rays come from the TorchXRayVision test set (NIH ChestX-ray14 / PadChest).
