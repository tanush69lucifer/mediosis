# Mediosis — evaluation results

## RxDecode · drug-name correction (top-1 accuracy)

| Test set | Edit distance only | fastText only | **Ensemble** |
|---|---|---|---|
| Synthetic OCR noise, 1–2 edits (n=1780) | 0.985 | 0.962 | **0.989** |
| Synthetic OCR noise, 3 edits (harder) | 0.923 | 0.805 | **0.926** |
| Hand-written real-style misreads (n=18) | 1.000 | 1.000 | **1.000** |

## RxDecode · BiLSTM-CRF NER (entity-level F1)

| Test set | F1 |
|---|---|
| Synthetic, unseen seed, heavier OCR noise (n=2000 lines) | 1.000 |
| Hand-written real-style lines, unseen formats (n=18 lines) | 0.939 |

```
precision    recall  f1-score   support

        DRUG      0.947     1.000     0.973        18
         DUR      0.667     0.800     0.727        10
        FORM      1.000     1.000     1.000        17
        FREQ      0.944     0.944     0.944        18
       INSTR      1.000     1.000     1.000         7
    STRENGTH      0.895     1.000     0.944        17

   micro avg      0.913     0.966     0.939        87
   macro avg      0.909     0.957     0.932        87
weighted avg      0.919     0.966     0.941        87
```

## XCheck · report assertion (present / absent / uncertain)

| Test set | NegEx rules acc / macro-F1 | BiGRU+self-attn acc / macro-F1 |
|---|---|---|
| synthetic held-out templates | 0.918 / 0.918 | 1.000 / 1.000 |
| hand-written real-style (n=30) | 0.900 / 0.893 | 0.900 / 0.897 |

## XCheck · image model

Pretrained TorchXRayVision DenseNet-121 (`densenet121-res224-all`), published AUROCs in Cohen et al., *TorchXRayVision* (MIDL 2022). Not re-trained here.

## Honest caveats

- Synthetic test sets come from generators we wrote; real-world accuracy will be lower.
- Real-style sets are small (tens of examples) — treat as sanity checks, not clinical validation.
- Not a medical device. Not clinically validated. For education/research demos only.

_Training time: 95s_