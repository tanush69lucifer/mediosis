"""OCR for prescriptions: EasyOCR (detection + printed text) with optional TrOCR re-read for handwriting."""
import numpy as np
import torch
from PIL import Image

_reader, _trocr = None, None


def _easy():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _reader


def _tr():
    global _trocr
    if _trocr is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        name = "microsoft/trocr-base-handwritten"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _trocr = (TrOCRProcessor.from_pretrained(name),
                  VisionEncoderDecoderModel.from_pretrained(name).to(dev), dev)
    return _trocr


def _group_lines(results):
    """results: [(box, text, conf)] -> list of lines [(text, conf, (x0,y0,x1,y1))] in reading order."""
    items = []
    for box, text, conf in results:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        items.append(dict(x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys), cy=(min(ys) + max(ys)) / 2,
                          h=max(ys) - min(ys), text=text, conf=conf))
    if not items:
        return []
    items.sort(key=lambda d: d["cy"])
    med_h = float(np.median([d["h"] for d in items]))
    lines, cur = [], [items[0]]
    for d in items[1:]:
        if abs(d["cy"] - np.mean([c["cy"] for c in cur])) < 0.6 * med_h:
            cur.append(d)
        else:
            lines.append(cur)
            cur = [d]
    lines.append(cur)
    out = []
    for ln in lines:
        ln.sort(key=lambda d: d["x0"])
        out.append((" ".join(d["text"] for d in ln), float(np.mean([d["conf"] for d in ln])),
                    (min(d["x0"] for d in ln), min(d["y0"] for d in ln),
                     max(d["x1"] for d in ln), max(d["y1"] for d in ln))))
    return out


def read_prescription(img, handwriting=False):
    """img: PIL.Image or np.ndarray. Returns list of (line_text, confidence)."""
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    lines = _group_lines(_easy().readtext(img))
    if not handwriting:
        return [(t, c) for t, c, _ in lines]
    proc, model, dev = _tr()
    out = []
    for t, c, (x0, y0, x1, y1) in lines:
        pad = 4
        crop = Image.fromarray(img[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad])
        with torch.no_grad():
            ids = model.generate(proc(images=crop, return_tensors="pt").pixel_values.to(dev), max_new_tokens=48)
        out.append((proc.batch_decode(ids, skip_special_tokens=True)[0], c))
    return out
