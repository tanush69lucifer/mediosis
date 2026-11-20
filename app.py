"""Mediosis — Gradio demo.   python app.py  (add --share for a public link)"""
import os
import sys

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gradio as gr
import pandas as pd
from gensim.models import FastText
from PIL import Image

from mediosis import rx_ner, xr_vision
from mediosis.rx_correct import DrugCorrector
from mediosis.rx_ocr import read_prescription
from mediosis.rx_pipeline import decode_lines, explain
from mediosis.xr_check import cross_check, final_verdict
from mediosis.xr_report import Asserter

ART = "artifacts"
if not os.path.exists(f"{ART}/ner.pt"):
    print("No trained artifacts found -> running train.py first")
    os.system(f"{sys.executable} train.py")

FT = FastText.load(f"{ART}/fasttext.model")
CORR = DrugCorrector(FT)
NER = rx_ner.load(f"{ART}/ner.pt")
ASSERT = Asserter.load(ART)
xr_vision.model()  # warm-up download

DISCLAIMER = ("⚠️ **Research / education demo — not a medical device, not clinically validated.** "
              "Never change medication or act on a scan based on this tool. Always consult a doctor or pharmacist.")


# ---------------- RxDecode ----------------
def run_rx(image, typed, handwriting, lang):
    ocr_rows = []
    if typed and typed.strip():
        lines = [l for l in typed.splitlines() if l.strip()]
    elif image is not None:
        res = read_prescription(image, handwriting=handwriting)
        lines = [t for t, _ in res]
        ocr_rows = [{"OCR line": t, "OCR conf": round(c, 2)} for t, c in res]
    else:
        return "Upload a prescription image or type/paste the lines.", None, None, None
    meds, alerts = decode_lines(lines, NER, FT, CORR)
    if not meds:
        return "No medicines detected.", pd.DataFrame(ocr_rows), None, None
    table = pd.DataFrame([{"OCR word": m["ocr_word"], "Matched drug": m["drug"], "Match conf": m["match_conf"],
                           "Generic (ask for Jan Aushadhi)": m["generic"], "Class": m["drug_class"],
                           "Strength": m["strength"], "How often": m["freq_en"], "Food": m["instr_en"],
                           "Duration": m["dur_en"]} for m in meds])
    al = pd.DataFrame(alerts) if alerts else pd.DataFrame([{"severity": "—", "drugs": "—",
                                                            "issue": "No interactions found in Mediosis' rule base (not exhaustive)."}])
    md = f"### {'दवा कैसे लें' if lang == 'हिन्दी' else 'How to take your medicines'}\n" + \
         explain(meds, "hi" if lang == "हिन्दी" else "en")
    if alerts:
        md += f"\n\n### ⚠️ {len(alerts)} safety alert(s) — show this to your doctor/pharmacist"
    return md, pd.DataFrame(ocr_rows) if ocr_rows else None, table, al


# ---------------- XCheck ----------------
def run_xr(image, report):
    if image is None:
        return "Upload a chest X-ray (frontal PA/AP view).", None, None, ""
    x, disp = xr_vision.preprocess(image)
    probs = xr_vision.predict(x)
    rows, _ = cross_check(probs, report, ASSERT)
    df = pd.DataFrame(rows)[["flag", "pathology", "image_prob", "tta_std", "image", "report", "evidence", "engine"]] \
        if rows else pd.DataFrame()
    top = sorted(probs.items(), key=lambda kv: -kv[1][0])[:3]
    flagged = [r["pathology"] for r in rows if r["severity"] >= 2]
    targets = list(dict.fromkeys(flagged + [k for k, _ in top]))[:4]
    gallery = [(xr_vision.overlay(disp, xr_vision.gradcam(x, t)), f"{t}  p={probs[t][0]:.2f}") for t in targets]
    n_red = sum(r["severity"] == 3 for r in rows)
    n_or = sum(r["severity"] == 2 for r in rows)
    summary = (f"### Cross-check: {n_red} contradiction(s), {n_or} possible miss/over-call(s)\n"
               f"Image model: TorchXRayVision DenseNet-121 · 0.5 = calibrated operating point · "
               f"uncertainty from 8× test-time augmentation.\n\n"
               + ("" if report and report.strip() else "_No report given — showing image findings only._"))
    return summary, df, gallery, final_verdict(rows, bool(report and report.strip()))


DEMO_REPORT = ("Heart size is normal. No pleural effusion or pneumothorax. "
               "Lungs are clear. No acute cardiopulmonary process.")

with gr.Blocks(title="Mediosis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 Mediosis\n**Prescription decoding + radiology second-reader** · by Genzis\n\n" + DISCLAIMER)
    with gr.Tab("💊 RxDecode — prescription reader"):
        with gr.Row():
            with gr.Column():
                rx_img = gr.Image(type="pil", label="Prescription photo")
                rx_txt = gr.Textbox(lines=5, label="…or type / paste prescription lines (overrides image)")
                with gr.Row():
                    hw = gr.Checkbox(label="Handwritten (TrOCR re-read, slower)")
                    lang = gr.Radio(["English", "हिन्दी"], value="English", label="Explain in")
                rx_btn = gr.Button("Decode prescription", variant="primary")
                gr.Examples([["samples/prescription_demo.png", "", False, "English"],
                             [None, "Tab Glycornet 500mg 1-0-1 after food x 30 days\nTab Ciplox 500 BD x 5 days\n"
                                    "Tab Shelca1 500 OD after lunch\nTab Thyronorm 50mcg 1-0-0 empty stomach", False, "हिन्दी"]],
                            [rx_img, rx_txt, hw, lang])
            with gr.Column():
                rx_md = gr.Markdown()
                rx_alerts = gr.Dataframe(label="Interaction & duplicate-drug alerts", wrap=True)
        rx_table = gr.Dataframe(label="Decoded medicines", wrap=True)
        rx_ocr = gr.Dataframe(label="Raw OCR lines")
        rx_btn.click(run_rx, [rx_img, rx_txt, hw, lang], [rx_md, rx_ocr, rx_table, rx_alerts])

    with gr.Tab("🫁 XCheck — X-ray vs report"):
        with gr.Row():
            with gr.Column():
                xr_img = gr.Image(type="pil", label="Chest X-ray")
                xr_rep = gr.Textbox(lines=5, label="Radiologist's report (optional)")
                xr_btn = gr.Button("Cross-check", variant="primary")
                gr.Examples([["samples/00000001_000.png", DEMO_REPORT],
                             ["samples/16747_3_1.jpg", "Small left upper lobe nodule. No effusion."]], [xr_img, xr_rep])
            with gr.Column():
                xr_md = gr.Markdown()
                xr_gal = gr.Gallery(label="Grad-CAM — where the model looked", columns=2, height=420)
        xr_df = gr.Dataframe(label="Finding-by-finding cross-check", wrap=True)
        xr_final = gr.Markdown()
        xr_btn.click(run_xr, [xr_img, xr_rep], [xr_md, xr_df, xr_gal, xr_final])

    with gr.Tab("📊 Model card"):
        gr.Markdown(open("results/metrics.md", encoding="utf-8").read() if os.path.exists("results/metrics.md") else "Run train.py")

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv, debug="--debug" in sys.argv)
