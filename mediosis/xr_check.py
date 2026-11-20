"""XCheck: cross-check what the image model sees against what the radiology report says."""
from .xr_report import analyse_report

POS, NEG = 0.60, 0.40    # model is calibrated so 0.5 = operating point
UNSTABLE_STD = 0.08      # TTA disagreement => unstable prediction

# labels the model is weaker on / that are often not reported -> lower severity
SOFT = {"Infiltration", "Lung Opacity", "Pleural_Thickening", "Fibrosis", "Lung Lesion"}


def image_status(p, sd):
    if sd > UNSTABLE_STD:
        return "unstable"
    return "positive" if p >= POS else "negative" if p <= NEG else "borderline"


def cross_check(probs, report_text, asserter):
    """probs: {pathology: (mean, std)}. Returns (rows, report_findings)."""
    rep = analyse_report(report_text, asserter) if report_text and report_text.strip() else {}
    rows = []
    for lab, (p, sd) in sorted(probs.items(), key=lambda kv: -kv[1][0]):
        img = image_status(p, sd)
        r_status, evidence, engine = rep.get(lab, ("not mentioned", "", ""))
        flag, sev = "✅ consistent", 0
        if img == "positive" and r_status == "absent":
            flag, sev = "🔴 CONTRADICTION: image suggests it, report says absent", 3
        elif img == "positive" and r_status == "not mentioned":
            flag, sev = ("🟠 possible MISSED finding (not in report)", 2) if lab not in SOFT else ("🟡 unmentioned (low-specificity label)", 1)
        elif img == "negative" and r_status == "present":
            flag, sev = "🟠 report claims it, image model does not support it", 2
        elif img in ("borderline", "unstable") and r_status in ("present", "absent"):
            flag, sev = f"🟡 image {img} — radiologist judgement prevails", 1
        elif r_status == "uncertain" and img == "positive":
            flag, sev = "🟡 report uncertain, image leans positive — consider follow-up", 1
        elif img == "negative" and r_status == "not mentioned":
            continue  # nothing to say
        rows.append(dict(pathology=lab, image_prob=round(p, 3), tta_std=round(sd, 3), image=img,
                         report=r_status, evidence=evidence, engine=engine, flag=flag, severity=sev))
    rows.sort(key=lambda r: (-r["severity"], -r["image_prob"]))
    return rows, rep
