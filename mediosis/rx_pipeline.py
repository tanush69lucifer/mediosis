"""RxDecode pipeline: text lines -> NER -> drug correction -> generics -> interactions -> explanation."""
import re

from .lexicon import BRANDS, FORMS, FREQ, GENERIC_CLASS, INSTR, SLOTS_EN, SLOTS_HI, check_interactions
from .rx_data import tokenize
from .rx_ner import predict

PHRASE_FREQ = [("three times", "tds"), ("thrice", "tds"), ("twice", "bd"), ("four times", "qid"),
               ("once", "od"), ("at night", "hs"), ("when required", "sos"), ("daily", "od")]
UNIT_HI = {"day": "दिन", "week": "हफ़्ते", "month": "महीने"}


def _spans(toks, tags):
    ents, cur = [], None
    for t, g in zip(toks, tags):
        if g.startswith("B-") or (g.startswith("I-") and (cur is None or cur[0] != g[2:])):
            cur = [g[2:], [t]]
            ents.append(cur)
        elif g.startswith("I-"):
            cur[1].append(t)
        else:
            cur = None
    return [(lab, " ".join(ts)) for lab, ts in ents]


def _freq(text):
    if not text:
        return None, None, None
    m = re.search(r"(\d)-(\d)-(\d)", text)
    if m:
        slots = [int(x) for x in m.groups()]
        en = ", ".join(f"{n} in the {s}" for n, s in zip(slots, SLOTS_EN) if n)
        hi = ", ".join(f"{s} {n}" for n, s in zip(slots, SLOTS_HI) if n)
        return sum(slots), en, hi
    for tok in text.split():
        if tok in FREQ:
            return FREQ[tok]
    for ph, code in PHRASE_FREQ:
        if ph in text:
            return FREQ[code]
    return None, text, text


def _dur(text):
    if not text:
        return None, None
    m = re.search(r"(\d+)\s*(d|day|days|w|wk|week|weeks|m|month|months)?", text)
    if not m:
        return None, None
    n, u = m.group(1), (m.group(2) or "d")[0]
    unit = {"d": "day", "w": "week", "m": "month"}[u]
    return f"{n} {unit}{'s' if n != '1' else ''}", f"{n} {UNIT_HI[unit]}"


def _instr(text):
    if not text:
        return None, None
    for k, v in INSTR.items():
        if k in text.split():
            return v
    return text, text


_CONFUSE = str.maketrans({"O": "0", "o": "0", "S": "5", "s": "5", "I": "1", "l": "1", "B": "8"})


def clean_line(line):
    """OCR clean-up before tagging: drop list numbering, repair digit/letter swaps in strengths (4Omg -> 40mg)."""
    line = re.sub(r"^\s*\d{1,2}\s*[._)\-:,]+\s*", "", line)
    line = re.sub(r"^\s*\d{1,2}\s+(?=(tab|cap|syp|syr|inj|susp)\b)", "", line, flags=re.I)
    return re.sub(r"\b([0-9OoSsIlB]{1,4})(mg|ml|mcg)\b",
                  lambda m: m.group(1).translate(_CONFUSE) + m.group(2), line, flags=re.I)


def _has_rx_context(d):
    strength_ok = bool(re.search(r"\d", d.get("STRENGTH", "")))
    return "FORM" in d or _freq(d.get("FREQ"))[0] is not None or strength_ok


def decode_lines(lines, ner, ft, corrector):
    """lines: list of str. Returns (meds list, alerts list)."""
    lines = [clean_line(l) for l in lines]
    token_lists = [tokenize(l) for l in lines]
    keep = [i for i, t in enumerate(token_lists) if t]
    tags = predict(ner, ft, [token_lists[i] for i in keep]) if keep else []
    meds = []
    for i, tg in zip(keep, tags):
        ents = _spans(token_lists[i], tg)
        d = {}
        for lab, txt in ents:
            d.setdefault(lab, txt)
        if "DRUG" not in d:
            continue
        raw = d["DRUG"].split()[0]
        canon, conf = corrector.correct(raw)
        if canon is None or conf < 0.75:  # NER may have picked the wrong word: scan the line
            best = max(((corrector.correct(t), t) for t in token_lists[i] if t.isalpha() and len(t) >= 3),
                       key=lambda z: z[0][1], default=((None, 0.0), raw))
            if best[0][0] and best[0][1] >= 0.85:
                (canon, conf), raw = best
        ctx = _has_rx_context(d)
        if (canon is None or conf < 0.75) and not ctx:
            continue  # header / signature / address line, not a medicine
        generics = BRANDS.get(canon, []) if canon else []
        _, f_en, f_hi = _freq(d.get("FREQ"))
        du_en, du_hi = _dur(d.get("DUR"))
        in_en, in_hi = _instr(d.get("INSTR"))
        form_en, form_hi = FORMS.get((d.get("FORM") or "").split()[0] if d.get("FORM") else "", ("", ""))
        meds.append(dict(line=lines[i], ocr_word=raw, drug=canon or "UNRECOGNISED", match_conf=round(conf, 2),
                         generic=", ".join(generics) or "-",
                         drug_class=", ".join(GENERIC_CLASS.get(g, "") for g in generics) or "-",
                         form=form_en, form_hi=form_hi, strength=d.get("STRENGTH", ""),
                         freq_en=f_en, freq_hi=f_hi, dur_en=du_en, dur_hi=du_hi, instr_en=in_en, instr_hi=in_hi))
    pairs = [(m["drug"], g) for m in meds if m["drug"] != "UNRECOGNISED" for g in m["generic"].split(", ")]
    return meds, check_interactions(pairs)


def explain(meds, lang="en"):
    out = []
    for i, m in enumerate(meds, 1):
        name = m["drug"].title()
        if lang == "hi":
            parts = [p for p in [m["freq_hi"], m["instr_hi"], (m["dur_hi"] + " तक") if m["dur_hi"] else None] if p]
            out.append(f"{i}. **{name}** ({m['generic']}) {m['form_hi']} {m['strength']} — " + ", ".join(parts) + "।")
        else:
            parts = [p for p in [m["freq_en"], m["instr_en"], ("for " + m["dur_en"]) if m["dur_en"] else None] if p]
            out.append(f"{i}. **{name}** ({m['generic']}; {m['drug_class']}) {m['form']} {m['strength']} — "
                       + ", ".join(parts) + ".")
        if m["drug"] == "UNRECOGNISED":
            out[-1] += "  ⚠️ could not confidently read this drug name - check with your pharmacist."
        elif m["match_conf"] < 0.75:
            out[-1] += f"  ⚠️ low read confidence ({m['match_conf']}) - verify the name."
    return "\n".join(out)
