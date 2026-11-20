"""Curated drug lexicon, interaction rules and Rx shorthand for Mediosis RxDecode.

Screening-level knowledge base only. NOT exhaustive. Always verify with a pharmacist.
"""

# brand / generic token -> list of generic ingredients
BRANDS = {
    # analgesics / antipyretics / NSAIDs
    "crocin": ["paracetamol"], "dolo": ["paracetamol"], "calpol": ["paracetamol"],
    "combiflam": ["ibuprofen", "paracetamol"], "brufen": ["ibuprofen"],
    "voveran": ["diclofenac"], "zerodol": ["aceclofenac"], "ecosprin": ["aspirin"],
    # antibiotics
    "augmentin": ["amoxicillin", "clavulanic acid"], "mox": ["amoxicillin"],
    "azithral": ["azithromycin"], "azee": ["azithromycin"], "ciplox": ["ciprofloxacin"],
    "zifi": ["cefixime"], "metrogyl": ["metronidazole"], "flagyl": ["metronidazole"],
    "claribid": ["clarithromycin"],
    # GI
    "pantocid": ["pantoprazole"], "omez": ["omeprazole"], "razo": ["rabeprazole"],
    "digene": ["antacid (aluminium/magnesium hydroxide)"], "ondem": ["ondansetron"],
    "emeset": ["ondansetron"], "domstal": ["domperidone"],
    # allergy / respiratory
    "allegra": ["fexofenadine"], "cetzine": ["cetirizine"], "okacet": ["cetirizine"],
    "levocet": ["levocetirizine"], "montair": ["montelukast"], "asthalin": ["salbutamol"],
    "deriphyllin": ["etofylline", "theophylline"], "wysolone": ["prednisolone"],
    # diabetes / cardio
    "glycomet": ["metformin"], "amaryl": ["glimepiride"], "januvia": ["sitagliptin"],
    "telma": ["telmisartan"], "losar": ["losartan"], "amlong": ["amlodipine"],
    "amlokind": ["amlodipine"], "clopilet": ["clopidogrel"], "atorva": ["atorvastatin"],
    "rosuvas": ["rosuvastatin"], "lasix": ["furosemide"], "aldactone": ["spironolactone"],
    "lanoxin": ["digoxin"], "warf": ["warfarin"],
    # thyroid / supplements
    "thyronorm": ["levothyroxine"], "eltroxin": ["levothyroxine"],
    "shelcal": ["calcium carbonate", "vitamin d3"], "becosules": ["vitamin b complex"],
    "limcee": ["vitamin c"],
}

GENERIC_CLASS = {
    "paracetamol": "analgesic/antipyretic", "ibuprofen": "NSAID", "diclofenac": "NSAID",
    "aceclofenac": "NSAID", "aspirin": "antiplatelet/NSAID", "amoxicillin": "antibiotic",
    "clavulanic acid": "beta-lactamase inhibitor", "azithromycin": "macrolide antibiotic",
    "ciprofloxacin": "fluoroquinolone antibiotic", "cefixime": "cephalosporin antibiotic",
    "metronidazole": "antibiotic/antiprotozoal", "clarithromycin": "macrolide antibiotic",
    "pantoprazole": "PPI (acidity)", "omeprazole": "PPI (acidity)", "rabeprazole": "PPI (acidity)",
    "antacid (aluminium/magnesium hydroxide)": "antacid", "ondansetron": "anti-emetic",
    "domperidone": "anti-emetic/prokinetic", "fexofenadine": "antihistamine",
    "cetirizine": "antihistamine", "levocetirizine": "antihistamine",
    "montelukast": "anti-asthma (leukotriene)", "salbutamol": "bronchodilator",
    "etofylline": "bronchodilator", "theophylline": "bronchodilator",
    "prednisolone": "corticosteroid", "metformin": "antidiabetic", "glimepiride": "antidiabetic",
    "sitagliptin": "antidiabetic", "telmisartan": "antihypertensive (ARB)",
    "losartan": "antihypertensive (ARB)", "amlodipine": "antihypertensive (CCB)",
    "clopidogrel": "antiplatelet", "atorvastatin": "statin", "rosuvastatin": "statin",
    "furosemide": "diuretic", "spironolactone": "diuretic (K-sparing)", "digoxin": "cardiac glycoside",
    "warfarin": "anticoagulant", "levothyroxine": "thyroid hormone",
    "calcium carbonate": "supplement", "vitamin d3": "supplement",
    "vitamin b complex": "supplement", "vitamin c": "supplement",
}

# generic names usable directly as tokens (single-word ones)
for _g in list(GENERIC_CLASS):
    if " " not in _g and "(" not in _g:
        BRANDS.setdefault(_g, [_g])

CANONICAL = sorted(BRANDS)  # all valid drug tokens

NSAIDS = {"ibuprofen", "diclofenac", "aceclofenac"}
ARBS = {"telmisartan", "losartan"}
QT_DRUGS = {"azithromycin", "clarithromycin", "ondansetron", "domperidone", "ciprofloxacin"}
CHELATORS = {"calcium carbonate", "antacid (aluminium/magnesium hydroxide)"}

# (set_a, set_b, severity, message)
PAIR_RULES = [
    ({"warfarin"}, NSAIDS | {"aspirin", "clopidogrel"}, "MAJOR", "Bleeding risk increases sharply when combined with an anticoagulant."),
    ({"warfarin"}, {"metronidazole"}, "MAJOR", "Metronidazole strongly raises warfarin levels (INR) - bleeding risk."),
    ({"warfarin"}, {"ciprofloxacin", "azithromycin", "clarithromycin"}, "MODERATE", "Antibiotic may raise INR; monitor."),
    ({"clopidogrel"}, {"omeprazole"}, "MODERATE", "Omeprazole reduces clopidogrel activation; pantoprazole is usually preferred."),
    ({"aspirin"}, NSAIDS, "MODERATE", "Extra GI-bleeding risk; ibuprofen can blunt aspirin's heart protection."),
    ({"prednisolone"}, NSAIDS | {"aspirin"}, "MODERATE", "Steroid + NSAID raises stomach ulcer/bleeding risk."),
    ({"spironolactone"}, ARBS, "MAJOR", "High potassium (hyperkalemia) risk; check K+ levels."),
    ({"digoxin"}, {"furosemide"}, "MODERATE", "Diuretic can lower potassium and trigger digoxin toxicity."),
    ({"digoxin"}, {"clarithromycin"}, "MAJOR", "Clarithromycin raises digoxin levels."),
    ({"atorvastatin"}, {"clarithromycin"}, "MAJOR", "Raised statin levels - muscle damage (myopathy) risk."),
    ({"levothyroxine"}, CHELATORS, "MODERATE", "Reduces thyroid tablet absorption - keep a 4 hour gap."),
    ({"ciprofloxacin"}, CHELATORS, "MODERATE", "Reduces ciprofloxacin absorption - take cipro 2 h before / 6 h after."),
    ({"glimepiride"}, {"ciprofloxacin"}, "MODERATE", "Fluoroquinolones can swing blood sugar (hypo/hyper)."),
    (NSAIDS, ARBS, "MODERATE", "NSAID may reduce BP control and stress kidneys."),
]


def check_interactions(generics):
    """generics: list of (source_label, generic). Returns list of dict alerts."""
    alerts, seen = [], set()
    names = [g for _, g in generics]
    uniq = set(names)
    # duplicate ingredient from different products (e.g. Crocin + Combiflam)
    for g in uniq:
        srcs = sorted({s for s, x in generics if x == g})
        if len(srcs) > 1:
            alerts.append({"severity": "MAJOR", "drugs": " + ".join(srcs),
                           "issue": f"Duplicate ingredient '{g}' - overdose risk."})
    for a, b, sev, msg in PAIR_RULES:
        for x in uniq & a:
            for y in uniq & b:
                if x == y:
                    continue
                key = tuple(sorted((x, y))) + (msg,)
                if key in seen:
                    continue
                seen.add(key)
                alerts.append({"severity": sev, "drugs": f"{x} + {y}", "issue": msg})
    nsaids = sorted(uniq & NSAIDS)
    if len(nsaids) > 1:
        alerts.append({"severity": "MAJOR", "drugs": " + ".join(nsaids),
                       "issue": "Two NSAIDs together - stomach bleeding & kidney risk."})
    if uniq & NSAIDS and uniq & ARBS and "furosemide" in uniq:
        alerts.append({"severity": "MAJOR", "drugs": "NSAID + ARB + furosemide",
                       "issue": "'Triple whammy' - acute kidney injury risk."})
    qt = sorted(uniq & QT_DRUGS)
    if len(qt) > 1:
        alerts.append({"severity": "MODERATE", "drugs": " + ".join(qt),
                       "issue": "Combined QT-prolongation risk (heart rhythm)."})
    order = {"MAJOR": 0, "MODERATE": 1}
    return sorted(alerts, key=lambda d: order[d["severity"]])


# ---------- Rx shorthand ----------
FREQ = {
    "od": (1, "once a day", "दिन में एक बार"), "qd": (1, "once a day", "दिन में एक बार"),
    "bd": (2, "twice a day", "दिन में दो बार"), "bid": (2, "twice a day", "दिन में दो बार"),
    "tds": (3, "three times a day", "दिन में तीन बार"), "tid": (3, "three times a day", "दिन में तीन बार"),
    "qid": (4, "four times a day", "दिन में चार बार"), "hs": (1, "at bedtime", "सोने से पहले"),
    "sos": (0, "only when needed", "ज़रूरत पड़ने पर"), "prn": (0, "only when needed", "ज़रूरत पड़ने पर"),
    "stat": (1, "immediately, once", "तुरंत, एक बार"),
}
INSTR = {
    "ac": ("before food", "खाने से पहले"), "pc": ("after food", "खाने के बाद"),
    "before": ("before food", "खाने से पहले"), "after": ("after food", "खाने के बाद"),
    "empty": ("on an empty stomach", "खाली पेट"),
}
FORMS = {"tab": ("tablet", "गोली"), "tabs": ("tablet", "गोली"), "cap": ("capsule", "कैप्सूल"),
         "syp": ("syrup", "सिरप"), "syr": ("syrup", "सिरप"), "inj": ("injection", "इंजेक्शन"),
         "drops": ("drops", "ड्रॉप्स"), "susp": ("suspension", "सस्पेंशन")}
SLOTS_EN = ["morning", "afternoon", "night"]
SLOTS_HI = ["सुबह", "दोपहर", "रात"]
