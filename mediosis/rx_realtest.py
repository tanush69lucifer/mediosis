"""Hand-written, real-style prescription lines (formats NOT produced by the generator templates,
plus realistic OCR errors). Used as an honest out-of-template test for the NER tagger."""
from .rx_data import tokenize

# each line = list of (text chunk, label)  ; label "O" = outside
REAL_LINES = [
    [("Tab.", "FORM"), ("Dolo", "DRUG"), ("650", "STRENGTH"), ("1 tab", "O"), ("thrice daily", "FREQ"), ("for 3 days", "DUR")],
    [("Rx", "O"), ("Augmentin", "DRUG"), ("625 mg", "STRENGTH"), ("BD", "FREQ"), ("after meals", "INSTR"), ("x 5 days", "DUR")],
    [("Cap", "FORM"), ("Pantocid", "DRUG"), ("40mg", "STRENGTH"), ("OD", "FREQ"), ("before breakfast", "INSTR")],
    [("Tab", "FORM"), ("Azithrornycin", "DRUG"), ("500 mg", "STRENGTH"), ("once daily", "FREQ"), ("x 3 days", "DUR")],
    [("Syp", "FORM"), ("Calpo1", "DRUG"), ("5 ml", "STRENGTH"), ("SOS", "FREQ")],
    [("Tab", "FORM"), ("Glycornet", "DRUG"), ("500mg", "STRENGTH"), ("1-0-1", "FREQ"), ("after food", "INSTR"), ("for 30 days", "DUR")],
    [("Tab", "FORM"), ("Telma", "DRUG"), ("40", "STRENGTH"), ("0-0-1", "FREQ"), ("continue", "O")],
    [("Tab", "FORM"), ("Ecosprin", "DRUG"), ("75mg", "STRENGTH"), ("HS", "FREQ"), ("x 1 month", "DUR")],
    [("Tab", "FORM"), ("Combiflarn", "DRUG"), ("SOS", "FREQ"), ("for pain", "O")],
    [("Cap", "FORM"), ("0mez", "DRUG"), ("20", "STRENGTH"), ("BD", "FREQ"), ("AC", "INSTR"), ("x 14 days", "DUR")],
    [("Tab", "FORM"), ("Montair", "DRUG"), ("10 mg", "STRENGTH"), ("at night", "FREQ"), ("for 2 weeks", "DUR")],
    [("Tab", "FORM"), ("Thyronorm", "DRUG"), ("50mcg", "STRENGTH"), ("1-0-0", "FREQ"), ("empty stomach", "INSTR")],
    [("Inj", "FORM"), ("Ondem", "DRUG"), ("4mg", "STRENGTH"), ("STAT", "FREQ")],
    [("Tab", "FORM"), ("Shelca1", "DRUG"), ("500", "STRENGTH"), ("OD", "FREQ"), ("after lunch", "INSTR"), ("x 1 month", "DUR")],
    [("Tab", "FORM"), ("Allegra", "DRUG"), ("120mg", "STRENGTH"), ("once daily", "FREQ"), ("x 5 d", "DUR")],
    [("Tab", "FORM"), ("Warf", "DRUG"), ("5", "STRENGTH"), ("OD", "FREQ"), ("as advised", "O")],
    [("Tab", "FORM"), ("Metrogy1", "DRUG"), ("400mg", "STRENGTH"), ("TDS", "FREQ"), ("x 5 days", "DUR")],
    [("Tab", "FORM"), ("Clopilet", "DRUG"), ("75", "STRENGTH"), ("0-1-0", "FREQ"), ("after lunch", "INSTR")],
]


def real_test_set():
    data = []
    for chunks in REAL_LINES:
        toks, tags = [], []
        for text, lab in chunks:
            ts = tokenize(text)
            toks += ts
            tags += ["O"] * len(ts) if lab == "O" else [("B-" if i == 0 else "I-") + lab for i in range(len(ts))]
        data.append((toks, tags))
    return data


# gold canonical names for the misspelt drugs above (for corrector eval)
REAL_DRUG_GOLD = {"dolo": "dolo", "augmentin": "augmentin", "pantocid": "pantocid",
                  "azithrornycin": "azithromycin", "calpo1": "calpol", "glycornet": "glycomet",
                  "telma": "telma", "ecosprin": "ecosprin", "combiflarn": "combiflam", "0mez": "omez",
                  "montair": "montair", "thyronorm": "thyronorm", "ondem": "ondem", "shelca1": "shelcal",
                  "allegra": "allegra", "warf": "warf", "metrogy1": "metrogyl", "clopilet": "clopilet"}
