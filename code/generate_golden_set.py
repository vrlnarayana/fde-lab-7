"""
generate_golden_set.py
----------------------
Builds a 50-question GOLDEN TEST SET for retrieval evaluation, and validates
that every 'relevant_doc_ids' entry exists in the generated corpus.

Question categories:
  - single_hop  : answer sits in ONE document
  - multi_hop   : answer needs TWO or more documents chained together
  - temporal    : answer depends on comparing VERSIONS across years
  - negative    : answer is NOT in the corpus (tests "I don't know" behaviour)

Output: ../golden_set/golden_set.json  and  ../golden_set/golden_set.csv
"""

import json, csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
GOLD = os.path.abspath(os.path.join(HERE, "..", "golden_set"))
os.makedirs(GOLD, exist_ok=True)

# Load corpus ids for validation
valid_ids = set()
with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
    for line in f:
        valid_ids.add(json.loads(line)["id"])

Q = []
def add(qid, question, category, relevant, answer, hops=1):
    Q.append({
        "qid": qid,
        "question": question,
        "category": category,
        "relevant_doc_ids": relevant,
        "reference_answer": answer,
        "hops": hops,
    })

# ---- single_hop (1 doc) ----
add("Q01", "What is the time target to deliver the Sepsis Six bundle after recognition?", "single_hop", ["SEP-v3_0"], "Within one hour of recognition.")
add("Q02", "Which score replaced SIRS for sepsis screening in the 2019 sepsis pathway?", "single_hop", ["SEP-v2_0"], "qSOFA together with NEWS2.")
add("Q03", "What is the first-line empirical antibiotic for severe sepsis of unknown source?", "single_hop", ["AMS-v3_0"], "Piperacillin-tazobactam.")
add("Q04", "How is Stage 3 acute kidney injury defined by KDIGO creatinine criteria?", "single_hop", ["AKI-v2_0"], "A 3.0 times or greater rise in serum creatinine, or starting dialysis.")
add("Q05", "What NT-proBNP level warrants urgent echocardiography in heart failure?", "single_hop", ["HF-v1_0"], "Above 2000 ng/L, with echo within two weeks.")
add("Q06", "At what eGFR should metformin be stopped?", "single_hop", ["DIA-v1_0"], "When eGFR is below 30 mL/min.")
add("Q07", "What imaging must be done before stroke thrombolysis?", "single_hop", ["STR-v1_0"], "Urgent non-contrast CT brain to exclude haemorrhage.")
add("Q08", "Which agent reverses dabigatran?", "single_hop", ["ANTI-v1_0"], "Idarucizumab.")
add("Q09", "Within how many hours of admission must VTE risk be assessed in the 2018 pathway?", "single_hop", ["VTE-v1_0"], "Within 14 hours.")
add("Q10", "Which medication prevents and treats eclamptic seizures?", "single_hop", ["MAT-v1_0"], "Magnesium sulphate.")
add("Q11", "Which tool is used to screen for delirium?", "single_hop", ["DEL-v1_0"], "The 4AT tool.")
add("Q12", "What is the four-pillar therapy for heart failure with reduced ejection fraction?", "single_hop", ["HF-v2_0"], "ARNI/ACE inhibitor, beta-blocker, mineralocorticoid receptor antagonist, and SGLT2 inhibitor.")
add("Q13", "What mean arterial pressure target defines the vasopressor goal in septic shock?", "single_hop", ["SEP-v2_0"], "65 mmHg or above.")
add("Q14", "Which scoring tools assess stroke risk and bleeding risk in atrial fibrillation?", "single_hop", ["ANTI-v1_0"], "CHA2DS2-VASc for stroke risk and HAS-BLED for bleeding risk.")
add("Q15", "What is the default antibiotic course length in the 2021 stewardship pathway?", "single_hop", ["AMS-v2_0"], "Five days for uncomplicated infections.")
add("Q16", "Which early warning score is used for pregnant patients?", "single_hop", ["MAT-v1_0"], "MEOWS (Modified Early Obstetric Warning Score).")
add("Q17", "What thrombolytic alternative to alteplase is accepted in the 2023 stroke pathway?", "single_hop", ["STR-v2_0"], "Tenecteplase.")
add("Q18", "What urine output threshold signals Stage 1 AKI in the 2022 pathway?", "single_hop", ["AKI-v2_0"], "Less than 0.5 mL/kg/hour for six hours.")
add("Q19", "What HbA1c value diagnoses type 2 diabetes?", "single_hop", ["DIA-v1_0"], "48 mmol/mol (6.5%) or above.")
add("Q20", "What first-line oral drug is used for severe hypertension in pre-eclampsia?", "single_hop", ["MAT-v2_0"], "Labetalol.")

# ---- multi_hop (2+ docs) ----
add("Q21", "A septic patient has worsening kidney function. Which pathway sets the antibiotic renal dose, and which one tells you to detect the falling eGFR?", "multi_hop", ["SEP-v3_0", "AMS-v3_0", "AKI-v2_0"], "Sepsis pathway directs to the Antimicrobial Stewardship Pathway for renal dosing, and the Acute Kidney Injury Pathway for eGFR detection.", hops=3)
add("Q22", "For a pregnant woman with sepsis, which two pathways together decide antibiotic choice?", "multi_hop", ["MAT-v2_0", "AMS-v3_0", "SEP-v3_0"], "The Maternal Deterioration Pathway plus the Antimicrobial Stewardship Pathway, triggered through the Adult Sepsis Pathway, using pregnancy-safe agents.", hops=3)
add("Q23", "A stroke patient is on a factor Xa inhibitor. Which pathway covers reversal before thrombolysis?", "multi_hop", ["STR-v2_0", "ANTI-v2_0"], "The Stroke Pathway sends you to the Anticoagulation and Bleeding Risk Pathway, which uses andexanet alfa or prothrombin complex concentrate.", hops=2)
add("Q24", "A heart failure patient develops sepsis. How should fluid resuscitation differ, and which pathways govern this?", "multi_hop", ["HF-v2_0", "SEP-v3_0", "AKI-v2_0"], "Use small 250-500 mL aliquots with reassessment to avoid overload, coordinated between the Heart Failure, Sepsis, and AKI pathways.", hops=3)
add("Q25", "After anticoagulation is held for bleeding, which pathway decides when to restart VTE prophylaxis?", "multi_hop", ["VTE-v2_0", "ANTI-v2_0"], "The VTE Prevention Pathway coordinates restart with the Anticoagulation and Bleeding Risk Pathway.", hops=2)
add("Q26", "A diabetic patient becomes acutely unwell with falling kidney function. Which drugs are held and which pathway guides eGFR monitoring?", "multi_hop", ["DIA-v3_0", "AKI-v2_0"], "Hold metformin and SGLT2 inhibitors per sick day rules; the Acute Kidney Injury Pathway guides eGFR monitoring.", hops=2)
add("Q27", "An older patient with infection develops delirium. Which pathways together cover the trigger and the correct antibiotic dosing?", "multi_hop", ["DEL-v1_0", "SEP-v2_0", "AMS-v2_0"], "The Delirium Pathway identifies infection/sepsis as a trigger; the Sepsis and Antimicrobial Stewardship pathways ensure correct renal antibiotic dosing.", hops=3)
add("Q28", "Which pathway holds the eGFR dose bands that the AKI pathway refers to for vancomycin and gentamicin?", "multi_hop", ["AKI-v2_0", "AMS-v2_0"], "The Antimicrobial Stewardship Pathway holds the eGFR dose bands referenced by the AKI Pathway.", hops=2)
add("Q29", "For a patient on a DOAC needing surgery, which pathway covers interruption timing and which covers restarting clot prevention?", "multi_hop", ["ANTI-v2_0", "VTE-v2_0"], "The Anticoagulation Pathway covers peri-procedural interruption; the VTE Prevention Pathway covers prophylaxis restart.", hops=2)
add("Q30", "Maternal sepsis is suspected. Which screening score is used and which pathway sets the bundle?", "multi_hop", ["MAT-v2_0", "SEP-v3_0"], "Use MEOWS triggers from the Maternal Pathway, applying the Sepsis Six bundle from the Adult Sepsis Pathway.", hops=2)
add("Q31", "A diabetic heart-failure patient needs glucose therapy with cardiac benefit. Which class is advised and which pathway flags fluid caution during illness?", "multi_hop", ["DIA-v2_0", "HF-v2_0"], "SGLT2 inhibitors are advised; the Heart Failure Pathway flags fluid caution during acute illness.", hops=2)
add("Q32", "When sepsis causes AKI, which pathway warns about fluid overload and which sets the resuscitation bundle?", "multi_hop", ["AKI-v2_0", "SEP-v3_0"], "The AKI Pathway warns on fluid overload; the Sepsis Pathway sets the Sepsis Six bundle.", hops=2)
add("Q33", "A patient with penicillin allergy has severe sepsis. Which pathway gives the alternative regimen and which sets the one-hour antibiotic target?", "multi_hop", ["AMS-v3_0", "SEP-v3_0"], "The Antimicrobial Stewardship Pathway gives ciprofloxacin plus metronidazole; the Sepsis Pathway sets the one-hour first dose.", hops=2)
add("Q34", "Which two pathways must be consulted to manage anticoagulation in a stroke patient who is also at venous thromboembolism risk?", "multi_hop", ["ANTI-v2_0", "VTE-v2_0", "STR-v2_0"], "The Anticoagulation, VTE Prevention, and Stroke pathways together.", hops=3)
add("Q35", "Where is contrast-related kidney injury covered, and where are the antibiotic renal doses for the same patient?", "multi_hop", ["AKI-v1_0", "AMS-v1_0"], "Contrast caution is in the AKI Pathway; antibiotic renal doses are in the Antimicrobial Stewardship Pathway.", hops=2)

# ---- temporal (compare versions across years) ----
add("Q36", "How did sepsis screening change between the 2014 and 2019 pathways?", "temporal", ["SEP-v1_0", "SEP-v2_0"], "It moved from SIRS criteria (2014) to qSOFA plus NEWS2 (2019).", hops=2)
add("Q37", "How did the initial fluid bolus advice change from the 2014 to the 2024 sepsis pathway?", "temporal", ["SEP-v1_0", "SEP-v3_0"], "From a single 30 mL/kg bolus to smaller 250-500 mL aliquots with reassessment using balanced crystalloid.", hops=2)
add("Q38", "What new drug classes were added to diabetes care between 2014 and 2025?", "temporal", ["DIA-v1_0", "DIA-v2_0", "DIA-v3_0"], "SGLT2 inhibitors (2020) and GLP-1 receptor agonists (2025) were added to metformin.", hops=3)
add("Q39", "How did heart failure first-line therapy change from 2015 to 2023?", "temporal", ["HF-v1_0", "HF-v2_0"], "From ACE inhibitor plus beta-blocker to four-pillar therapy adding an MRA and an SGLT2 inhibitor.", hops=2)
add("Q40", "How did the stroke time window change between 2016 and 2023?", "temporal", ["STR-v1_0", "STR-v2_0"], "Thrombolysis within 4.5 hours plus mechanical thrombectomy up to 24 hours in selected patients.", hops=2)
add("Q41", "What changed in default antibiotic duration between the 2015 and 2021 stewardship pathways?", "temporal", ["AMS-v1_0", "AMS-v2_0"], "Default moved from 5-7 days to 5 days for uncomplicated infections.", hops=2)
add("Q42", "What new AKI detection method was introduced in 2022 compared with 2016?", "temporal", ["AKI-v1_0", "AKI-v2_0"], "An automated electronic AKI alert plus urine-output staging criteria.", hops=2)
add("Q43", "What reversal-stock change happened between the 2017 and 2024 anticoagulation pathways?", "temporal", ["ANTI-v1_0", "ANTI-v2_0"], "Idarucizumab and andexanet alfa are now stocked in the Emergency Department.", hops=2)
add("Q44", "What was added to VTE prophylaxis options between 2018 and 2025?", "temporal", ["VTE-v1_0", "VTE-v2_0"], "Direct oral anticoagulants for extended prophylaxis after major orthopaedic surgery.", hops=2)
add("Q45", "What diagnostic test was added to pre-eclampsia care between 2017 and 2024?", "temporal", ["MAT-v1_0", "MAT-v2_0"], "Placental growth factor based testing.")
add("Q46", "Which is the current version of the Antimicrobial Stewardship Pathway and what did it add?", "temporal", ["AMS-v3_0"], "v3.0 (2025), which added OPAT and a maternal antibiotics annex.")
add("Q47", "Between 2015 and 2025, when were sick day rules introduced for diabetes drugs?", "temporal", ["DIA-v2_0", "DIA-v3_0"], "Sick day rules were introduced in the 2020 version and reinforced with a written card in 2025.", hops=2)

# ---- negative (not in corpus) ----
add("Q48", "What is the recommended insulin pump brand for paediatric type 1 diabetes?", "negative", [], "Not covered in this corpus; the model should say it cannot find this.")
add("Q49", "What is the network policy on organ transplantation immunosuppression?", "negative", [], "Not covered in this corpus; the model should decline.")
add("Q50", "What are the COVID-19 vaccination intervals in the current schedule?", "negative", [], "Not covered in this corpus; the model should decline.")

# Validate
bad = []
for q in Q:
    for d in q["relevant_doc_ids"]:
        if d not in valid_ids:
            bad.append((q["qid"], d))
if bad:
    raise SystemExit(f"INVALID doc ids referenced: {bad}")

assert len(Q) == 50, f"Expected 50 questions, got {len(Q)}"

with open(os.path.join(GOLD, "golden_set.json"), "w", encoding="utf-8") as f:
    json.dump(Q, f, indent=2, ensure_ascii=False)

with open(os.path.join(GOLD, "golden_set.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["qid", "category", "hops", "question", "relevant_doc_ids", "reference_answer"])
    for q in Q:
        w.writerow([q["qid"], q["category"], q["hops"], q["question"],
                    "|".join(q["relevant_doc_ids"]), q["reference_answer"]])

from collections import Counter
cats = Counter(q["category"] for q in Q)
print(f"Wrote {len(Q)} questions. Categories: {dict(cats)}")
print("All relevant_doc_ids validated against corpus.")
