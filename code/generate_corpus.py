"""
generate_corpus.py
------------------
Generates a SYNTHETIC clinical "care-pathway guideline" corpus for the
Azure AI Search / Hybrid Retrieval / RAG lab (Lab 7).

IMPORTANT: This data is 100% synthetic and invented for teaching retrieval.
It is NOT medical advice and must NEVER be used for real clinical decisions.

What it produces (inside ../data):
  - guidelines/*.md      : one human-readable Markdown file per guideline version
  - corpus.jsonl         : one JSON record per guideline version (full text + metadata)
  - corpus_manifest.csv  : quick index of every document (for eyeballing)

The corpus is designed so that:
  * Pathways have MULTIPLE VERSIONS across 2014..2025 (>= 12 years) -> temporal questions
  * Guidelines CROSS-REFERENCE each other -> multi-hop questions
  * Every section carries a citation handle (guideline_id + version + section)
"""

import json
import csv
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
GDIR = os.path.join(DATA, "guidelines")
os.makedirs(GDIR, exist_ok=True)

ISSUER = "Meridian Health Network — Clinical Standards Office (SYNTHETIC)"

# ---------------------------------------------------------------------------
# Each pathway = a clinical topic. Each pathway has several versions over years.
# A version = one searchable DOCUMENT. Sections carry the real teaching content.
# cross_refs point to OTHER guideline IDs (used for multi-hop questions).
# ---------------------------------------------------------------------------

PATHWAYS = [
    {
        "code": "SEP",
        "title": "Adult Sepsis Recognition and Management Pathway",
        "specialty": "Emergency & Critical Care",
        "versions": [
            {
                "version": "v1.0", "year": 2014,
                "sections": [
                    ("Scope", "This pathway covers recognition and first-hour management of suspected sepsis in adults aged 18 and above presenting to the Emergency Department or inpatient wards."),
                    ("Recognition", "Screen any patient with suspected infection using SIRS criteria: temperature above 38C or below 36C, heart rate above 90, respiratory rate above 20, white cell count abnormal. Two or more criteria with suspected infection should trigger a sepsis alert."),
                    ("Sepsis Six Bundle", "Within one hour of recognition, deliver: (1) high-flow oxygen, (2) blood cultures, (3) intravenous broad-spectrum antibiotics, (4) intravenous fluid challenge, (5) measure serum lactate, (6) measure hourly urine output."),
                    ("Antibiotics", "Start empirical broad-spectrum antibiotics within one hour. For specific antibiotic choice and dose adjustment in kidney impairment, follow the Antimicrobial Stewardship Pathway."),
                    ("Fluids", "Give an initial crystalloid bolus of 30 mL/kg for hypotension or lactate above 4 mmol/L. Reassess after each bolus."),
                    ("Escalation", "If lactate remains above 4 mmol/L or hypotension persists after fluids, escalate to critical care for consideration of vasopressors."),
                ],
                "cross_refs": ["AMS", "AKI"],
            },
            {
                "version": "v2.0", "year": 2019,
                "sections": [
                    ("Scope", "This version replaces SIRS-based screening with qSOFA and the National Early Warning Score 2 (NEWS2) for adults with suspected infection."),
                    ("Recognition", "Use qSOFA: respiratory rate 22 or above, altered mentation, systolic blood pressure 100 mmHg or below. A qSOFA score of 2 or more indicates higher risk. NEWS2 of 5 or more should also trigger a sepsis screen."),
                    ("Sepsis Six Bundle", "The Sepsis Six bundle is retained and must still be completed within one hour of recognition. Lactate is now mandatory at recognition and repeated at 2 hours if initially raised."),
                    ("Antibiotics", "Empirical antibiotics within one hour remain the standard. Renal dose adjustment and de-escalation at 48-72 hours are governed by the Antimicrobial Stewardship Pathway."),
                    ("Fluids", "Initial bolus reduced to 250-500 mL aliquots with reassessment, up to 30 mL/kg, to avoid fluid overload in heart failure and kidney patients. See Acute Kidney Injury Pathway for fluid caution."),
                    ("Escalation", "Persistent lactate above 2 mmol/L after resuscitation now defines septic shock when vasopressors are required to keep mean arterial pressure at 65 mmHg or above."),
                ],
                "cross_refs": ["AMS", "AKI", "HF"],
            },
            {
                "version": "v3.0", "year": 2024,
                "sections": [
                    ("Scope", "Current version. Adds a dedicated section on sepsis in pregnancy and links to the Maternal Sepsis annex of the Pre-eclampsia and Maternal Deterioration Pathway."),
                    ("Recognition", "NEWS2 remains the primary inpatient trigger. For maternal patients use MEOWS (Modified Early Obstetric Warning Score) because normal pregnancy physiology alters vital sign thresholds."),
                    ("Sepsis Six Bundle", "Bundle unchanged but documentation must record the exact recognition timestamp to measure the one-hour target. Lactate point-of-care testing is preferred."),
                    ("Antibiotics", "First dose within one hour. For penicillin allergy and for renal dosing, defer to the Antimicrobial Stewardship Pathway current version."),
                    ("Fluids", "Balanced crystalloid preferred over 0.9% saline. Continue 250-500 mL aliquots with reassessment. In acute kidney injury, follow fluid-balance guidance in the Acute Kidney Injury Pathway."),
                    ("Escalation", "Septic shock definition unchanged. Maternal sepsis with deterioration must trigger a senior obstetric and critical care review per the Maternal Deterioration Pathway."),
                ],
                "cross_refs": ["AMS", "AKI", "MAT"],
            },
        ],
    },
    {
        "code": "AMS",
        "title": "Antimicrobial Stewardship Pathway",
        "specialty": "Infection & Pharmacy",
        "versions": [
            {
                "version": "v1.0", "year": 2015,
                "sections": [
                    ("Scope", "Governs empirical antibiotic selection, duration, renal dose adjustment, and de-escalation across the network."),
                    ("Empirical Choice", "For undifferentiated severe infection start piperacillin-tazobactam. For suspected meningitis add ceftriaxone. Always send cultures before the first dose where it does not delay treatment beyond one hour."),
                    ("Renal Dose Adjustment", "Estimate kidney function using eGFR. Reduce dose or extend interval for renally cleared agents when eGFR is below 30 mL/min. The Acute Kidney Injury Pathway defines how to detect falling eGFR."),
                    ("Penicillin Allergy", "For confirmed severe penicillin allergy use a non-beta-lactam regimen such as ciprofloxacin plus metronidazole."),
                    ("De-escalation", "Review all empirical antibiotics at 48-72 hours against culture results and narrow the spectrum."),
                    ("Duration", "Default course is 5-7 days for most infections unless a deep-seated source requires longer."),
                ],
                "cross_refs": ["AKI"],
            },
            {
                "version": "v2.0", "year": 2021,
                "sections": [
                    ("Scope", "Updated to align with rising carbapenem resistance and to add explicit sepsis cross-links."),
                    ("Empirical Choice", "First-line for severe sepsis of unknown source remains piperacillin-tazobactam. Reserve meropenem for known extended-spectrum beta-lactamase risk or recent failure of first-line therapy."),
                    ("Renal Dose Adjustment", "Use eGFR bands: 30-50, 15-29, and below 15 mL/min, each with specific dose tables for piperacillin-tazobactam, vancomycin, and gentamicin. Recheck eGFR daily during acute kidney injury."),
                    ("Penicillin Allergy", "Adopt allergy de-labelling: many reported penicillin allergies are not true allergies. A pharmacist-led assessment can release first-line agents safely."),
                    ("De-escalation", "Mandatory antibiotic review at 48 hours documented in the electronic record, with stop, switch to oral, or continue decision."),
                    ("Duration", "Move to 5 days for uncomplicated infections in line with shorter-course evidence."),
                ],
                "cross_refs": ["AKI", "SEP"],
            },
            {
                "version": "v3.0", "year": 2025,
                "sections": [
                    ("Scope", "Current version. Adds outpatient parenteral antibiotic therapy (OPAT) and a maternal antibiotics annex."),
                    ("Empirical Choice", "Unchanged first-line piperacillin-tazobactam for severe sepsis. New guidance: for maternal sepsis avoid agents contraindicated in pregnancy and coordinate with the Maternal Deterioration Pathway."),
                    ("Renal Dose Adjustment", "eGFR bands retained. Vancomycin now requires area-under-curve monitoring rather than trough-only. During acute kidney injury follow daily eGFR checks per the Acute Kidney Injury Pathway."),
                    ("Penicillin Allergy", "Pharmacist-led de-labelling expanded network-wide. Confirmed severe allergy regimen unchanged."),
                    ("De-escalation", "48-hour review unchanged. Add procalcitonin-guided stopping for respiratory infections where available."),
                    ("Duration", "Default 5 days. OPAT pathway allows safe completion of longer courses at home with monitoring."),
                ],
                "cross_refs": ["AKI", "SEP", "MAT"],
            },
        ],
    },
    {
        "code": "AKI",
        "title": "Acute Kidney Injury (AKI) Detection and Management Pathway",
        "specialty": "Nephrology & Acute Medicine",
        "versions": [
            {
                "version": "v1.0", "year": 2016,
                "sections": [
                    ("Scope", "Detection, staging, and early management of acute kidney injury in adults."),
                    ("Detection and Staging", "Stage AKI using KDIGO creatinine criteria: Stage 1 is a 1.5-1.9 times rise in serum creatinine, Stage 2 is 2.0-2.9 times, Stage 3 is 3.0 times or more, or initiation of dialysis."),
                    ("Common Causes", "Most inpatient AKI is pre-renal from hypovolaemia or sepsis. Review nephrotoxic drugs and contrast exposure."),
                    ("Management", "Treat the cause, optimise fluid balance, and stop nephrotoxic drugs. For sepsis-related AKI, resuscitate per the Adult Sepsis Pathway while avoiding fluid overload."),
                    ("Drug Dosing", "Falling eGFR requires antibiotic and analgesic dose review. The Antimicrobial Stewardship Pathway holds the renal dose tables."),
                    ("Referral", "Refer to nephrology for Stage 3 AKI, suspected intrinsic disease, or need for renal replacement therapy."),
                ],
                "cross_refs": ["SEP", "AMS"],
            },
            {
                "version": "v2.0", "year": 2022,
                "sections": [
                    ("Scope", "Adds an automated electronic AKI alert driven by laboratory creatinine trends."),
                    ("Detection and Staging", "KDIGO staging retained and now also uses urine output criteria: less than 0.5 mL/kg/hour for 6 hours signals Stage 1. The electronic alert fires automatically on a qualifying creatinine rise."),
                    ("Common Causes", "Pre-renal causes still dominate. Add immune checkpoint inhibitor nephritis to the differential for oncology patients."),
                    ("Management", "Fluid strategy now emphasises balanced crystalloid and careful assessment to avoid overload, especially in heart failure (see Heart Failure Pathway). Stop nephrotoxins and review contrast."),
                    ("Drug Dosing", "Daily eGFR review during AKI. Vancomycin and gentamicin require level monitoring; see the Antimicrobial Stewardship Pathway dose bands."),
                    ("Referral", "Stage 3 AKI, refractory hyperkalaemia, severe acidosis, or fluid overload unresponsive to treatment require urgent nephrology and critical care review."),
                ],
                "cross_refs": ["SEP", "AMS", "HF"],
            },
        ],
    },
    {
        "code": "HF",
        "title": "Chronic Heart Failure Management Pathway",
        "specialty": "Cardiology",
        "versions": [
            {
                "version": "v1.0", "year": 2015,
                "sections": [
                    ("Scope", "Diagnosis and long-term management of chronic heart failure with reduced ejection fraction in adults."),
                    ("Diagnosis", "Measure NT-proBNP. A level above 2000 ng/L warrants urgent echocardiography within 2 weeks."),
                    ("First-line Therapy", "Start an ACE inhibitor and a beta-blocker, titrating to target doses as tolerated."),
                    ("Fluid Caution", "Patients are sensitive to fluid overload. When such patients develop sepsis or AKI, fluid resuscitation must be cautious and closely monitored."),
                    ("Monitoring", "Check kidney function and potassium within 1-2 weeks of starting or increasing ACE inhibitors."),
                ],
                "cross_refs": ["AKI", "SEP"],
            },
            {
                "version": "v2.0", "year": 2023,
                "sections": [
                    ("Scope", "Updated to four-pillar therapy for heart failure with reduced ejection fraction."),
                    ("Diagnosis", "NT-proBNP thresholds retained. Echocardiography remains the confirmatory test."),
                    ("Four-Pillar Therapy", "Combine an ARNI or ACE inhibitor, a beta-blocker, a mineralocorticoid receptor antagonist, and an SGLT2 inhibitor. Introduce all four early and titrate."),
                    ("Fluid Caution", "Fluid overload risk persists. Sepsis resuscitation should use small aliquots with reassessment; coordinate with the Adult Sepsis Pathway and Acute Kidney Injury Pathway."),
                    ("Monitoring", "Monitor kidney function and potassium after starting mineralocorticoid receptor antagonists and SGLT2 inhibitors."),
                ],
                "cross_refs": ["AKI", "SEP"],
            },
        ],
    },
    {
        "code": "DIA",
        "title": "Type 2 Diabetes Inpatient and Outpatient Pathway",
        "specialty": "Endocrinology",
        "versions": [
            {
                "version": "v1.0", "year": 2014,
                "sections": [
                    ("Scope", "Glycaemic management of adults with type 2 diabetes across inpatient and outpatient settings."),
                    ("Diagnosis", "Diagnose with HbA1c of 48 mmol/mol (6.5%) or above, confirmed on repeat unless symptomatic with high glucose."),
                    ("First-line", "Start metformin unless contraindicated by kidney function. Lifestyle advice for all."),
                    ("Renal Caution", "Metformin must be reviewed if eGFR falls. Stop metformin if eGFR is below 30 mL/min. Detection of falling eGFR follows the Acute Kidney Injury Pathway."),
                    ("Targets", "Aim for an individualised HbA1c target, commonly 53 mmol/mol (7%)."),
                ],
                "cross_refs": ["AKI"],
            },
            {
                "version": "v2.0", "year": 2020,
                "sections": [
                    ("Scope", "Adds cardiovascular and renal protective drug classes."),
                    ("Diagnosis", "Diagnostic thresholds unchanged."),
                    ("First-line", "Metformin remains first-line. Add an SGLT2 inhibitor early for patients with established cardiovascular disease, heart failure, or chronic kidney disease."),
                    ("Renal Caution", "Continue to review metformin against eGFR. SGLT2 inhibitors also have eGFR thresholds for initiation. Use the Acute Kidney Injury Pathway for eGFR monitoring during illness."),
                    ("Sick Day Rules", "During acute illness, advise temporary stopping of metformin and SGLT2 inhibitors to reduce the risk of lactic acidosis and diabetic ketoacidosis."),
                    ("Targets", "Targets individualised; relax for frail older adults."),
                ],
                "cross_refs": ["AKI", "HF"],
            },
            {
                "version": "v3.0", "year": 2025,
                "sections": [
                    ("Scope", "Current version. Adds GLP-1 receptor agonists for weight and cardiovascular benefit."),
                    ("Diagnosis", "Unchanged."),
                    ("First-line", "Metformin plus early SGLT2 inhibitor where indicated. Consider a GLP-1 receptor agonist for obesity or high cardiovascular risk."),
                    ("Renal Caution", "eGFR-based review unchanged. During acute kidney injury, hold SGLT2 inhibitors and metformin per sick day rules and the Acute Kidney Injury Pathway."),
                    ("Sick Day Rules", "Reinforced. Provide every patient a written sick day rules card."),
                    ("Targets", "Individualised, with emphasis on avoiding hypoglycaemia in frail patients."),
                ],
                "cross_refs": ["AKI", "HF"],
            },
        ],
    },
    {
        "code": "STR",
        "title": "Acute Ischaemic Stroke Pathway",
        "specialty": "Neurology",
        "versions": [
            {
                "version": "v1.0", "year": 2016,
                "sections": [
                    ("Scope", "Hyperacute assessment and reperfusion for adults with suspected acute ischaemic stroke."),
                    ("Recognition", "Use the FAST tool (Face, Arm, Speech, Time). Record the time the patient was last known well."),
                    ("Imaging", "Perform urgent non-contrast CT brain to exclude haemorrhage before any thrombolysis."),
                    ("Thrombolysis", "Offer intravenous alteplase within 4.5 hours of onset if no contraindication, after excluding haemorrhage."),
                    ("Anticoagulation Interaction", "Check whether the patient takes anticoagulants, because this affects thrombolysis eligibility. See the Anticoagulation and Bleeding Risk Pathway for reversal."),
                ],
                "cross_refs": ["ANTI"],
            },
            {
                "version": "v2.0", "year": 2023,
                "sections": [
                    ("Scope", "Adds mechanical thrombectomy and extended time windows with advanced imaging."),
                    ("Recognition", "FAST retained. Pre-hospital large vessel occlusion screening triggers direct transfer to a thrombectomy centre."),
                    ("Imaging", "Non-contrast CT plus CT angiography to identify large vessel occlusion. CT perfusion guides extended-window selection."),
                    ("Thrombolysis", "Alteplase within 4.5 hours; tenecteplase is an accepted alternative. Thrombectomy is offered up to 24 hours in selected patients with favourable perfusion imaging."),
                    ("Anticoagulation Interaction", "For patients on direct oral anticoagulants, follow the Anticoagulation and Bleeding Risk Pathway for reversal agents before any intervention."),
                ],
                "cross_refs": ["ANTI"],
            },
        ],
    },
    {
        "code": "ANTI",
        "title": "Anticoagulation and Bleeding Risk Pathway",
        "specialty": "Haematology",
        "versions": [
            {
                "version": "v1.0", "year": 2017,
                "sections": [
                    ("Scope", "Safe use of anticoagulants and management of bleeding and reversal in adults."),
                    ("Indications", "Common indications include atrial fibrillation and venous thromboembolism. Assess stroke risk with CHA2DS2-VASc and bleeding risk with HAS-BLED."),
                    ("Agents", "Direct oral anticoagulants (apixaban, rivaroxaban, dabigatran, edoxaban) are preferred over warfarin for most non-valvular atrial fibrillation."),
                    ("Reversal", "For dabigatran use idarucizumab. For factor Xa inhibitors use andexanet alfa or prothrombin complex concentrate. For warfarin use vitamin K and prothrombin complex concentrate."),
                    ("Stroke Link", "Before stroke thrombolysis, anticoagulant status must be checked; reversal may be required. See the Acute Ischaemic Stroke Pathway."),
                ],
                "cross_refs": ["STR", "VTE"],
            },
            {
                "version": "v2.0", "year": 2024,
                "sections": [
                    ("Scope", "Updated reversal stock guidance and peri-procedural management."),
                    ("Indications", "Risk scoring unchanged. Add explicit guidance on anticoagulation in cancer-associated thrombosis."),
                    ("Agents", "DOACs remain first-line for most patients. Warfarin retained for mechanical valves and severe kidney impairment."),
                    ("Reversal", "Idarucizumab and andexanet alfa stocked in the Emergency Department. Prothrombin complex concentrate remains the fallback for factor Xa inhibitors when andexanet is unavailable."),
                    ("Peri-procedural", "Define interruption timing before surgery based on drug and kidney function. Coordinate venous thromboembolism prophylaxis restart with the VTE Prevention Pathway."),
                ],
                "cross_refs": ["STR", "VTE"],
            },
        ],
    },
    {
        "code": "VTE",
        "title": "Venous Thromboembolism (VTE) Prevention Pathway",
        "specialty": "Acute Medicine & Surgery",
        "versions": [
            {
                "version": "v1.0", "year": 2018,
                "sections": [
                    ("Scope", "Risk assessment and prophylaxis to prevent hospital-associated venous thromboembolism."),
                    ("Risk Assessment", "Assess every adult admission within 14 hours using the network VTE risk tool, balancing thrombosis risk against bleeding risk."),
                    ("Prophylaxis", "Offer low molecular weight heparin to at-risk patients unless contraindicated. Use mechanical prophylaxis when bleeding risk is high."),
                    ("Renal Adjustment", "Reduce low molecular weight heparin dose in significant kidney impairment; detection follows the Acute Kidney Injury Pathway."),
                    ("Restart After Bleeding", "When anticoagulation is held for bleeding, restart timing is coordinated with the Anticoagulation and Bleeding Risk Pathway."),
                ],
                "cross_refs": ["AKI", "ANTI"],
            },
            {
                "version": "v2.0", "year": 2025,
                "sections": [
                    ("Scope", "Adds extended prophylaxis for selected surgical and cancer patients."),
                    ("Risk Assessment", "Reassessment now required at 24 hours and whenever clinical condition changes."),
                    ("Prophylaxis", "Low molecular weight heparin remains first-line. Direct oral anticoagulants are options for extended prophylaxis after major orthopaedic surgery."),
                    ("Renal Adjustment", "Dose reduction thresholds retained; follow the Acute Kidney Injury Pathway for eGFR monitoring."),
                    ("Restart After Bleeding", "Coordinate restart with the Anticoagulation and Bleeding Risk Pathway, balancing rebleeding against thrombosis."),
                ],
                "cross_refs": ["AKI", "ANTI"],
            },
        ],
    },
    {
        "code": "MAT",
        "title": "Pre-eclampsia and Maternal Deterioration Pathway",
        "specialty": "Obstetrics",
        "versions": [
            {
                "version": "v1.0", "year": 2017,
                "sections": [
                    ("Scope", "Recognition and management of pre-eclampsia and general maternal deterioration in pregnancy and the postnatal period."),
                    ("Recognition", "Use the Modified Early Obstetric Warning Score (MEOWS) because normal pregnancy changes vital sign thresholds. Blood pressure of 140/90 mmHg or above with proteinuria suggests pre-eclampsia."),
                    ("Pre-eclampsia Management", "Control severe hypertension with labetalol. Use magnesium sulphate to prevent and treat eclamptic seizures."),
                    ("Maternal Sepsis", "Maternal sepsis can progress rapidly. Apply the Adult Sepsis Pathway but use MEOWS triggers and pregnancy-safe antibiotics."),
                    ("Escalation", "Senior obstetric and anaesthetic review for severe features. Plan timing of delivery as the definitive treatment of pre-eclampsia."),
                ],
                "cross_refs": ["SEP", "AMS"],
            },
            {
                "version": "v2.0", "year": 2024,
                "sections": [
                    ("Scope", "Current version. Strengthens maternal sepsis links and adds postnatal blood pressure follow-up."),
                    ("Recognition", "MEOWS retained. Add placental growth factor based testing to support pre-eclampsia diagnosis where available."),
                    ("Pre-eclampsia Management", "Labetalol first-line; nifedipine as alternative. Magnesium sulphate unchanged for seizure prevention."),
                    ("Maternal Sepsis", "Maternal sepsis must trigger the Adult Sepsis Pathway current version with pregnancy-safe antibiotic selection from the Antimicrobial Stewardship Pathway."),
                    ("Escalation", "Severe features require consultant-led review. Provide structured postnatal blood pressure monitoring and community follow-up."),
                ],
                "cross_refs": ["SEP", "AMS"],
            },
        ],
    },
    {
        "code": "DEL",
        "title": "Delirium Prevention and Management Pathway",
        "specialty": "Geriatric Medicine",
        "versions": [
            {
                "version": "v1.0", "year": 2018,
                "sections": [
                    ("Scope", "Prevention, detection, and management of delirium in hospitalised older adults."),
                    ("Detection", "Screen at-risk patients with the 4AT tool on admission and when behaviour changes."),
                    ("Common Triggers", "Common precipitants include infection, dehydration, constipation, pain, and medications. Sepsis and acute kidney injury are frequent medical triggers."),
                    ("Management", "Treat the underlying cause first. Use non-drug measures: reorientation, sleep, hydration, mobility. Reserve low-dose antipsychotics for severe distress."),
                    ("Medication Review", "Review and minimise deliriogenic drugs. For patients with infection, ensure antibiotic doses are correct for kidney function via the Antimicrobial Stewardship Pathway."),
                ],
                "cross_refs": ["SEP", "AKI", "AMS"],
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Build documents
# ---------------------------------------------------------------------------

def doc_id(code, version):
    return f"{code}-{version.replace('.', '_')}"

# Map code -> human title (for nicer cross-ref text)
TITLE_BY_CODE = {p["code"]: p["title"] for p in PATHWAYS}

# Find the latest year per code (used to mark "current")
LATEST_YEAR = {p["code"]: max(v["year"] for v in p["versions"]) for p in PATHWAYS}

records = []
for p in PATHWAYS:
    versions_sorted = sorted(p["versions"], key=lambda v: v["year"])
    for i, v in enumerate(versions_sorted):
        did = doc_id(p["code"], v["version"])
        supersedes = doc_id(p["code"], versions_sorted[i - 1]["version"]) if i > 0 else None
        is_current = (v["year"] == LATEST_YEAR[p["code"]])
        status = "CURRENT" if is_current else "SUPERSEDED"

        # Build section text + a plain-text body
        section_objs = [{"heading": h, "text": t} for (h, t) in v["sections"]]
        body_lines = [f"## {s['heading']}\n{s['text']}" for s in section_objs]
        cross_titles = [TITLE_BY_CODE[c] for c in v["cross_refs"]]

        rec = {
            "id": did,
            "title": f"{p['title']} ({v['version']}, {v['year']})",
            "pathway_code": p["code"],
            "pathway_title": p["title"],
            "specialty": p["specialty"],
            "version": v["version"],
            "year": v["year"],
            "status": status,
            "supersedes": supersedes,
            "issuing_body": ISSUER,
            "cross_references": v["cross_refs"],
            "cross_reference_titles": cross_titles,
            "sections": section_objs,
            "content": "\n\n".join(body_lines),
            "citation": f"{p['title']} {v['version']} ({v['year']}), {ISSUER}",
            "source_uri": f"meridian-kb://guidelines/{did}.md",
        }
        records.append(rec)

# Write JSONL corpus
jsonl_path = os.path.join(DATA, "corpus.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Write one Markdown file per document
for r in records:
    md = []
    md.append(f"# {r['title']}")
    md.append("")
    md.append(f"- **Document ID:** {r['id']}")
    md.append(f"- **Specialty:** {r['specialty']}")
    md.append(f"- **Version:** {r['version']}  |  **Year:** {r['year']}  |  **Status:** {r['status']}")
    if r["supersedes"]:
        md.append(f"- **Supersedes:** {r['supersedes']}")
    md.append(f"- **Issuing body:** {r['issuing_body']}")
    if r["cross_reference_titles"]:
        md.append(f"- **Related pathways:** {', '.join(r['cross_reference_titles'])}")
    md.append(f"- **Citation handle:** {r['citation']}")
    md.append("")
    md.append("> SYNTHETIC TEACHING DATA — not medical advice. Do not use for real patient care.")
    md.append("")
    md.append(r["content"])
    with open(os.path.join(GDIR, f"{r['id']}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

# Write a manifest CSV
csv_path = os.path.join(DATA, "corpus_manifest.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "pathway_code", "title", "version", "year", "status", "supersedes", "cross_references"])
    for r in records:
        w.writerow([r["id"], r["pathway_code"], r["title"], r["version"], r["year"],
                    r["status"], r["supersedes"] or "", "|".join(r["cross_references"])])

print(f"Generated {len(records)} guideline documents across {len(PATHWAYS)} pathways.")
print(f"Years span: {min(r['year'] for r in records)} .. {max(r['year'] for r in records)}")
print(f"JSONL : {jsonl_path}")
print(f"MD dir: {GDIR}")
print(f"CSV   : {csv_path}")
