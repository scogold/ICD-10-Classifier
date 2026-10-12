"""Generate synthetic MIMIC-IV files (discharge notes + diagnoses) for tests.

Everything here is made up from public ICD code lists and public schema docs.
No real MIMIC data is read. Output layout mirrors MIMIC so paths.mimic_dir can point at it:

    <out>/note/discharge.csv
    <out>/hosp/diagnoses_icd.csv
    <out>/hosp/d_icd_diagnoses.csv
    <out>/README.md

Usage:
    python scripts/make_synthetic_mimic.py [--seed 0] [--n-patients 200] [--out-dir DIR]
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_mimic"

# (icd_code, long_title, phrase used in note text). The first HEAD_SIZE codes are
# very common; the last N_SINGLETON codes are used exactly once.
ICD10_CODES = [
    ("I10", "Essential (primary) hypertension", "hypertension"),
    ("E785", "Hyperlipidemia, unspecified", "hyperlipidemia"),
    ("Z87891", "Personal history of nicotine dependence", "remote tobacco use, former smoker"),
    ("I2510", "Atherosclerotic heart disease of native coronary artery without angina pectoris", "coronary artery disease"),
    ("N179", "Acute kidney failure, unspecified", "acute kidney injury"),
    ("E119", "Type 2 diabetes mellitus without complications", "type 2 diabetes mellitus"),
    ("I4891", "Unspecified atrial fibrillation", "atrial fibrillation"),
    ("J189", "Pneumonia, unspecified organism", "pneumonia"),
    ("K219", "Gastro-esophageal reflux disease without esophagitis", "gastroesophageal reflux disease"),
    ("E039", "Hypothyroidism, unspecified", "hypothyroidism"),
    ("I509", "Heart failure, unspecified", "congestive heart failure"),
    ("F329", "Major depressive disorder, single episode, unspecified", "depression"),
    ("I5023", "Acute on chronic systolic (congestive) heart failure", "acute on chronic systolic heart failure"),
    ("N390", "Urinary tract infection, site not specified", "urinary tract infection"),
    ("E871", "Hypo-osmolality and hyponatremia", "hyponatremia"),
    ("D649", "Anemia, unspecified", "anemia"),
    ("A419", "Sepsis, unspecified organism", "sepsis"),
    ("J449", "Chronic obstructive pulmonary disease, unspecified", "chronic obstructive pulmonary disease"),
    ("F419", "Anxiety disorder, unspecified", "anxiety"),
    ("G4733", "Obstructive sleep apnea (adult) (pediatric)", "obstructive sleep apnea"),
    ("E669", "Obesity, unspecified", "obesity"),
    ("Z794", "Long term (current) use of insulin", "chronic insulin therapy"),
    ("Z7901", "Long term (current) use of anticoagulants", "on chronic anticoagulation"),
    ("Z7982", "Long term (current) use of aspirin", "daily aspirin use"),
    ("Z79899", "Other long term (current) drug therapy", "long-term medication therapy"),
    ("J9601", "Acute respiratory failure with hypoxia", "acute hypoxic respiratory failure"),
    ("N1830", "Chronic kidney disease, stage 3 unspecified", "chronic kidney disease stage 3"),
    ("I129", "Hypertensive chronic kidney disease with stage 1 through stage 4 chronic kidney disease, or unspecified chronic kidney disease", "hypertensive chronic kidney disease"),
    ("E1122", "Type 2 diabetes mellitus with diabetic chronic kidney disease", "type 2 diabetes with diabetic nephropathy"),
    ("E1165", "Type 2 diabetes mellitus with hyperglycemia", "diabetes with hyperglycemia"),
    ("I480", "Paroxysmal atrial fibrillation", "paroxysmal atrial fibrillation"),
    ("I4820", "Chronic atrial fibrillation, unspecified", "chronic atrial fibrillation"),
    ("I214", "Non-ST elevation (NSTEMI) myocardial infarction", "NSTEMI"),
    ("I219", "Acute myocardial infarction, unspecified", "acute myocardial infarction"),
    ("I5022", "Chronic systolic (congestive) heart failure", "chronic systolic heart failure"),
    ("I5032", "Chronic diastolic (congestive) heart failure", "chronic diastolic heart failure"),
    ("I5033", "Acute on chronic diastolic (congestive) heart failure", "acute on chronic diastolic heart failure"),
    ("I5043", "Acute on chronic combined systolic and diastolic (congestive) heart failure", "acute on chronic combined systolic and diastolic heart failure"),
    ("I110", "Hypertensive heart disease with heart failure", "hypertensive heart disease with heart failure"),
    ("I130", "Hypertensive heart and chronic kidney disease with heart failure and stage 1 through stage 4 chronic kidney disease, or unspecified chronic kidney disease", "hypertensive heart and kidney disease with heart failure"),
    ("I255", "Ischemic cardiomyopathy", "ischemic cardiomyopathy"),
    ("I420", "Dilated cardiomyopathy", "dilated cardiomyopathy"),
    ("I350", "Nonrheumatic aortic (valve) stenosis", "severe aortic stenosis"),
    ("I340", "Nonrheumatic mitral (valve) insufficiency", "mitral regurgitation"),
    ("I4720", "Ventricular tachycardia, unspecified", "ventricular tachycardia"),
    ("I442", "Atrioventricular block, complete", "complete heart block"),
    ("I739", "Peripheral vascular disease, unspecified", "peripheral vascular disease"),
    ("I2720", "Pulmonary hypertension, unspecified", "pulmonary hypertension"),
    ("I951", "Orthostatic hypotension", "orthostatic hypotension"),
    ("I959", "Hypotension, unspecified", "hypotension"),
    ("I161", "Hypertensive emergency", "hypertensive emergency"),
    ("I2699", "Other pulmonary embolism without acute cor pulmonale", "pulmonary embolism"),
    ("I82401", "Acute embolism and thrombosis of unspecified deep veins of right lower extremity", "acute deep vein thrombosis of the right leg"),
    ("I82402", "Acute embolism and thrombosis of unspecified deep veins of left lower extremity", "acute deep vein thrombosis of the left leg"),
    ("I639", "Cerebral infarction, unspecified", "acute ischemic stroke"),
    ("I619", "Nontraumatic intracerebral hemorrhage, unspecified", "intracerebral hemorrhage"),
    ("G459", "Transient cerebral ischemic attack, unspecified", "transient ischemic attack"),
    ("Z951", "Presence of aortocoronary bypass graft", "prior coronary artery bypass grafting"),
    ("Z955", "Presence of coronary angioplasty implant and graft", "prior coronary stent placement"),
    ("Z8673", "Personal history of transient ischemic attack (TIA), and cerebral infarction without residual deficits", "prior stroke without residual deficit"),
    ("Z86718", "Personal history of other venous thrombosis and embolism", "history of venous thromboembolism"),
    ("Z853", "Personal history of malignant neoplasm of breast", "history of breast cancer"),
    ("Z66", "Do not resuscitate", "DNR status"),
    ("Z515", "Encounter for palliative care", "palliative care involvement"),
    ("Z7984", "Long term (current) use of oral hypoglycemic drugs", "chronic oral hypoglycemic therapy"),
    ("Z7902", "Long term (current) use of antithrombotics/antiplatelets", "chronic antiplatelet therapy"),
    ("Z7952", "Long term (current) use of systemic steroids", "chronic steroid therapy"),
    ("Z23", "Encounter for immunization", "vaccination given this admission"),
    ("Z6841", "Body Mass Index (BMI) 40.0-44.9, adult", "BMI in the low 40s"),
    ("Z8249", "Family history of ischemic heart disease and other diseases of the circulatory system", "family history of early heart disease"),
    ("Z992", "Dependence on renal dialysis", "hemodialysis dependence"),
    ("J441", "Chronic obstructive pulmonary disease with (acute) exacerbation", "COPD exacerbation"),
    ("J440", "Chronic obstructive pulmonary disease with (acute) lower respiratory infection", "COPD with lower respiratory infection"),
    ("J45909", "Unspecified asthma, uncomplicated", "asthma"),
    ("J9622", "Acute and chronic respiratory failure with hypercapnia", "acute on chronic hypercapnic respiratory failure"),
    ("J690", "Pneumonitis due to inhalation of food and vomit", "aspiration pneumonitis"),
    ("J90", "Pleural effusion, not elsewhere classified", "pleural effusion"),
    ("J9811", "Atelectasis", "atelectasis"),
    ("J810", "Acute pulmonary edema", "acute pulmonary edema"),
    ("J22", "Unspecified acute lower respiratory infection", "lower respiratory tract infection"),
    ("J069", "Acute upper respiratory infection, unspecified", "upper respiratory infection"),
    ("U071", "COVID-19", "COVID-19 infection"),
    ("E1110", "Type 2 diabetes mellitus with ketoacidosis without coma", "diabetic ketoacidosis in type 2 diabetes"),
    ("E1010", "Type 1 diabetes mellitus with ketoacidosis without coma", "diabetic ketoacidosis in type 1 diabetes"),
    ("E1142", "Type 2 diabetes mellitus with diabetic polyneuropathy", "diabetic peripheral neuropathy"),
    ("E10649", "Type 1 diabetes mellitus with hypoglycemia without coma", "hypoglycemia in type 1 diabetes"),
    ("E860", "Dehydration", "dehydration"),
    ("E875", "Hyperkalemia", "hyperkalemia"),
    ("E876", "Hypokalemia", "hypokalemia"),
    ("E870", "Hyperosmolality and hypernatremia", "hypernatremia"),
    ("E8351", "Hypocalcemia", "hypocalcemia"),
    ("E6601", "Morbid (severe) obesity due to excess calories", "morbid obesity"),
    ("E559", "Vitamin D deficiency, unspecified", "vitamin D deficiency"),
    ("D509", "Iron deficiency anemia, unspecified", "iron deficiency anemia"),
    ("D62", "Acute posthemorrhagic anemia", "acute blood loss anemia"),
    ("D696", "Thrombocytopenia, unspecified", "thrombocytopenia"),
    ("D630", "Anemia in neoplastic disease", "anemia of malignancy"),
    ("C3490", "Malignant neoplasm of unspecified part of unspecified bronchus or lung", "lung cancer"),
    ("C50919", "Malignant neoplasm of unspecified site of unspecified female breast", "breast cancer"),
    ("C61", "Malignant neoplasm of prostate", "prostate cancer"),
    ("C189", "Malignant neoplasm of colon, unspecified", "colon cancer"),
    ("C259", "Malignant neoplasm of pancreas, unspecified", "pancreatic cancer"),
    ("C787", "Secondary malignant neoplasm of liver and intrahepatic bile duct", "liver metastases"),
    ("C7951", "Secondary malignant neoplasm of bone", "bone metastases"),
    ("K5732", "Diverticulitis of large intestine without perforation or abscess without bleeding", "acute sigmoid diverticulitis"),
    ("K922", "Gastrointestinal hemorrhage, unspecified", "gastrointestinal bleeding"),
    ("K921", "Melena", "melena"),
    ("K920", "Hematemesis", "hematemesis"),
    ("K8000", "Calculus of gallbladder with acute cholecystitis without obstruction", "acute calculous cholecystitis"),
    ("K3580", "Unspecified acute appendicitis", "acute appendicitis"),
    ("K5650", "Intestinal adhesions [bands], unspecified as to partial versus complete obstruction", "adhesive small bowel obstruction"),
    ("K560", "Paralytic ileus", "ileus"),
    ("K7031", "Alcoholic cirrhosis of liver with ascites", "alcoholic cirrhosis with ascites"),
    ("K7469", "Other cirrhosis of liver", "cirrhosis"),
    ("K8590", "Acute pancreatitis without necrosis or infection, unspecified", "acute pancreatitis"),
    ("K5900", "Constipation, unspecified", "constipation"),
    ("K766", "Portal hypertension", "portal hypertension"),
    ("L03115", "Cellulitis of right lower limb", "cellulitis of the right leg"),
    ("L03116", "Cellulitis of left lower limb", "cellulitis of the left leg"),
    ("N170", "Acute kidney failure with tubular necrosis", "acute tubular necrosis"),
    ("N184", "Chronic kidney disease, stage 4 (severe)", "chronic kidney disease stage 4"),
    ("N186", "End stage renal disease", "end stage renal disease"),
    ("N401", "Benign prostatic hyperplasia with lower urinary tract symptoms", "benign prostatic hyperplasia with urinary symptoms"),
    ("N200", "Calculus of kidney", "kidney stone"),
    ("N132", "Hydronephrosis with renal and ureteral calculous obstruction", "obstructing ureteral stone with hydronephrosis"),
    ("G309", "Alzheimer's disease, unspecified", "Alzheimer's disease"),
    ("F0390", "Unspecified dementia, unspecified severity, without behavioral disturbance, psychotic disturbance, mood disturbance, and anxiety", "dementia"),
    ("G40909", "Epilepsy, unspecified, not intractable, without status epilepticus", "epilepsy"),
    ("R569", "Unspecified convulsions", "seizure"),
    ("G894", "Chronic pain syndrome", "chronic pain syndrome"),
    ("G629", "Polyneuropathy, unspecified", "polyneuropathy"),
    ("R55", "Syncope and collapse", "syncope"),
    ("R079", "Chest pain, unspecified", "chest pain"),
    ("R0602", "Shortness of breath", "shortness of breath"),
    ("R112", "Nausea with vomiting, unspecified", "nausea and vomiting"),
    ("R1084", "Generalized abdominal pain", "diffuse abdominal pain"),
    ("R4182", "Altered mental status, unspecified", "altered mental status"),
    ("R531", "Weakness", "generalized weakness"),
    ("R001", "Bradycardia, unspecified", "bradycardia"),
    ("R509", "Fever, unspecified", "fever"),
    ("R6520", "Severe sepsis without septic shock", "severe sepsis"),
    ("R6521", "Severe sepsis with septic shock", "septic shock requiring vasopressors"),
    ("R570", "Cardiogenic shock", "cardiogenic shock"),
    ("F1020", "Alcohol dependence, uncomplicated", "alcohol use disorder"),
    ("F10239", "Alcohol dependence with withdrawal, unspecified", "alcohol withdrawal"),
    ("F17210", "Nicotine dependence, cigarettes, uncomplicated", "current cigarette smoking"),
    ("F1120", "Opioid dependence, uncomplicated", "opioid use disorder"),
    ("F4310", "Post-traumatic stress disorder, unspecified", "post-traumatic stress disorder"),
    ("F319", "Bipolar disorder, unspecified", "bipolar disorder"),
    ("F209", "Schizophrenia, unspecified", "schizophrenia"),
    ("F05", "Delirium due to known physiological condition", "delirium"),
    ("F332", "Major depressive disorder, recurrent severe without psychotic features", "recurrent severe major depression"),
    ("S72001A", "Fracture of unspecified part of neck of right femur, initial encounter for closed fracture", "right hip fracture after a fall"),
    ("S0990XA", "Unspecified injury of head, initial encounter", "head injury"),
    ("W19XXXA", "Unspecified fall, initial encounter", "mechanical fall at home"),
    ("M810", "Age-related osteoporosis without current pathological fracture", "osteoporosis"),
    ("M5450", "Low back pain, unspecified", "low back pain"),
    ("M1711", "Unilateral primary osteoarthritis, right knee", "right knee osteoarthritis"),
    ("M069", "Rheumatoid arthritis, unspecified", "rheumatoid arthritis"),
    ("M109", "Gout, unspecified", "gout"),
    ("M6282", "Rhabdomyolysis", "rhabdomyolysis"),
    ("T814XXA", "Infection following a procedure, initial encounter", "postprocedural infection"),
    ("T7840XA", "Allergy, unspecified, initial encounter", "allergic reaction"),
    # Singletons: each of these is attached to exactly one admission.
    ("A0472", "Enterocolitis due to Clostridium difficile, not specified as recurrent", "Clostridium difficile colitis"),
    ("A4102", "Sepsis due to Methicillin resistant Staphylococcus aureus", "MRSA bacteremia"),
    ("B20", "Human immunodeficiency virus [HIV] disease", "HIV infection"),
    ("B182", "Chronic viral hepatitis C", "chronic hepatitis C"),
    ("G9341", "Metabolic encephalopathy", "metabolic encephalopathy"),
    ("L89154", "Pressure ulcer of sacral region, stage 4", "stage 4 sacral pressure ulcer"),
    ("K432", "Incisional hernia without obstruction or gangrene", "incisional hernia"),
    ("K652", "Spontaneous bacterial peritonitis", "spontaneous bacterial peritonitis"),
    ("I8501", "Esophageal varices with bleeding", "bleeding esophageal varices"),
    ("M726", "Necrotizing fasciitis", "necrotizing fasciitis"),
    ("E8342", "Hypomagnesemia", "hypomagnesemia"),
    ("Q211", "Atrial septal defect", "atrial septal defect"),
]
HEAD_SIZE = 8
N_SINGLETON = 12

ICD9_CODES = [
    ("4019", "Unspecified essential hypertension", "hypertension"),
    ("25000", "Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled", "type 2 diabetes mellitus"),
    ("4280", "Congestive heart failure, unspecified", "congestive heart failure"),
    ("42731", "Atrial fibrillation", "atrial fibrillation"),
    ("486", "Pneumonia, organism unspecified", "pneumonia"),
    ("5849", "Acute kidney failure, unspecified", "acute kidney injury"),
    ("41401", "Coronary atherosclerosis of native coronary artery", "coronary artery disease"),
    ("2724", "Other and unspecified hyperlipidemia", "hyperlipidemia"),
    ("53081", "Esophageal reflux", "acid reflux"),
    ("311", "Depressive disorder, not elsewhere classified", "depression"),
    ("5990", "Urinary tract infection, site not specified", "urinary tract infection"),
    ("49390", "Asthma, unspecified type, unspecified", "asthma"),
    ("99592", "Severe sepsis", "severe sepsis"),
    ("2859", "Anemia, unspecified", "anemia"),
    ("V4582", "Percutaneous transluminal coronary angioplasty status", "prior coronary angioplasty"),
    ("3051", "Tobacco use disorder", "tobacco use"),
    ("5859", "Chronic kidney disease, unspecified", "chronic kidney disease"),
    ("51881", "Acute respiratory failure", "acute respiratory failure"),
    ("78650", "Chest pain, unspecified", "chest pain"),
]

# Share of admissions per edge case (each has a floor of 3 admissions).
EDGE_FRACTIONS = {"icd9_only": 0.06, "mixed": 0.03, "no_diagnoses": 0.03, "no_note": 0.03, "long_note": 0.02}
LONG_NOTE_MIN_CHARS = 40_000

FILLER = [
    "The patient was seen and examined by the primary team, and the plan was discussed with the family at the bedside.",
    "Vital signs remained stable overnight, with no events reported by nursing, and the patient tolerated a regular diet.",
    "Laboratory studies were reviewed daily, including a basic metabolic panel, complete blood count, and magnesium level.",
    "Physical therapy evaluated the patient, and recommended a short course of rehabilitation, \"as tolerated\" per protocol.",
    "The patient stated, \"I feel much better than when I came in,\" and asked about going home.",
    "Imaging was obtained, and the radiology read was reviewed with the consulting service, who agreed with the plan.",
    "Medications were reconciled on admission, and several home doses were adjusted, held, or discontinued, as noted below.",
    "Case management arranged outpatient follow-up, and transportation was confirmed with the family prior to discharge.",
    "An electrocardiogram showed no acute changes compared with the prior tracing, and telemetry was unremarkable.",
    "The patient was counseled on diet, activity, and medication adherence, and verbalized understanding, with teach-back.",
]
MEDICATIONS = [
    "Aspirin 81 mg PO DAILY", "Atorvastatin 40 mg PO QPM", "Metoprolol Succinate XL 50 mg PO DAILY",
    "Lisinopril 10 mg PO DAILY", "Furosemide 40 mg PO DAILY", "Omeprazole 20 mg PO DAILY",
    "Levothyroxine Sodium 75 mcg PO DAILY", "Acetaminophen 650 mg PO Q6H:PRN pain",
    "Apixaban 5 mg PO BID", "Insulin Glargine 20 units SC QHS", "Sertraline 50 mg PO DAILY",
    "Tamsulosin 0.4 mg PO QHS", "Senna 8.6 mg PO BID:PRN constipation",
]
COMPLAINTS = ["shortness of breath", "chest pain", "abdominal pain", "weakness", "fever and chills",
              "altered mental status", "fall at home", "nausea and vomiting", "leg swelling", "cough"]


def pick(rng: np.random.Generator, items: list, k: int) -> list:
    """Pick k distinct items in random order."""
    idx = rng.choice(len(items), size=min(k, len(items)), replace=False)
    return [items[i] for i in idx]


def code_weights(n_codes: int) -> np.ndarray:
    """Long-tail sampling weights: a few very common codes, then a Zipf-like tail."""
    ranks = np.arange(1, n_codes + 1)
    weights = 1.0 / ranks
    weights[:HEAD_SIZE] *= 8
    return weights / weights.sum()


def assign_edge_cases(rng: np.random.Generator, n_admissions: int) -> list[str]:
    """Return one label per admission: an edge case name or 'normal'."""
    kinds = ["normal"] * n_admissions
    order = rng.permutation(n_admissions)
    pos = 0
    for kind, fraction in EDGE_FRACTIONS.items():
        count = max(3, round(fraction * n_admissions))
        for i in order[pos:pos + count]:
            kinds[i] = kind
        pos += count
    return kinds


def make_admission_codes(rng: np.random.Generator, kind: str, regular: list, weights: np.ndarray) -> list[tuple]:
    """Return [(code, version, title, phrase)] in seq_num order."""
    if kind == "no_diagnoses":
        return []
    n_codes = int(rng.integers(3, 16))
    icd9 = [(c, 9, t, p) for c, t, p in pick(rng, ICD9_CODES, n_codes if kind == "icd9_only" else int(rng.integers(1, 4)))]
    if kind == "icd9_only":
        return icd9
    idx = rng.choice(len(regular), size=n_codes, replace=False, p=weights)
    icd10 = [(regular[i][0], 10, regular[i][1], regular[i][2]) for i in idx]
    return icd10 + icd9 if kind == "mixed" else icd10


def make_note_text(rng: np.random.Generator, codes: list[tuple], noise_phrases: list[str], long_note: bool) -> str:
    """Build a discharge-summary-like note that mentions most of the coded conditions."""
    # ~10% of coded conditions are left out of the text (code not clearly mentioned).
    mentioned = [c[3] for c in codes if rng.random() > 0.1]
    # Conditions that are mentioned but not coded.
    mentioned_uncoded = noise_phrases[: int(rng.integers(0, 3))]
    all_mentions = mentioned + mentioned_uncoded
    rng.shuffle(all_mentions)
    principal = codes[0][3] if codes and codes[0][3] in mentioned else (all_mentions[0] if all_mentions else "unclear etiology")

    sex = "F" if rng.random() < 0.5 else "M"
    age = int(rng.integers(25, 95))
    complaint = COMPLAINTS[int(rng.integers(len(COMPLAINTS)))]
    pmh = all_mentions[: len(all_mentions) // 2 + 1]
    course = [f"# {m.capitalize()}: managed per the primary team, with improvement noted during the stay." for m in all_mentions]
    meds = pick(rng, MEDICATIONS, int(rng.integers(3, 9)))
    n_filler = int(rng.integers(1, 12))

    lines = [
        "Name:  ___                     Unit No:   ___", "",
        "Admission Date:  ___              Discharge Date:   ___", "",
        f"Date of Birth:  ___             Sex:   {sex}", "",
        "Service: MEDICINE", "",
        "Allergies: ", "No Known Allergies / Adverse Drug Reactions", "",
        "Attending: ___", "",
        "Chief Complaint:", complaint.capitalize(), "",
        "History of Present Illness:",
        f"___ is a {age} year old {'woman' if sex == 'F' else 'man'} with a history of {', '.join(pmh) or 'no significant history'}, "
        f"who presented with {complaint}. On arrival to the ED, vitals were notable for ___. "
        f"The patient stated, \"it has been getting worse for ___ days.\"", "",
        "Past Medical History:",
    ]
    lines += [f"{m}" for m in pmh] + ["", "Social History:", "___", "Family History:", "Non-contributory.", "",
                                       "Physical Exam:", "General: NAD, comfortable, in bed.", "Lungs: CTAB, no wheezes.", "", "Hospital Course:",
                                       f"Briefly, this is a {age} year old with {principal}, admitted for further evaluation and management."]
    lines += course
    lines += [" ".join(rng.choice(FILLER, size=n_filler)), "", "Discharge Medications:"]
    lines += [f"{i + 1}. {m}" for i, m in enumerate(meds)]
    lines += ["", "Discharge Disposition:", "Home", "", "Discharge Diagnosis:"]
    lines += [m for m in all_mentions] or ["___"]
    lines += ["", "Discharge Condition:", "Mental Status: Clear and coherent.", "", "Discharge Instructions:",
              "You were hospitalized, and treated. Please take all medications as prescribed, and keep your follow-up appointments.",
              "Return to the ED for worsening symptoms, fever, or any new concerns.", "", "Followup Instructions:", "___"]
    text = "\n".join(lines)

    if long_note:
        extra = []
        while len(text) + sum(len(e) for e in extra) < LONG_NOTE_MIN_CHARS:
            extra.append("\n\nPertinent Results:\n" + "\n".join(rng.choice(FILLER, size=8)))
        text += "".join(extra)
    return text


def make_dates(rng: np.random.Generator, n_admissions: int) -> list[tuple[datetime, datetime]]:
    """Return (charttime, storetime) per admission, increasing for one patient."""
    start = datetime(int(rng.integers(2100, 2190)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
    out = []
    for _ in range(n_admissions):
        discharge = start + timedelta(days=int(rng.integers(1, 15)))
        chart = discharge + timedelta(hours=int(rng.integers(8, 20)), minutes=int(rng.integers(60)), seconds=int(rng.integers(60)))
        store = chart + timedelta(minutes=int(rng.integers(30, 24 * 60)))
        out.append((chart, store))
        start = discharge + timedelta(days=int(rng.integers(20, 700)))
    return out


def unique_ids(rng: np.random.Generator, low: int, high: int, n: int) -> list[int]:
    return [int(x) for x in rng.choice(np.arange(low, high), size=n, replace=False)]


def generate(seed: int, n_patients: int) -> dict:
    rng = np.random.default_rng(seed)
    singletons = ICD10_CODES[-N_SINGLETON:]
    regular = ICD10_CODES[:-N_SINGLETON]
    weights = code_weights(len(regular))

    subject_ids = unique_ids(rng, 10_000_000, 20_000_000, n_patients)
    adm_counts = rng.choice([1, 2, 3, 4, 5], size=n_patients, p=[0.5, 0.25, 0.12, 0.08, 0.05])
    n_admissions = int(adm_counts.sum())
    hadm_ids = unique_ids(rng, 20_000_000, 30_000_000, n_admissions)
    kinds = assign_edge_cases(rng, n_admissions)

    # Each singleton code goes on one distinct, ordinary ICD-10 admission.
    normal_adms = [i for i, k in enumerate(kinds) if k == "normal"]
    singleton_adm = dict(zip(rng.choice(normal_adms, size=len(singletons), replace=False).tolist(), singletons))

    notes, diagnoses, admissions = [], [], []
    adm = 0
    for subject_id, n_adm in zip(subject_ids, adm_counts):
        dates = make_dates(rng, int(n_adm))
        for note_seq, (chart, store) in enumerate(dates, start=1):
            kind, hadm_id = kinds[adm], hadm_ids[adm]
            codes = make_admission_codes(rng, kind, regular, weights)
            if adm in singleton_adm:
                c, t, p = singleton_adm[adm]
                codes.insert(int(rng.integers(1, len(codes) + 1)), (c, 10, t, p))

            for seq_num, (code, version, _, _) in enumerate(codes, start=1):
                diagnoses.append((subject_id, hadm_id, seq_num, code, version))

            if kind != "no_note":
                # With no codes, the note still describes some conditions.
                text_codes = codes or [(c, 10, t, p) for c, t, p in pick(rng, regular, int(rng.integers(3, 8)))]
                coded = {c[0] for c in codes}
                noise = [p for c, _, p in pick(rng, regular, 4) if c not in coded] if rng.random() < 0.4 else []
                text = make_note_text(rng, text_codes, noise, kind == "long_note")
                notes.append((f"{subject_id}-DS-{note_seq}", subject_id, hadm_id, "DS", note_seq,
                              chart.strftime("%Y-%m-%d %H:%M:%S"), store.strftime("%Y-%m-%d %H:%M:%S"), text))
            admissions.append((subject_id, hadm_id, kind))
            adm += 1

    used = {(c, v) for _, _, _, c, v in diagnoses}
    dictionary = [(c, 10, t) for c, t, _ in ICD10_CODES] + [(c, 9, t) for c, t, _ in ICD9_CODES]
    return {
        "discharge": pd.DataFrame(notes, columns=["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"]),
        "diagnoses_icd": pd.DataFrame(diagnoses, columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]),
        "d_icd_diagnoses": pd.DataFrame(dictionary, columns=["icd_code", "icd_version", "long_title"]),
        "admissions": pd.DataFrame(admissions, columns=["subject_id", "hadm_id", "kind"]),
        "n_used_codes": len(used),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a plain UTF-8 CSV with LF line endings (write_bytes avoids Windows CRLF translation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def write_readme(path: Path, seed: int, n_patients: int, data: dict) -> None:
    kinds = data["admissions"]["kind"].value_counts()
    edge_lines = "\n".join(f"  - `{k}`: {kinds.get(k, 0)} admissions" for k in EDGE_FRACTIONS)
    path.write_text(f"""# Synthetic MIMIC-IV fixture

This data is **synthetic**. It is generated from public ICD code lists and the public
MIMIC-IV schema docs, and contains no real patient data. It is safe to print and commit.

Same layout as MIMIC-IV, so `paths.mimic_dir` can point at this directory:
`note/discharge.csv`, `hosp/diagnoses_icd.csv`, `hosp/d_icd_diagnoses.csv`.
Real MIMIC ships these as `.csv.gz`; the project loader accepts either.

Regenerate (deterministic, byte-identical for the same arguments):

    python scripts/make_synthetic_mimic.py --seed {seed} --n-patients {n_patients}

Read `icd_code` as a string (`dtype={{"icd_code": str}}`) to keep the code format.

## Contents (seed={seed}, patients={n_patients})

- Admissions: {len(data['admissions'])}; notes: {len(data['discharge'])}; diagnosis rows: {len(data['diagnoses_icd'])}
- Distinct codes used in `diagnoses_icd`: {data['n_used_codes']}
- Singleton ICD-10 codes (appear exactly once): {N_SINGLETON}
- Edge cases (mutually exclusive; the rest are ordinary ICD-10 admissions with a note):
{edge_lines}
- Edge case meanings: `icd9_only` = ICD-9 codes only (no ICD-10), with a note; `mixed` = ICD-10 plus a few
  ICD-9 codes; `no_diagnoses` = note but no `diagnoses_icd` rows; `no_note` = diagnoses but no note;
  `long_note` = note over {LONG_NOTE_MIN_CHARS} characters (more than 8192 tokens).
- Noise: about 10% of coded conditions are not mentioned in the note, and some notes mention
  conditions that are not coded.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-patients", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    data = generate(args.seed, args.n_patients)
    write_csv(data["discharge"], args.out_dir / "note" / "discharge.csv")
    write_csv(data["diagnoses_icd"], args.out_dir / "hosp" / "diagnoses_icd.csv")
    write_csv(data["d_icd_diagnoses"], args.out_dir / "hosp" / "d_icd_diagnoses.csv")
    write_readme(args.out_dir / "README.md", args.seed, args.n_patients, data)
    print(f"Wrote synthetic MIMIC files to {args.out_dir}")


if __name__ == "__main__":
    main()
