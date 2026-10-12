# Synthetic MIMIC-IV fixture

This data is **synthetic**. It is generated from public ICD code lists and the public
MIMIC-IV schema docs, and contains no real patient data. It is safe to print and commit.

Same layout as MIMIC-IV, so `paths.mimic_dir` can point at this directory:
`note/discharge.csv`, `hosp/diagnoses_icd.csv`, `hosp/d_icd_diagnoses.csv`.
Real MIMIC ships these as `.csv.gz`; the project loader accepts either.

Regenerate (deterministic, byte-identical for the same arguments):

    python scripts/make_synthetic_mimic.py --seed 0 --n-patients 200

Read `icd_code` as a string (`dtype={"icd_code": str}`) to keep the code format.

## Contents (seed=0, patients=200)

- Admissions: 395; notes: 383; diagnosis rows: 3539
- Distinct codes used in `diagnoses_icd`: 193
- Singleton ICD-10 codes (appear exactly once): 12
- Edge cases (mutually exclusive; the rest are ordinary ICD-10 admissions with a note):
  - `icd9_only`: 24 admissions
  - `mixed`: 12 admissions
  - `no_diagnoses`: 12 admissions
  - `no_note`: 12 admissions
  - `long_note`: 8 admissions
- Edge case meanings: `icd9_only` = ICD-9 codes only (no ICD-10), with a note; `mixed` = ICD-10 plus a few
  ICD-9 codes; `no_diagnoses` = note but no `diagnoses_icd` rows; `no_note` = diagnoses but no note;
  `long_note` = note over 40000 characters (more than 8192 tokens).
- Noise: about 10% of coded conditions are not mentioned in the note, and some notes mention
  conditions that are not coded.
