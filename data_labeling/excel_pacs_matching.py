import os
import re
import pandas as pd
import shutil
import sys
import pydicom
from tqdm import tqdm

# ---------------- BASE FOLDER ----------------
if getattr(sys, 'frozen', False):
    BASE_FOLDER = os.path.dirname(sys.executable)
else:
    BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

print(f"Running from: {BASE_FOLDER}")

# ---------------- INPUT ----------------
# PACS root folder
PACS_ROOT = input("Enter PACS root folder: ").strip().strip('"')

# Output folder
OUTPUT_ROOT = os.path.join(BASE_FOLDER, "EXTRACTED_THYROID")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------- HELPERS ----------------
def normalize_prn(prn):
    """Keep digits only from PRN, remove spaces and special chars"""
    if pd.isna(prn):
        return ""
    return "".join(re.findall(r"\d+", str(prn)))

def normalize_date(val):
    """Return date in ddmmyyyy format without separators"""
    if isinstance(val, pd.Timestamp):
        return val.strftime("%d%m%Y")
    return str(val).replace("-", "").replace("/", "").replace(" ", "").strip()

def dicom_matches(dcm_path, prn_set, date_str):
    """Check if DICOM file matches PRN, date, and thyroid study"""
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        dcm_prn = normalize_prn(dcm.get("PatientID", ""))
        study_desc = str(dcm.get("StudyDescription", "")).lower()
        study_date = normalize_date(dcm.get("StudyDate", ""))

        if dcm_prn in prn_set and study_date == date_str and "thyroid" in study_desc:
            return True
    except:
        return False
    return False

# ---------------- READ EXCEL ----------------
excel_files = [f for f in os.listdir(BASE_FOLDER) if f.endswith(".xlsx")]
if not excel_files:
    print("❌ No Excel file found")
    sys.exit(1)

df = pd.read_excel(excel_files[0])
df.columns = df.columns.str.strip()
print(f"Excel rows: {len(df)}")

# ---------------- INDEX EXCEL ----------------
# We allow duplicates, so group by (PRN, Date)
excel_index = {}
for _, row in df.iterrows():
    prn_norm = normalize_prn(row["P-ID"])
    scan_date = normalize_date(row["Mgt date"])
    key = (prn_norm, scan_date)
    excel_index.setdefault(key, []).append(row)

print(f"Unique PRN-Date combinations: {len(excel_index)}")

# ---------------- WALK PACS ----------------
checked = 0
copied = 0
missing_count = 0

for root, dirs, files in os.walk(PACS_ROOT):
    for file in tqdm(files, desc="Processing DICOMs"):
        if not file.lower().endswith(".dcm"):
            continue

        dicom_path = os.path.join(root, file)

        # Try matching with each PRN-Date combination
        for (prn_norm, scan_date), rows in excel_index.items():
            if dicom_matches(dicom_path, {prn_norm}, scan_date):
                # Create output folder preserving tree structure
                rel_path = os.path.relpath(dicom_path, PACS_ROOT)
                dest_path = os.path.join(OUTPUT_ROOT, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(dicom_path, dest_path)
                copied += 1
                break  # stop checking other PRNs for this file

        checked += 1

# ---------------- SUMMARY ----------------
print("\n✅ DONE")
print(f"DICOMs checked : {checked}")
print(f"DICOMs copied  : {copied}")
print(f"Output folder  : {OUTPUT_ROOT}")
