import os
import pandas as pd
import pydicom
from tqdm import tqdm
import shutil

# ======================
# CONFIG
# ======================
ROOT_DIR = "D:/"
OUTPUT_DIR = "thyroid_dataset"
LOG_FILE = "processed_files.txt"
EXCEL_FILE = "patient_ids.xlsx"
OUTPUT_EXCEL = "saved_files.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# LOAD PATIENT IDs
# ======================
df = pd.read_excel(EXCEL_FILE)
patient_ids = set(df['PatientID'].astype(str))

# ======================
# LOAD PROGRESS LOG
# ======================
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        processed = set(f.read().splitlines())
else:
    processed = set()

# ======================
# STORAGE FOR EXCEL
# ======================
saved_records = []

keywords = {"thyroid", "uptake", "scan"}

# ======================
# MAIN LOOP
# ======================
with open(LOG_FILE, "a") as log:

    for root, dirs, files in os.walk(ROOT_DIR):

        for file in tqdm(files, desc="Scanning DICOMs", leave=False):

            if not file.lower().endswith(".dcm"):
                continue

            file_path = os.path.join(root, file)

            if file_path in processed:
                continue

            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                patient_id = str(ds.get("PatientID", ""))
                modality = str(ds.get("Modality", ""))
                study_desc = str(ds.get("StudyDescription", "")).lower()
                series_desc = str(ds.get("SeriesDescription", "")).lower()

                # FILTER CONDITION
                if (
                    patient_id in patient_ids and
                    modality == "NM" and
                    (
                        any(k in study_desc for k in keywords) or
                        any(k in series_desc for k in keywords)
                    )
                ):
                    unique_name = f"{patient_id}_{file}"
                    dest_path = os.path.join(OUTPUT_DIR, unique_name)

                    shutil.copy2(file_path, dest_path)

                    # SAVE RECORD FOR EXCEL
                    saved_records.append({
                        "PatientID": patient_id,
                        "FileName": unique_name,
                        "OriginalPath": file_path,
                        "Modality": modality,
                        "StudyDescription": study_desc,
                        "SeriesDescription": series_desc
                    })

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

            # LOG PROGRESS
            log.write(file_path + "\n")
            log.flush()
            processed.add(file_path)

# ======================
# SAVE EXCEL FILE
# ======================
if saved_records:
    df_out = pd.DataFrame(saved_records)
    df_out.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nExcel saved: {OUTPUT_EXCEL}")
else:
    print("\nNo matching files found.")