import os
import shutil
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import generate_uid
from openpyxl import Workbook

# =============================
# USER INPUT
# =============================
input_root = input("Enter INPUT DICOM folder path: ").strip()
if not os.path.isdir(input_root):
    raise ValueError("Input path does not exist.")

output_root = input("Enter OUTPUT folder path: ").strip()
os.makedirs(output_root, exist_ok=True)

# =============================
# EXCEL SETUP
# =============================
excel_path = os.path.join(output_root, "Study_Index.xlsx")

fields_to_extract = [
    "PatientID",
    "StudyDate",
    "StudyDescription",
    "AccessionNumber",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality"
]

wb = Workbook()
ws = wb.active
ws.title = "DICOM_Index"
ws.append(["FilePath"] + fields_to_extract)

# =============================
# MAIN PROCESS
# =============================
processed = 0
dicom_files = 0

for root, dirs, files in os.walk(input_root):
    for file in files:
        input_path = os.path.join(root, file)

        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, file)

        try:
            ds = pydicom.dcmread(input_path)

            dicom_files += 1

            # -------- Extract original info for Excel
            row = [input_path]
            for tag in fields_to_extract:
                row.append(str(ds.get(tag, "")))

            # -------- Remove private tags
            ds.remove_private_tags()

            # -------- Remove PHI
            if "PatientName" in ds:
                del ds.PatientName
            ds.PatientBirthDate = ""
            ds.PatientSex = ""
            ds.InstitutionName = ""
            ds.ReferringPhysicianName = ""

            # -------- Update SOPInstanceUID (safe)
            ds.SOPInstanceUID = generate_uid()
            # Keep StudyInstanceUID and SeriesInstanceUID unchanged

            # -------- Save the anonymized file
            ds.save_as(output_path, write_like_original=False)

            # -------- Add row to Excel
            ws.append(row)

            processed += 1

        except InvalidDicomError:
            # Copy non-DICOM files as-is
            shutil.copy2(input_path, output_path)

# =============================
# SAVE EXCEL
# =============================
wb.save(excel_path)

print("\n===================================")
print(f"Total DICOM files found: {dicom_files}")
print(f"Total DICOM files anonymized: {processed}")
print(f"Output folder: {output_root}")
print(f"Excel log saved at: {excel_path}")
print("===================================\n")
