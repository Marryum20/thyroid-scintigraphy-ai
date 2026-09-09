import os
import numpy as np
import pydicom
import cv2
from tqdm import tqdm

# ==============================
# SETTINGS
# ==============================Labeled Data/Multinodular goiter/EXTRACTED_THYROID
input_root = r"C:\Users\PMLS\Documents\Thyroid\All Data\relable\Normal\EXTRACTED_THYROID"
output_root = r"C:\Users\PMLS\Documents\Thyroid\All Data\relable\Normal\png"
image_size = 320
save_format = ".png"   # ".png" or ".jpg"
projection_method = "max"  # "max" for MIP, "mean" for average projection

# ==============================
# COUNTERS
# ==============================
total_files = 0
processed_count = 0
failed_count = 0

# ==============================
# NORMALIZATION FUNCTION
# ==============================
def normalize_image(img):
    """Normalize a 2D image to 0-255 as uint8."""
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img = (img * 255.0).astype(np.uint8)
    return img

# ==============================
# PROCESS FUNCTION
# ==============================
def process_dicom(dicom_path, save_path):
    global processed_count, failed_count
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array

        # Apply slope/intercept if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        # Handle multi-frame or SPECT images
        if len(img.shape) == 3:
            if projection_method == "max":
                img = np.max(img, axis=0)  # Maximum intensity projection
            elif projection_method == "mean":
                img = np.mean(img, axis=0)  # Average projection
            else:
                # Fallback: take middle slice
                img = img[img.shape[0] // 2]

        # Ensure 2D image after projection
        if len(img.shape) != 2:
            print(f"Skipping non-2D DICOM after projection: {dicom_path}, shape={img.shape}")
            failed_count += 1
            return

        # Normalize image
        img = normalize_image(img)

        # Resize for CNN
        img = cv2.resize(img, (image_size, image_size))

        # Convert grayscale to 3-channel RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Save image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        processed_count += 1

    except Exception as e:
        failed_count += 1
        print(f"❌ Error processing {dicom_path}: {e}")

# ==============================
# GET ALL DICOM FILES
# ==============================
dicom_files = []

for root, dirs, files in os.walk(input_root):
    for file in files:
        dicom_path = os.path.join(root, file)
        try:
            # Fast check to include only DICOM files
            pydicom.dcmread(dicom_path, stop_before_pixels=True)
            dicom_files.append(dicom_path)
        except:
            continue

total_files = len(dicom_files)
print(f"\n📊 Total DICOM files found: {total_files}\n")

# ==============================
# PROCESS ALL FILES
# ==============================
with tqdm(total=total_files, desc="Processing DICOMs") as pbar:
    for dicom_path in dicom_files:
        relative_path = os.path.relpath(dicom_path, input_root)
        new_relative_path = os.path.splitext(relative_path)[0] + save_format
        save_path = os.path.join(output_root, new_relative_path)

        process_dicom(dicom_path, save_path)
        pbar.update(1)

# ==============================
# FINAL SUMMARY
# ==============================
print("\n==============================")
print("✅ PROCESS COMPLETED")
print("==============================")
print(f"Total files found      : {total_files}")
print(f"Successfully processed : {processed_count}")
print(f"Failed                 : {failed_count}")
print("==============================")
