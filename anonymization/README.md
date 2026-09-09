# DICOM Anonymization

This folder contains the scripts and documentation used for anonymizing thyroid scintigraphy DICOM files before their use in the research workflow.

## Purpose

Anonymization was performed to protect patient privacy and remove personally identifiable information from the imaging data.

## Workflow

The anonymization process included:

1. Loading the original DICOM files.
2. Identifying patient-related metadata.
3. Removing or modifying personally identifiable information.
4. Preserving the imaging data and required non-identifying metadata where appropriate.
5. Saving the anonymized DICOM files for subsequent processing.

## Data Privacy

Original and anonymized patient DICOM files are **not included in this repository**.

No patient names, patient IDs, dates of birth, clinical reports, or other identifiable information should be uploaded.

## Next Step

The anonymized DICOM files were converted into image formats and prepared for labeling and preprocessing.
