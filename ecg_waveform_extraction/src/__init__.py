"""All source code for ECG waveform extraction.

Subpackages: preprocessing, features, hsmm, segmentation, extraction, utils.
Core modules: limb_lead_processor, limb_lead_reversal, chest_lead_analyzer,
plot_segmentation, plot_qrs_c1.
Entry-point scripts: batch_*, train_*, eval_*, export_*, download_*,
process_*, plot_*, save_*, compare_*, p_wave_three_methods.

Run entry points as modules from the repo root (ECG_engineering), e.g.:
    python -m ecg_waveform_extraction.src.batch_limb_leads --n 5
Data (data/), models (models/) and tests/ live at the package root, one
level up; ALL output results go under output/ (e.g. output/rala_full/...,
output/trained, output/three_methods) — never root-level output_* dirs.
Scripts anchor these via Path(__file__).
"""
