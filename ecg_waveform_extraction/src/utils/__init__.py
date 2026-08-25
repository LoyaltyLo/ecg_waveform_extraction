"""Utility functions for visualization, data loading, and dashboard generation."""

from .vis import (
    plot_segmentation,
    plot_p_wave_detail,
    plot_duration_distributions,
    plot_transition_matrix,
    plot_training_progress,
)
from .data_loader import (
    load_csv_ecg,
    load_wfdb_record,
    generate_synthetic_ecg,
)
from .aecg_parser import parse_aecg, get_default_leads
from .dashboard import build_dashboard

__all__ = [
    "plot_segmentation",
    "plot_p_wave_detail",
    "plot_duration_distributions",
    "plot_transition_matrix",
    "plot_training_progress",
    "load_csv_ecg",
    "load_wfdb_record",
    "generate_synthetic_ecg",
    "parse_aecg",
    "get_default_leads",
    "build_dashboard",
]
