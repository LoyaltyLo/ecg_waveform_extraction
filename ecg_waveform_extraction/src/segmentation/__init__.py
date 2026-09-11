"""Waveform segmentation: HSMM state sequence → per-beat boundaries, plus a
training-free classical DSP delineator (DSPECGSegmenter) with the same API."""

from .segmenter import ECGSegmenter, SegmentResult
from .dsp_segmenter import DSPECGSegmenter, DSPDelineationConfig

__all__ = ["ECGSegmenter", "SegmentResult", "DSPECGSegmenter",
           "DSPDelineationConfig"]
