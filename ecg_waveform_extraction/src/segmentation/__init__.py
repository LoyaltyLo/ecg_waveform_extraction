"""Waveform segmentation: HSMM state sequence → per-beat boundaries."""

from .segmenter import ECGSegmenter, SegmentResult

__all__ = ["ECGSegmenter", "SegmentResult"]
