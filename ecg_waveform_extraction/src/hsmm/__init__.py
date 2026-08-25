"""HSMM module for ECG waveform segmentation.

Contains:
- distributions: GMM observation model and Gaussian duration distribution
- hsmm_model: HSMM topology and parameter container
- hsmm_trainer: Baum-Welch EM training
- hsmm_decoder: Modified Viterbi decoding with explicit durations
- initializer: Smart GMM initialization from signal characteristics
"""

from .distributions import GaussianMixtureModel, DurationDistribution
from .hsmm_model import HSMMModel
from .hsmm_trainer import HSMMTrainer
from .hsmm_decoder import HSMMDecoder
from .initializer import smart_initialize_gmms

__all__ = [
    "GaussianMixtureModel",
    "DurationDistribution",
    "HSMMModel",
    "HSMMTrainer",
    "HSMMDecoder",
    "smart_initialize_gmms",
]
