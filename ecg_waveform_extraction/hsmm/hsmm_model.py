"""HSMM model definition: topology, parameters, initialization, serialization.

Left-Right topology with 9 states capturing the canonical P-QRS-T sequence.
"""

import json
import numpy as np
from .distributions import GaussianMixtureModel, DurationDistribution


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES = 9

STATE_LABELS = ["ISO", "P", "PR", "Q", "R", "S", "ST", "T", "TP"]

STATE_TO_IDX = {label: i for i, label in enumerate(STATE_LABELS)}

# Left-right transition adjacency (no self-loops — duration handles self-transitions)
# ISO is the only state that can self-transition (for long inter-beat intervals).
ALLOWED_TRANSITIONS = [
    # (from_idx, to_idx)
    (0, 0),   # ISO -> ISO  (self-loop for diastole)
    (0, 1),   # ISO -> P
    (1, 2),   # P   -> PR
    (2, 3),   # PR  -> Q
    (3, 4),   # Q   -> R
    (4, 5),   # R   -> S
    (5, 6),   # S   -> ST
    (6, 7),   # ST  -> T
    (7, 8),   # T   -> TP
    (8, 0),   # TP  -> ISO
]

# Predecessor list per state (computed once, used in forward/Viterbi)
PREDECESSORS = {j: [i for i, tgt in ALLOWED_TRANSITIONS if tgt == j]
                for j in range(N_STATES)}

# Successor list per state (for backward pass)
SUCCESSORS = {i: [j for src, j in ALLOWED_TRANSITIONS if src == i]
              for i in range(N_STATES)}

# Global D_max cap (2 seconds worth of samples at 250 Hz)
GLOBAL_D_MAX = 500


class HSMMModel:
    """Hidden Semi-Markov Model for ECG waveform segmentation.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 9 for ECG).
    state_labels : list[str]
        Human-readable labels for states.
    n_features : int
        Dimensionality of observation vectors.
    n_gmm_components : int
        Number of Gaussian components per state's observation GMM.
    fs : float
        Sampling frequency (Hz), used for physiological priors.
    """

    def __init__(self, n_states: int = N_STATES,
                 state_labels: list[str] | None = None,
                 n_features: int = 3, n_gmm_components: int = 2,
                 fs: float = 250.0):
        self.n_states = n_states
        self.state_labels = state_labels or STATE_LABELS
        self.state_to_idx = {label: i for i, label in enumerate(self.state_labels)}
        self.n_features = n_features
        self.n_gmm_components = n_gmm_components
        self.fs = fs

        if len(self.state_labels) != self.n_states:
            raise ValueError(
                f"state_labels length ({len(self.state_labels)}) != n_states ({n_states})"
            )

        # ---- Parameters ----
        self.pi = np.ones(n_states) / n_states              # initial probs
        self.A = np.zeros((n_states, n_states))              # transition matrix

        self.obs_dists: list[GaussianMixtureModel] = [
            GaussianMixtureModel(n_components=n_gmm_components, n_features=n_features)
            for _ in range(n_states)
        ]

        self.dur_dists: list[DurationDistribution] = [
            DurationDistribution() for _ in range(n_states)
        ]

        self.D_max = np.full(n_states, GLOBAL_D_MAX, dtype=int)

    # ------------------------------------------------------------------
    # Topology setup
    # ------------------------------------------------------------------
    def set_left_right_topology(self):
        """Configure the canonical left-right transition matrix for ECG.

        After this call, A[i,j] > 0 only for physiologically allowed transitions.
        Probabilities are initialized uniformly over outgoing edges.
        """
        self.A = np.zeros((self.n_states, self.n_states))

        for src, tgt in ALLOWED_TRANSITIONS:
            self.A[src, tgt] = 1.0

        # Normalize rows (each row sums to 1)
        row_sums = self.A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.A = self.A / row_sums

    @property
    def predecessors(self) -> dict[int, list[int]]:
        """Return predecessor mapping computed from the current A matrix."""
        preds = {j: [] for j in range(self.n_states)}
        for i in range(self.n_states):
            for j in range(self.n_states):
                if self.A[i, j] > 0:
                    preds[j].append(i)
        return preds

    @property
    def successors(self) -> dict[int, list[int]]:
        """Return successor mapping computed from the current A matrix."""
        succs = {i: [] for i in range(self.n_states)}
        for i in range(self.n_states):
            for j in range(self.n_states):
                if self.A[i, j] > 0:
                    succs[i].append(j)
        return succs

    # ------------------------------------------------------------------
    # Initialization with physiological priors
    # ------------------------------------------------------------------
    def initialize_with_priors(self):
        """Initialize duration distributions with ECG physiological priors."""
        self.set_left_right_topology()
        self.pi = np.ones(self.n_states) / self.n_states

        for i, label in enumerate(self.state_labels):
            self.dur_dists[i].set_physiological_prior(label, self.fs)

        self._compute_D_max()

    def _compute_D_max(self):
        """Set D_max per state as mu + 4*sigma, clamped to GLOBAL_D_MAX."""
        for i in range(self.n_states):
            dd = self.dur_dists[i]
            d_max = int(np.round(dd.mu + 4.0 * dd.sigma))
            d_max = min(d_max, GLOBAL_D_MAX)
            d_max = max(d_max, dd.d_min * 2)  # at least 2x d_min
            self.D_max[i] = d_max

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """Check model validity. Returns list of issues (empty = valid)."""
        issues = []

        # Check A is row-stochastic
        row_sums = self.A.sum(axis=1)
        for i, s in enumerate(row_sums):
            if abs(s - 1.0) > 0.01 and s > 0:
                issues.append(f"Row {i} ({self.state_labels[i]}) of A sums to {s:.4f}")

        # Check pi sums to 1
        if abs(self.pi.sum() - 1.0) > 0.01:
            issues.append(f"pi sums to {self.pi.sum():.4f}")

        # Check all GMMs are fitted
        for i, gmm in enumerate(self.obs_dists):
            if not gmm._fitted:
                issues.append(f"GMM for state {i} ({self.state_labels[i]}) not fitted")

        # Check duration distributions have valid parameters
        for i, dd in enumerate(self.dur_dists):
            if dd.mu <= 0 or dd.sigma <= 0:
                issues.append(
                    f"Duration for state {i} ({self.state_labels[i]}) has "
                    f"invalid params: mu={dd.mu}, sigma={dd.sigma}"
                )

        # Check for NaN in parameters
        for i, gmm in enumerate(self.obs_dists):
            if np.any(np.isnan(gmm.means)):
                issues.append(f"NaN in GMM means for state {i}")
            if np.any(np.isnan(gmm.covars)):
                issues.append(f"NaN in GMM covars for state {i}")

        return issues

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize all model parameters to a JSON-compatible dict."""
        return {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "n_gmm_components": self.n_gmm_components,
            "fs": self.fs,
            "state_labels": self.state_labels,
            "pi": self.pi.tolist(),
            "A": self.A.tolist(),
            "D_max": self.D_max.tolist(),
            "obs_dists": [g.get_params() for g in self.obs_dists],
            "dur_dists": [d.get_params() for d in self.dur_dists],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HSMMModel":
        """Restore model from dict."""
        model = cls(
            n_states=d["n_states"],
            state_labels=d["state_labels"],
            n_features=d["n_features"],
            n_gmm_components=d["n_gmm_components"],
            fs=d["fs"],
        )
        model.pi = np.array(d["pi"])
        model.A = np.array(d["A"])
        model.D_max = np.array(d["D_max"], dtype=int)

        for i, params in enumerate(d["obs_dists"]):
            model.obs_dists[i].set_params(params)

        for i, params in enumerate(d["dur_dists"]):
            model.dur_dists[i].set_params(params)

        return model

    def save(self, filepath: str):
        """Save model to a .npz file."""
        model_dict = self.to_dict()
        # np.savez doesn't handle nested dicts well, so we use JSON inside npz
        json_str = json.dumps(model_dict)
        npz_path = filepath if filepath.endswith(".npz") else filepath + ".npz"
        np.savez(npz_path, model_json=json_str)

    @classmethod
    def load(cls, filepath: str) -> "HSMMModel":
        """Load model from a .npz file."""
        data = np.load(filepath, allow_pickle=True)
        json_str = str(data["model_json"])
        return cls.from_dict(json.loads(json_str))

    def get_state_name(self, idx: int) -> str:
        """Return human-readable state label."""
        return self.state_labels[idx] if 0 <= idx < self.n_states else "UNKNOWN"

    def __repr__(self) -> str:
        return (f"HSMMModel(n_states={self.n_states}, "
                f"states={self.state_labels}, fs={self.fs})")
