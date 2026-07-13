"""Core probability distributions for HSMM: GMM observations and Gaussian durations.

No external HMM library dependency — all distributions are implemented from scratch.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Small helper: safe log (avoids log(0) -> -inf without warning)
# ---------------------------------------------------------------------------
def _safe_log(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(x, eps))


# ===========================================================================
# Gaussian Mixture Model (diagonal covariance)
# ===========================================================================
class GaussianMixtureModel:
    """Diagonal-covariance Gaussian Mixture Model with EM training.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components (K).
    n_features : int
        Dimensionality of observation vectors.
    covariance_type : str
        Only 'diag' is currently supported.
    random_state : int or None
        Seed for reproducible random initialization.
    """

    def __init__(self, n_components: int = 2, n_features: int = 3,
                 covariance_type: str = "diag", random_state: int | None = None):
        if covariance_type != "diag":
            raise NotImplementedError("Only diag covariance is supported.")

        self.n_components = n_components
        self.n_features = n_features
        self.covariance_type = covariance_type

        self.weights = np.ones(n_components) / n_components  # (K,)
        self.means = np.zeros((n_components, n_features))     # (K, D)
        self.covars = np.ones((n_components, n_features))     # (K, D) — diagonal
        self._fitted = False

        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_random(self, X: np.ndarray):
        """Randomly select data points as initial means."""
        n = X.shape[0]
        idx = self._rng.choice(n, self.n_components, replace=False)
        self.means = X[idx].copy()
        self.covars = np.full((self.n_components, self.n_features),
                               np.var(X, axis=0) / self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------
    def _estimate_log_gaussians(self, X: np.ndarray) -> np.ndarray:
        """Compute log-prob of each sample under each Gaussian component.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        np.ndarray, shape (N, K)
            log N(x_n | mu_k, Sigma_k) for each (n, k).
        """
        N = X.shape[0]
        K = self.n_components
        D = self.n_features

        log_gauss = np.zeros((N, K))

        for k in range(K):
            diff = X - self.means[k]               # (N, D)
            log_gauss[:, k] = -0.5 * np.sum(
                diff ** 2 / self.covars[k] + np.log(2.0 * np.pi * self.covars[k]),
                axis=1,
            )

        return log_gauss

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """Log-probability of each sample under the GMM (log-sum-exp over components).

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        np.ndarray, shape (N,)
            log p(x_n) = log Σ_k w_k N(x_n | μ_k, Σ_k).
        """
        log_gauss = self._estimate_log_gaussians(X)      # (N, K)
        log_weights = _safe_log(self.weights)             # (K,)
        return logsumexp(log_gauss + log_weights, axis=1)  # (N,)

    # ------------------------------------------------------------------
    # EM Training
    # ------------------------------------------------------------------
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior responsibilities.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        responsibilities : np.ndarray, shape (N, K)
            Posterior probability of component k given sample n.
        """
        log_gauss = self._estimate_log_gaussians(X)       # (N, K)
        log_weights = _safe_log(self.weights)              # (K,)
        log_joint = log_gauss + log_weights                # (N, K)
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        log_resp = log_joint - log_norm
        return np.exp(log_resp)                            # (N, K)

    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        """Update GMM parameters from responsibilities."""
        N_k = resp.sum(axis=0) + 1e-12                     # (K,) — avoid div by 0

        self.weights = N_k / N_k.sum()
        self.means = (resp.T @ X) / N_k[:, np.newaxis]     # (K, D)

        for k in range(self.n_components):
            diff = X - self.means[k]                        # (N, D)
            self.covars[k] = (resp[:, k] @ (diff ** 2)) / N_k[k]

        self.covars = np.maximum(self.covars, 1e-6)

    def fit(self, X: np.ndarray, max_iter: int = 100, tol: float = 1e-3,
            sample_weight: np.ndarray | None = None) -> "GaussianMixtureModel":
        """Train GMM via EM.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Training data.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence threshold on log-likelihood change.
        sample_weight : np.ndarray, shape (N,) or None
            Weight of each sample in the likelihood (for weighted EM in HSMM M-step).

        Returns
        -------
        self
        """
        if X.shape[0] < self.n_components:
            raise ValueError(
                f"Fewer samples ({X.shape[0]}) than components ({self.n_components})"
            )

        self._init_random(X)

        prev_ll = -np.inf

        for _it in range(max_iter):
            resp = self._e_step(X)                          # (N, K)

            if sample_weight is not None:
                resp = resp * sample_weight[:, np.newaxis]
                row_sum = resp.sum(axis=1, keepdims=True)
                resp = np.where(row_sum > 1e-12, resp / row_sum, 1.0 / self.n_components)

            self._m_step(X, resp)

            ll = self.log_prob(X)
            if sample_weight is not None:
                ll = np.sum(ll * sample_weight) / np.sum(sample_weight)
            else:
                ll = np.mean(ll)

            if np.abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        """Return parameters as a plain dict for serialization."""
        return {
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
        }

    def set_params(self, params: dict):
        """Restore parameters from dict."""
        self.weights = np.array(params["weights"])
        self.means = np.array(params["means"])
        self.covars = np.array(params["covars"])
        self._fitted = True

    # ------------------------------------------------------------------
    # Sampling (for debugging)
    # ------------------------------------------------------------------
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate samples from the GMM.

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        """
        if not self._fitted:
            raise RuntimeError("GMM not fitted. Call fit() first.")

        comp = self._rng.choice(self.n_components, size=n_samples, p=self.weights)
        samples = np.zeros((n_samples, self.n_features))
        for k in range(self.n_components):
            mask = comp == k
            nk = mask.sum()
            if nk > 0:
                samples[mask] = self._rng.normal(
                    self.means[k], np.sqrt(self.covars[k]), size=(nk, self.n_features)
                )
        return samples


# ===========================================================================
# Duration Distribution (truncated Gaussian)
# ===========================================================================
class DurationDistribution:
    """Gaussian duration distribution with minimum-duration floor.

    p(d) ∝ N(d | mu, sigma²) for d >= d_min, else 0.

    Used in HSMM to model the expected number of samples a state persists.

    Parameters
    ----------
    mu : float or None
        Mean duration (in samples).
    sigma : float or None
        Standard deviation (in samples).
    d_min : int
        Hard minimum duration. Durations below this get zero probability.
    """

    def __init__(self, mu: float | None = None, sigma: float | None = None,
                 d_min: int = 1):
        self.mu = mu if mu is not None else 10.0
        self.sigma = sigma if sigma is not None else 5.0
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        self.d_min = max(d_min, 1)
        self._log_Z = None

    def _compute_log_Z(self):
        """Compute log normalization constant for truncated Gaussian over [d_min, ∞)."""
        z = (self.d_min - 0.5 - self.mu) / self.sigma
        cdf_val = norm.cdf(z)
        self._log_Z = np.log(max(1.0 - cdf_val, 1e-15))

    def log_prob(self, d: int) -> float:
        """Log-probability of a single duration value.

        Returns -inf if d < d_min.
        """
        if d < self.d_min:
            return -np.inf
        if self._log_Z is None:
            self._compute_log_Z()
        return norm.logpdf(d, loc=self.mu, scale=self.sigma) - self._log_Z

    def log_prob_range(self, d_min_val: int, d_max_val: int) -> np.ndarray:
        """Vectorized log-prob for durations in [d_min_val, d_max_val] inclusive.

        Returns np.ndarray of length (d_max_val - d_min_val + 1).
        """
        if d_min_val > d_max_val:
            return np.array([])

        ds = np.arange(d_min_val, d_max_val + 1, dtype=np.float64)
        if self._log_Z is None:
            self._compute_log_Z()

        log_p = norm.logpdf(ds, loc=self.mu, scale=self.sigma) - self._log_Z
        log_p[ds.astype(int) < self.d_min] = -np.inf
        return log_p

    def fit(self, durations: np.ndarray):
        """MLE: estimate mu and sigma from duration samples. Keeps priors if < 3 samples."""
        if len(durations) < 3:
            return
        self.mu = float(np.mean(durations))
        self.sigma = float(np.std(durations))
        if self.sigma < 1.0:
            self.sigma = 1.0
        self._log_Z = None

    @staticmethod
    def physiological_prior(state_name: str, fs: float = 250.0) -> dict:
        """Return (mu, sigma, d_min) in samples for a given ECG state.

        Parameters
        ----------
        state_name : str
            One of 'ISO', 'P', 'PR', 'Q', 'R', 'S', 'ST', 'T', 'TP'.
        fs : float
            Sampling frequency.

        Returns
        -------
        dict with keys 'mu', 'sigma', 'd_min' (all in samples).
        """
        phys_durations_ms = {
            "ISO":  (80,   120),
            "P":    (100,  20),
            "PR":   (160,  60),
            "Q":    (30,   10),
            "R":    (60,   15),
            "S":    (30,   10),
            "ST":   (120,  60),
            "T":    (200,  40),
            "TP":   (200,  200),
        }

        d_min_ms = {
            "ISO":  1,  "P": 30,  "PR": 20, "Q": 8,
            "R":   12, "S": 8,   "ST": 20,  "T": 60, "TP": 20,
        }

        mu_ms, sigma_ms = phys_durations_ms.get(state_name, (100, 50))
        dmin_ms = d_min_ms.get(state_name, 1)

        ms_to_samples = fs / 1000.0
        return {
            "mu": mu_ms * ms_to_samples,
            "sigma": sigma_ms * ms_to_samples,
            "d_min": max(1, int(np.round(dmin_ms * ms_to_samples))),
        }

    def set_physiological_prior(self, state_name: str, fs: float = 250.0):
        """Set parameters from known ECG physiological priors."""
        prior = self.physiological_prior(state_name, fs)
        self.mu = prior["mu"]
        self.sigma = prior["sigma"]
        self.d_min = prior["d_min"]
        self._log_Z = None

    def get_params(self) -> dict:
        """Return parameters for serialization."""
        return {"mu": self.mu, "sigma": self.sigma, "d_min": self.d_min}

    def set_params(self, params: dict):
        """Restore from dict."""
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.d_min = params["d_min"]
        self._log_Z = None
