"""HSMM Baum-Welch (EM) training with explicit duration distributions.

Implements the forward-backward algorithm specialized for HSMM, where
self-transitions are modeled via duration distributions rather than
the standard HMM transition matrix.
"""

import numpy as np
from scipy.special import logsumexp
from .hsmm_model import HSMMModel
from .distributions import _safe_log


class HSMMTrainer:
    """Baum-Welch EM trainer for Hidden Semi-Markov Models.

    Parameters
    ----------
    model : HSMMModel
        The model to train.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold on log-likelihood.
    verbose : bool
        Print per-iteration log-likelihood.
    """

    def __init__(self, model: HSMMModel, max_iter: int = 50,
                 tol: float = 1e-4, verbose: bool = False):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self._log_likelihood_history: list[float] = []

    # ==================================================================
    # Public API
    # ==================================================================
    def fit(self, features: np.ndarray) -> list[float]:
        """Run Baum-Welch EM training.

        Parameters
        ----------
        features : np.ndarray, shape (T, D)
            Observation feature vectors.

        Returns
        -------
        list[float]
            Log-likelihood at each iteration.
        """
        T = features.shape[0]
        N = self.model.n_states

        # Precompute per-sample per-state log observation likelihood
        log_B = self._compute_obs_log_likelihood(features)  # (T, N)

        self._log_likelihood_history = []
        prev_ll = -np.inf

        for it in range(self.max_iter):
            # ---- E-step ----
            log_alpha, ll = self.forward(log_B)
            if np.isinf(ll) or np.isnan(ll):
                if self.verbose:
                    print(f"  Iter {it}: LL = -inf — model collapsed, stopping")
                break

            log_beta = self.backward(log_B, ll)
            stats = self._e_step_collect(log_alpha, log_beta, log_B, ll)

            self._log_likelihood_history.append(ll)

            if self.verbose:
                print(f"  Iter {it}: log-likelihood = {ll:.2f}")

            # Check convergence
            if abs(ll - prev_ll) < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {it}")
                break
            prev_ll = ll

            # ---- M-step ----
            self._m_step(features, stats)

        # Update D_max after training (duration dists may have changed)
        self.model._compute_D_max()

        return self._log_likelihood_history

    # ==================================================================
    # Observation log-likelihood matrix
    # ==================================================================
    def _compute_obs_log_likelihood(self, features: np.ndarray) -> np.ndarray:
        """Precompute log b_j(o_t) for all states and time steps.

        Parameters
        ----------
        features : np.ndarray, shape (T, D)

        Returns
        -------
        log_B : np.ndarray, shape (T, N)
            log_B[t, j] = log P(o_t | state=j)
        """
        T = features.shape[0]
        N = self.model.n_states
        log_B = np.zeros((T, N))

        for j in range(N):
            log_B[:, j] = self.model.obs_dists[j].log_prob(features)

        return log_B

    # ==================================================================
    # HSMM Forward Algorithm (log-space)
    # ==================================================================
    def forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """HSMM forward pass in log-space.

        Computes log_alpha[t, j] = log P(O_{1:t}, state j ends at t).

        Recurrence:
            α_t(j) = Σ_{i≠j} Σ_d α_{t-d}(i) · a_ij · p_j(d) · b_j(o_{t-d+1:t})

        Parameters
        ----------
        log_B : np.ndarray, shape (T, N)
            log b_j(o_t) for each (t, j).

        Returns
        -------
        log_alpha : np.ndarray, shape (T, N)
        log_likelihood : float
            log P(O | model).
        """
        T, N = log_B.shape
        log_alpha = np.full((T, N), -np.inf)
        log_pi = _safe_log(self.model.pi)

        # Precompute cumulative log_B for O(1) segmental likelihood
        cum_log_B = np.zeros((T + 1, N))
        cum_log_B[1:] = np.cumsum(log_B, axis=0)  # cum_log_B[t, j] = Σ_{s=0}^{t-1} log_B[s, j]

        predecessors = self.model.predecessors

        for t in range(T):
            for j in range(N):
                D_j = self.model.D_max[j]
                max_d = min(D_j, t + 1)

                candidates = []

                for i in predecessors.get(j, []):
                    log_a_ij = _safe_log(np.array([self.model.A[i, j]]))[0]
                    if np.isinf(log_a_ij):
                        continue

                    dur_dist = self.model.dur_dists[j]

                    for d in range(dur_dist.d_min, max_d + 1):
                        # Segment from (t-d+1) to t inclusive in state j
                        seg_ll = self._segment_log_likelihood(cum_log_B, j, t - d + 1, t)

                        # Duration log-prob
                        log_dur = dur_dist.log_prob(d)
                        if np.isinf(log_dur):
                            continue

                        if t - d < 0:
                            # Initial segment: state j starts the sequence
                            log_prev = log_pi[j]
                        else:
                            log_prev = log_alpha[t - d, i]

                        if np.isinf(log_prev):
                            continue

                        candidate = log_prev + log_a_ij + log_dur + seg_ll
                        candidates.append(candidate)

                if candidates:
                    log_alpha[t, j] = logsumexp(candidates)

        # Total log-likelihood
        log_likelihood = logsumexp(log_alpha[T - 1, :])
        return log_alpha, float(log_likelihood)

    # ==================================================================
    # HSMM Backward Algorithm (log-space)
    # ==================================================================
    def backward(self, log_B: np.ndarray, log_likelihood: float) -> np.ndarray:
        """HSMM backward pass in log-space.

        Computes log_beta[t, i] = log P(O_{t+1:T} | state i ends at t).

        Recurrence:
            β_t(i) = Σ_{j≠i} Σ_d a_ij · p_j(d) · b_j(o_{t+1:t+d}) · β_{t+d}(j)

        Parameters
        ----------
        log_B : np.ndarray, shape (T, N)
        log_likelihood : float
            Forward log-likelihood (for numerical scaling).

        Returns
        -------
        log_beta : np.ndarray, shape (T, N)
        """
        T, N = log_B.shape
        log_beta = np.full((T, N), -np.inf)

        # Precompute cumulative log_B
        cum_log_B = np.zeros((T + 1, N))
        cum_log_B[1:] = np.cumsum(log_B, axis=0)

        successors = self.model.successors

        # Initialize: at t = T-1, β_{T-1}(i) = 0 for all i (no future obs)
        log_beta[T - 1, :] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(N):
                candidates = []

                for j in successors.get(i, []):
                    log_a_ij = _safe_log(np.array([self.model.A[i, j]]))[0]
                    if np.isinf(log_a_ij):
                        continue

                    dur_dist = self.model.dur_dists[j]
                    D_j = self.model.D_max[j]
                    max_d = min(D_j, T - t - 1)

                    for d in range(dur_dist.d_min, max_d + 1):
                        end_t = t + d
                        if end_t >= T:
                            continue

                        # Segment: observations from t+1 to t+d in state j
                        seg_ll = self._segment_log_likelihood(cum_log_B, j, t, end_t - 1)

                        log_dur = dur_dist.log_prob(d)
                        if np.isinf(log_dur):
                            continue

                        log_beta_next = log_beta[end_t, j]
                        if np.isinf(log_beta_next):
                            continue

                        candidate = log_a_ij + log_dur + seg_ll + log_beta_next
                        candidates.append(candidate)

                if candidates:
                    log_beta[t, i] = logsumexp(candidates)

        # Note: we compute unnormalized beta. Full normalization would
        # use log_likelihood, but the E-step only uses alpha*beta ratios,
        # so the normalization cancels out.
        return log_beta

    # ==================================================================
    # Segment log-likelihood helper
    # ==================================================================
    def _segment_log_likelihood(self, cum_log_B: np.ndarray, state: int,
                                  start: int, end: int) -> float:
        """O(1) segmental log-likelihood: Σ_{t=start}^{end} log b_j(o_t).

        Parameters
        ----------
        cum_log_B : np.ndarray, shape (T+1, N)
            Cumulative sum of log_B.
        state : int
            State index.
        start : int
            Start index (0-based, inclusive).
        end : int
            End index (0-based, inclusive).

        Returns
        -------
        float
        """
        if start > end:
            return 0.0
        if start < 0:
            start = 0
        # cum_log_B[k, j] = Σ_{s=0}^{k-1} log_B[s, j]
        return float(cum_log_B[end + 1, state] - cum_log_B[start, state])

    # ==================================================================
    # E-step: collect sufficient statistics
    # ==================================================================
    def _e_step_collect(self, log_alpha: np.ndarray, log_beta: np.ndarray,
                        log_B: np.ndarray, ll: float) -> dict:
        """Collect expected sufficient statistics from alpha/beta.

        Returns dict with keys:
            gamma: (T, N) — state occupancy posteriors
            xi: list of (from_state, to_state, end_time, duration) tuples with weights
            obs_weights: list of (state, time, weight) for GMM training
            durations_per_state: dict mapping state -> list of (duration, weight)
        """
        T, N = log_B.shape
        cum_log_B = np.zeros((T + 1, N))
        cum_log_B[1:] = np.cumsum(log_B, axis=0)

        # Compute gamma[t, j] = P(state j active at time t | O)
        # gamma[t, j] = Σ_d Σ_{start} P(segment covers t in state j | O)
        gamma = np.zeros((T, N))

        # Duration tracking per state
        durations_per_state: dict[int, list[tuple[int, float]]] = {j: [] for j in range(N)}

        # Observation weights per state
        obs_weights: dict[int, list[tuple[int, float]]] = {j: [] for j in range(N)}

        # Transition counts (for M-step A update)
        xi_counts = np.zeros((N, N))

        # Initial state posteriors
        pi_posterior = np.zeros(N)

        predecessors = self.model.predecessors

        # Iterate over all possible segmentations
        for t in range(T):
            for j in range(N):
                D_j = self.model.D_max[j]
                dur_dist = self.model.dur_dists[j]
                max_d = min(D_j, t + 1)

                for d in range(dur_dist.d_min, max_d + 1):
                    seg_start = t - d + 1

                    # Segment log-likelihood
                    seg_ll = self._segment_log_likelihood(cum_log_B, j, seg_start, t)

                    # Duration log-prob
                    log_dur = dur_dist.log_prob(d)
                    if np.isinf(log_dur):
                        continue

                    for i in predecessors.get(j, []):
                        log_a_ij = _safe_log(np.array([self.model.A[i, j]]))[0]
                        if np.isinf(log_a_ij):
                            continue

                        if seg_start == 0:
                            log_prev = _safe_log(np.array([self.model.pi[j]]))[0]
                        else:
                            log_prev = log_alpha[seg_start - 1, i]

                        if np.isinf(log_prev):
                            continue

                        log_beta_t = log_beta[t, j]
                        if np.isinf(log_beta_t):
                            continue

                        # Log posterior weight for this (i, j, d, seg_start..t)
                        log_weight = log_prev + log_a_ij + log_dur + seg_ll + log_beta_t - ll
                        weight = np.exp(log_weight)

                        if weight < 1e-15:
                            continue

                        # Accumulate to gamma: all samples in [seg_start, t] get weight
                        gamma[seg_start:t + 1, j] += weight

                        # Transition count
                        xi_counts[i, j] += weight

                        # Initial state (if seg_start == 0, this is first state visited)
                        if seg_start == 0:
                            pi_posterior[j] += weight

                        # Duration sample
                        durations_per_state[j].append((d, weight))

                        # Observation weights (per sample in segment)
                        for s in range(seg_start, t + 1):
                            obs_weights[j].append((s, weight))

        # Normalize gamma row-wise
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        gamma = gamma / row_sums

        return {
            "gamma": gamma,
            "xi_counts": xi_counts,
            "pi_posterior": pi_posterior,
            "durations_per_state": durations_per_state,
            "obs_weights": obs_weights,
        }

    # ==================================================================
    # M-step: update model parameters
    # ==================================================================
    def _m_step(self, features: np.ndarray, stats: dict):
        """Update all model parameters from sufficient statistics.

        Parameters
        ----------
        features : np.ndarray, shape (T, D)
        stats : dict
            Output of _e_step_collect.
        """
        N = self.model.n_states

        # ---- Update pi ----
        pi_sum = stats["pi_posterior"].sum()
        if pi_sum > 1e-12:
            self.model.pi = stats["pi_posterior"] / pi_sum

        # ---- Update A (transitions) ----
        for i in range(N):
            row_sum = stats["xi_counts"][i, :].sum()
            if row_sum > 1e-12:
                self.model.A[i, :] = stats["xi_counts"][i, :] / row_sum
            # Zero out physiologically invalid transitions
            for j in range(N):
                if j not in self.model.successors.get(i, []):
                    self.model.A[i, j] = 0.0
            # Re-normalize
            row_sum = self.model.A[i, :].sum()
            if row_sum > 0:
                self.model.A[i, :] /= row_sum

        # ---- Update observation GMMs ----
        gamma = stats["gamma"]  # (T, N)
        for j in range(N):
            weights = gamma[:, j]  # (T,)
            w_sum = weights.sum()
            if w_sum > 1e-12 and features.shape[0] > self.model.n_gmm_components:
                try:
                    self.model.obs_dists[j].fit(
                        features,
                        max_iter=30,
                        tol=1e-3,
                        sample_weight=weights,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    # Keep old parameters if fit fails
                    pass

        # ---- Update duration distributions ----
        for j in range(N):
            dur_samples = stats["durations_per_state"][j]
            if len(dur_samples) >= 3:
                durations = np.array([d for d, w in dur_samples])
                weights = np.array([w for d, w in dur_samples])
                w_sum = weights.sum()
                if w_sum > 0:
                    weighted_mean = np.average(durations, weights=weights)
                    weighted_var = np.average((durations - weighted_mean) ** 2, weights=weights)
                    self.model.dur_dists[j].mu = float(weighted_mean)
                    self.model.dur_dists[j].sigma = float(np.sqrt(max(weighted_var, 1.0)))
                    # Invalidate cache
                    self.model.dur_dists[j]._log_Z = None

    # ==================================================================
    # Convenience
    # ==================================================================
    @property
    def log_likelihood_history(self) -> list[float]:
        return self._log_likelihood_history
