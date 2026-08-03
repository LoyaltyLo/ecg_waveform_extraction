"""HSMM Baum-Welch (EM) training with explicit duration distributions.

Implements the forward-backward algorithm specialized for HSMM, where
self-transitions are modeled via duration distributions rather than
the standard HMM transition matrix.
"""

import numpy as np
from scipy.special import logsumexp
from .hsmm_model import HSMMModel


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

            log_beta = self.backward(log_B)
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
    # Precomputed log caches (shared by forward/backward/E-step)
    # ==================================================================
    def _precompute_caches(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Precompute log transition matrix, log initial probs, and
        per-state duration log-prob arrays for the current model parameters.

        Zero-probability transitions map to -inf (not a small epsilon), so
        invalid moves are truly skipped in the DP recurrences.

        Returns
        -------
        log_A : np.ndarray, shape (N, N)
        log_pi : np.ndarray, shape (N,)
        log_dur_cache : list of np.ndarray
            log_dur_cache[j][d - 1] = log p_j(d) for d in [1, D_max[j]].
        """
        N = self.model.n_states

        log_A = np.full((N, N), -np.inf)
        pos = self.model.A > 0
        log_A[pos] = np.log(self.model.A[pos])

        log_pi = np.full(N, -np.inf)
        pos_pi = self.model.pi > 0
        log_pi[pos_pi] = np.log(self.model.pi[pos_pi])

        log_dur_cache = [
            self.model.dur_dists[j].log_prob_range(1, int(self.model.D_max[j]))
            for j in range(N)
        ]
        return log_A, log_pi, log_dur_cache

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

        log_A, log_pi, log_dur_cache = self._precompute_caches()

        # Precompute cumulative log_B for O(1) segmental likelihood
        cum_log_B = np.zeros((T + 1, N))
        cum_log_B[1:] = np.cumsum(log_B, axis=0)  # cum_log_B[t, j] = Σ_{s=0}^{t-1} log_B[s, j]

        predecessors = self.model.predecessors

        for t in range(T):
            for j in range(N):
                dur_dist = self.model.dur_dists[j]
                D_j = self.model.D_max[j]
                max_d = min(D_j, t + 1)
                d_min = dur_dist.d_min
                if max_d < d_min:
                    continue

                log_durs = log_dur_cache[j]
                preds_j = [i for i in predecessors.get(j, [])
                           if not np.isinf(log_A[i, j])]

                candidates = []

                for d in range(d_min, max_d + 1):
                    log_dur = log_durs[d - 1]
                    if np.isinf(log_dur):
                        continue

                    # Segment from (t-d+1) to t inclusive in state j
                    seg_ll = cum_log_B[t + 1, j] - cum_log_B[t - d + 1, j]

                    if t - d < 0:
                        # Initial segment: state j starts the sequence.
                        # Counted once, with pi only (no transition term).
                        candidates.append(log_pi[j] + log_dur + seg_ll)
                    else:
                        for i in preds_j:
                            log_prev = log_alpha[t - d, i]
                            if np.isinf(log_prev):
                                continue
                            candidates.append(log_prev + log_A[i, j]
                                              + log_dur + seg_ll)

                if candidates:
                    log_alpha[t, j] = logsumexp(candidates)

        # Total log-likelihood
        log_likelihood = logsumexp(log_alpha[T - 1, :])
        return log_alpha, float(log_likelihood)

    # ==================================================================
    # HSMM Backward Algorithm (log-space)
    # ==================================================================
    def backward(self, log_B: np.ndarray) -> np.ndarray:
        """HSMM backward pass in log-space.

        Computes log_beta[t, i] = log P(O_{t+1:T} | state i ends at t).

        Recurrence:
            β_t(i) = Σ_{j≠i} Σ_d a_ij · p_j(d) · b_j(o_{t+1:t+d}) · β_{t+d}(j)

        Parameters
        ----------
        log_B : np.ndarray, shape (T, N)

        Returns
        -------
        log_beta : np.ndarray, shape (T, N)
        """
        T, N = log_B.shape
        log_beta = np.full((T, N), -np.inf)

        log_A, _, log_dur_cache = self._precompute_caches()

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
                    log_a_ij = log_A[i, j]
                    if np.isinf(log_a_ij):
                        continue

                    dur_dist = self.model.dur_dists[j]
                    D_j = self.model.D_max[j]
                    max_d = min(D_j, T - t - 1)
                    d_min = dur_dist.d_min
                    if max_d < d_min:
                        continue

                    log_durs = log_dur_cache[j]

                    for d in range(d_min, max_d + 1):
                        end_t = t + d
                        if end_t >= T:
                            continue

                        log_dur = log_durs[d - 1]
                        if np.isinf(log_dur):
                            continue

                        # Segment: observations t+1 .. t+d (= end_t) in state j
                        seg_ll = cum_log_B[end_t + 1, j] - cum_log_B[t + 1, j]

                        log_beta_next = log_beta[end_t, j]
                        if np.isinf(log_beta_next):
                            continue

                        candidates.append(log_a_ij + log_dur + seg_ll
                                          + log_beta_next)

                if candidates:
                    log_beta[t, i] = logsumexp(candidates)

        # Note: we compute unnormalized beta. Full normalization would
        # use the forward log-likelihood, but the E-step only uses
        # alpha*beta ratios, so the normalization cancels out.
        return log_beta

    # ==================================================================
    # E-step: collect sufficient statistics
    # ==================================================================
    def _e_step_collect(self, log_alpha: np.ndarray, log_beta: np.ndarray,
                        log_B: np.ndarray, ll: float) -> dict:
        """Collect expected sufficient statistics from alpha/beta.

        Returns dict with keys:
            gamma: (T, N) — state occupancy posteriors
            xi_counts: (N, N) — expected transition counts
            pi_posterior: (N,) — expected initial-state counts
            durations_per_state: dict mapping state -> list of (duration, weight)
        """
        T, N = log_B.shape
        cum_log_B = np.zeros((T + 1, N))
        cum_log_B[1:] = np.cumsum(log_B, axis=0)

        log_A, log_pi, log_dur_cache = self._precompute_caches()

        # gamma accumulated via difference array: each segment contributes
        # O(1) instead of O(d); one cumsum at the end restores per-sample values.
        gamma_diff = np.zeros((T + 1, N))

        # Duration tracking per state
        durations_per_state: dict[int, list[tuple[int, float]]] = {j: [] for j in range(N)}

        # Transition counts (for M-step A update)
        xi_counts = np.zeros((N, N))

        # Initial state posteriors
        pi_posterior = np.zeros(N)

        predecessors = self.model.predecessors

        # Iterate over all possible segmentations
        for t in range(T):
            for j in range(N):
                log_beta_t = log_beta[t, j]
                if np.isinf(log_beta_t):
                    continue

                dur_dist = self.model.dur_dists[j]
                D_j = self.model.D_max[j]
                max_d = min(D_j, t + 1)
                d_min = dur_dist.d_min
                if max_d < d_min:
                    continue

                log_durs = log_dur_cache[j]
                preds_j = [i for i in predecessors.get(j, [])
                           if not np.isinf(log_A[i, j])]

                for d in range(d_min, max_d + 1):
                    log_dur = log_durs[d - 1]
                    if np.isinf(log_dur):
                        continue

                    seg_start = t - d + 1
                    seg_ll = cum_log_B[t + 1, j] - cum_log_B[seg_start, j]

                    if seg_start == 0:
                        # Initial segment: no predecessor, no transition term.
                        log_weight = log_pi[j] + log_dur + seg_ll + log_beta_t - ll
                        weight = np.exp(log_weight)
                        if weight < 1e-15:
                            continue

                        gamma_diff[0, j] += weight
                        gamma_diff[t + 1, j] -= weight
                        pi_posterior[j] += weight
                        durations_per_state[j].append((d, weight))
                    else:
                        for i in preds_j:
                            log_prev = log_alpha[seg_start - 1, i]
                            if np.isinf(log_prev):
                                continue

                            # Log posterior weight for this (i, j, d, seg_start..t)
                            log_weight = (log_prev + log_A[i, j] + log_dur
                                          + seg_ll + log_beta_t - ll)
                            weight = np.exp(log_weight)
                            if weight < 1e-15:
                                continue

                            gamma_diff[seg_start, j] += weight
                            gamma_diff[t + 1, j] -= weight
                            xi_counts[i, j] += weight
                            durations_per_state[j].append((d, weight))

        # Restore per-sample gamma from the difference array
        gamma = np.cumsum(gamma_diff, axis=0)[:T, :]

        # Normalize gamma row-wise
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        gamma = gamma / row_sums

        return {
            "gamma": gamma,
            "xi_counts": xi_counts,
            "pi_posterior": pi_posterior,
            "durations_per_state": durations_per_state,
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
        # A changed: predecessor/successor cache must be rebuilt lazily
        self.model._invalidate_topology_cache()

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
