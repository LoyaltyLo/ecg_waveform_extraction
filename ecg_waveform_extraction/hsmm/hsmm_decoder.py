"""HSMM Modified Viterbi Decoder with explicit duration modeling.

Finds the MAP (maximum a posteriori) state-duration sequence through
log-space dynamic programming. Optimized with precomputed caches.
"""

import numpy as np
from scipy.special import logsumexp
from .hsmm_model import HSMMModel


class HSMMDecoder:
    """Maximum a posteriori decoder for HSMM with explicit durations."""

    def __init__(self):
        pass

    # ==================================================================
    # Public API
    # ==================================================================
    def decode(self, model: HSMMModel, features: np.ndarray) -> dict:
        """Decode the most likely state sequence.

        Returns dict with keys:
            state_sequence: list of (state_idx, start_sample, end_sample)
            state_labels: np.ndarray [T] int — per-sample state assignment
            log_likelihood: float — Viterbi path log-probability
        """
        T = features.shape[0]
        N = model.n_states

        if T == 0:
            return {
                "state_sequence": [],
                "state_labels": np.array([], dtype=int),
                "log_likelihood": -np.inf,
            }

        log_B = self._compute_obs_log_likelihood(model, features)
        log_delta, psi = self._viterbi_log(model, log_B)
        segments = self._backtrack(log_delta, psi, T, model)
        state_labels = self._segments_to_labels(segments, T, model.n_states)
        ll = float(logsumexp(log_delta[T - 1, :]))

        return {
            "state_sequence": segments,
            "state_labels": state_labels,
            "log_likelihood": ll,
        }

    def decode_to_labels(self, model: HSMMModel, features: np.ndarray) -> np.ndarray:
        result = self.decode(model, features)
        return result["state_labels"]

    # ==================================================================
    # Observation log-likelihood
    # ==================================================================
    @staticmethod
    def _compute_obs_log_likelihood(model, features):
        T = features.shape[0]
        N = model.n_states
        log_B = np.zeros((T, N))
        for j in range(N):
            log_B[:, j] = model.obs_dists[j].log_prob(features)
        return log_B

    # ==================================================================
    # Fast Viterbi with precomputed caches
    # ==================================================================
    def _viterbi_log(self, model: HSMMModel, log_B: np.ndarray):
        """Optimized HSMM Viterbi with precomputed duration and transition caches."""
        T, N = log_B.shape

        # ---- Precompute caches ----
        # (1) Cumulative log_B for O(1) segment likelihood
        cum_log_B = np.zeros((T + 1, N))
        cum_log_B[1:] = np.cumsum(log_B, axis=0)

        # (2) Log transition matrix
        log_A = np.full((N, N), -np.inf)
        for i in range(N):
            for j in range(N):
                if model.A[i, j] > 0:
                    log_A[i, j] = np.log(model.A[i, j])

        # (3) Log initial probabilities
        log_pi = np.full(N, -np.inf)
        for j in range(N):
            if model.pi[j] > 0:
                log_pi[j] = np.log(model.pi[j])

        # (4) Duration log-prob caches per state [d_min .. D_max]
        D_max_arr = model.D_max
        log_dur_cache = {}
        for j in range(N):
            dd = model.dur_dists[j]
            D_j = D_max_arr[j]
            log_dur_cache[j] = dd.log_prob_range(1, D_j)

        # (5) Predecessors list per state
        preds = model.predecessors

        # ---- DP ----
        log_delta = np.full((T, N), -np.inf)
        psi = np.full((T, N, 2), -1, dtype=int)  # psi[t, j] = [prev_state, duration]

        for t in range(T):
            for j in range(N):
                dd = model.dur_dists[j]
                D_j = D_max_arr[j]
                max_d = min(D_j, t + 1)
                d_min = dd.d_min

                if max_d < d_min:
                    continue

                best_val = -np.inf
                best_prev = -1
                best_d = -1

                for d in range(d_min, max_d + 1):
                    seg_start = t - d + 1

                    # Duration log-prob from cache (index = d-1)
                    log_dur = log_dur_cache[j][d - 1]
                    if np.isinf(log_dur):
                        continue

                    # Segment log-likelihood O(1)
                    seg_ll = cum_log_B[t + 1, j] - cum_log_B[seg_start, j]

                    if seg_start == 0:
                        # Initial segment
                        val = log_pi[j] + log_dur + seg_ll
                        if val > best_val:
                            best_val = val
                            best_prev = -1
                            best_d = d
                    else:
                        for i in preds.get(j, []):
                            if np.isinf(log_A[i, j]):
                                continue
                            prev = log_delta[seg_start - 1, i]
                            if np.isinf(prev):
                                continue
                            val = prev + log_A[i, j] + log_dur + seg_ll
                            if val > best_val:
                                best_val = val
                                best_prev = i
                                best_d = d

                log_delta[t, j] = best_val
                if best_d > 0:
                    psi[t, j, 0] = best_prev
                    psi[t, j, 1] = best_d

        return log_delta, psi

    # ==================================================================
    # Backtracking
    # ==================================================================
    def _backtrack(self, log_delta, psi, T, model):
        N = model.n_states
        best_final = int(np.argmax(log_delta[T - 1, :]))
        if np.isinf(log_delta[T - 1, best_final]):
            return []

        segments = []
        t = T - 1
        j = best_final

        while t >= 0:
            prev_state = int(psi[t, j, 0])
            duration = int(psi[t, j, 1])
            if duration <= 0:
                break

            seg_start = t - duration + 1
            segments.append((j, max(0, seg_start), t))

            if prev_state == -1:
                break
            t = seg_start - 1
            j = prev_state

        segments.reverse()
        return segments

    # ==================================================================
    # Convert segments to per-sample labels
    # ==================================================================
    @staticmethod
    def _segments_to_labels(segments, T, n_states):
        labels = np.full(T, -1, dtype=int)
        for state, start, end in segments:
            if 0 <= start <= end < T:
                labels[start:end + 1] = state
        return labels
