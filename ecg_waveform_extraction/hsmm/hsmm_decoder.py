"""HSMM Modified Viterbi Decoder with explicit duration modeling.

Finds the MAP (maximum a posteriori) state-duration sequence through
log-space dynamic programming. The duration dimension of the DP is
vectorized with numpy; per-state caches are computed once per decode.
"""

import numpy as np
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
        ll = float(np.max(log_delta[T - 1, :]))

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
    # Fast Viterbi with precomputed caches + vectorized duration loop
    # ==================================================================
    def _viterbi_log(self, model: HSMMModel, log_B: np.ndarray):
        """HSMM Viterbi with per-state caches and numpy-vectorized durations.

        Recurrence (per t, j):
            δ_t(j) = max_d [ p_j(d) · b_j(o_{t-d+1:t}) ·
                             max( π_j·1{start=0},
                                  max_i δ_{t-d}(i)·a_ij ) ]
        The max over d is computed with array ops instead of a Python loop.
        Tie-breaking matches the plain triple loop: durations ascending,
        predecessors in list order, the initial-segment candidate last.
        """
        T, N = log_B.shape

        # ---- Precompute caches ----
        # (1) Cumulative log_B for O(1) segment likelihood
        cum_log_B = np.zeros((T + 1, N))
        cum_log_B[1:] = np.cumsum(log_B, axis=0)

        # (2) Log transition matrix (zero-prob transitions -> -inf)
        log_A = np.full((N, N), -np.inf)
        pos_A = model.A > 0
        log_A[pos_A] = np.log(model.A[pos_A])

        # (3) Log initial probabilities
        log_pi = np.full(N, -np.inf)
        pos_pi = model.pi > 0
        log_pi[pos_pi] = np.log(model.pi[pos_pi])

        # (4) Per-state hoisted arrays (constant across t)
        D_max_arr = model.D_max
        preds = model.predecessors
        d_full = {}        # j -> np.ndarray of durations [d_min..D_j] (finite log_dur only)
        log_dur_full = {}  # j -> log p_j(d) aligned with d_full[j]
        preds_arr = {}     # j -> np.ndarray of predecessor state indices
        logA_preds = {}    # j -> log_A[preds, j] aligned with preds_arr[j]
        for j in range(N):
            dd = model.dur_dists[j]
            D_j = int(D_max_arr[j])
            d_j = np.arange(dd.d_min, D_j + 1)
            # log_prob_range(d_min, D_j) aligns element-wise with d_j
            ld_j = dd.log_prob_range(dd.d_min, D_j)
            finite = ~np.isinf(ld_j)
            d_full[j] = d_j[finite]
            log_dur_full[j] = ld_j[finite]
            p_j = np.array([i for i in preds.get(j, [])
                            if not np.isinf(log_A[i, j])], dtype=int)
            preds_arr[j] = p_j
            logA_preds[j] = log_A[p_j, j] if p_j.size else np.array([])

        # ---- DP ----
        log_delta = np.full((T, N), -np.inf)
        psi = np.full((T, N, 2), -1, dtype=int)  # psi[t, j] = [prev_state, duration]

        for t in range(T):
            for j in range(N):
                # Durations valid at this t: d <= t + 1
                d_j = d_full[j]
                if d_j.size == 0:
                    continue
                n_valid = np.searchsorted(d_j, t + 1, side="right")
                if n_valid == 0:
                    continue
                d_arr = d_j[:n_valid]

                # Keep the terms separate and add them in exactly the order of
                # the plain loop — ((prev + log_A) + log_dur) + seg_ll — so the
                # DP values are bit-identical to the scalar implementation
                # (a different association order rounds differently and can
                # flip near-tied argmax decisions downstream).
                log_dur_v = log_dur_full[j][:n_valid]
                seg_start = t - d_arr + 1
                # Segment log-likelihood O(1) per duration (vectorized)
                seg_ll_v = cum_log_B[t + 1, j] - cum_log_B[seg_start, j]

                best_val = -np.inf
                best_prev = -1
                best_d = -1

                # ---- Transition candidates (seg_start > 0) ----
                tr = seg_start > 0
                p_j = preds_arr[j]
                if tr.any() and p_j.size:
                    ss = seg_start[tr]
                    prev_mat = log_delta[ss - 1][:, p_j]      # (n_d, n_preds)
                    vals = prev_mat + logA_preds[j][None, :]
                    vals = vals + log_dur_v[tr][:, None]
                    vals = vals + seg_ll_v[tr][:, None]
                    k = int(np.argmax(vals))                  # d-major order
                    v = vals.flat[k]
                    if v > -np.inf:
                        best_val = v
                        best_prev = int(p_j[k % p_j.size])
                        best_d = int(d_arr[tr][k // p_j.size])

                # ---- Initial-segment candidate (seg_start == 0) ----
                # Only possible at d = t + 1 (the largest duration), so the
                # plain loop considers it after all transition candidates;
                # the strict '>' here reproduces that tie order.
                init_idx = np.nonzero(seg_start == 0)[0]
                if init_idx.size:
                    k = int(init_idx[0])
                    v = log_pi[j] + log_dur_v[k] + seg_ll_v[k]
                    if v > best_val:
                        best_val = v
                        best_prev = -1
                        best_d = int(d_arr[k])

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
