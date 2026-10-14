#!/usr/bin/env python
"""Stage 6: per-record overview of the QRS / T / P situation.

One row per record (666 cases) aggregating the four downstream stages into
a case-level table a clinician can read in Excel:

- rhythm: n_beats, median HR from RR, RR IQR, device HR, median noise;
- coverage: QRS window rate, T found rate, P gate-pass rate;
- durations: median QRS/P/T durations and PR/QT vs the device
  measurements (device = reference, never an input upstream);
- amplitudes: median R / T / P amplitude (mV) on the reference lead;
- polarity: pos/neg/biphasic/flat shares per wave, lead-II T inversion
  share, V1 P '+-'/'-+' pattern counts, the full 12-lead x 3-wave
  majority matrix, and the aVR reversal probe
  (aVR QRS dominant-positive AND device P axis in [0, 90]).

Pure pandas over the existing stage CSVs - no signal access, no
re-detection; the whole 666-record run takes seconds.

Inputs
------
output/qrs_wavedet_reflead/{qrs_by_beat,summary_by_record}.csv
output/twave_wavedet_reflead/{t_by_beat,summary_by_record}.csv
output/pwave_reflead/summary_by_record.csv
output/polarity_reflead/{polarity_by_beat_lead,summary_by_record}.csv

Outputs (output/record_overview/)
---------------------------------
record_overview.csv : 666 rows x ~90 columns, the case-level table.
cohort_summary.txt  : cohort-level totals (also printed to console).

Validation (full 666-record run, 2026-10)
-----------------------------------------
666 rows x 81 columns, no duplicate records, no all-NaN columns.
Cross-checks against the stage reports (identical numbers): QRS dur
|d| median 10.0 ms; QT 21.0 ms / 72.5% within 30 ms; PR 11.0 ms /
83.8% within 30 ms; aVR reversal probe 15 records (same definition
and list as the polarity stage: aVR QRS+ AND T+, device P_axis
0-90).  Cohort: 13,319 beats, median HR 78.4 bpm, 95.5% of records
within 5 bpm of the device HR; median coverage QRS windows 90% /
T found 90% / P gate-pass 80%; median per-record polarity share
P 51.6/19.6/4.7% and QRS 33.3/29.6/21.5% and T 66.7/19.0/2.3%
(pos/neg/bip); 111 records with lead-II T majority negative; 381
records with V1 P '+-' pattern dominant.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QRS_DIR = ROOT / "output" / "qrs_wavedet_reflead"
T_DIR = ROOT / "output" / "twave_wavedet_reflead"
P_DIR = ROOT / "output" / "pwave_reflead"
POL_DIR = ROOT / "output" / "polarity_reflead"
OUT_DIR = ROOT / "output" / "record_overview"

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
WAVES = ["P", "QRS", "T"]
# RR guard: physiologic beat-to-beat distances only (ms)
RR_MIN_MS, RR_MAX_MS = 300.0, 3000.0


def _save(frame: pd.DataFrame, path: Path) -> None:
    """to_csv with the locked-file fallback (Excel open -> .new.csv)."""
    try:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        alt = path.with_name(path.stem + ".new.csv")
        frame.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"WARNING: {path.name} locked, wrote {alt.name} instead")


def _rhythm(qrs_beats: pd.DataFrame) -> pd.DataFrame:
    """HR / RR variability from the PT R peaks (one row per record)."""
    rows = {}
    for rec, g in qrs_beats.groupby("record"):
        rr = np.diff(np.sort(g["r_ms"].to_numpy(dtype=float)))
        rr = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]
        if len(rr) == 0:
            rows[rec] = (np.nan, np.nan, 0, 0.0)
            continue
        iqr = float(np.percentile(rr, 75) - np.percentile(rr, 25))
        rows[rec] = (float(60000.0 / np.median(rr)), iqr,
                     int(len(g)), 100.0 * len(rr) / max(len(g) - 1, 1))
    return pd.DataFrame.from_dict(
        rows, orient="index",
        columns=["hr_median_bpm", "rr_iqr_ms", "n_beats_qrs",
                 "rr_coverage_pct"]).rename_axis("record").reset_index()


def _coverage(qrs_beats: pd.DataFrame, t_beats: pd.DataFrame,
              p_beats: pd.DataFrame) -> pd.DataFrame:
    """Per-record window rates from the three per-beat tables."""
    q_ok = qrs_beats["qrs_on_ms"].notna() & qrs_beats["qrs_off_ms"].notna()
    qrs_rate = qrs_beats.assign(ok=q_ok).groupby("record")["ok"].mean() * 100.0

    t_ok = t_beats["in_window"] == True  # noqa: E712 (NaN -> False)
    t_rate = t_beats.assign(ok=t_ok).groupby("record")["ok"].mean() * 100.0

    p_ok = ((p_beats["in_window"] == True)        # noqa: E712
            & (p_beats["passes_snr"] == True)     # noqa: E712
            & p_beats["p_on_ms"].notna())
    p_rate = p_beats.assign(ok=p_ok).groupby("record")["ok"].mean() * 100.0

    out = pd.DataFrame({"qrs_window_rate_pct": qrs_rate,
                        "t_found_rate_pct": t_rate,
                        "p_pass_rate_pct": p_rate})
    return out.rename_axis("record").reset_index()


def _polarity_stats(pol: pd.DataFrame, ref_lead: pd.Series) -> pd.DataFrame:
    """Amplitudes on the reference lead + lead-II / V1 diagnostics."""
    ii = pol[pol["lead"] == "II"].dropna(subset=["T_on_ms"])
    t_inv = ii.groupby("record")["T_polarity"].apply(
        lambda s: 100.0 * float((s == "negative").mean()))

    v1 = pol[(pol["lead"] == "V1") & (pol["P_polarity"] == "biphasic")]
    v1_pm = v1.groupby("record")["P_pattern"].apply(
        lambda s: int((s == "+-").sum()))
    v1_mp = v1.groupby("record")["P_pattern"].apply(
        lambda s: int((s == "-+").sum()))

    ref = pol[pol["lead"] == pol["record"].map(ref_lead)]
    q_amp = ref.dropna(subset=["QRS_on_ms"]).groupby("record")["QRS_abs_amp"].median()
    t_amp = ref.dropna(subset=["T_on_ms"]).groupby("record")["T_abs_amp"].median()
    p_amp = ref.dropna(subset=["P_on_ms"]).groupby("record")["P_abs_amp"].median()
    noise = ref.groupby("record")["noise_mv"].median()

    out = pd.DataFrame({
        "qrs_amp_median_mv": q_amp, "t_amp_median_mv": t_amp,
        "p_amp_median_mv": p_amp, "noise_median_mv": noise,
        "t_inverted_II_pct": t_inv,
        # records with no biphasic V1 P simply have zero pattern beats
        "p_V1_pm_beats": v1_pm.fillna(0).astype(int),
        "p_V1_mp_beats": v1_mp.fillna(0).astype(int),
    })
    return out.rename_axis("record").reset_index()


def build(qrs_dir: Path, t_dir: Path, p_dir: Path, pol_dir: Path) -> pd.DataFrame:
    qrs_sum = pd.read_csv(qrs_dir / "summary_by_record.csv", encoding="utf-8-sig")
    t_sum = pd.read_csv(t_dir / "summary_by_record.csv", encoding="utf-8-sig")
    p_sum = pd.read_csv(p_dir / "summary_by_record.csv", encoding="utf-8-sig")
    pol_sum = pd.read_csv(pol_dir / "summary_by_record.csv", encoding="utf-8-sig")

    qrs_beats = pd.read_csv(qrs_dir / "qrs_by_beat.csv", encoding="utf-8-sig",
                            usecols=["record", "beat_i", "r_ms",
                                     "qrs_on_ms", "qrs_off_ms"])
    t_beats = pd.read_csv(t_dir / "t_by_beat.csv", encoding="utf-8-sig",
                          usecols=["record", "beat_i", "in_window"])
    p_beats = pd.read_csv(p_dir / "p_by_beat.csv", encoding="utf-8-sig",
                          usecols=["record", "beat_i", "in_window",
                                   "passes_snr", "p_on_ms"])
    pol = pd.read_csv(pol_dir / "polarity_by_beat_lead.csv", encoding="utf-8-sig",
                      usecols=["record", "lead", "noise_mv",
                               "QRS_on_ms", "QRS_abs_amp",
                               "T_on_ms", "T_abs_amp", "T_polarity",
                               "P_on_ms", "P_abs_amp", "P_polarity",
                               "P_pattern"])

    ov = qrs_sum[["record", "ref_lead", "device_HR_bpm", "device_QRS_dur_ms",
                  "dur_median_ms", "dur_iqr_ms"]].rename(columns={
        "dur_median_ms": "qrs_dur_median_ms",
        "dur_iqr_ms": "qrs_dur_iqr_ms"})
    ov = ov.merge(t_sum[["record", "t_dur_median_ms", "qt_median_ms",
                         "device_QT_ms", "device_QTc_ms"]], on="record")
    ov = ov.merge(p_sum[["record", "p_dur_median_ms", "pr_median_ms",
                         "device_PR_ms", "device_P_dur_ms"]], on="record")
    ov = ov.merge(_rhythm(qrs_beats), on="record", how="left")
    ov = ov.merge(_coverage(qrs_beats, t_beats, p_beats), on="record", how="left")
    ov = ov.merge(_polarity_stats(pol, pol_sum.set_index("record")["ref_lead"]),
                  on="record", how="left")

    # polarity shares from the polarity summary's per-wave counts
    count_cols = [f"{w}_{state}" for w in WAVES for state in
                  ("positive", "negative", "biphasic", "flat", "n_classified")]
    ov = ov.merge(pol_sum[["record"] + count_cols], on="record")
    for w in WAVES:
        den = ov[f"{w}_n_classified"].replace(0, np.nan)
        for state, short in [("positive", "pos"), ("negative", "neg"),
                             ("biphasic", "bip"), ("flat", "flat")]:
            ov[f"{w}_{short}_pct"] = 100.0 * ov[f"{w}_{state}"] / den

    lead_cols = [f"{lead}_{w}" for lead in LEADS for w in WAVES]
    ov = ov.merge(pol_sum[["record", "device_P_axis", "device_QRS_axis",
                           "device_T_axis"] + lead_cols], on="record")

    # aVR reversal probe: aVR QRS AND T dominant-positive with a normal
    # device P axis (same definition as the polarity stage's console check)
    sm_idx = pol_sum.set_index("record")
    probe = ((sm_idx["AVR_QRS"] == "positive")
             & (sm_idx["AVR_T"] == "positive")
             & sm_idx["device_P_axis"].between(0, 90).fillna(False))
    ov["reversal_probe"] = ov["record"].map(
        probe).map({True: "yes", False: ""}).fillna("")

    front = ["record", "ref_lead", "n_beats_qrs", "hr_median_bpm", "rr_iqr_ms",
             "rr_coverage_pct", "device_HR_bpm", "noise_median_mv",
             "qrs_window_rate_pct", "qrs_dur_median_ms", "qrs_dur_iqr_ms",
             "device_QRS_dur_ms", "qrs_amp_median_mv",
             "QRS_pos_pct", "QRS_neg_pct", "QRS_bip_pct", "QRS_flat_pct",
             "t_found_rate_pct", "t_dur_median_ms", "qt_median_ms",
             "device_QT_ms", "device_QTc_ms", "t_amp_median_mv",
             "t_inverted_II_pct",
             "T_pos_pct", "T_neg_pct", "T_bip_pct", "T_flat_pct",
             "p_pass_rate_pct", "p_dur_median_ms", "pr_median_ms",
             "device_PR_ms", "device_P_dur_ms", "p_amp_median_mv",
             "p_V1_pm_beats", "p_V1_mp_beats",
             "P_pos_pct", "P_neg_pct", "P_bip_pct", "P_flat_pct"]
    tail = ["reversal_probe", "device_P_axis", "device_QRS_axis",
            "device_T_axis", "interpretation"]
    interp = pol_sum[["record", "interpretation"]]
    ov = ov.merge(interp, on="record")
    return ov[front + lead_cols + tail]


def cohort_report(ov: pd.DataFrame) -> str:
    lines = []
    add = lines.append
    add(f"records: {len(ov)}  beats total: {int(ov['n_beats_qrs'].sum())}")
    hr_dev = (ov["hr_median_bpm"] - ov["device_HR_bpm"]).abs()
    add(f"HR: median {ov['hr_median_bpm'].median():.1f} bpm, "
        f"|vs device| within 5 bpm: {100 * (hr_dev <= 5).mean():.1f}%")
    for name, col, dev in [
            ("QRS dur", "qrs_dur_median_ms", "device_QRS_dur_ms"),
            ("QT", "qt_median_ms", "device_QT_ms"),
            ("PR", "pr_median_ms", "device_PR_ms")]:
        d = (ov[col] - ov[dev]).abs().dropna()
        add(f"{name} vs device: median |d| {d.median():.1f} ms, "
            f"within 30 ms {100 * (d <= 30).mean():.1f}%")
    for rate in ["qrs_window_rate_pct", "t_found_rate_pct", "p_pass_rate_pct"]:
        add(f"{rate}: median {ov[rate].median():.1f}%")
    for w in WAVES:
        shares = ", ".join(
            f"{s} {ov[f'{w}_{s}_pct'].median():.1f}%"
            for s in ("pos", "neg", "bip", "flat"))
        add(f"{w} polarity share (median across records): {shares}")
    t_neg_II = (ov["II_T"] == "negative")
    add(f"records with lead-II T majority negative: {int(t_neg_II.sum())}")
    v1_pm_dom = ov["p_V1_pm_beats"].fillna(0) > ov["p_V1_mp_beats"].fillna(0)
    add(f"records with V1 P '+-' pattern dominant: {int(v1_pm_dom.sum())}")
    n_probe = int((ov["reversal_probe"] == "yes").sum())
    add(f"aVR reversal probe records: {n_probe} -> "
        + ", ".join(ov.loc[ov["reversal_probe"] == "yes", "record"].head(20)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qrs-dir", type=Path, default=QRS_DIR)
    ap.add_argument("--t-dir", type=Path, default=T_DIR)
    ap.add_argument("--p-dir", type=Path, default=P_DIR)
    ap.add_argument("--pol-dir", type=Path, default=POL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--records", type=str, default="",
                    help="comma-separated record filter (debug)")
    args = ap.parse_args(argv)

    ov = build(args.qrs_dir, args.t_dir, args.p_dir, args.pol_dir)
    if args.records:
        keep = set(args.records.split(","))
        ov = ov[ov["record"].isin(keep)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _save(ov, args.out_dir / "record_overview.csv")
    report = cohort_report(ov)
    print(report)
    (args.out_dir / "cohort_summary.txt").write_text(report + "\n",
                                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
