# -*- coding: utf-8 -*-
"""Cache-wide audit of P/T delineation quality over output/rala_full/_limb_leads (200 records).

Questions answered with exact numbers:
 1. P/T duration distributions per lead, split by refinement source (prominence vs hsmm),
    non-physiological rates (P<40/>160 ms, T<80/>350 ms), window-saturation rates.
 2. Our per-record MEDIAN durations vs machine XML measurements (P_dur, PR_ms, QT_ms, QRS):
    median bias, MAD, % records within +-20 ms, per lead.
 3. T evidence: (a) T-peak mislocalization vs the decoded T state, (b) peak-position
    asymmetry inside the T window, (c) HSMM T-state run-length floor, (d) t_offset ->
    next-beat p_onset collisions.
 4. False-P evidence: wide/ectopic beats that got a P window, P windows overlapping the
    previous beat's T window, P windows overlapping the QRS.

Run from repo root:  /c/ProgramData/anaconda3/python.exe src/audit_pt_delineation.py
"""
import json
import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.join("output", "rala_full", "_limb_leads")
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF"]
FS = 1000.0
N_T_SAMPLE = 20
SEED = 0
P_LO, P_HI = 40.0, 160.0
T_LO, T_HI = 80.0, 350.0


def clean(a):
    return np.asarray([v for v in a if v is not None and np.isfinite(v)], dtype=float)


def pct(a, q):
    a = clean(a)
    return float(np.percentile(a, q)) if a.size else float("nan")


def med(a):
    a = clean(a)
    return float(np.median(a)) if a.size else float("nan")


def valid_beat(b):
    """Reject sentinel/truncated beats (boundaries of -1 at signal end)."""
    keys = ("p_onset", "p_offset", "q_onset", "r_peak", "s_offset", "t_onset", "t_offset")
    return all(b.get(k) is not None and b[k] >= 0 for k in keys)


records = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)))
print("=" * 104)
print(f"PT DELINEATION AUDIT  |  {len(records)} records x {len(LEADS)} leads  |  fs={FS:.0f} Hz")
print("=" * 104)

dur = {w: {L: {"prominence": [], "hsmm": []} for L in LEADS} for w in ("P", "T")}
durs_all = {w: {L: [] for L in LEADS} for w in ("P", "T")}
rec_cmp, sentinels = [], 0
p_sat = {L: {"prominence": [0, 0], "hsmm": [0, 0]} for L in LEADS}      # n_at_99_101, n
part4 = {L: dict(n_beats=0, n_wide=0, wide_with_p=0, wide_p_ovl_qrs=0, wide_p_ovl_prevT=0,
                 wide_p_dur=[], narrow_p_dur=[], all_ovl_prevT=0, all_with_p=0,
                 short_rr_with_p_ovl_prevT=0, n_short_rr=0) for L in LEADS}
gap_stats = {L: [] for L in LEADS}
coll = {L: dict(n=0, neg=0, neg_tsrc={"prominence": 0, "hsmm": 0},
                neg_psrc={"prominence": 0, "hsmm": 0}) for L in LEADS}

for rec in records:
    rdir = os.path.join(BASE, rec)
    sf = os.path.join(rdir, "summary.json")
    meas = json.load(open(sf, encoding="utf-8")).get("measurements", {}) if os.path.exists(sf) else {}

    for L in LEADS:
        d = os.path.join(rdir, f"lead_{L}")
        try:
            beats = json.load(open(os.path.join(d, "beats.json"), encoding="utf-8"))
            pw = json.load(open(os.path.join(d, "p_waves.json"), encoding="utf-8"))
            tw = json.load(open(os.path.join(d, "t_waves.json"), encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pw_by = {e["beat_id"]: e for e in pw}
        tw_by = {e["beat_id"]: e for e in tw}
        vb = [b for b in beats if valid_beat(b)]
        sentinels += len(beats) - len(vb)

        # ---- part 1: durations by source + fixed-window saturation ------------
        for name, waves in (("P", pw_by), ("T", tw_by)):
            for e in waves.values():
                du, src = e.get("duration_ms"), e.get("source", "hsmm")
                if du is None:
                    continue
                dur[name][L][src].append(du)
                durs_all[name][L].append(du)
                if name == "P" and 99.0 <= du <= 101.0:
                    p_sat[L][src][0] += 1
                if name == "P":
                    p_sat[L][src][1] += 1

        # ---- part 2: per-record medians ---------------------------------------
        qt = [b["t_offset"] - b["q_onset"] for b in vb if b["t_offset"] > b["q_onset"]]
        pr = [b["q_onset"] - b["p_onset"] for b in vb if 0 < b["q_onset"] - b["p_onset"] < 500]
        qrs = [b["s_offset"] - b["q_onset"] for b in vb if b["s_offset"] > b["q_onset"]]
        rec_cmp.append(dict(rec=rec, lead=L, n=len(vb),
                            P_m=meas.get("P_dur"), PR_m=meas.get("PR_ms"),
                            QT_m=meas.get("QT_ms"), QRS_m=meas.get("QRS_dur"),
                            P_o=med([pw_by[b["beat_id"]].get("duration_ms") for b in vb
                                     if b["beat_id"] in pw_by]),
                            QT_o=med(qt), PR_o=med(pr), QRS_o=med(qrs),
                            QRS_med=np.median(qrs) if qrs else np.nan))

        # ---- part 4: false-P geometry ----------------------------------------
        st = part4[L]
        for i, b in enumerate(vb):
            qrs_b = b["s_offset"] - b["q_onset"]
            pe = pw_by.get(b["beat_id"], {})
            p_dur = pe.get("duration_ms")
            prev = vb[i - 1] if i > 0 else None
            in_prevT = (prev is not None and b["p_offset"] > prev["t_onset"]
                        and b["p_onset"] < prev["t_offset"])
            wide = qrs_b > max(160.0, (st and rec_cmp[-1]["QRS_med"]) + 40.0) \
                if np.isfinite(rec_cmp[-1]["QRS_med"]) else qrs_b > 160.0
            st["n_beats"] += 1
            st["all_with_p"] += 1
            st["all_ovl_prevT"] += in_prevT
            rr = (b["q_onset"] - prev["r_peak"]) / FS * 1000.0 if prev else np.inf
            if rr < 400:
                st["n_short_rr"] += 1
                st["short_rr_with_p_ovl_prevT"] += in_prevT
            if wide:
                st["n_wide"] += 1
                st["wide_with_p"] += 1
                st["wide_p_dur"].append(p_dur)
                st["wide_p_ovl_qrs"] += b["p_offset"] > b["q_onset"]
                st["wide_p_ovl_prevT"] += in_prevT
            else:
                st["narrow_p_dur"].append(p_dur)

        # ---- part 3d: t_offset -> next p_onset gap ---------------------------
        for i in range(len(vb) - 1):
            b, nb = vb[i], vb[i + 1]
            gap = (nb["p_onset"] - b["t_offset"]) / FS * 1000.0
            gap_stats[L].append(gap)
            g = coll[L]
            g["n"] += 1
            if gap < 0:
                g["neg"] += 1
                g["neg_tsrc"][b.get("t_source", "hsmm")] += 1
                g["neg_psrc"][nb.get("p_source", "hsmm")] += 1

# ==================================================================================
print("\n" + "#" * 104)
print("PART 1  P/T duration distributions per lead, split by source (ms)")
print("#" * 104)
for name, lo, hi in (("P", P_LO, P_HI), ("T", T_LO, T_HI)):
    print(f"\n  --- {name} (physiological {lo:.0f}-{hi:.0f} ms) ---")
    for L in LEADS:
        cells = []
        for src in ("prominence", "hsmm"):
            a = dur[name][L][src]
            cells.append(f"{src:>10}: n={len(a):6d} p5={pct(a,5):6.1f} med={med(a):6.1f} "
                         f"p95={pct(a,95):6.1f} bad={100*np.mean((clean(a)<lo)|(clean(a)>hi)):5.1f}%")
        a = durs_all[name][L]
        cells.append(f"ALL: n={len(a):6d} med={med(a):6.1f} p5={pct(a,5):6.1f} p95={pct(a,95):6.1f}")
        print(f"  {name} {L:>3}  " + "  ".join(cells))
print(f"\n  sentinel/truncated beats excluded (boundary = -1 at signal end): {sentinels}")
print("  fixed-window saturation (the delineator's hard window, not the wave):")
print(f"  {'lead':>5}{'P prom %at 99-101ms':>22}{'P hsmm %at 99-101ms':>22}")
for L in LEADS:
    pr_, pn_ = p_sat[L]["prominence"], p_sat[L]["hsmm"]
    print(f"  {L:>5}{100*pr_[0]/max(pr_[1],1):>21.1f}%{100*pn_[0]/max(pn_[1],1):>21.1f}%")

# ==================================================================================
print("\n" + "#" * 104)
print("PART 2  our per-record MEDIAN vs machine XML measurement (bias = ours - machine, ms)")
print("#" * 104)
for key, mk, ok in (("P duration", "P_m", "P_o"), ("PR", "PR_m", "PR_o"),
                    ("QT", "QT_m", "QT_o"), ("QRS", "QRS_m", "QRS_o")):
    print(f"\n  --- our median {key} vs measurements.{mk[:-2]} ---")
    print(f"  {'lead':>5}{'n_rec':>6}{'excl*mach<=0':>14}{'median bias':>13}{'MAD':>8}"
          f"{'IQR bias':>14}{'%|b|<=20':>10}{'%|b|<=40':>10}")
    for L in LEADS:
        bs, excl = [], 0
        for r in rec_cmp:
            if r["lead"] != L:
                continue
            m, o = r[mk], r[ok]
            if m is None or o is None or not np.isfinite(o):
                continue
            if m <= 0:
                excl += 1
                continue
            bs.append(o - m)
        n = len(bs)
        if n == 0:
            print(f"  {L:>5}{0:>6}   no data")
            continue
        a = np.array(bs)
        print(f"  {L:>5}{n:>6}{excl:>14}{med(a):>13.1f}{med(np.abs(a)):>8.1f}"
              f"{f'[{pct(a,25):.0f},{pct(a,75):.0f}]':>14}{100*np.mean(np.abs(a)<=20):>9.1f}%"
              f"{100*np.mean(np.abs(a)<=40):>9.1f}%")

# ==================================================================================
print("\n" + "#" * 104)
print(f"PART 3  T-wave evidence ({N_T_SAMPLE} sampled records for waveform work)")
print("#" * 104)
rng = np.random.default_rng(SEED)
sample = list(rng.choice(records, size=min(N_T_SAMPLE, len(records)), replace=False))
mis = {L: {"prominence": [0, 0], "hsmm": [0, 0]} for L in LEADS}
pos = {L: [] for L in LEADS}
runlen = {L: {"hsmm": [], "prominence": []} for L in LEADS}
floor60 = {L: {"hsmm": 0, "prominence": 0} for L in LEADS}
for rec in sample:
    for L in LEADS:
        d = os.path.join(BASE, rec, f"lead_{L}")
        try:
            beats = [b for b in json.load(open(os.path.join(d, "beats.json"), encoding="utf-8"))
                     if valid_beat(b)]
            x = np.load(os.path.join(d, "filtered_ecg.npy"))
            lab = np.load(os.path.join(d, "state_labels.npy"))
        except (OSError, json.JSONDecodeError):
            continue
        for i, b in enumerate(beats):
            t0, t1 = b["t_onset"], b["t_offset"]
            nxt = beats[i + 1] if i + 1 < len(beats) else None
            if t1 <= t0 or t1 > len(x):
                continue
            # search window for the true T peak: from QRS end to next beat's P/QRS
            s0 = b["s_offset"] + 5
            s1 = min(nxt["p_onset"] if nxt else len(x), b["r_peak"] + 600, len(x))
            if s1 - s0 < 20:
                continue
            seg = x[s0:s1]
            pk = s0 + int(np.argmax(np.abs(seg)))
            inside = t0 <= pk < t1
            src = b.get("t_source", "hsmm")
            mis[L][src][1] += 1
            mis[L][src][0] += 0 if inside else 1
            if inside:
                w = x[t0:t1]
                pos[L].append(int(np.argmax(np.abs(w))) / max(len(w) - 1, 1))
            # HSMM T-state run length at the decoded T
            if 0 <= t0 < len(lab) and 0 < t1 <= len(lab):
                run = t1 - t0
                runlen[L][src].append(run)
                if run == 60:
                    floor60[L][src] += 1
print(f"  sampled records: {len(sample)}   (T-peak search window: s_offset+5 .. min(next p_onset, r_peak+600))")
print(f"\n  {'lead':>5}{'src':>11}{'n_T':>6}{'% peak OUTSIDE decoded T':>27}"
      f"{'peak-pos p5':>13}{'med':>7}{'p95':>7}{'%pos>=0.8':>11}{'%dur==60ms':>12}")
for L in LEADS:
    for src in ("prominence", "hsmm"):
        m = mis[L][src]
        r = np.array(runlen[L][src], dtype=float)
        p = np.array(pos[L])
        print(f"  {L:>5}{src:>11}{m[1]:>6}{100*m[0]/max(m[1],1):>26.1f}%"
              f"{pct(p,5) if p.size else float('nan'):>13.2f}{med(p):>7.2f}{pct(p,95):>7.2f}"
              f"{100*np.mean(p>=0.8) if p.size else float('nan'):>10.1f}%"
              f"{100*np.mean(r==60) if r.size else float('nan'):>11.1f}%")

print("\n  t_offset -> NEXT beat p_onset gap (ms), all records:")
print(f"  {'lead':>5}{'n':>7}{'p5':>7}{'med':>7}{'p95':>7}{'%gap<0':>9}{'%gap<40':>9}")
for L in LEADS:
    a = np.array(gap_stats[L])
    print(f"  {L:>5}{a.size:>7}{pct(a,5):>7.0f}{med(a):>7.0f}{pct(a,95):>7.0f}"
          f"{100*np.mean(a<0):>9.2f}{100*np.mean(a<40):>9.2f}")
print("  collisions (gap<0) by source  [T src of the beat | P src of the next beat]:")
for L in LEADS:
    g = coll[L]
    print(f"    {L:>3}: {g['neg']:>4}/{g['n']}  T: prom={g['neg_tsrc']['prominence']:>4}"
          f" hsmm={g['neg_tsrc']['hsmm']:>4} | next-P: prom={g['neg_psrc']['prominence']:>4}"
          f" hsmm={g['neg_psrc']['hsmm']:>4}")

# ==================================================================================
print("\n" + "#" * 104)
print("PART 4  false-P evidence")
print("#" * 104)
print("  wide/ectopic beat := qrs(s_offset-q_onset) > max(160, record_median_qrs + 40) ms")
print(f"  {'lead':>5}{'beats':>7}{'wide':>7}{'%wide':>8}{'wide w/ P':>11}{'%of wide':>10}"
      f"{'%P ovl QRS':>12}{'%P ovl prevT':>14}{'medPdur wide':>14}{'medPdur narrow':>16}")
for L in LEADS:
    s = part4[L]
    print(f"  {L:>5}{s['n_beats']:>7}{s['n_wide']:>7}{100*s['n_wide']/max(s['n_beats'],1):>7.1f}%"
          f"{s['wide_with_p']:>11}{100*s['wide_with_p']/max(s['n_wide'],1):>9.1f}%"
          f"{100*s['wide_p_ovl_qrs']/max(s['wide_with_p'],1):>11.1f}%"
          f"{100*s['wide_p_ovl_prevT']/max(s['wide_with_p'],1):>13.1f}%"
          f"{med(s['wide_p_dur']):>14.1f}{med(s['narrow_p_dur']):>16.1f}")
print("\n  ALL beats: P window overlapping the PREVIOUS beat's T window (any overlap):")
print(f"  {'lead':>5}{'%all beats':>12}{'%of short-RR(<400ms) beats':>28}")
for L in LEADS:
    s = part4[L]
    print(f"  {L:>5}{100*s['all_ovl_prevT']/max(s['all_with_p'],1):>11.1f}%"
          f"{100*s['short_rr_with_p_ovl_prevT']/max(s['n_short_rr'],1):>27.1f}%")

# ==================================================================================
print("\n" + "#" * 104)
print("APPENDIX  record-level extremes (lead II)")
print("#" * 104)
r2 = [r for r in rec_cmp if r["lead"] == "II"]
short = sorted(r2, key=lambda r: r["QT_o"])[:8]
print("  8 records with the SHORTEST median T window (q_onset..t_offset):")
for r in short:
    print(f"    {r['rec']:>12}: med T win={r['QT_o']:6.1f} ms  machine QT={r['QT_m']} ms  n={r['n']}")
print("  records whose median T window is < 200 ms while machine QT >= 300 ms (T clipped):")
n_clip = sum(1 for r in r2 if (r["QT_m"] or 0) >= 300 and r["QT_o"] < 200)
n_tot = sum(1 for r in r2 if (r["QT_m"] or 0) >= 300)
print(f"    {n_clip}/{n_tot} records ({100*n_clip/max(n_tot,1):.1f}%)")
print("  8 records with largest |P bias| vs machine P_dur (machine>0):")
out = sorted((r for r in r2 if r["P_m"] and r["P_o"] is not None and np.isfinite(r["P_o"])),
             key=lambda r: -abs(r["P_o"] - r["P_m"]))[:8]
for r in out:
    print(f"    {r['rec']:>12}: ours={r['P_o']:6.1f}  machine={r['P_m']:>6.0f}  "
          f"(bias {r['P_o']-r['P_m']:+.1f})")

if "1805185J6U" in records:
    print("\n  symptomatic record 1805185J6U lead II (first 10 valid beats):")
    d = os.path.join(BASE, "1805185J6U", "lead_II")
    beats = [b for b in json.load(open(os.path.join(d, "beats.json"), encoding="utf-8"))
             if valid_beat(b)]
    pw_by = {e["beat_id"]: e for e in json.load(open(os.path.join(d, "p_waves.json"),
                                                     encoding="utf-8"))}
    tw_by = {e["beat_id"]: e for e in json.load(open(os.path.join(d, "t_waves.json"),
                                                     encoding="utf-8"))}
    for b in beats[:10]:
        p, t = pw_by.get(b["beat_id"], {}), tw_by.get(b["beat_id"], {})
        print(f"    beat {b['beat_id']:>2}: qrs={b['s_offset']-b['q_onset']:>4}ms "
              f"P={p.get('duration_ms')}[{p.get('source','?')[:4]}] "
              f"T={t.get('duration_ms')}[{t.get('source','?')[:4]}] "
              f"t_on={b['t_onset']:>5} t_off={b['t_offset']:>5} r={b['r_peak']:>5}")
print("\n(done)")
