"""Data loading utilities: CSV import, PhysioNet WFDB import, synthetic ECG generation."""

import numpy as np
import csv


def load_csv_ecg(filepath: str, fs: float = 250.0,
                 signal_col: int = 0, skip_header: bool = True) -> tuple[np.ndarray, float]:
    """Load ECG signal from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    fs : float
        Sampling frequency.
    signal_col : int
        Column index (0-based) containing the ECG signal.
    skip_header : bool
        Whether to skip the first line.

    Returns
    -------
    signal : np.ndarray, 1-D
    fs : float
    """
    values = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        for row in reader:
            if row and len(row) > signal_col:
                try:
                    values.append(float(row[signal_col]))
                except ValueError:
                    continue
    return np.array(values, dtype=np.float64), fs


def load_wfdb_record(record_name: str,
                     pn_dir: str | None = None) -> tuple[np.ndarray, float, dict]:
    """Load an ECG record from PhysioNet using the wfdb library.

    Requires: pip install wfdb

    For databases that require authentication (e.g., QTDB), set
    pn_dir to a directory containing downloaded PhysioNet files,
    or use wfdb.dl_database() first.

    Parameters
    ----------
    record_name : str
        PhysioNet record name (e.g., 'mitdb/100', 'qtdb/sel100').
    pn_dir : str or None
        PhysioNet data directory. If None, uses wfdb default.

    Returns
    -------
    signal : np.ndarray, shape (N, M) or (N,)
        ECG signal(s). For multi-lead, first lead is returned.
    fs : float
        Sampling frequency.
    info : dict
        Record metadata.
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError(
            "wfdb library is required. Install with: pip install wfdb"
        )

    # wfdb 4.x uses pn_dir, older versions use dl_dir
    try:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    except TypeError:
        record = wfdb.rdrecord(record_name, dl_dir=pn_dir)

    signal = record.p_signal
    fs = record.fs

    # If multi-lead, take first lead
    if signal.ndim > 1:
        signal = signal[:, 0]

    info = {
        "record_name": record.record_name,
        "fs": fs,
        "n_samples": record.sig_len,
        "units": record.units[0] if record.units else "",
        "comments": record.comments,
    }

    return signal.astype(np.float64), fs, info


def download_physionet_db(db_name: str, pn_dir: str) -> list[str]:
    """Download a PhysioNet database.

    Parameters
    ----------
    db_name : str
        Database name (e.g., 'mitdb', 'qtdb').
    pn_dir : str
        Local directory to store downloaded files.

    Returns
    -------
    list[str]
        List of downloaded record names.
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb library is required. Install with: pip install wfdb")
    wfdb.dl_database(db_name, dl_dir=pn_dir)
    records = wfdb.get_record_list(db_name)
    return records


# ===========================================================================
# Synthetic ECG generation for testing
# ===========================================================================
def generate_synthetic_ecg(fs: float = 250.0, duration_sec: float = 10.0,
                           heart_rate: float = 60.0, noise_std: float = 0.02,
                           random_state: int | None = 42) -> dict:
    """Generate synthetic ECG with known ground-truth boundaries.

    Builds a P-QRS-T cycle from shifted/scaled Gaussian pulses at known positions.
    Returns the signal and exact onset/offset for each waveform component.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    duration_sec : float
        Total recording duration in seconds.
    heart_rate : float
        Heart rate in BPM.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    random_state : int or None
        Random seed.

    Returns
    -------
    dict with keys:
        ecg : np.ndarray
            Synthetic ECG signal.
        fs : float
        true_boundaries : list[dict]
            Per-beat ground-truth boundaries with keys:
            'beat_id', 'P_onset', 'P_offset', 'P_peak',
            'Q_onset', 'R_peak', 'S_offset', 'T_onset', 'T_offset', 'T_peak'
    """
    rng = np.random.RandomState(random_state)
    total_samples = int(duration_sec * fs)
    t = np.arange(total_samples) / fs

    ecg = np.zeros(total_samples, dtype=np.float64)

    # Beat parameters
    beat_interval_sec = 60.0 / heart_rate
    beat_interval_samples = int(beat_interval_sec * fs)

    # Waveform timing relative to beat start (in ms) — canonical ECG
    # These are the ground-truth positions
    wave_timing_ms = {
        "P_onset":  0,
        "P_peak":   50,
        "P_offset": 100,
        "Q_onset":  160,
        "R_peak":   190,
        "S_offset": 230,
        "T_onset":  280,
        "T_peak":   380,
        "T_offset": 480,
    }

    # Convert to samples relative to beat start
    wave_timing = {k: int(np.round(v / 1000.0 * fs))
                   for k, v in wave_timing_ms.items()}

    # Gaussian waveform shapes
    def _gaussian(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

    # Generate beats
    true_boundaries = []
    beat_id = 0
    beat_start_sample = 0

    # Add a small initial delay (ISO before first beat)
    initial_delay = int(0.2 * fs)  # 200ms

    while beat_start_sample + beat_interval_samples < total_samples:
        bs = beat_start_sample + initial_delay if beat_id == 0 else beat_start_sample
        bs_t = bs / fs

        # P wave
        p_center = bs + wave_timing["P_peak"]
        p_width = wave_timing["P_offset"] - wave_timing["P_onset"]
        p_amp = 0.15
        p_idx = np.arange(
            max(0, bs + wave_timing["P_onset"] - p_width),
            min(total_samples, bs + wave_timing["P_offset"] + p_width),
        )
        p_time = p_idx / fs
        ecg[p_idx] += _gaussian(p_time, p_center / fs, p_width / fs / 3.0 * 0.8, p_amp)

        # Q wave
        q_center = bs + wave_timing["Q_onset"] + (wave_timing["R_peak"] - wave_timing["Q_onset"]) // 3
        q_width = 3
        q_amp = -0.15
        q_idx = np.arange(
            max(0, bs + wave_timing["Q_onset"] - 2),
            min(total_samples, bs + wave_timing["R_peak"]),
        )
        ecg[q_idx] += _gaussian(q_idx / fs, q_center / fs, q_width / fs, q_amp)

        # R wave
        r_center = bs + wave_timing["R_peak"]
        r_width = 5
        r_amp = 1.0
        r_idx = np.arange(
            max(0, r_center - 2 * r_width),
            min(total_samples, r_center + 2 * r_width),
        )
        ecg[r_idx] += _gaussian(r_idx / fs, r_center / fs, r_width / fs, r_amp)

        # S wave
        s_center = bs + wave_timing["R_peak"] + (wave_timing["S_offset"] - wave_timing["R_peak"]) // 2
        s_width = 3
        s_amp = -0.25
        s_idx = np.arange(
            max(0, bs + wave_timing["R_peak"]),
            min(total_samples, bs + wave_timing["S_offset"] + 2),
        )
        ecg[s_idx] += _gaussian(s_idx / fs, s_center / fs, s_width / fs, s_amp)

        # T wave
        t_center = bs + wave_timing["T_peak"]
        t_width = wave_timing["T_offset"] - wave_timing["T_onset"]
        t_amp = 0.3
        t_idx = np.arange(
            max(0, bs + wave_timing["T_onset"] - t_width // 2),
            min(total_samples, bs + wave_timing["T_offset"] + t_width // 2),
        )
        ecg[t_idx] += _gaussian(t_idx / fs, t_center / fs, t_width / fs / 4.0, t_amp)

        # Record ground truth
        true_boundaries.append({
            "beat_id": beat_id,
            "P_onset": bs + wave_timing["P_onset"],
            "P_peak": bs + wave_timing["P_peak"],
            "P_offset": bs + wave_timing["P_offset"],
            "Q_onset": bs + wave_timing["Q_onset"],
            "R_peak": bs + wave_timing["R_peak"],
            "S_offset": bs + wave_timing["S_offset"],
            "T_onset": bs + wave_timing["T_onset"],
            "T_peak": bs + wave_timing["T_peak"],
            "T_offset": bs + wave_timing["T_offset"],
        })

        beat_id += 1
        beat_start_sample += beat_interval_samples

    # Add Gaussian noise
    ecg += rng.randn(total_samples) * noise_std

    return {
        "ecg": ecg,
        "fs": fs,
        "true_boundaries": true_boundaries,
        "heart_rate": heart_rate,
        "duration_sec": duration_sec,
    }
