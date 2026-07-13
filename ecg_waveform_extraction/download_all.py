"""Download all training data: complete MIT-BIH + QTDB + NSRDB + STDB.

Downloads PhysioNet databases for ECG waveform extraction training.
Bypasses SSL revocation check (common Windows issue with PhysioNet).
"""

import os
import ssl
import urllib.request
import time

DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
os.makedirs(DATA_DIR, exist_ok=True)

# ---- SSL setup ----
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

def download(url, fpath, retries=3):
    if os.path.exists(fpath):
        return True, 0
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as r:
                data = r.read()
            with open(fpath, 'wb') as f:
                f.write(data)
            return True, len(data)
        except Exception as e:
            if i < retries - 1:
                time.sleep(3)
            else:
                return False, str(e)
    return False, "unknown"

def download_db(db_name, version, record_list, exts=None):
    """Download multiple records from a PhysioNet database."""
    if exts is None:
        exts = ['.hea', '.dat', '.atr']
    base = f"https://physionet.org/files/{db_name}/{version}"
    ok = 0
    fail = []
    for rec in record_list:
        for ext in exts:
            fname = f"{rec}{ext}"
            url = f"{base}/{fname}"
            fpath = os.path.join(DATA_DIR, fname)
            success, result = download(url, fpath)
            if not success:
                fail.append(fname)
        if all(os.path.exists(os.path.join(DATA_DIR, f"{rec}{e}")) for e in exts):
            ok += 1
    return ok, fail

# =====================================================================
# 1. Complete missing MIT-BIH records
# =====================================================================
print("=" * 60)
print("1. Complete MIT-BIH: missing records + missing .dat files")
print("=" * 60)

# All MIT-BIH records
ALL_MITBIH = (
    [str(i) for i in range(100, 110)] +
    [str(i) for i in range(111, 125)] +
    [str(i) for i in range(200, 235)]
)

# Download only records with <3 files present
to_download = []
for rec in ALL_MITBIH:
    existing = sum(1 for e in ['.hea','.dat','.atr']
                   if os.path.exists(os.path.join(DATA_DIR, rec + e)))
    if existing < 3:
        to_download.append(rec)

print(f"  {len(to_download)} records need downloading")
ok, fail = download_db('mitdb', '1.0.0', to_download)
print(f"  Done: {ok} complete, {len(fail)} failed")
if fail:
    print(f"  Failed: {fail[:5]}...")

# =====================================================================
# 2. Download QT Database (gold-standard P-QRS-T boundaries)
# =====================================================================
print("\n" + "=" * 60)
print("2. QT Database (manual P/QRS/T boundary annotations)")
print("=" * 60)

# QTDB records with manual annotations (subset that exists)
QTDB_RECORDS = [
    "sel100", "sel102", "sel103", "sel104",
    "sel114", "sel116", "sel117", "sel123",
    "sel213", "sel221", "sel223", "sel230",
    "sel231", "sel232", "sel233",
    "sel301", "sel302", "sel306", "sel307", "sel308",
    "sel310", "sel803", "sel808", "sel811", "sel820",
    "sel821", "sel840", "sel847", "sel853", "sel871",
    "sel872", "sel873", "sel883", "sel891",
    "sele0104", "sele0106", "sele0107", "sele0111",
    "sele0112", "sele0114", "sele0116", "sele0121",
    "sele0122", "sele0124", "sele0126", "sele0129",
    "sele0133", "sele0136", "sele0166", "sele0170",
    "sele0203", "sele0210", "sele0211", "sele0303",
]

print(f"  {len(QTDB_RECORDS)} QTDB records to download...")
qtdb_ok, qtdb_fail = download_db('qtdb', '1.0.0', QTDB_RECORDS, ['.hea','.dat'])
print(f"  QTDB: {qtdb_ok} complete, {len(qtdb_fail)} failed")

# =====================================================================
# 3. Download MIT-BIH Normal Sinus Rhythm Database
# =====================================================================
print("\n" + "=" * 60)
print("3. MIT-BIH Normal Sinus Rhythm Database (18 long records)")
print("=" * 60)

NSRDB_RECORDS = [
    "16265", "16272", "16273", "16420", "16483",
    "16539", "16773", "16786", "16795", "17052",
    "17453", "18177", "18184", "19088", "19090",
    "19093", "19140", "19830",
]
print(f"  {len(NSRDB_RECORDS)} NSRDB records to download...")
nsrdb_ok, nsrdb_fail = download_db('nsrdb', '1.0.0', NSRDB_RECORDS, ['.hea','.dat'])
print(f"  NSRDB: {nsrdb_ok} complete, {len(nsrdb_fail)} failed")

# =====================================================================
# 4. Download MIT-BIH ST Change Database
# =====================================================================
print("\n" + "=" * 60)
print("4. MIT-BIH ST Change Database")
print("=" * 60)

STDB_RECORDS = [str(i) for i in range(300, 328)]
print(f"  {len(STDB_RECORDS)} STDB records to download...")
stdb_ok, stdb_fail = download_db('stdb', '1.0.0', STDB_RECORDS, ['.hea','.dat'])
print(f"  STDB: {stdb_ok} complete, {len(stdb_fail)} failed")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 60)
print("  DOWNLOAD SUMMARY")
print("=" * 60)

def count_db(pattern):
    """Count files matching pattern prefix."""
    # Collect unique record names from .hea files
    recs = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('.hea'):
            recs.add(f[:-4])
    return len(recs)

total_hea = count_db(None)
# Count only records with all 3 files
complete = 0
for f in os.listdir(DATA_DIR):
    if f.endswith('.hea'):
        rec = f[:-4]
        if (os.path.exists(os.path.join(DATA_DIR, rec + '.dat')) and
            os.path.exists(os.path.join(DATA_DIR, rec + '.hea'))):
            complete += 1

print(f"  Total records with headers: {total_hea}")
print(f"  Records with .hea + .dat: {complete}")
print(f"  Total files: {len(os.listdir(DATA_DIR))}")
print(f"  Data directory: {DATA_DIR}")
