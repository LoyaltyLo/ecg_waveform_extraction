"""Download all MIT-BIH Arrhythmia database records (48 records: 100-124, 200-234)."""
import os
import ssl
import urllib.request
import time

DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
os.makedirs(DATA_DIR, exist_ok=True)

MITBIH_RECORDS = (
    [str(i) for i in range(100, 110)] +
    [str(i) for i in range(111, 125)] +
    [str(i) for i in range(200, 235)]
)

PHYSIONET_BASE = "https://physionet.org/files/mitdb/1.0.0"

# SSL context bypassing CRL check
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

def download_file(url, fpath, max_retries=3):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
                data = resp.read()
            with open(fpath, 'wb') as f:
                f.write(data)
            return True, len(data)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, str(e)
    return False, "unknown"

total = len(MITBIH_RECORDS)
success = 0
failed = []

for idx, rec in enumerate(MITBIH_RECORDS):
    for ext in ['.hea', '.dat', '.atr']:
        fname = rec + ext
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            continue
        url = f"{PHYSIONET_BASE}/{fname}"
        ok, result = download_file(url, fpath)
        if not ok:
            failed.append(fname)

    downloaded = sum(1 for ext in ['.hea', '.dat', '.atr']
                     if os.path.exists(os.path.join(DATA_DIR, rec + ext)))
    if downloaded == 3:
        success += 1

    print(f"  [{idx+1}/{total}] {rec}: {downloaded}/3 files", flush=True)

print(f"\nDownload complete: {success}/{total} records OK")
if failed:
    print(f"Failed files: {failed}")
