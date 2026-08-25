"""Download the icentia11k-continuous-ecg dataset from PhysioNet.

Dataset: https://physionet.org/content/icentia11k-continuous-ecg/1.0/
Layout:  pXX/pXXXXX/pXXXXX_sXX.{dat,hea,atr}
         11 groups x 1000 patients x ~50 segments/patient

The script crawls the PhysioNet file tree (RECORDS + per-patient directory
listings) and downloads every file, mirroring the source layout under
--dest. It is fully resumable: files that are already complete are skipped,
partial files are resumed via HTTP Range requests, and each file is verified
against its expected size.

IMPORTANT: PhysioNet throttles anonymous downloads to ~16 KB/s per IP. The
full dataset is ~188 GB compressed (~1.1 TB as individual files), which is
impractical anonymously (weeks to months). Either provide PhysioNet account
credentials (--user/--password, HTTP Basic Auth) to lift the throttle, or
download a subset (--groups / --limit).

Examples:
    # everything (slow!)
    python download_icentia11k.py

    # first 10 patients only
    python download_icentia11k.py --limit 10

    # a single group
    python download_icentia11k.py --groups p03

    # authenticated (faster)
    python download_icentia11k.py --user alice --password '...'
"""

import argparse
import base64
import os
import re
import ssl
import sys
import time
import urllib.request

BASE_URL = "https://physionet.org/files/icentia11k-continuous-ecg/1.0"
DEFAULT_DEST = r"C:\LoyaltyLo\datasets\Icentia11k_continuous_ecg"
USER_AGENT = "Mozilla/5.0"
CHUNK = 65536

# Bypass SSL certificate revocation check (common Windows issue with PhysioNet).
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# <a href="p00000_s00.dat">p00000_s00.dat</a>  13-Feb-2022 16:55  2097154
_LINK_RE = re.compile(r'href="([^"]+)"')
_SIZE_RE = re.compile(r'(\d+)\s*$')


def build_headers(user=None, password=None):
    """Return a request-header dict, optionally with HTTP Basic Auth."""
    headers = {"User-Agent": USER_AGENT}
    if user:
        token = base64.b64encode(f"{user}:{password or ''}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    return headers


def fetch_text(url, headers, timeout=120):
    """GET a URL and return its decoded text."""
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_listing(html):
    """Parse a PhysioNet directory listing into {name: size} (size None for dirs)."""
    entries = {}
    for line in html.splitlines():
        m = _LINK_RE.search(line)
        if not m:
            continue
        name = m.group(1)
        if name == "../":
            continue
        if name.endswith("/"):
            entries[name] = None  # directory
            continue
        s = _SIZE_RE.search(line)
        entries[name] = int(s.group(1)) if s else None
    return entries


def list_patients(headers, groups=None):
    """Return patient paths like 'p00/p00000' from the RECORDS file."""
    text = fetch_text(f"{BASE_URL}/RECORDS", headers)
    patients = []
    for line in text.splitlines():
        line = line.strip().rstrip("/")
        if not line:
            continue
        group = line.split("/")[0]
        if groups and group not in groups:
            continue
        patients.append(line)
    return patients


def download_file(url, dest, expected_size, headers, retries=5):
    """Download one file with resume. Return 'skip', 'ok', or 'error: ...'."""
    if os.path.exists(dest):
        existing = os.path.getsize(dest)
        if expected_size is not None and existing == expected_size:
            return "skip"
        if expected_size is not None and existing > expected_size:
            os.remove(dest)
            existing = 0
    else:
        existing = 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    for attempt in range(retries):
        hdrs = dict(headers)
        if existing > 0:
            hdrs["Range"] = f"bytes={existing}-"
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=300) as r:
                # If the server ignored Range and returned the whole file, restart.
                if existing > 0 and r.status == 200:
                    existing = 0
                mode = "ab" if existing > 0 else "wb"
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)

            existing = os.path.getsize(dest)
            if expected_size is not None and existing != expected_size:
                # Incomplete: loop to resume from where we left off.
                continue
            return "ok"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                if os.path.exists(dest):
                    existing = os.path.getsize(dest)
            else:
                return f"error: {e}"
    return "error: max retries"


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Download the icentia11k-continuous-ecg dataset from PhysioNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("IMPORTANT:")[1].strip() if "IMPORTANT:" in __doc__ else "",
    )
    p.add_argument("--dest", default=DEFAULT_DEST,
                   help="destination directory (default: %(default)s)")
    p.add_argument("--groups", default=None,
                   help="comma-separated groups to download, e.g. p00,p01 (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="max number of patients to download (default: all)")
    p.add_argument("--user", default=None, help="PhysioNet username (optional)")
    p.add_argument("--password", default=None, help="PhysioNet password (optional)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel download workers (default: %(default)s; note the "
                        "anonymous throttle is per-IP, so more workers won't help)")
    p.add_argument("--verbose", action="store_true", help="log every file")
    args = p.parse_args(argv)

    groups = set(args.groups.split(",")) if args.groups else None
    headers = build_headers(args.user, args.password)

    print("Fetching patient list (RECORDS)...")
    patients = list_patients(headers, groups)
    if args.limit is not None:
        patients = patients[: args.limit]
    total = len(patients)
    print(f"{total} patient(s) to download -> {args.dest}")
    print("NOTE: anonymous PhysioNet downloads are throttled to ~16 KB/s.")
    print(f"      ~{total * 100 / 1024:.1f} GiB estimated (individual files).")

    ok = skip = err = 0
    for i, patient in enumerate(patients, 1):
        listing_url = f"{BASE_URL}/{patient}/"
        try:
            files = parse_listing(fetch_text(listing_url, headers))
        except Exception as e:
            print(f"[{i}/{total}] {patient}: failed to list ({e})")
            err += 1
            continue

        file_items = [(n, s) for n, s in files.items() if s is not None]
        for name, size in file_items:
            url = f"{BASE_URL}/{patient}/{name}"
            dest = os.path.join(args.dest, patient, name)
            status = download_file(url, dest, size, headers)
            if status == "skip":
                skip += 1
            elif status == "ok":
                ok += 1
                if args.verbose:
                    print(f"  {patient}/{name}  ({size} bytes)")
            else:
                err += 1
                print(f"  FAILED {patient}/{name}: {status}")

        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] ok={ok} skip={skip} err={err}", flush=True)

    print(f"\nDone. downloaded={ok} skipped={skip} failed={err}")

    if err:
        # Re-run to retry failures (resume picks up where it left off).
        print("Re-run the script to retry failed files.")


if __name__ == "__main__":
    main()
