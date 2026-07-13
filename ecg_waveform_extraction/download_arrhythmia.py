"""Download arrhythmia-specific PhysioNet databases for training diversity."""
import os, ssl, urllib.request, time

DATA_DIR = 'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/data'
os.makedirs(DATA_DIR, exist_ok=True)

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

def download(url, fpath, retries=3):
    if os.path.exists(fpath):
        return True
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as r:
                data = r.read()
            with open(fpath, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            if i < retries - 1:
                time.sleep(3)
    return False

def download_db(name, version, records, exts=['.hea','.dat','.atr']):
    base = f"https://physionet.org/files/{name}/{version}"
    ok = 0
    for rec in records:
        for ext in exts:
            url = f"{base}/{rec}{ext}"
            path = os.path.join(DATA_DIR, rec + ext)
            download(url, path)
        if all(os.path.exists(os.path.join(DATA_DIR, rec + e)) for e in exts):
            ok += 1
    return ok

print("=" * 60)
print("1. MIT-BIH Supraventricular Arrhythmia DB (svdb)")
print("=" * 60)
# SVDB: 78 half-hour records with supraventricular arrhythmias
svdb_records = []
for r in ['800','801','802','803','804','805','806','807','808','809',
          '810','811','812','820','821','822','823','824','825','826',
          '827','828','829','840','841','842','843','844','845','846',
          '847','848','849','850','851','852','853','854','855','856',
          '857','858','859','860','861','862','863','864','865','866',
          '867','868','869','870','871','872','873','874','875','876',
          '877','878','879','880','881','882','883','884','885','886',
          '887','888','889','890','891','892','893','894']:
    svdb_records.append(r)
ok = download_db('svdb', '1.0.0', svdb_records[:20], ['.hea','.dat','.atr'])  # First 20
print(f"  Downloaded {ok}/{20} (sampling subset of 78)")

print()
print("=" * 60)
print("2. MIT-BIH Malignant Ventricular Arrhythmia DB (vfdb)")
print("=" * 60)
vfdb_records = [str(i) for i in range(418, 439)]
ok = download_db('vfdb', '1.0.0', vfdb_records[:10], ['.hea','.dat','.atr'])
print(f"  Downloaded {ok}/{10}")

print()
print("=" * 60)
print("3. MIT-BIH Atrial Fibrillation DB (afdb)")
print("=" * 60)
afdb_records = ['04015','04043','04048','04126','04746','04908','04936',
                '05091','05121','05261','06426','06453','06995','07162',
                '07859','07879','07910','08215','08219','08378','08405',
                '08434','08455']
ok = download_db('afdb', '1.0.0', afdb_records[:10], ['.hea','.dat'])
print(f"  Downloaded {ok}/{10} (AF records, no .atr needed)")

print()
print("=" * 60)
print("4. CU Ventricular Tachyarrhythmia DB (cudb)")
print("=" * 60)
cudb_records = ['cu01','cu02','cu03','cu04','cu05','cu06','cu07','cu08',
                'cu09','cu10','cu11','cu12','cu13','cu14','cu15','cu16',
                'cu17','cu18','cu19','cu20','cu21','cu22','cu23','cu24',
                'cu25','cu26','cu27','cu28','cu29','cu30','cu31','cu32',
                'cu33','cu34','cu35']
ok = download_db('cudb', '1.0.0', cudb_records[:10], ['.hea','.dat'])
print(f"  Downloaded {ok}/{10}")

print()
# Count all .hea+.dat records now
hea_count = sum(1 for f in os.listdir(DATA_DIR) if f.endswith('.hea'))
dat_count = sum(1 for f in os.listdir(DATA_DIR) if f.endswith('.dat'))
both = 0
for f in os.listdir(DATA_DIR):
    if f.endswith('.hea'):
        rec = f[:-4]
        if os.path.exists(os.path.join(DATA_DIR, rec + '.dat')):
            both += 1
print(f"Total: {hea_count} .hea files, {dat_count} .dat files")
print(f"Records with .hea+.dat: {both}")
