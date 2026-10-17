"""Download pinned public tables and verify their exact SHA-256 hashes."""
import argparse,concurrent.futures,hashlib,json
from pathlib import Path
import requests
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));a=p.parse_args()
a.data.mkdir(parents=True,exist_ok=True)
entries=[]
for f in ['bnab_manifest.json','ch65_manifest.json']:
 entries.extend(json.loads((BASE/'provenance'/f).read_text()))
def get(x):
 dest=a.data/x['file']
 if not dest.exists():
  r=requests.get(x['url'],timeout=120);r.raise_for_status();dest.write_bytes(r.content)
 actual=hashlib.sha256(dest.read_bytes()).hexdigest()
 if actual!=x['sha256']:raise ValueError(f'Hash mismatch: {dest}')
 return x['file']
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for f in ex.map(get,entries):print('Verified',f,flush=True)
(a.data/'concentration_grids.json').write_text((BASE/'provenance/concentration_grids.json').read_text())
print('All source tables verified. About 830 MB of source data are retained locally.')
