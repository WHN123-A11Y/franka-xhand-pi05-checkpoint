#!/usr/bin/env python3
"""Download, verify and restore the pinned step-1500 + Student/K4 release."""
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, tempfile, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def verify_lines(manifest,base):
 for line in manifest.splitlines():
  if not line.strip():continue
  digest,name=line.split('  ',1);p=Path(name)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
  f=base/p
  if not f.is_file() or sha(f)!=digest:raise ValueError('Missing or corrupt file: '+str(f))

def asset(a,cache):
 dst=cache/a['name']
 if dst.is_file() and dst.stat().st_size==a['size'] and sha(dst)==a['sha256']:
  print('Verified cache:',a['name'],flush=True);return dst
 partial=dst.with_name(dst.name+'.download')
 if partial.exists() and partial.stat().st_size>=a['size']:partial.unlink()
 subprocess.run(['curl','--fail','--location','--retry','5','--retry-all-errors','--connect-timeout','30','--continue-at','-','--silent','--show-error','--output',str(partial),a['url']],check=True)
 if partial.stat().st_size!=a['size'] or sha(partial)!=a['sha256']:raise ValueError('Downloaded asset failed verification: '+a['name'])
 partial.replace(dst);print('Downloaded and verified:',a['name'],flush=True);return dst

def restore(cache):
 # Refuse to merge or overwrite any existing user-owned model tree.
 targets=[ROOT/'models/pi05_selected_1500',ROOT/'models/student',ROOT/'models/k4',ROOT/'assets/openpi_cache/big_vision/paligemma_tokenizer.model']
 if any(p.exists() for p in targets):raise ValueError('Model assets already exist. Use check_assets.py --weights-only --full-policy-hash to verify; restoration refuses overwrite.')
 (ROOT/'models').mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.paired-restore-',dir=ROOT/'models') as tmp:
  stage=Path(tmp)
  with zipfile.ZipFile(cache/'paired_assets.zip') as z:
   for name in z.namelist():
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe ZIP path')
   bad=z.testzip()
   if bad:raise ValueError('Corrupt ZIP entry: '+bad)
   z.extractall(stage)
  verify_lines((stage/'PAIRED_ASSETS_SHA256.txt').read_text(),stage)
  # Stream the exact seven parts; extra cache files cannot enter the archive.
  parts=[str(cache/f'pi05_selected_1500.tar.zst.part-{i:02d}') for i in range(7)]
  joined=subprocess.Popen(['cat',*parts],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  decomp=subprocess.Popen(['zstd','-dc'],stdin=joined.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  joined.stdout.close()
  unpack=subprocess.run(['tar','-xf','-','-C',str(stage/'models')],stdin=decomp.stdout)
  decomp.stdout.close();stderr=decomp.stderr.read();decomp.wait()
  join_error=joined.stderr.read();joined.wait()
  if joined.returncode:raise RuntimeError('Reading archive parts failed: '+join_error.decode(errors='replace'))
  if unpack.returncode or decomp.returncode:raise RuntimeError('Checkpoint extraction failed: '+stderr.decode(errors='replace'))
  verify_lines((ROOT/'selected_files_sha256.txt').read_text(),stage/'models/pi05_selected_1500')
  for rel in ['models/pi05_selected_1500','models/student','models/k4','assets/openpi_cache/big_vision/paligemma_tokenizer.model']:
   dst=ROOT/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(stage/rel),str(dst))
  # Carry the original tokenizer license/notice supplied with these assets.
  licenses=ROOT/'licenses';licenses.mkdir(exist_ok=True)
  for f in (stage/'licenses').glob('*'):
   dst=licenses/f.name
   if dst.exists() and dst.read_bytes()!=f.read_bytes():raise ValueError('Existing license differs: '+str(dst))
   if not dst.exists():shutil.copy2(f,dst)
  shutil.copy2(stage/'PAIRED_ASSETS_SHA256.txt',ROOT/'PAIRED_ASSETS_SHA256.txt')
 print('Restored: baseline step 1500 + Student step 2000 + K4 step 12273 + tokenizer + normalization.')
 print('Recorded camera samples and reference predictions are supplied separately by the research team.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',type=Path,default=ROOT/'weights');p.add_argument('--download-only',action='store_true');p.add_argument('--verify-cache-only',action='store_true');args=p.parse_args()
 m=json.loads((ROOT/'assets/paired_release_manifest.json').read_text());cache=args.cache_dir.resolve();cache.mkdir(parents=True,exist_ok=True)
 for tool in (('cat','zstd','tar') if args.verify_cache_only else ('curl','cat','zstd','tar')):
  if not shutil.which(tool):raise SystemExit('Required tool missing: '+tool)
 if args.verify_cache_only:
  for a in m['assets']:
   f=cache/a['name']
   if not f.is_file() or f.stat().st_size!=a['size'] or sha(f)!=a['sha256']:raise ValueError('Cache check failed: '+a['name'])
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda a:asset(a,cache),m['assets']))
 if not args.download_only:restore(cache)
if __name__=='__main__':main()
