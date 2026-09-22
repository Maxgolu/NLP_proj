"""Build the isolated, hashed upload for the Section-1.5 readout (code + frozen plan; no weights, no results)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from stage3_common import digest,json_write
from stage3_readout import CODE_FILES,READOUT_POLICY

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,default=ROOT.parent/'results/stage3_readout_inputs_v1')
    ap.add_argument('--out',type=Path,default=ROOT/'stage3_readout_v1_update.tar.gz');a=ap.parse_args()
    plan=json.loads((a.plan/'readout_plan.json').read_text(encoding='utf-8'))
    if plan['policy']!=READOUT_POLICY:raise ValueError('Plan policy differs from the code; run prepare again')
    files={n:ROOT/n for n in CODE_FILES+['test_stage3_readout.py','submit_stage3_readout.sh','stage3_readout.sbatch','RUNBOOK_stage3_readout.md']}
    files.update({'inputs/'+p.name:p for p in a.plan.iterdir() if p.is_file()})
    hashes={}
    with tarfile.open(a.out,'w:gz') as archive:
        for name,path in sorted(files.items()):
            data=path.read_bytes()
            if path.suffix in ['.py','.sh','.sbatch','.md','.json']:data=data.replace(b'\r\n',b'\n')
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo('stage3_readout_v1/'+name);info.size=len(data);info.mode=0o755 if path.suffix=='.sh' else 0o644
            archive.addfile(info,io.BytesIO(data))
        data=(json.dumps(hashes,indent=2)+'\n').encode();info=tarfile.TarInfo('stage3_readout_v1/bundle_hashes.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    with tarfile.open(a.out) as archive:
        for name,sha in hashes.items():
            if hashlib.sha256(archive.extractfile('stage3_readout_v1/'+name).read()).hexdigest()!=sha:raise ValueError('Archive verification failed')
    json_write(a.out.with_suffix(a.out.suffix+'.sha256.json'),dict(archive=a.out.name,sha256=digest(a.out),bytes=a.out.stat().st_size,files=len(hashes),items=len(plan['items'])))
    print(f'Built and verified {a.out} ({a.out.stat().st_size:,} bytes), SHA256 {digest(a.out)}')


if __name__=='__main__':main()
