"""Create an isolated Slurm update: sources + frozen discovery inputs, no model weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from stage3_common import digest,json_write
from stage4_plan import load
from stage4_run import CODE_FILES

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=ROOT.parent/'results/stage4_inputs_v1')
    p.add_argument('--out',type=Path,default=ROOT/'stage4_s41_v2_update.tar.gz');a=p.parse_args()
    load(a.inputs,a.inputs/'coverage.json');load(a.inputs,a.inputs/'seed.json')
    files={n:ROOT/n for n in CODE_FILES+['test_stage4.py','submit_stage4.sh','stage4.sbatch','RUNBOOK_stage4_s41.md',
             'stage4_pipeline.py','stage4_pipeline.sbatch','submit_stage4_pipeline.sh','test_stage4_pipeline.py']}
    files.update({'inputs/'+name:a.inputs/name for name in ['plan.json','pairs.jsonl.gz','coverage.json','seed.json']})
    hashes={}
    with tarfile.open(a.out,'w:gz') as tar:
        for name,path in sorted(files.items()):
            data=path.read_bytes()
            # Plan/schedule JSON bytes are frozen and must NOT be newline-normalized.
            if path.suffix in ['.sh','.sbatch','.py','.md']:data=data.replace(b'\r\n',b'\n')
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo('stage4_s41_v2/'+name);info.size=len(data);info.mode=0o755 if path.suffix=='.sh' else 0o644
            tar.addfile(info,io.BytesIO(data))
        data=(json.dumps(hashes,indent=2)+'\n').encode();info=tarfile.TarInfo('stage4_s41_v2/bundle_hashes.json');info.size=len(data)
        tar.addfile(info,io.BytesIO(data))
    with tarfile.open(a.out) as tar:
        for name,sha in hashes.items():
            assert hashlib.sha256(tar.extractfile('stage4_s41_v2/'+name).read()).hexdigest()==sha
    json_write(a.out.with_suffix(a.out.suffix+'.sha256.json'),dict(archive=a.out.name,sha256=digest(a.out),bytes=a.out.stat().st_size,files=len(hashes)))
    print('Built and verified',a.out,'bytes',a.out.stat().st_size)

if __name__=='__main__':main()
