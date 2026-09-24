"""Package only executable code and frozen discovery inputs; no runbook or weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from stage3_common import digest,json_write
from s42_plan import load
from s42_run import CODE_FILES

ROOT=Path(__file__).resolve().parent

def build(inputs,out):
    load(inputs,inputs/'initial.json')
    names=CODE_FILES+['s42_pipeline.sbatch','submit_s42_pipeline.sh']
    files={n:ROOT/n for n in names}
    files.update({'inputs/'+n:inputs/n for n in ['plan.json','pairs.jsonl.gz','initial.json']})
    hashes={}
    with tarfile.open(out,'w:gz') as tar:
        for name,path in sorted(files.items()):
            data=path.read_bytes()
            if path.suffix in ['.py','.sh','.sbatch']:data=data.replace(b'\r\n',b'\n')
            hashes[name]=hashlib.sha256(data).hexdigest();info=tarfile.TarInfo('stage4_s42_v1/'+name)
            info.size=len(data);info.mode=0o755 if path.suffix=='.sh' else 0o644;tar.addfile(info,io.BytesIO(data))
        data=(json.dumps(hashes,indent=2)+'\n').encode();info=tarfile.TarInfo('stage4_s42_v1/bundle_hashes.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    with tarfile.open(out) as tar:
        for name,sha in hashes.items():
            if hashlib.sha256(tar.extractfile('stage4_s42_v1/'+name).read()).hexdigest()!=sha:raise ValueError('Archive verification')
    json_write(out.with_suffix(out.suffix+'.sha256.json'),dict(archive=out.name,sha256=digest(out),bytes=out.stat().st_size,files=len(hashes)))
    print('Built and verified',out,'bytes',out.stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=ROOT.parent/'results/stage4_s42_inputs_v1')
    p.add_argument('--out',type=Path,default=ROOT/'stage4_s42_v1_update.tar.gz');a=p.parse_args();build(a.inputs,a.out)
