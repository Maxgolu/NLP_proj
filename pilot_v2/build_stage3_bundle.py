"""Build an isolated, hashed upload; no model weights, old results, packages or secrets."""
import argparse
import io
import json
from pathlib import Path
import tarfile
from stage3_common import digest,json_write
from stage3_run import validate_inputs,CODE_FILES

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,default=ROOT.parent/'results/stage3_inputs_v1')
    ap.add_argument('--out',type=Path,default=ROOT/'stage3_v1_update.tar.gz');a=ap.parse_args()
    plan,_,_,work=validate_inputs(a.inputs)
    files={n:ROOT/n for n in CODE_FILES+['test_stage3.py','test_stage3_analysis.py','submit_stage3.sh','stage3.sbatch','RUNBOOK_stage3_v1.md']}
    files.update({'inputs/'+p.name:p for p in a.inputs.iterdir() if p.is_file()})
    files['methodology.tex']=ROOT.parent/'חומר כתוב/Stage3_Head_Level_Characterization_revised.tex'
    hashes={};a.out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.out,'w:gz') as archive:
        for name,path in sorted(files.items()):
            data=path.read_bytes()
            if path.suffix in ['.py','.sh','.sbatch','.md','.tex']:data=data.replace(b'\r\n',b'\n')
            import hashlib
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo('stage3_v1/'+name);info.size=len(data);info.mode=0o755 if path.suffix=='.sh' else 0o644
            archive.addfile(info,io.BytesIO(data))
        data=(json.dumps(hashes,indent=2)+'\n').encode();info=tarfile.TarInfo('stage3_v1/bundle_hashes.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    # Verify the archive itself, including names and bytes, before reporting ready.
    with tarfile.open(a.out) as archive:
        for name,sha in hashes.items():
            if hashlib.sha256(archive.extractfile('stage3_v1/'+name).read()).hexdigest()!=sha:raise ValueError('Archive verification failed')
    json_write(a.out.with_suffix(a.out.suffix+'.sha256.json'),dict(archive=a.out.name,sha256=digest(a.out),bytes=a.out.stat().st_size,
               files=len(hashes),heads=len(plan['heads']),workload=work))
    print(f'Built and verified {a.out} ({a.out.stat().st_size:,} bytes), SHA256 {digest(a.out)}')


if __name__=='__main__':main()
