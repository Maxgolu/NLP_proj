"""Build the isolated code-only Slurm release without changing existing scripts."""
import io
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
FILES=['ri_test_audit.py','ri_test_diagnostics.py','test_ri_test_audit.py','test_ri_test_diagnostics.py','ri_test_audit.sbatch',
       'submit_ri_test_audit.sh','pack_ri_test_results.sh','verify_ri_test_results.py',
       'README_RI_TEST_RETRY.md','stage1_scan.py','stage1_audit.py','model_lock_olmo2.json']


def build():
    out=ROOT/'ri_test_audit_v2_update.tar.gz'
    with tarfile.open(out,'w:gz') as tar:
        for name in FILES:
            data=(ROOT/name).read_bytes()
            if name.endswith(('.sh','.sbatch')) and b'\r' in data:
                raise ValueError('Shell scripts must use LF: '+name)
            info=tarfile.TarInfo('ri_test_audit_v2/'+name)
            info.size=len(data)
            info.mode=0o755 if name.endswith(('.sh','.sbatch')) else 0o644
            tar.addfile(info,io.BytesIO(data))
    with tarfile.open(out,'r:gz') as tar:
        if tar.getnames()!=['ri_test_audit_v2/'+n for n in FILES]:
            raise ValueError('Archive member mismatch')
        for name in FILES:
            if tar.extractfile('ri_test_audit_v2/'+name).read()!=(ROOT/name).read_bytes():
                raise ValueError('Archive byte mismatch: '+name)
    print(f'Verified {out.name}: {len(FILES)} files, {out.stat().st_size:,} bytes.')


if __name__=='__main__':
    build()
