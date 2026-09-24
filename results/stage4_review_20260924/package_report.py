"""Package only the portable report assets; preserve source research results."""
from pathlib import Path
import hashlib,json,re,zipfile,shutil
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
R=Path(__file__).resolve().parent
O=ROOT/'חומר כתוב/Stage4_Overleaf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

source=(O/'main.tex').read_text(encoding='utf-8')
refs=re.findall(r'\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}',source)
for ref in refs:
    p=O/ref
    if not p.suffix:p=p.with_suffix('.tex')
    assert p.is_file(),ref
labels=set(re.findall(r'\\label\{([^}]+)\}',source))
assert set(re.findall(r'\\ref\{([^}]+)\}',source))<=labels
log=(O/'main.log').read_text(encoding='utf-8',errors='replace')
assert 'Overfull' not in log and 'undefined references' not in log.lower()
reader=PdfReader(O/'main.pdf')
assert len(reader.pages)==13
assert all('??' not in (p.extract_text() or '') for p in reader.pages)
assert len(list((O/'figures').glob('*.png')))==6
assert len(list((O/'tables').glob('*.tex')))==4

shutil.copyfile(R/'verification.json',O/'data/verification.json')
shutil.copyfile(R/'report_facts.json',O/'data/report_facts.json')
assets=[O/'main.tex',O/'README.txt']
for folder in ['figures','tables','data']:assets+=sorted((O/folder).glob('*'))
hashes={p.relative_to(O).as_posix():sha(p) for p in assets if p.is_file()}
(O/'package_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
assets.append(O/'package_hashes.json')
archive=ROOT/'חומר כתוב/Stage4_Experiment_Overleaf.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in assets:z.write(p,p.relative_to(O).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert 'main.tex' in z.namelist()
    for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
dest=ROOT/'output/pdf/Stage4_Experiment_Report.pdf';dest.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(O/'main.pdf',dest)
delivery=dict(pdf=str(dest.relative_to(ROOT)),pdf_sha256=sha(dest),pages=len(reader.pages),
    overleaf_zip=str(archive.relative_to(ROOT)),zip_sha256=sha(archive),zip_files=len(assets),
    main_tex_sha256=sha(O/'main.tex'),figures=6,tables=4,compile_engine='Tectonic/XeTeX',
    visual_review='All final pages rendered at 100 dpi and inspected; figures also inspected separately',
    overfull_boxes=0,unresolved_references=0,original_gpu_artifacts_modified=False,new_gpu_jobs=0)
(R/'delivery_verification.json').write_text(json.dumps(delivery,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in delivery.items() if k not in ['pdf','overleaf_zip']},indent=2))
