"""Validate and package the combined report, without modifying experimental data."""
from pathlib import Path
import hashlib, json, re, shutil, zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = ROOT / 'חומר כתוב/Stage4_Overleaf'
DEST = ROOT / 'output/pdf/Stage4_Experiment_Report.pdf'
ZIP = ROOT / 'חומר כתוב/Stage4_Experiment_Overleaf.zip'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

seen = set()
def expand(p):
    assert p.is_file(), p
    seen.add(p)
    text = p.read_text(encoding='utf-8')
    return re.sub(r'\\input\{([^}]+)\}', lambda m: expand(SRC / (m[1] + '.tex')), text)

text = expand(SRC / 'main.tex')
labels = re.findall(r'\\label\{([^}]+)\}', text)
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)
assert len(labels) == len(set(labels)), 'Duplicate labels'
assert set(refs) <= set(labels), set(refs) - set(labels)
figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text)
assert all((SRC / p).is_file() for p in figures)
log = (SRC / 'main.log').read_text(encoding='utf-8', errors='replace')
for bad in ['Overfull', 'undefined references', 'Missing character:', 'LaTeX Error']:
    assert bad not in log, bad
pdf = PdfReader(SRC / 'main.pdf')
extracted = '\n'.join(p.extract_text() for p in pdf.pages)
assert len(pdf.pages) == 27
assert '??' not in extracted and '\ufffd' not in extracted
assert len(extracted) > 60000
for heading in ['G1: joint dependence', 'G2: conditional RI', 'G3: the L26H31', '249', '305']:
    assert heading in extracted, heading
plan = json.loads((HERE / 'next_stage_plan.json').read_text())
assert len(plan['C0']) == 33 and len(plan['C50_if_C0_fails']) == 50
assert len(plan['RI23']) == 23 and len(plan['R17']) == 17
assert set(plan['RI14']) | set(plan['nonRI19']) == set(plan['C0'])
assert not set(plan['RI14']) & set(plan['nonRI19'])
assert sum(p['configs'] for p in plan['local_S43'] + plan['composed_chains']) + plan['controls']['configs'] == 249
assert plan['S43_config_cap'] + plan['S43_additional_direct_comparator_cap'] == plan['S43_scientific_registry_cap'] == 305

files = [p for p in SRC.rglob('*') if p.is_file() and (p.parts[-2] in ['figures', 'tables', 'data'] or p.name in ['main.tex', 's42_results.tex', 'README.txt'])]
manifest = {p.relative_to(SRC).as_posix(): digest(p) for p in sorted(files)}
(SRC / 'package_hashes.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
DEST.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SRC / 'main.pdf', DEST)
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files + [SRC / 'package_hashes.json']):
        z.write(p, p.relative_to(SRC).as_posix())
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    for name, expected in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
checks = dict(pdf_pages=len(pdf.pages), figures=len(figures), labels=len(labels), resolved_references=len(refs),
    packaged_files=len(files)+1, pdf_sha256=digest(DEST), zip_sha256=digest(ZIP),
    raw_archive_sha256=json.loads((ROOT/'results/stage4_s42_readiness_20260924/readiness.json').read_text())['archive_sha256'],
    visual_review='All 27 rendered pages reviewed; charts and dense tables additionally inspected at page resolution.',
    scope='CPU analysis and report only; no GPU submission or S4.3-S4.5 implementation.')
(HERE / 'report_verification.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks, indent=2))
