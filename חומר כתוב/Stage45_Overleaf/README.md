# S45 discovery report — Overleaf package

Upload this directory as a new Overleaf project (or upload the supplied ZIP).
Set `main.tex` as the main document and use pdfLaTeX. All figures are local vector
PDFs; there is no bibliography or external asset dependency.

This is a standalone results-and-analysis report, not a replacement for the
main paper's Results section. It covers the completed discovery experiment only.
Held-out validation is not included.

Contents:
- `main.tex`: full audited report, interpretation corrections and conclusions.
- `tables/`: six numerical tables generated from checked exports.
- `figures/`: three two-panel vector PDF figures.

The source analysis and frozen run files were not edited. Reproduction scripts,
check results, recomputed contrasts and a Hebrew review are in
`results/stage4_s45_review_20260927/` relative to the project root.

Verification: numeric consistency, references, input files and TeX environments
were checked locally. Figures were visually inspected. Local compilation was
blocked by Windows Application Control for the available TeX executable, so the
complete document's pagination must be checked in Overleaf.

Python reproduction (from the project root, with numpy, pandas and matplotlib):
```
python results/stage4_s45_review_20260927/audit_s45.py --figures
python results/stage4_s45_review_20260927/build_tables.py
```
