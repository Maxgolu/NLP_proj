# Stage 4.5 — השתתפות RI במבנים שנמדדו: הוראות הרצה (CPU, GPU, Slurm, Git)

מימוש של `חומר כתוב/RI_Structure_Final_Protocol_20260926.md` (המפרט מ־26.9.2026).
**טרם הורץ על מודל המחקר.** כל מה שרץ עד כה הוא בדיקות מימוש על מודל OLMo2 אקראי קטן (CPU);
אלה אינן תוצאות מחקריות. החבילה מבודדת בתיקייה `pilot_v2/s45/`; אין לשנות קוד בתוך ריצה קיימת
(זהות הקוד נרשמת ב־manifest של כל worker, ו־`--resume` מסרב לקוד שהשתנה).

## 1. מה יש בחבילה

| קובץ | תפקיד |
|---|---|
| `s45_plan.py` | רשימות קפואות (T1–T5, C33, C50, שישה מועמדים, ייחוסים, חיבורים ובקרות), בניית ארבעת התאים `x00/x10/x01/x11` עם **בנייה מחדש מלאה** של כל השדות הנגזרים, מצבי רקע (`state`), לוחות זמנים לכל שלב, חתימת קלטים, איטום held-out. |
| `s45_means.py` | ממוצעי תפקיד לכל 1,024 הראשים; שני מצבים: `lofo` (discovery, הורדת המשפחה הנבדקת) ו־`full` (held-out, בנק קבוע ללא חיסור/הוספה). |
| `s45_engine.py` | שכבת רקע מעל `S43Engine`: hook ראשון על כל `o_proj` (קדימות: המסכות מגדירות את B; החלפת מקור/קליטה דורסות רק את הקואורדינטות שלהן); כל רכיבי מסלול (recipient, donor, hybrid, endpoint) נבנים בתוך אותו רקע. |
| `s45_run.py` | worker: `prepare`, `prepare-heldout`, `check`, `means`, `run`; שערים על המודל האמיתי; chunks עם `.ok`; reuse של רשומות זהות; חריג תאימות משפחה 034 מפורש. |
| `s45_analyze.py` | אגרגציה משפחתית (ממוצע שני הסדרים ואז משפחות; כל סדר בנפרד; הפרשי סדר מוחלטים), שומרי F/L/דיוק, `d_b`, מטריצת Γ 6×5, בחירה דטרמיניסטית, Γ דו־כיווני, חיבור מול בקרה, רגישות baseline, תוויות (a)–(e). bootstrap: 20,000 דגימות משפחתיות מזווגות, seed 20260926, תיאורי בלבד. |
| `s45_freeze.py` | manifest הקפאה חתום: שלבי discovery שהושלמו, זהויות B/מועמדים/בקרות/בנק, גרסאות קוד ונתונים, מפרט validation, סימנים ראשיים, ספים. **הוא הדבר היחיד שפותח את 87 המשפחות החתומות.** |
| `s45_pipeline.py`, `s45_pipeline.sbatch`, `submit_s45_pipeline.sh`, `s45_submit.py` | הקצאת Slurm אחת, שלוש רפליקות של זוג GPU, החלטות CPU קפואות בין שלבים; מצב `discovery` ומצב `heldout` הם **שתי הגשות נפרדות**. |
| `test_s45.py` | שערי מודל קטן (רצים אוטומטית בתחילת כל worker): all-live משחזר full; self-donor זהות בכל סוג רקע; אפס מבני ל־Q חוצה־מיקום ללא מתווכים; שינוי שאלה לא משנה מיקומים קודמים; מפתחות ממוצע ללא זהות/רלוונטיות; קולט בקרה מושתק אינו מתפרש כאפס; שער manifest ההקפאה. |
| `test_s45_analysis.py` | בדיקות CPU של נוסחאות הניתוח על רשומות סינתטיות עם תשובות ידועות: g/F/L/דיוק ובחירת B (A), `d_b` וכלל השימור (B), מטריצת Γ, זכאות ובחירה דטרמיניסטית (C), Γ דו־כיווני וחיבור מול בקרה (D), תוויות (a)–(e) (validation). |
| `test_s45_pipeline.py` | הרצה מקצה לקצה על מודל קטן: Stage D עם בחירה כפויה (מסלולים בשני הכיוונים, חיבור+בקרה ברקע האבחוני), הקפאה, איטום held-out סינתטי, worker במצב `full`, תוויות סופיות. איטי (דקות); להריץ ידנית. |
| `stage*_*.py`, `s42_*.py`, `s43_engine.py`, `test_s42.py`, `test_s43.py`, `test_stage4.py`, `model_lock_olmo2.json` | עותקים בייט־זהים של התלויות הקפואות (S4.1–S4.3). |

## 2. העלאה לאשכול (scp של חבילה, כמו בשלבים הקודמים)

הקוד באשכול **לא** מתעדכן מ־git; מעלים חבילה. החבילה `pilot_v2/s45_bundle_v1.tar.gz` (עם
`s45_bundle_v1.tar.gz.sha256.json`) כוללת את הקוד, את הקלטים הקפואים `s45/inputs_s45/` (כבר הוכנו
מקומית עם הטוקנייזר הנעול; ראה §3) ואת `s45/reference/` — הדאטה הגולמי, תוצאות ריצת הבסיס, אבחון
התאימות של S4.3 ו־`candidates.json` — כך שאיטום ה־held-out לא תלוי בקבצים שאולי חסרים באשכול.

PowerShell משורש הפרויקט, החיבור שכבר שימש בפרויקט:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\s45_bundle_v1.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

בשרת, Bash (CPU בלבד, כדקה):

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
test ! -e s45 || { echo 'Package already exists: use a new bundle version instead of overwriting'; exit 1; }
sha256sum s45_bundle_v1.tar.gz     # להשוות ל־s45_bundle_v1.tar.gz.sha256.json
tar -xzf s45_bundle_v1.tar.gz
source runtime.sh
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('s45')
for name,expected in json.loads((p/'bundle_hashes.json').read_text()).items():
    if name=='bundle_hashes.json':continue
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
print('Bundle verified')
PY
cd s45 && python3 -m unittest -q test_s45 test_s45_analysis test_s43 test_s42.TinyJoint test_stage4.TinyRoutes && cd ..
python3 s45/s45_run.py check --inputs s45/inputs_s45
```

צפוי: `Bundle verified`, `OK`, ואז `{'pairs': 40, 'states': 8, 'jobs': 8, 'records': 1280}`.
על שרת ה־GPU יש לוודא שבדיקות המודל הקטן אינן מדולגות (torch/transformers קיימים דרך `runtime.sh`).

## 3. הקלטים הקפואים (כבר בוצע; לתיעוד ולשחזור בלבד)

`inputs_s45/` נוצר מקומית ב־26.9 בפקודה הבאה (7 שניות, CPU), מתוך `s45/`:

```bash
python3 s45_run.py prepare \
  --stage3-inputs ../../results/stage3_inputs_v1 \
  --s42-inputs    ../../results/stage4_s42_inputs_v1 \
  --candidates    reference/ri_test_v2_candidates.json \
  --dataset       reference/singlehop_v1_4shot.jsonl \
  --baseline      reference/olmo2_singlehop_4shot_results.jsonl \
  --compatibility reference/a100_40gb_full178_compatibility.json \
  --tokenizer     ../../pilot_v3/olmo2_tokenizer/tokenizer.json \
  --out inputs_s45
```

כל שערי השחזור ההיסטורי עברו (token_ids, offsets, מסכות, `mean_keys`, קידומת — בייט־בבייט מול S4.2),
89 משפחות, ליבה 20, 108 מפתחות ממוצע, חריג משפחה 034 נטען, ו־87 משפחות validation מזוהות מהבסיס
ההתנהגותי. עותק של הקלטים נמצא גם ב־`results/stage4_s45_inputs_v1/` בריפו. אין להריץ `prepare` שוב
לאותה תיקייה (מסרב לדרוס); גרסה חדשה = תיקייה חדשה + חבילה חדשה.

מה `prepare` עושה: משחזר את 178 זוגות ה־discovery מהקלטים הקפואים של S4.2 (מסכות, `mean_keys`, baselines
שמורים), **מאמת** שהטוקניזציה וההערות ההיסטוריות משוחזרות בייט־בבייט, ובונה לכל זוג ארבעה תאים:

- `x00` עובדות מקוריות/שאלה מקורית (זהב a), `x10` אימהות מוחלפות/שאלה מקורית (זהב b),
- `x01` עובדות מקוריות/שאלה חלופית (זהב b), `x11` מוחלפות/חלופית (זהב a).

לכל תא נבנים מחדש: `query_source/question_entity`, `gold`, `prompt`, `token_ids`, `offsets`, `tokens`,
מסכות תפקיד (כולל `all_sentence` = ארבעת משפטי העובדות ללא שאלה/נקודתיים), `mean_keys`, `input_ids`
עם הקידומת המשותפת, `n`, `tok_a/tok_b`, וזהות התא. נבדק שהזוגות `x00/x10` ו־`x01/x11` מיושרים
(אורך זהה, ≤6 טוקנים שונים, מסכות ומפתחות זהים, קידומת זהה), ושהשאלה החדשה מזיזה את העובדה הנשאלת.
המדד בכל התאים: `logit(a) − logit(b)` בטוקן התשובה הראשון השונה, עם הקידומת המשותפת ב־teacher forcing.
`--compatibility` מייבא את חריג משפחה 034 של S4.3 כערך ייחוס חלופי מפורש (סובלנות 1e-3 מול הערך
הדטרמיניסטי מהאבחון; לתא המוחלף סובלנות 0.10) — אין הרפיה אוטומטית של הסף 0.05.

נוצרים: `plan.json` (כולל `control_rosters` — קולט בקרה אחד לכל חיבור אפשרי, `Random(20260926)` על
רשימת ראשים ממוינת, בעיבוד מפתחות בסדר לקסיקלי, לפני כל מדידה), `structure_registry.json`,
`candidate_registry.json`, `stage_a.json`, `stage_a_c50.json`, `pairs.jsonl.gz`.

## 4. הרצת discovery (הגשה אחת)

```bash
bash s45/submit_s45_pipeline.sh --mode discovery --name s45_discovery_v1 --gpus 6
```

הרצף בתוך ההקצאה: **means** (356 forwards, כל 1,024 הראשים, x00/x10 בשני הסדרים, 89 משפחות) →
**gate** (רפליקה אחת, משפחות השער `[0,16,168]`) → **Stage A** (full/empty/T1–T5/C33 × 4 תאים × 40 זוגות)
→ ניתוח שומרים; אם C33 נכשל: **Stage A C50** → בחירת B (`B_decision.json`) → **Stage B**
(B±h לשישה מועמדים, B−L18H19, B−L8H15, B−R(B); B עצמו נעשה reuse) → **Stage C** (7 מצבי B × 5 עוגנים × 40,
כיוון noising, + 5 עוגנים במודל המלא כרגרסיה מול הממוצעים ההיסטוריים) → `frozen_selection.json` →
**Stage D** (לזוגות שנבחרו: restoration בשני מצבי B; חיבור מוצהר + בקרה, ברקע אבחוני `B+h+control`,
שני הכיוונים) → **full discovery** (178 זוגות, ציר העובדות: full/empty/B/B−R(B)/מצבי המועמדים שנבחרו
בממוצעים; B/B−R(B)/מועמדים שנבחרו ב־paired-donor) → **freeze_manifest.json** + `ri_membership_ledger.csv`.

שערים שעוצרים את הרצף (fail closed): baseline שמור (0.05 או חריג תאימות מפורש), instrumentation
אינרטי, שחזור SDPA, אינווריאנטיות קידומת בשינוי שאלה, all-live≡full, self-donor זהות בכל סוג רקע
(C33, C33−h, C33+h, אבחוני), אפס מבני ל־Q חוצה־מיקום, קולט בקרה מושתק = אפס (מתועד),
וסטיית עוגן במודל המלא > 0.05 מהממוצע ההיסטורי (T1 +1.635, T2 +0.172, T3 +0.876, T4 +0.465, T5 −0.236).
במקרה האחרון יש לבדוק את `analysis/stage_c/frozen_selection.json` → `full_background_regression`
(סטייה לכל עוגן), ורק אז `S45_ALLOW_ANCHOR_DRIFT=1` ו־`--resume`.

אומדן עומס (forwards, לא רק endpoints): means 356; A ≤1,440; B 1,440; C 3,360 (+480 רגרסיה);
D ≤1,520; full discovery ≤4,628 (+356 לכידות donor). סה"כ discovery ≈ 13.5k forwards על 3 רפליקות —
לפי S4.2/S4.3 צפוי כשעתיים–שלוש. מגבלת הזמן המבוקשת 24 שעות (או MaxTime של המחיצה).

ניטור:

```bash
source runtime.sh
cat "$PILOT_RUNS/s45_discovery_v1/pipeline_state.json"
tail -n 5 "$PILOT_RUNS/s45_discovery_v1/logs/"*.log
```

`--resume` (אותו שם) ממשיך מ־chunks שנשמרו; החלטות אדפטיביות מחושבות מחדש ומושוות ללוחות הזמנים
הקפואים — שינוי מזוהה עוצר.

## 5. איטום ופתיחת held-out (הגשה שנייה, רק אחרי הקפאה)

```bash
source runtime.sh
python3 s45/s45_run.py prepare-heldout \
  --inputs   s45/inputs_s45 \
  --freeze   "$PILOT_RUNS/s45_discovery_v1/freeze_manifest.json" \
  --dataset  s45/reference/singlehop_v1_4shot.jsonl \
  --baseline s45/reference/olmo2_singlehop_4shot_results.jsonl \
  --out s45/inputs_s45_heldout
bash s45/submit_s45_pipeline.sh --mode heldout --name s45_heldout_v1 --gpus 6 \
  --freeze    "$PILOT_RUNS/s45_discovery_v1/freeze_manifest.json" \
  --bank      "$PILOT_RUNS/s45_discovery_v1/mean_bank" \
  --discovery "$PILOT_RUNS/s45_discovery_v1"
```

`prepare-heldout` מסרב בלי manifest שמאמת (חתימה, כל שלבי discovery הושלמו, זהויות, קוד). הוא בונה
87 משפחות validation (סט העבודה מהבסיס ההתנהגותי, ללא חפיפה ל־89) → 174 זוגות × 4 תאים, וכותב
`validation.json` קפוא: ממוצעים בארבעה תאים (≤7 תצורות), Γ לזוגות שנבחרו בשני מצבי B ובשני הכיוונים,
חיבורים+בקרות ברקעים האבחוניים הקפואים, ו־paired-donor (≤5). הבנק במצב `full` (מסרב למשפחות discovery).
אומדן: ≈ 18k forwards.

## 6. תוצרים

ב־`$PILOT_RUNS/<name>/`: `B_decision.json`, `frozen_selection.json`, `freeze_manifest.json`,
`ri_membership_ledger.csv`, `discovery_signs.json`, ותחת `analysis/`: `stage_a/core_behavior.csv`,
`stage_b/stage_b_summary.json`, `stage_c/core_route_effects.csv`, `stage_c/candidate_structure_matrix.csv`,
`stage_d/core_bidirectional_gamma.csv`, `stage_d/attachment_controls.csv`,
`full_discovery/extension_behavior.csv`, ובריצת held-out: `validation/validation_behavior.csv`,
`validation_labels.csv`, `validation_summary.json`. הקלטים הקפואים (`plan.json`, `structure_registry.json`,
`candidate_registry.json`, `mean_bank/mean_bank_manifest.json`) נמצאים בחבילה/בריצה.

כל רשומה נושאת: זוג/תא/סדר, מצב רקע (hash + תווית), baseline (mean/donor) והאש הבנק, מקור/אתר/קולט/ערוץ/
ראשים חיים/כיוון/סטטוס בקרה, שני הלוגיטים, margin, טוקן־על, נורמות ההחלפה וההזרקה, ומקור (`new`/`prior`).
לכידות ביניים אינן נשמרות לדיסק (מאות MB לכל מצב); הן משוחזרות דטרמיניסטית ומשותפות בתוך תהליך לכל (זוג, מצב).

## 7. הגבלות מובנות בקוד

- לא יותר משלושה זוגות נבחרים/שלושה מועמדים, שישה מועמדים, חמישה עוגנים; אין חיפוש מחודש אחרי validation.
- `load()` מסרב למסלול שמקורו/קולטו/מתווכיו אינם חיים ברקע שלו — בקרה מושתקת לעולם אינה "אפס".
- `MeanBank('full')` מסרב למשפחת discovery; `MeanBank('lofo')` מסרב למשפחה מחוץ לבנק.
- T2 נמדד רק בענף child-last→L18H18 (העוגן הראשי); T3 מקונן ב־T1; מקטעי השרשרת ללא מתווכים הם אפס מבני
  ונבדקים בשער, לא מדווחים כ"תוספת".
- `S45_TINY_MODEL` קיים לבדיקות בלבד; ה־sbatch מבטל אותו במפורש, וה־manifest של כל worker מציין `tiny_model`.

## 8. Git

```bash
git add pilot_v2/s45/*.py pilot_v2/s45/*.sh pilot_v2/s45/*.sbatch pilot_v2/s45/*.md pilot_v2/s45/model_lock_olmo2.json
git add pilot_v2/s45_bundle_v1.tar.gz.sha256.json results/stage4_s45_inputs_v1
git commit -m "S4.5: RI participation in measured structures (implementation, frozen inputs; not yet executed)"
```

הארכיון `s45_bundle_v1.tar.gz` עצמו אינו ב־git (כמו קודמיו); רק ה־sha256 שלו.

תוצאות ריצה (`$PILOT_RUNS/...`) אינן בריפו; לאחר ריצה — `results/stage4_s45_*` לפי הנוהל של S4.2/S4.3.
