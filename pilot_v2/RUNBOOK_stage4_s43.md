# RUNBOOK — Stage 4.3 (בוצע; מסמך שחזור)

שלב `S4.3` בוצע במלואו על ידי דניאלה סימונובסקי ב־Google Colab (25–26 בספטמבר 2026) ושולב
בריפו ב־26 בספטמבר. מסמך זה מתאר מה הורץ, היכן כל דבר יושב, ואיך להריץ מחדש על האשכול.
דוחות: `חומר כתוב/Stage4_3_Methodology_and_Results.pdf` (מלא), `Stage4_3_Summary.pdf` (סיכום).
ביקורת עצמאית: `results/stage4_s43_review_20260926/S43_review.md`.

## 1. מה הורץ

| שלב | קלט קפוא | קוד | פלט | סטטוס |
|---|---|---|---|---|
| smoke על זוג אחד | `results/stage4_s43_inputs_v1/` | (סקריפט לא נכלל) | `results/stage4_s43_all_v1/smoke_v1/smoke_report.json` | עבר |
| סינון ראשוני, `305` תצורות, `40` זוגות, noising | `results/stage4_s43_inputs_v1/` | `s43_run_v2.py` | `results/stage4_s43_all_v1/initial_v2/` (`12,200` רשומות) | הושלם |
| ניתוח הסינון ובחירת `16` ניגודים + `11` prerequisites | — | (קוד לא נכלל; הכלל: `τ=0.10`, coherent / heterogeneous, מכסת `16`) | `results/stage4_s43_all_v1/analysis_v1/`, `results/stage4_s43_extension_inputs_v1/selection.json` | הושלם |
| הרחבה דו־כיוונית, `27` תצורות, `178` זוגות | `results/stage4_s43_extension_inputs_v1/` + `analysis_v1/` + `diagnostics/` | `s43_extension_run_v3.py` | `results/stage4_s43_all_v1/extension_v4_family034_final/` (`9,612` רשומות) | הושלם |
| ניתוח ההרחבה | — | (קוד לא נכלל; שוחזר ב־`verify_s43.py`) | `results/stage4_s43_all_v1/extension_analysis_v1/` | הושלם |

ריצות שלא הושלמו, נשמרו לתיעוד בלבד: `initial_v1` (`26/40` זוגות, הופסקה; ה־chunks שלה
לא הועתקו), `extension_v1` ו־`extension_v3_original_80gb` (נכשלו בשער ההתאימות של משפחה `034`;
manifests בלבד).

זהויות ריצות הייצור: סינון `a971c12f…` (`initial_v2/run_manifest.json`), הרחבה
`1f6d18be…` (`extension_v4_family034_final/run_manifest.json`).

## 2. הקוד

- `pilot_v2/s43_engine.py` — `S43Engine`, יורש מ־`S42Engine` → `Stage4Engine` → `Stage3Engine`.
  שחרור MLP/בלוקים בתוך hybrid קפוא, ראשי־ביניים "חיים" (chain), comparator ישיר.
- `pilot_v2/s43_run_v2.py` — הרץ של הסינון הראשוני (גרסת הייצור; בדיקת התאימות ההיסטורית לפי margin, סבילות `0.05`).
- `pilot_v2/s43_extension_run_v3.py` — הרץ ההרחבה (מייבא את `s43_run_v2` כבסיס); מיישם את החריג של משפחה `034`.
- `pilot_v2/test_s43.py` — שמונה מבחני tiny gate; הרצים מפעילים אותם אוטומטית לפני טעינת המשקולות.
- `pilot_v2/s43_superseded/` — `s43_run.py`, `s43_extension_run.py`, `s43_extension_run_v2.py`: גרסאות שקדמו לייצור; לא להריץ.

הקוד טוען את המודל דרך `stage1_scan.load_model()` שלנו (`model_lock_olmo2.json`,
`$PILOT_STORAGE/model/<revision>`), ולכן עובד עם `runtime.sh` של האשכול ללא שינוי.
ארבעת קובצי התשתית שהוא מייבא זהים לבייט לקבצים בריפו (נבדק ב־26.9).

## 3. הרצה מחדש על האשכול (Bash, מתוך `pilot_v2/`)

```bash
source runtime.sh
# 1. סינון ראשוני (נדרש smoke_report.json קיים; ראה סעיף 5)
python s43_run_v2.py --inputs ../results/stage4_s43_inputs_v1 \
  --smoke-report ../results/stage4_s43_all_v1/smoke_v1/smoke_report.json \
  --out $PILOT_RUNS/stage4_s43_initial_rerun --audit-only     # בדיקה ללא מודל
python s43_run_v2.py --inputs ../results/stage4_s43_inputs_v1 \
  --smoke-report ../results/stage4_s43_all_v1/smoke_v1/smoke_report.json \
  --out $PILOT_RUNS/stage4_s43_initial_rerun                  # ריצה; --resume להמשך
# 2. הרחבה
python s43_extension_run_v3.py \
  --extension-inputs ../results/stage4_s43_extension_inputs_v1 \
  --frozen-inputs ../results/stage4_s43_inputs_v1 \
  --analysis-dir ../results/stage4_s43_all_v1/analysis_v1 \
  --smoke-report ../results/stage4_s43_all_v1/smoke_v1/smoke_report.json \
  --compat-diagnostic ../results/stage4_s43_all_v1/diagnostics/a100_40gb_full178_compatibility.json \
  --out $PILOT_RUNS/stage4_s43_extension_rerun --audit
```

הערות:
- הריצה המקורית: Colab, GPU יחיד (`L4` בהקפאה, `A100-40GB`/`80GB` בייצור), `torch 2.11`,
  `transformers 4.51.3`, `fp16`, `sdpa` במצב math. על האשכול (`torch 2.6`, שני GPU לעובד)
  יש לצפות להבדלי fp16 קטנים; שער ה־margin (`0.05`) הוא הבדיקה הרלוונטית. חריג משפחה `034`
  נקבע מול reference שנוצר על `A100`; על חומרה אחרת ייתכן שיידרש reference חדש.
- `--max-new-pairs N` מאפשר עצירה בטוחה אחרי `N` זוגות ו־`--resume` להמשך באותה זהות.
- אין לגעת ב־`results/stage4_s43_inputs_v1/` ו־`…_extension_inputs_v1/`: ה־hashes שלהם
  מקודדים בזהות הריצה.

## 4. ניתוח מחדש (CPU)

```bash
python results/stage4_s43_review_20260926/verify_s43.py results/stage4_s43_all_v1
```
משחזר את טבלת `16` המסלולים, סיווגי coherent/heterogeneous, התאמת סימנים, `69` המשפחות,
שחזור `common-20`, ומטבלץ את `11` ה־prerequisites. דורש `pandas`.

## 5. מה חסר לשחזור מלא

לא נכללו בחבילה (כנראה תאי Colab): בניית הקלטים הקפואים מ־`next_stage_plan.json`; סקריפט
ה־smoke; קוד הניתוח והבחירה (`analysis_v1`, `selection.json`, `extension_analysis_v1`);
סקריפט הדיאגנוסטיקה של ההתאימות. בלעדיהם הניסוי ניתן להרצה חוזרת אך לא לשינוי מתוך התוכנית.

## 6. Git ו־OneDrive

בגיט: הקוד, שתי חבילות הקלט, manifests/gates/done/state של כל ריצה, קובצי `.ok` (hash לכל
chunk), קובצי הסיכום והניתוח, הדוחות. מחוץ לגיט (OneDrive): `initial_v2/chunks/` (`40`
קבצים, כ־`23MB`) ו־`extension_v4_family034_final/chunks/` (`178` קבצים, כ־`17MB`),
`analysis_v1/pair_order_level.csv` ו־`family_level.csv`, `extension_analysis_v1/pair_order_matched.csv`
ו־`family_direction_level.csv` (טבלאות גולמיות גדולות).
