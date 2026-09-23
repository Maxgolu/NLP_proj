# Stage 4.1 — מיפוי מסלולים: הוראות CPU, GPU, Slurm ו־Git

מימוש לתכנון `Stage4_Protocol.tex`, גרסה 2. **טרם הורץ על מודל המחקר ב־GPU.**
בדיקות CPU על מודל OLMo2 קטן הן בדיקות מימוש, לא תוצאות מחקריות.
החבילה מבודדת בשם `stage4_s41_v2`; אין לשנות קוד בתוך ריצה קיימת.

## הרצה מומלצת: כל S4.1 בפקודה אחת

אין צורך לשלוח ידנית כל שלב. גרסה 2 מוסיפה מנהל רצף אוטומטי:
**gate → Coverage → לוקליזציה אם נדרשת → P1–P4 → refinement → הרחבות נבחרות → ניתוח CPU**.
הוא ממתין לסיום כל הרפליקות ולבדיקת שלמות לפני המעבר. כישלון gate או worker עוצר את הרצף;
היעדר מסלולים שעברו את כלל הבחירה נרשם כתוצאה ללא הרחבות, ולא כתקלה.
ההרצה מסתיימת ב־S4.1; אינה מפעילה G1–G3 או את S4.3–S4.5 שטרם מומשו.

לאחר העלאת וחילוץ **החבילה החדשה** לפי סעיף 2 (שמור חבילה/ריצות v1 ללא שינוי):

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
bash stage4_s41_v2/submit_stage4_pipeline.sh --name stage4_all_v2 --gpus 6
```

זו הקצאת Slurm אחת על node אחד: שלוש רפליקות עצמאיות, כל אחת על זוג GPU נפרד.
שער הפתיחה רץ ברפליקה אחת; מדידות הכיסוי והמיפוי מתחלקות בין שלושתן.
CPU selection/analysis רצים בתוך אותה הקצאה. המשקולות נטענות מחדש בתחילת שלב worker,
לא לכל התערבות. ברירת המחדל היא 12 ליבות CPU ו־144GB RAM; אפשר להתאים `--mem`.
אפשר לבחור `--gpus 4` או `--gpus 2`; 6 GPU על node אחד עשויים לדרוש המתנה ארוכה יותר בתור.

מגבלת הזמן המבוקשת היא **48 שעות**, מוגבלת אוטומטית ל־MaxTime של המחיצה אם הוא נמוך יותר.
הסקריפט מדפיס את המגבלה שנבחרה. אפשר להעביר `--time 2-00:00:00` במפורש אם המחיצה מאפשרת זאת.
מגבלות QOS/חשבון נוספות עדיין עשויות לגרום ל־Slurm לדחות בקשה; במקרה כזה מתקבלת השגיאה המקורית.
מחיצת studentkillable יכולה לבצע preemption גם לפני תום הזמן; אין הבטחה לריצה בלתי מופרעת.

ניטור:

```bash
source runtime.sh
cat "$PILOT_RUNS/stage4_all_v2/pipeline_state.json"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,NodeList
```

חידוש באותה תצורת GPU ובאותה גרסת קוד:

```bash
bash stage4_s41_v2/submit_stage4_pipeline.sh --name stage4_all_v2 --gpus 6 --resume
```

שלבים שלמים מאומתים ומדולגים; worker שנקטע ממשיך מה־chunks המאומתים שלו.
`pipeline_done.json` מסמן השלמת כל הרצף. הפלטים נמצאים תחת `runs/`, `analysis/`,
`schedules/`, ולוגים נפרדים לכל שלב/רפליקה תחת `logs/`. אין צורך ליצור או להפעיל את
ה־schedules ידנית במסלול זה. רצוי לעקוב אחרי השער הראשוני, אך אין צורך בפעולה שלך כדי להתקדם ממנו.
אם SIGKILL הותיר נעילה, יש לאמת שהמשימה אינה פעילה ולפעול לפי הוראות הנעילה בסעיף 6;
אין להפעיל שני controllers עם אותו שם ריצה במקביל.

הסעיפים המפורטים בהמשך הם חלופה ידנית/לצורכי איתור תקלה. **אין להפעיל אותם בנוסף להרצה המאוחדת**.

## מה ממומש ומה לא

- הכנת 178 זוגות discovery, עם מסכות טוקנים מדויקות, הפניות למדידות קודמות ובדיקת התאמה בין clean/corrupt.
- שערי תקינות על שש דוגמאות קבועות, לרבות משפחה 168 עם prefix משותף.
- Coverage check לשני המועמדים L13H18/L24H19; שימוש חוזר ב־Scope P על 40 הזוגות שבהם הוא כבר נמדד. מדידות P/F לשאר הנדרש. לוקליזציה מותנית בשישה אזורים ופירוק אזורי עובדות לפי slot.
- P1/P2/P4: שינוי מקור בעמדות העובדה → K/V של ראש מאוחר → פלט הראש בפוזיציית הנקודתיים בלבד; גם MLP באותה שכבה ובשכבה הבאה.
- P3: מקור בנקודתיים → Q/KV של ראשים מאוחרים, MLP מקומי או מסלול residual ישיר לפלט. נכלל גם L17H3.
- השלמות מותנות: KV משותף, פיצול KV ל־K/V, איחוד אתרי L17H1, ביקורת `is` של L15H25, אתרי ביקורת ומקבלים אקראיים תואמי שכבה.
- הרחבת מסלולים נבחרים ל־178 זוגות בשני הכיוונים; ניתוח ברמת משפחה, רווחי bootstrap תיאוריים ורגישות ל־prefix.
- חידוש ריצה לפי checksum וזהות קוד/קלט, חלוקה לפי זוגות בין רפליקות, אימות כיסוי מלא לפני ניתוח.

**לא ממומשים בחבילה זו:** G1/G2/G3, חסימת קבוצות מקבלים, mean ablation,
חיפוש המתווכים וההרחבות של S4.3, ו־S4.4/S4.5. מסלול ישיר שלא נמצא אינו שולל תיווך.
שער all-retained עבור mean masks יתווסף עם תשתית ה־mean; כאן נבדקים self/joint-self של ההתערבויות בפועל.
הוולידציה החתומה אינה נפתחת.

## כיצד החישוב עובד

המקור מוחלף ב־`z = A V` לפני `o_proj`. תרומת הקשב של שכבת המקור עוברת את הנרמול המשותף כרגיל.
בהרצת hybrid, יתר ענפי ה־attention וה־MLP מאותה שכבה והלאה מוקפאים **אחרי** הנרמול לערכי recipient.
כך שינוי ב־residual יכול להגיע ישירות למקבל בלי להפעיל מתווכים בלתי מתועדים.
לוכדים את Q/K אחרי QK-norm ו־RoPE ואת V בפועל. בהרצה חדשה מזריקים רק את הערוץ המבוקש,
ומשחררים רק את שורת פלט הקורא בנקודתיים; יתר שורות פלט הראש נשארות תקינות. ההמשך רץ רגיל.
בהזרקת MLP משתנה הקלט שלו רק בעמדות המקור. במסלול bypass כל הענפים המאוחרים מוקפאים,
והנרמול הסופי נשאר חי. prefix משותף הופך bypass מהנקודתיים לאפס מבני, ומדווח בנפרד.

`effect` בכיוון noise הוא margin תקין פחות margin לאחר התערבות.
ב־restore הוא margin לאחר התערבות פחות margin מושחת, תמיד עם סימן clean-answer קבוע.
השוליים נמדדים בטוקן הראשון שמבדיל בין שמות המועמדות, לאחר teacher forcing של prefix משותף.
שומרים את שני הלוגיטים, ההסתברויות, top token, נורמת שינוי הערוץ, המסכות והכיוון.
מדידת Scope P שנעשה בה שימוש חוזר מסומנת `saved_stage2_exact_P`; אין להמציא עבורה לוגיטים שלא נשמרו.

## 1. CPU מקומי — הכנה ובדיקה

הרץ משורש הפרויקט ב־PowerShell. הקלטים כבר הוכנו ב־`results/stage4_inputs_v1`.
אין להריץ שוב prepare לאותה תיקייה; הוא מסרב לדרוס קלט קפוא.

```powershell
$stage4Python = 'C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $stage4Python pilot_v2/stage4_run.py check --inputs results/stage4_inputs_v1 --schedule results/stage4_inputs_v1/coverage.json
& $stage4Python pilot_v2/stage4_run.py check --inputs results/stage4_inputs_v1 --schedule results/stage4_inputs_v1/seed.json
$env:PYTHONPATH = 'tmp/stage3_test_deps'
& $stage4Python -m unittest discover -s pilot_v2 -p "test_stage4*.py" -v
& $stage4Python pilot_v2/build_stage4_bundle.py
```

רק אם צריך ליצור גרסת קלט חדשה:

```powershell
& $stage4Python pilot_v2/stage4_run.py prepare --stage3 results/stage3_inputs_v1 --design results/stage4_design_v2 --out results/stage4_inputs_v2
& $stage4Python pilot_v2/build_stage4_bundle.py --inputs results/stage4_inputs_v2
```

אין להריץ את מודל 7B ב־CPU. `prepare`, `check`, `analyze`, `next` ובניית החבילה הם חישובי CPU.
ב־CPU שאין בו torch/transformers בדיקות המודל הזעיר ידולגו; על שרת ה־GPU יש לוודא שאינן מדולגות.

## 2. העלאה לשרת ואימות החבילה

PowerShell, לפי חיבור השרת שכבר שימש בפרויקט:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\stage4_s41_v2_update.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

בשרת, Bash:

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
test ! -e stage4_s41_v2 || { echo 'Package already exists: use a new package name/version instead of overwriting'; exit 1; }
tar -xzf stage4_s41_v2_update.tar.gz
source runtime.sh
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('stage4_s41_v2')
for name,expected in json.loads((p/'bundle_hashes.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
print('Bundle verified')
PY
PYTHONPATH="$PWD/stage4_s41_v2:$PYTHONPATH" python3 -m unittest test_stage4 test_stage4_pipeline -v
mkdir -p stage4_s41_v2/schedules
```

החבילה משתמשת ב־`runtime.sh` הקיים ובמשקולות pinned הקיימות ב־storage; אינה מורידה מודל או משנה סביבה.

## 3. GPU — שער תקינות חובה

```bash
bash stage4_s41_v2/submit_stage4.sh --name stage4_gate_v1 --schedule stage4_s41_v2/inputs/coverage.json --gate-only --time 02:00:00
```

שמור את מספר המשימה. לבדיקה:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,NodeList
cat "$PILOT_RUNS/stage4_gate_v1/gate.json"
cat "$PILOT_RUNS/stage4_gate_v1/runtime.json"
```

יש לדרוש `passed: true` וכל ששת הזוגות. בכישלון עוצרים; לא מרחיבים tolerance כדי להעביר תוצאה.
נבדקים self/no-hook drift עד 0.001, שכפול P/לוגיטים עד 0.05,
שחזור SDPA עד 0.02, החלפת QKV מול החלפת פלט הראש עד 0.05,
ומסלולים בלתי אפשריים/שינוי בין פוזיציות ללא מתווך עד 0.001.
הבדיקות משוות גם את הלוגיטים בנפרד, לא רק הפרש שעשוי להסתיר ביטול.
כל ריצת מדידה חוזרת על השער; אין דגל שמדלג עליו.

## 4. GPU — השלמת Coverage, ואז CPU — בחירת השלב הבא

```bash
bash stage4_s41_v2/submit_stage4.sh --name stage4_coverage_v1 --schedule stage4_s41_v2/inputs/coverage.json --time 02:00:00
```

רק לאחר `done.json` תקין:

```bash
python3 stage4_s41_v2/stage4_run.py analyze --inputs stage4_s41_v2/inputs --schedule stage4_s41_v2/inputs/coverage.json --runs "$PILOT_RUNS/stage4_coverage_v1" --out "$PILOT_RUNS/stage4_coverage_v1/analysis"
python3 stage4_s41_v2/stage4_run.py next --mode after-coverage --inputs stage4_s41_v2/inputs --schedule stage4_s41_v2/inputs/coverage.json --runs "$PILOT_RUNS/stage4_coverage_v1" --out stage4_s41_v2/schedules/after_coverage.json
python3 stage4_s41_v2/stage4_run.py check --inputs stage4_s41_v2/inputs --schedule stage4_s41_v2/schedules/after_coverage.json
```

הפקודה האחרונה מציגה `phase`:

- **seed:** אפשר לעבור לסעיף 5 עם הקובץ שנוצר.
- **localize_roles:** הרץ אותו כמשימת GPU, למשל `stage4_localize_roles_v1`.
  לאחר הסיום, הרץ `next --mode after-localize` עם schedule/run אלה ליצירת הקובץ הבא.
- אם הקובץ הבא הוא **localize_slots**, הרץ אותו והפעל שוב `after-localize` על תוצאותיו.
  הקובץ הבא יהיה **seed**. כך בדיקות ה־slots נעשות רק אם אזור העובדות עבר את הכלל.

דוגמה לסבב לוקליזציה:

```bash
bash stage4_s41_v2/submit_stage4.sh --name stage4_localize_roles_v1 --schedule stage4_s41_v2/schedules/after_coverage.json --time 02:00:00
# After completion:
python3 stage4_s41_v2/stage4_run.py next --mode after-localize --inputs stage4_s41_v2/inputs --schedule stage4_s41_v2/schedules/after_coverage.json --runs "$PILOT_RUNS/stage4_localize_roles_v1" --out stage4_s41_v2/schedules/after_roles.json
```

בכל מעבר שמור את ה־schedule שהורץ לצד תוצאותיו. `inputs/seed.json` הוא תצוגה מקדימה בלבד;
הקוד מסרב להריץ אותו למדידות לפני החלטת Coverage. אין צורך בעריכה ידנית של JSON.

## 5. GPU — מיפוי P1–P4

קבע את הנתיב ל־schedule האחרון שה־phase שלו seed (בדוגמה ללא לוקליזציה):

```bash
S4_SEED=stage4_s41_v2/schedules/after_coverage.json
python3 stage4_s41_v2/stage4_run.py check --inputs stage4_s41_v2/inputs --schedule "$S4_SEED"
bash stage4_s41_v2/submit_stage4.sh --name stage4_seed_v1 --schedule "$S4_SEED" --time 08:00:00
```

ברשימת הפתיחה, לפני תוספות Coverage: **584 תצורות × 40 זוגות = 23,360 מדידות endpoint**.
זה אינו כולל לכידות clean/corrupt, לכידת hybrid משותפת למקור/אתר, שערים ו־self controls.
המודל נטען פעם אחת לכל תהליך, לא מחדש לכל מסלול. המימוש הנוכחי הוא batch=1;
חלוקה בין רפליקות מקצרת זמן, אבל אין כאן עדיין אופטימיזציית batching.
אין התחייבות ש־8 שעות יספיקו; זהו גבול ההקצאה, וריצה שנקטעה ניתנת לחידוש.

אם מוקצים שישה GPU, אפשר במקום המשימה היחידה להריץ שלוש רפליקות עצמאיות של שני GPU:

```bash
for shard in 0 1 2; do
  bash stage4_s41_v2/submit_stage4.sh --name "stage4_seed_v1_r${shard}" --schedule "$S4_SEED" --shard "$shard" --shards 3 --time 08:00:00
done
```

אין להריץ גם את האפשרות היחידה וגם את שלוש הרפליקות. החלוקה דטרמיניסטית לפי זוג;
אפשר ששתי ההזמנות של משפחה יהיו ברפליקות שונות, והניתוח מאחד אותן לפני הסטטיסטיקה.

## 6. ניטור וחידוש

```bash
cat "$PILOT_RUNS/stage4_seed_v1/state.json"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
bash stage4_s41_v2/submit_stage4.sh --name stage4_seed_v1 --schedule "$S4_SEED" --resume --time 08:00:00
```

בחידוש רפליקה יש להעביר בדיוק את אותם `--shard` ו־`--shards`. קוד/תוכנית שהשתנו מחייבים ריצה חדשה.
כל chunk מאומת ב־checksum. chunk ללא `.ok` אינו נחשב הושלם ויחושב מחדש.
`done.json` נוצר רק לאחר התאמה מלאה של IDs צפויים ונמדדים. exit code 75 מציין הפסקה מסודרת שדורשת resume.
אם נשאר `.running.lock` עקב SIGKILL, בדוק ב־Slurm שהמשימה הקודמת אינה פעילה,
קרא את תוכן הנעילה, ורק אז הסר **את הקובץ הזה בלבד** וחזור על resume.

```bash
cat "$PILOT_RUNS/stage4_seed_v1/.running.lock"
# Only after verifying that its recorded job has ended:
rm -- "$PILOT_RUNS/stage4_seed_v1/.running.lock"
```

## 7. CPU — ניתוח ויצירת השלמות

משימה יחידה:

```bash
python3 stage4_s41_v2/stage4_run.py analyze --inputs stage4_s41_v2/inputs --schedule "$S4_SEED" --runs "$PILOT_RUNS/stage4_seed_v1" --out "$PILOT_RUNS/stage4_seed_v1/analysis"
python3 stage4_s41_v2/stage4_run.py next --mode refine --inputs stage4_s41_v2/inputs --schedule "$S4_SEED" --runs "$PILOT_RUNS/stage4_seed_v1" --out stage4_s41_v2/schedules/refine.json
python3 stage4_s41_v2/stage4_run.py next --mode extend --inputs stage4_s41_v2/inputs --schedule "$S4_SEED" --runs "$PILOT_RUNS/stage4_seed_v1" --out stage4_s41_v2/schedules/extend_seed.json
```

בשלוש רפליקות מחליפים בכל הפקודות את `--runs` ב:

```bash
--runs "$PILOT_RUNS/stage4_seed_v1_r0" "$PILOT_RUNS/stage4_seed_v1_r1" "$PILOT_RUNS/stage4_seed_v1_r2"
```

אלו ארגומנטים בתוך הפקודה, לא פקודת Bash עצמאית. אין לנתח רפליקה בודדת כאילו היא המדגם המלא.
אם אין אף קשר שעומד בכלל, `next` מסרב ליצור משימה ריקה. זו תוצאה המחייבת בדיקת גבולות/תיווך,
לא הרשאה לשנות את סף הבחירה.

לאחר שנוצרו, אלו שוב הרצות GPU:

```bash
bash stage4_s41_v2/submit_stage4.sh --name stage4_refine_v1 --schedule stage4_s41_v2/schedules/refine.json --time 08:00:00
bash stage4_s41_v2/submit_stage4.sh --name stage4_extend_seed_v1 --schedule stage4_s41_v2/schedules/extend_seed.json --time 08:00:00
```

לאחר השלמת refinement, יוצרים את הרחבתו תוך שמירת התקציב המשותף:

```bash
python3 stage4_s41_v2/stage4_run.py next --mode extend --inputs stage4_s41_v2/inputs --schedule stage4_s41_v2/schedules/refine.json --runs "$PILOT_RUNS/stage4_refine_v1" --prior-extension stage4_s41_v2/schedules/extend_seed.json --out stage4_s41_v2/schedules/extend_refine.json
bash stage4_s41_v2/submit_stage4.sh --name stage4_extend_refine_v1 --schedule stage4_s41_v2/schedules/extend_refine.json --time 08:00:00
```

הסקריפט מקצה עד 72 מסלולים מה־seed ועד יתרת התקרה 96 מה־refinement, מסיר כפילויות,
ומדווח overflow כלא־נבדק. הבחירה מתעדפת ייצוג למקורות ולמועמדי RI לפני דירוג יתר האפקטים.
מדידות refinement core משמשות לביקורת ולבחירה; השוואות KV מול K/V ואיחוד מול אתרים
בודדים נעשות על אותה אוכלוסייה/כיוון. אין להסיק אינטראקציה מחיבור ממוצעים באוכלוסיות שונות.
את כל ריצות ההרחבה מנתחים עם פקודת `analyze` וה־schedule שלהן; נשמרים גם שני הכיוונים בנפרד.

לאחר ניתוח seed ו־refine על 40 הזוגות, אפשר לחשב את האינטראקציות המזווגות ב־CPU:

```bash
python3 stage4_s41_v2/stage4_run.py analyze --inputs stage4_s41_v2/inputs --schedule stage4_s41_v2/schedules/refine.json --runs "$PILOT_RUNS/stage4_refine_v1" --out "$PILOT_RUNS/stage4_refine_v1/analysis"
python3 stage4_s41_v2/stage4_run.py compare --analyses "$PILOT_RUNS/stage4_seed_v1/analysis" "$PILOT_RUNS/stage4_refine_v1/analysis" --out "$PILOT_RUNS/stage4_interactions_core_v1"
```

הפלט כולל `KV − K − V` ו־`union − sum(single sites)` רק כשכל האיברים נמדדו
בדיוק על אותן משפחות ובאותו כיוון. איברים חסרים מדווחים ב־`unavailable.json`, לא מושלמים באפס.

## 8. פלטים והורדה לניתוח מקומי

- `manifest.json`: hashes של קוד, תוכנית, schedule וחלוקת עבודה.
- `gate.json`, `runtime.json`, `state.json`, `done.json`: תקינות, סביבה, התקדמות ושלמות.
- `chunks/*.json` ו־`.ok`: הרשומות הגולמיות המאומתות; נדרשים לשחזור ניתוח.
- `profiles/*.json`: נורמות ומרחקי clean/corrupt אחרי כל בלוק בחמישה אתרים, שנאספו בהרצות הבסיס הנדרשות ללא הרצות נוספות. אלו אבחונים נלווים, לא בדיקת סיבתיות.
- `analysis/events.csv`: לוגיטים ואפקטים לכל התערבות.
- `analysis/family_effects.csv`: ממוצע שתי ההזמנות בכל משפחה.
- `analysis/route_summary.csv`: mean, mean absolute, median/range, sign fractions, כלל שימור ורווח bootstrap תיאורי (20,000 דגימות משפחות).
- `analysis/context_summary.csv`: strata שנקבעו מראש לפי סדר עובדת השאלה, כשיש לפחות 10 משפחות.
- `analysis/verification.json`: אימות כיסוי וקישור למקורות.

דוגמת אריזה של ריצה אחת, מהשרת:

```bash
tar -czf stage4_seed_v1_results.tar.gz -C "$PILOT_RUNS" stage4_seed_v1
tar -czf stage4_schedules.tar.gz -C stage4_s41_v2 schedules
```

PowerShell מקומי:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/stage4_seed_v1_results.tar.gz" .\results\
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/stage4_schedules.tar.gz" .\results\
tar -xzf results/stage4_seed_v1_results.tar.gz -C results
New-Item -ItemType Directory -Force results/stage4_download | Out-Null
tar -xzf results/stage4_schedules.tar.gz -C results/stage4_download
& $stage4Python pilot_v2/stage4_run.py analyze --inputs results/stage4_inputs_v1 --schedule results/stage4_download/schedules/after_coverage.json --runs results/stage4_seed_v1 --out results/stage4_seed_v1/analysis_local
```

התאם את שם ה־schedule למסלול הלוקליזציה שבוצע. זיהוי הניתוח מבוסס hashes ולא הנתיב המוחלט בשרת.
הורד כל רפליקה שהשתתפה; תיקיית analysis לבדה אינה מספיקה לשחזור.

## 9. Git — עדכון ממוקד, ללא ערבוב קבצים אחרים

הוכן `update_stage4_git.ps1`. ברירת המחדל היא preview בלבד:

```powershell
powershell -NoProfile -File pilot_v2/update_stage4_git.ps1
powershell -NoProfile -File pilot_v2/update_stage4_git.ps1 -Stage
git diff --cached --stat
```

אפשר לבצע commit רגיל לאחר בדיקה. לחלופין, `-Commit` מבצע stage ו־commit רק אם ה־index היה ריק מלכתחילה:

```powershell
powershell -NoProfile -File pilot_v2/update_stage4_git.ps1 -Commit
```

אין push אוטומטי, שינוי branch, reset או `git add .`. משקולות ותוצאות גולמיות אינן מצורפות.
חבילת ההעלאה היא אמצעי ההעברה לשרת; checksum שלה מצורף לעדכון Git. קובצי המקור וה־runbook
הם המקור הנשמר ב־Git. ה־protocol עצמו ועדכוני דוחות אחרים נשארים לבדיקתך בנפרד.

אם PowerShell חוסם קובצי ps1 לפי ExecutionPolicy, אין צורך לשנות את המדיניות. אפשר לבצע ידנית:

```powershell
git add -- pilot_v2/stage4_pipeline.py pilot_v2/test_stage4_pipeline.py pilot_v2/stage4_pipeline.sbatch pilot_v2/submit_stage4_pipeline.sh pilot_v2/stage4_engine.py pilot_v2/stage4_plan.py pilot_v2/stage4_run.py pilot_v2/stage4_analyze.py pilot_v2/test_stage4.py pilot_v2/stage4.sbatch pilot_v2/submit_stage4.sh pilot_v2/build_stage4_bundle.py pilot_v2/RUNBOOK_stage4_s41.md pilot_v2/update_stage4_git.ps1 pilot_v2/stage4_s41_v2_update.tar.gz.sha256.json results/stage4_design_v2/manifest_proposal.json results/stage4_design_v2/ri_reduced_selection.csv
git diff --cached --stat
```

ודא שאין ב־index קבצים אחרים לפני commit. אם מתקבלת הודעת ownership, בדוק שהנתיב אכן
הפרויקט שלך והשתמש באפשרות `git -c safe.directory=...` לפקודה הבודדת; אין צורך בשינוי גלובלי.
