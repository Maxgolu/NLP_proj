# העלאה סופית ל־Git

נבדק ב־28.9.2026: גם `HEAD` המקומי וגם הענף `main` ב־GitHub מצביעים על
`ff627501b7265c66aeb16a4dbc122d588d4d5c1a`, מה־27.9.
לא בוצעו staging, commit או push במהלך הבדיקה.

אין שינויים במנועי הניסוי המנוהלים ב־`pilot_v2` וב־`pilot_v3` מאז הקומיט הזה.
כן יש לעדכן את התיעוד, קובצי המאמר והעיצוב ששונו, שלוש תמונות, תוצאות האימות,
תוכנית האימות, קוד הביקורת שלו, פרק המסקנות ותמונת המצב החדשה מהשותפה.
קוד ה־CPU שנוסף לביקורת האימות אינו ניסוי GPU חדש.

## פקודות PowerShell

להריץ מהנתיב הבא:

```powershell
Set-Location -LiteralPath 'C:\Users\User\OneDrive\Documents\computer science\4B\NLP\final proj\project'
```

הוספה מפורשת של הקבצים הנדרשים:

```powershell
git add -- README.md RESEARCH_HANDOFF.md results/README.md
git add -- results/heldout_plan.json results/stage4_s45_heldout_received_20260928 results/paper_final_review_20260928
git add -- 'חומר כתוב/paper/acl.sty' 'חומר כתוב/paper/main.tex'
git add -- 'חומר כתוב/paper/appendix/G_stage3.tex' 'חומר כתוב/paper/appendix/H_stage4.tex'
git add -- 'חומר כתוב/paper/sections/03_setting.tex' 'חומר כתוב/paper/sections/04_method.tex' 'חומר כתוב/paper/sections/05_results.tex' 'חומר כתוב/paper/sections/06_conclusions.tex'
git add -- 'חומר כתוב/paper/figs/figH_final_effects.pdf' 'חומר כתוב/paper/figs/figH_final_retention.pdf' 'חומר כתוב/paper/figs/fig_stage4_chains.pdf'
git add -- 'חומר כתוב/paper/paper3'
git diff --cached --stat
git diff --cached --check
```

לאחר בדיקת רשימת הקבצים:

```powershell
git commit -m "Finalize experiment status, held-out evidence and manuscript review snapshot"
git push origin main
git status --short
```

אין צורך להשתמש ב־`git add .`. הקבצים שלא נכללו במכוון יכולים להישאר
untracked; אין צורך למחוק אותם כדי לבצע את ההעלאה.

## מה לא להוסיף בהעלאה זו

- `Claude outputs/main_preview.pdf` — תצוגה מקדימה, לא מקור המאמר העדכני.
- שלושת קובצי `*_pdfcheck.png` תחת ביקורת הגילוי — תמונות בדיקת עימוד.
- `Stage45_Overleaf.pdf`, `Stage4_results & analysis.pdf` — עותקי דוחות נלווים;
  המקורות והמספרים כבר מנוהלים. אם רוצים גרסת PDF להפצה, לבחור רק את ה־PDF הסופי.
- `results/stage4_design_v1/revise_protocol_v2.py`,
  `results/stage4_design_v2/Stage4_Protocol_before_revision.tex`,
  `results/stage4_implementation_review/stage4_all_v2_remote_completion_20260924.txt`
  — שרידים מקומיים מהשלבים ההיסטוריים, לא עדכון למימוש הניסוי האחרון.
- ZIP/TAR, משקלים, mean banks וקובצי עבודה גולמיים שהוחרגו ב־`.gitignore`.

## ההבחנה בין גיבוי המאמר להגשה הסופית

הפקודות מגבות גם את העותק המקומי המודולרי וגם את הייצוא החדש ב־`paper3`;
ה־README מסביר במפורש שהעותק המודולרי הישן אינו מסונכרן לחלוטין עם אוברליפ.
אין להציג העלאה זו כהוכחה שהמקורות המקומיים מייצרים בדיוק את ה־PDF להגשה.

אחרי סיום התיקונים במאמר, הדרך לשמור ב־Git את **הגרסה המדויקת שהוגשה** היא
לייצא מאוברליפ את הפרויקט המלא, לשמור את מבנה `sections/appendix/figs`, ולסנכרן
את המקורות ב־`חומר כתוב/paper/`. אין צורך לפרסם את ZIP ההעברה עצמו.
קובצי `paper3` השטוחים לבדם אינם חבילה מלאה שניתן להעלות ולבנות.
