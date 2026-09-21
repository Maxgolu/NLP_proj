"""Freeze Stage-3 descriptive groups from existing results; no effect recomputation."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage3_plan"
SOURCES = {
    "head_table": ROOT / "results/stage2_v2_analysis/head_table_v2.csv",
    "candidates": ROOT / "results/ri_test_v2/candidates.json",
    "shortlist": ROOT / "results/stage2_v1/shortlist.json",
    "coverage": ROOT / "results/stage2_v2_coverage/extension_v2_heads.json",
}
HISTORICAL = {3 * 32 + 11, 9 * 32 + 22}
SMALL, STRONG = 0.1, 0.3


def main():
    with SOURCES["head_table"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = json.loads(SOURCES["candidates"].read_text(encoding="utf-8"))["candidates"]
    shortlist = json.loads(SOURCES["shortlist"].read_text(encoding="utf-8"))
    new_ri = {c["layer"] * 32 + c["head"] for c in candidates}
    ri = new_ri | HISTORICAL
    controls = set(shortlist["random_controls"])
    assert len(new_ri) == 59 and len(controls) == 25
    assert len(rows) == 148 and len({int(r["flat_id"]) for r in rows}) == 148
    assert ri | controls <= {int(r["flat_id"]) for r in rows}
    groups = {k: [] for k in ["RI-selected_small-effect", "strong-positive_outside-RI",
              "strong-positive_RI-selected", "strong-negative", "random-controls",
              "moderate-effect_supplement"]}
    output = []
    for r in rows:
        h = int(r["flat_id"])
        assert r["head"] == f"L{h // 32}H{h % 32}"
        assert (r["in_A"] == "True") == (h in new_ri)
        assert (r["random_control"] == "True") == (h in controls)
        effect = float(r["I_P_178"])
        assert math.isfinite(effect)
        checks = {
            "RI-selected_small-effect": h in ri and abs(effect) < SMALL,
            "strong-positive_outside-RI": h not in ri and effect >= STRONG,
            "strong-positive_RI-selected": h in ri and effect >= STRONG,
            "strong-negative": effect <= -STRONG,
            "random-controls": h in controls,
            "moderate-effect_supplement": SMALL <= abs(effect) < STRONG,
        }
        membership = [k for k, yes in checks.items() if yes]
        for k in membership:
            groups[k].append(r["head"])
        output.append(dict(head=r["head"], flat_id=h, exact_scopeP_mean_178=effect,
                           in_new_RI=h in new_ri, in_historical_RI=h in HISTORICAL,
                           final_QK_group=r["in_B"] == "True",
                           has_saved_scopeF=bool(r.get("I_F_178")),
                           included=bool(membership), groups=";".join(membership)))
    selected = [r for r in output if r["included"]]
    # Named candidates are checked, never inserted as exceptions.
    expected = {
        "L8H15": "moderate-effect_supplement", "L14H23": "moderate-effect_supplement",
        "L6H24": "moderate-effect_supplement", "L9H16": "RI-selected_small-effect",
        "L11H4": "RI-selected_small-effect", "L17H5": "RI-selected_small-effect",
        "L18H19": "strong-positive_RI-selected", "L16H21": "strong-positive_RI-selected",
        "L16H1": "strong-positive_RI-selected", "L22H5": "strong-positive_RI-selected",
    }
    assert all(h in groups[g] for h, g in expected.items())
    assert all(any(int(r["flat_id"]) == h for r in selected) for h in ri | controls)
    strongest = sorted(selected, key=lambda r: (-abs(r["exact_scopeP_mean_178"]), r["flat_id"]))[:10]
    result = dict(status="prospective Stage-3 plan; selected using already observed Stage-1/2 data",
                  population="148 heads with exact Scope-P means on the same 178 pairs / 89 families",
                  small_abs_effect_lt=SMALL, strong_abs_effect_ge=STRONG,
                  ri_selection="59-head test-only union plus L3H11 and L9H22; not all historical top-25 heads",
                  groups=groups, group_counts={k: len(v) for k, v in groups.items()},
                  selected_unique_heads=len(selected), selected_flat_ids=[r["flat_id"] for r in selected],
                  scopeF_missing_heads=[r["head"] for r in selected if not r["has_saved_scopeF"]],
                  reverse_patch_heads=[r["head"] for r in strongest],
                  overlaps=[dict(head=r["head"], groups=r["groups"].split(";"))
                            for r in selected if ";" in r["groups"]],
                  source_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in SOURCES.items()},
                  note="Only memberships and selection summaries computed. No model calls, effects, correlations or intervals recomputed.")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "head_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (OUT / "inventory_policy.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["group_counts", "selected_unique_heads", "scopeF_missing_heads", "reverse_patch_heads", "overlaps"]}, indent=2))


if __name__ == "__main__":
    main()
