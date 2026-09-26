"""S4.3 full-discovery bidirectional extension runner.

Runs only the prospectively selected S4.3 extension contrasts and their matched
prerequisites over all 178 discovery prompt pairs, in both directions:

  noise   : clean recipient <- corrupt donor
            I = M(clean) - M(clean with corrupt route)

  restore : corrupt recipient <- clean donor
            J = M(corrupt with clean route) - M(corrupt)

Held-out families are never loaded. Checkpoints are atomic at pair boundaries.
"""

from pathlib import Path
import argparse
import collections
import gc
import gzip
import hashlib
import json
import math
import os
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import s43_run_v2 as base


EXPECTED_PAIRS = 178
EXPECTED_FAMILIES = 89
EXPECTED_SELECTED = 16
EXPECTED_PREREQS = 11
EXPECTED_CONFIGS = 27
EXPECTED_DIRECTIONS = ("noise", "restore")
EXPECTED_RECORDS_PER_PAIR = EXPECTED_CONFIGS * len(EXPECTED_DIRECTIONS)
EXPECTED_TOTAL_RECORDS = EXPECTED_PAIRS * EXPECTED_RECORDS_PER_PAIR
EXPECTED_HYBRID_GROUPS_PER_DIRECTION = 16
EXPECTED_BYPASS_PER_DIRECTION = 5
EXPECTED_FRESH_ENDPOINTS_PER_DIRECTION = EXPECTED_CONFIGS - EXPECTED_BYPASS_PER_DIRECTION
HISTORICAL_MARGIN_TOL = 0.05
HISTORICAL_EXCEPTION_IDS = frozenset({
    "034/direct/base/0/order0",
    "034/direct/base/0/order1",
})
EXCEPTION_REFERENCE_TOL = 1e-6
SDPA_TOL = 0.02
SELF_TOL = 0.001


def read_json(path):
    return base.read_json(Path(path))


def write_json_atomic(path, obj):
    return base.write_json_atomic(Path(path), obj)


def sha256_path(path):
    return base.sha256_path(Path(path))


def read_jsonl(path):
    return base.read_jsonl(Path(path))


def digest_object(obj):
    return base.digest_object(obj)


def load_compatibility_reference(path, model_id, revision):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    report = read_json(path)

    if report.get("scientific_intervention") is not False:
        raise ValueError("Compatibility diagnostic must be baseline-only")
    if report.get("heldout_accessed") is not False:
        raise ValueError("Compatibility diagnostic indicates held-out access")

    runtime = report.get("runtime", {})
    if runtime.get("model") != model_id:
        raise ValueError("Compatibility diagnostic model ID mismatch")
    if runtime.get("revision") != revision:
        raise ValueError("Compatibility diagnostic model revision mismatch")
    if runtime.get("gpu") != "NVIDIA A100-SXM4-40GB":
        raise ValueError("Compatibility diagnostic was not produced on A100-SXM4-40GB")

    summary = report.get("summary", {})
    if int(summary.get("pairs", -1)) != EXPECTED_PAIRS:
        raise ValueError("Compatibility diagnostic pair count mismatch")
    if int(summary.get("families", -1)) != EXPECTED_FAMILIES:
        raise ValueError("Compatibility diagnostic family count mismatch")
    if int(summary.get("historical_margin_error_gt_0_05", -1)) != 2:
        raise ValueError("Compatibility diagnostic must contain exactly two >0.05 pairs")
    if float(summary.get("max_error_excluding_family_034", math.inf)) > HISTORICAL_MARGIN_TOL:
        raise ValueError("A non-family-034 diagnostic pair exceeds the 0.05 tolerance")

    cross = summary.get("initial_v2_cross_runtime", {})
    if float(cross.get("max_margin_error", math.inf)) != 0.0:
        raise ValueError("Frozen compatibility diagnostic does not exactly reproduce initial_v2 margins")
    if int(cross.get("gt_0_05", -1)) != 0:
        raise ValueError("Frozen compatibility diagnostic disagrees with initial_v2 above 0.05")

    rows = report.get("rows", [])
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError("Compatibility diagnostic row count mismatch")

    by_id = {row["pair_id"]: row for row in rows}
    if len(by_id) != EXPECTED_PAIRS:
        raise ValueError("Duplicate pair IDs in compatibility diagnostic")

    bad = {
        row["pair_id"]
        for row in rows
        if float(row["historical_margin_error"]) > HISTORICAL_MARGIN_TOL
    }
    if bad != HISTORICAL_EXCEPTION_IDS:
        raise ValueError(
            "Compatibility diagnostic exceptions changed: "
            + json.dumps(sorted(bad))
        )

    repeats = report.get("family_034_repeats", [])
    if {row.get("pair_id") for row in repeats} != HISTORICAL_EXCEPTION_IDS:
        raise ValueError("Family-034 repeat diagnostic is incomplete")

    for row in repeats:
        for key in [
            "clean_logit_spread",
            "corr_logit_spread",
            "margin_spread",
            "historical_error_spread",
        ]:
            if float(row.get(key, math.inf)) != 0.0:
                raise ValueError(
                    f"Family-034 diagnostic is not deterministic: {row['pair_id']} {key}"
                )

    reference = {
        pair_id: {
            "current_margin": float(by_id[pair_id]["current_margin"]),
            "historical_margin_error": float(by_id[pair_id]["historical_margin_error"]),
        }
        for pair_id in HISTORICAL_EXCEPTION_IDS
    }

    return {
        "path": str(path.resolve()),
        "sha256": sha256_path(path),
        "reference": reference,
    }


def check_historical_compatibility(pair, current_margin, compat_reference):
    pair_id = pair["id"]
    saved_margin = float(pair["clean_logits"][0]) - float(pair["clean_logits"][1])
    historical_error = abs(float(current_margin) - saved_margin)

    if not math.isfinite(historical_error):
        raise ValueError("Nonfinite historical margin error")

    if pair_id in HISTORICAL_EXCEPTION_IDS:
        ref = compat_reference["reference"][pair_id]
        reference_error = abs(float(current_margin) - float(ref["current_margin"]))
        base.require_le(
            reference_error,
            EXCEPTION_REFERENCE_TOL,
            "family-034 diagnostic reference " + pair_id,
        )
        mode = "frozen_family034_diagnostic_exception"
    else:
        base.require_le(
            historical_error,
            HISTORICAL_MARGIN_TOL,
            "historical clean margin " + pair_id,
        )
        reference_error = None
        mode = "strict_historical_margin_0.05"

    return {
        "historical_margin_error": historical_error,
        "compatibility_mode": mode,
        "diagnostic_reference_error": reference_error,
    }


def pair_chunk_paths(chunks_dir, pair):
    stem = f"f{int(pair['family']):03d}_o{int(pair['order'])}"
    return chunks_dir / f"{stem}.json", chunks_dir / f"{stem}.ok"


def verify_extension_inputs(extension_inputs, frozen_inputs, smoke_report, analysis_dir):
    extension_inputs = Path(extension_inputs)
    frozen_inputs = Path(frozen_inputs)
    analysis_dir = Path(analysis_dir)

    # Reuse the production verifier for the original frozen scientific bundle,
    # discovery-only population, smoke provenance, and current S43 engine/tests.
    (
        frozen_manifest,
        plan,
        all_pairs,
        common,
        full_registry,
        smoke,
    ) = base.verify_frozen_bundle(frozen_inputs, smoke_report)

    if len(all_pairs) != EXPECTED_PAIRS:
        raise ValueError("Discovery pair count changed")
    if len({int(p['family']) for p in all_pairs}) != EXPECTED_FAMILIES:
        raise ValueError("Discovery family count changed")
    if any(
        p['clean']['split'] != 'discovery' or p['corr']['split'] != 'discovery'
        for p in all_pairs
    ):
        raise ValueError("Held-out leakage in extension population")

    manifest_path = extension_inputs / "manifest.json"
    selection_path = extension_inputs / "selection.json"
    registry_path = extension_inputs / "extension_registry.jsonl"
    pairs_path = extension_inputs / "pairs.jsonl.gz"

    for path in [manifest_path, selection_path, registry_path, pairs_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    for name, expected in manifest["files"].items():
        path = extension_inputs / name
        if sha256_path(path) != expected:
            raise ValueError(f"Extension input changed: {name}")

    if manifest["source_frozen_manifest_sha256"] != sha256_path(
        frozen_inputs / "manifest.json"
    ):
        raise ValueError("Extension bundle points to a different frozen S4.3 bundle")

    selection = read_json(selection_path)
    selected_ids = selection["selected_configs"]
    prereq_ids = selection["required_direct_comparators"]

    if len(selected_ids) != EXPECTED_SELECTED or len(set(selected_ids)) != EXPECTED_SELECTED:
        raise ValueError("Expected exactly 16 selected extension contrasts")
    if len(prereq_ids) != EXPECTED_PREREQS or len(set(prereq_ids)) != EXPECTED_PREREQS:
        raise ValueError("Expected exactly 11 prerequisite comparators")
    if set(selected_ids) & set(prereq_ids):
        raise ValueError("Selected contrasts and prerequisites overlap")

    analysis_manifest_path = analysis_dir / "analysis_manifest.json"
    if sha256_path(analysis_manifest_path) != selection["analysis_manifest_sha256"]:
        raise ValueError("Analysis manifest changed after extension selection")

    analysis_manifest = read_json(analysis_manifest_path)
    ext = analysis_manifest["extension"]
    if ext["selected_configs"] != selected_ids:
        raise ValueError("Selected config order differs from analysis manifest")
    if ext["required_direct_comparators"] != prereq_ids:
        raise ValueError("Prerequisite list differs from analysis manifest")
    if ext["full_178_run"] is not False or ext["reverse_restoration_run"] is not False:
        raise ValueError("Analysis manifest does not represent the pre-extension state")
    if analysis_manifest["population"]["heldout_accessed"] is not False:
        raise ValueError("Analysis provenance indicates held-out access")

    # The extension pair archive is copied byte-for-byte from the frozen bundle.
    if sha256_path(pairs_path) != sha256_path(frozen_inputs / "pairs.jsonl.gz"):
        raise ValueError("Extension pair archive is not the frozen discovery archive")

    registry_raw = read_jsonl(registry_path)
    if len(registry_raw) != EXPECTED_CONFIGS:
        raise ValueError(f"Expected {EXPECTED_CONFIGS} extension configs")

    registry_ids = [cfg["id"] for cfg in registry_raw]
    expected_ids = selected_ids + prereq_ids
    if registry_ids != expected_ids:
        raise ValueError("Extension registry order/identity changed")
    if len(set(registry_ids)) != EXPECTED_CONFIGS:
        raise ValueError("Duplicate extension config IDs")
    if any(cfg.get("registry_section") == "slot_control" for cfg in registry_raw):
        raise ValueError("Slot controls may not enter the extension run")

    registry = [base.runtime_config(cfg) for cfg in registry_raw]
    groups = base.group_registry(registry)
    if len(groups) != EXPECTED_HYBRID_GROUPS_PER_DIRECTION:
        raise ValueError(
            f"Expected {EXPECTED_HYBRID_GROUPS_PER_DIRECTION} hybrid groups; got {len(groups)}"
        )

    bypass = sum(cfg["channel"] == "bypass" for cfg in registry)
    if bypass != EXPECTED_BYPASS_PER_DIRECTION:
        raise ValueError(f"Expected {EXPECTED_BYPASS_PER_DIRECTION} bypass configs; got {bypass}")

    selected_set = set(selected_ids)
    roles = {
        cfg["id"]: ("selected" if cfg["id"] in selected_set else "prerequisite")
        for cfg in registry
    }

    priority = {
        cid: i + 1
        for i, cid in enumerate(selected_ids)
    }

    return {
        "frozen_manifest": frozen_manifest,
        "plan": plan,
        "pairs": sorted(all_pairs, key=lambda p: (int(p["family"]), int(p["order"]))),
        "registry": registry,
        "groups": groups,
        "smoke": smoke,
        "manifest": manifest,
        "selection": selection,
        "roles": roles,
        "priority": priority,
    }


def audit(extension_inputs, frozen_inputs, smoke_report, analysis_dir):
    x = verify_extension_inputs(extension_inputs, frozen_inputs, smoke_report, analysis_dir)
    registry = x["registry"]
    groups = x["groups"]

    fresh = sum(cfg["channel"] != "bypass" for cfg in registry)
    per_pair_forwards = 2 + 2 * (len(groups) + fresh)

    info = {
        "pairs": EXPECTED_PAIRS,
        "families": EXPECTED_FAMILIES,
        "selected": EXPECTED_SELECTED,
        "prerequisites": EXPECTED_PREREQS,
        "configs": EXPECTED_CONFIGS,
        "directions": len(EXPECTED_DIRECTIONS),
        "records_per_pair": EXPECTED_RECORDS_PER_PAIR,
        "records": EXPECTED_TOTAL_RECORDS,
        "hybrid_groups_per_direction": len(groups),
        "bypass_per_direction": sum(cfg["channel"] == "bypass" for cfg in registry),
        "fresh_endpoints_per_direction": fresh,
        "forwards_per_pair": per_pair_forwards,
        "science_forwards": per_pair_forwards * EXPECTED_PAIRS,
        "baseline_gate_forwards": EXPECTED_PAIRS,
        "self_gate_forwards": per_pair_forwards,
        "heldout": "sealed",
    }

    print("=" * 76)
    print("S4.3 EXTENSION — DRY AUDIT")
    print("=" * 76)
    print(f"Discovery pairs                 : {info['pairs']}")
    print(f"Discovery families              : {info['families']}")
    print(f"Selected contrasts              : {info['selected']}")
    print(f"Direct prerequisites            : {info['prerequisites']}")
    print(f"Configs / direction             : {info['configs']}")
    print(f"Directions                      : {EXPECTED_DIRECTIONS}")
    print(f"Records / pair                  : {info['records_per_pair']}")
    print(f"Total records                   : {info['records']:,}")
    print(f"Hybrid groups / direction       : {info['hybrid_groups_per_direction']}")
    print(f"Bypass configs / direction      : {info['bypass_per_direction']}")
    print(f"Fresh endpoints / direction     : {info['fresh_endpoints_per_direction']}")
    print(f"Estimated forwards / pair       : {info['forwards_per_pair']}")
    print(f"Estimated science forwards      : {info['science_forwards']:,}")
    print("Direction definitions:")
    print("  noise   = M(clean) - M(clean with corrupt route)")
    print("  restore = M(corrupt with clean route) - M(corrupt)")
    print("Held-out                       : SEALED")
    print("\nAUDIT ONLY — no model measurements were run.")
    return info


def capture_pair(engine, pair, heads, full_z_layers, compat_reference):
    spec = base.checked_spec(engine, pair)
    g, d = spec["g"], spec["d"]

    clean = engine.capture_s43(
        spec["clean"], heads, g, d, full_z_layers=full_z_layers
    )
    corr = engine.capture_s43(
        spec["corr"], heads, g, d, full_z_layers=full_z_layers
    )

    compatibility = check_historical_compatibility(
        pair,
        clean["output"]["margin"],
        compat_reference,
    )
    historical_margin_error = compatibility["historical_margin_error"]

    base.require_le(
        clean["reconstruction_error"],
        SDPA_TOL,
        "clean SDPA reconstruction " + pair["id"],
    )
    base.require_le(
        corr["reconstruction_error"],
        SDPA_TOL,
        "corrupt SDPA reconstruction " + pair["id"],
    )
    return spec, clean, corr, historical_margin_error, compatibility


def measure_direction(
    engine,
    pair,
    spec,
    recipient,
    donor,
    ids,
    direction,
    registry,
    groups,
    heads,
    roles,
    priority,
    stop_requested,
):
    if direction not in EXPECTED_DIRECTIONS:
        raise ValueError("Unknown direction")

    g, d, n = spec["g"], spec["d"], spec["n"]
    baseline = recipient["output"]
    rows = []
    max_clamp = 0.0

    for group_index, (key, configs) in enumerate(groups.items(), start=1):
        if stop_requested[0]:
            raise InterruptedError("Stop requested")

        cfg0 = configs[0]
        positions = pair["masks"][cfg0["site"]]
        hybrid = engine.hybrid_for_config(
            ids=ids,
            config=cfg0,
            positions=positions,
            donor=donor,
            recipient=recipient,
            heads=heads,
            g=g,
            d=d,
        )

        clamp = base.clamp_error(hybrid, recipient, cfg0, engine)
        if clamp is not None:
            max_clamp = max(max_clamp, float(clamp))

        group_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

        for cfg in configs:
            result = engine.endpoint_for_config(
                ids=ids,
                config=cfg,
                positions=positions,
                hybrid=hybrid,
                g=g,
                d=d,
                n=n,
            )
            for value in result.values():
                if not math.isfinite(float(value)):
                    raise ValueError("Nonfinite endpoint " + cfg["id"])

            if direction == "noise":
                effect = float(baseline["margin"] - result["margin"])
            else:
                effect = float(result["margin"] - baseline["margin"])

            channel_norm = engine.channel_norm_for_config(
                config=cfg,
                positions=positions,
                hybrid=hybrid,
                recipient=recipient,
                n=n,
            )
            if channel_norm is not None and not math.isfinite(float(channel_norm)):
                raise ValueError("Nonfinite channel norm " + cfg["id"])

            rows.append({
                "pair_id": pair["id"],
                "family": int(pair["family"]),
                "order": int(pair["order"]),
                "query_first": bool(pair["masks"]["_query_first"]),
                "config_id": cfg["id"],
                "config": cfg,
                "extension_role": roles[cfg["id"]],
                "extension_priority": priority.get(cfg["id"]),
                "direction": direction,
                "hybrid_group": group_id,
                "source_positions": positions,
                "receiver_row": n - 1,
                "shared_prefix": pair["shared_prefix"],
                "baseline": baseline,
                "result": result,
                "effect": effect,
                "channel_norm": None if channel_norm is None else float(channel_norm),
                "nonlive_clamp_error": clamp,
                "freeze_semantics": (
                    "post-normalization branches >= source frozen except source "
                    "attention and named releases"
                ),
            })

        del hybrid

        if group_index % 8 == 0 or group_index == len(groups):
            print(
                f"    {direction}: hybrid groups {group_index}/{len(groups)}",
                flush=True,
            )

    if len(rows) != EXPECTED_CONFIGS:
        raise ValueError(
            f"{pair['id']} {direction}: expected {EXPECTED_CONFIGS} rows; got {len(rows)}"
        )
    if len({r["config_id"] for r in rows}) != EXPECTED_CONFIGS:
        raise ValueError("Duplicate config measurement")
    return rows, max_clamp


def measure_pair(
    engine,
    pair,
    registry,
    groups,
    heads,
    full_z_layers,
    roles,
    priority,
    compat_reference,
    stop_requested,
):
    spec, clean, corr, historical_margin_error, compatibility = capture_pair(
        engine, pair, heads, full_z_layers, compat_reference
    )

    noise_rows, noise_clamp = measure_direction(
        engine=engine,
        pair=pair,
        spec=spec,
        recipient=clean,
        donor=corr,
        ids=spec["clean"],
        direction="noise",
        registry=registry,
        groups=groups,
        heads=heads,
        roles=roles,
        priority=priority,
        stop_requested=stop_requested,
    )

    restore_rows, restore_clamp = measure_direction(
        engine=engine,
        pair=pair,
        spec=spec,
        recipient=corr,
        donor=clean,
        ids=spec["corr"],
        direction="restore",
        registry=registry,
        groups=groups,
        heads=heads,
        roles=roles,
        priority=priority,
        stop_requested=stop_requested,
    )

    rows = noise_rows + restore_rows
    if len(rows) != EXPECTED_RECORDS_PER_PAIR:
        raise ValueError("Unexpected records per pair")

    keys = {(r["direction"], r["config_id"]) for r in rows}
    if len(keys) != EXPECTED_RECORDS_PER_PAIR:
        raise ValueError("Duplicate direction/config row")

    return {
        "pair_id": pair["id"],
        "family": int(pair["family"]),
        "order": int(pair["order"]),
        "historical_margin_error": historical_margin_error,
        "historical_compatibility_mode": compatibility["compatibility_mode"],
        "diagnostic_reference_error": compatibility["diagnostic_reference_error"],
        "clean_sdpa_error": float(clean["reconstruction_error"]),
        "corrupt_sdpa_error": float(corr["reconstruction_error"]),
        "max_nonlive_clamp_error": max(noise_clamp, restore_clamp),
        "records": rows,
    }


def valid_existing_chunk(chunks_dir, pair, identity_hash):
    chunk, ok = pair_chunk_paths(chunks_dir, pair)
    if not chunk.exists() or not ok.exists():
        return False
    try:
        marker = read_json(ok)
        if marker.get("pair_id") != pair["id"]:
            return False
        if marker.get("identity_hash") != identity_hash:
            return False
        if int(marker.get("records", -1)) != EXPECTED_RECORDS_PER_PAIR:
            return False
        if marker.get("sha256") != sha256_path(chunk):
            return False

        data = read_json(chunk)
        if data.get("identity_hash") != identity_hash:
            return False
        if data.get("pair_id") != pair["id"]:
            return False
        records = data.get("records", [])
        if len(records) != EXPECTED_RECORDS_PER_PAIR:
            return False
        keys = {(r.get("direction"), r.get("config_id")) for r in records}
        if len(keys) != EXPECTED_RECORDS_PER_PAIR:
            return False
        if {r.get("direction") for r in records} != set(EXPECTED_DIRECTIONS):
            return False
        return True
    except Exception:
        return False


def run_baseline_gate(engine, pairs, compat_reference):
    errors = []

    for i, pair in enumerate(pairs, start=1):
        spec = base.checked_spec(engine, pair)
        out = engine.output(spec["clean"], spec["g"], spec["d"])

        compatibility = check_historical_compatibility(
            pair,
            out["margin"],
            compat_reference,
        )

        errors.append({
            "pair_id": pair["id"],
            "error": compatibility["historical_margin_error"],
            "compatibility_mode": compatibility["compatibility_mode"],
            "diagnostic_reference_error": compatibility["diagnostic_reference_error"],
        })

        if i % 25 == 0 or i == len(pairs):
            print(f"  baseline gate: {i}/{len(pairs)}", flush=True)

    maximum = max(x["error"] for x in errors)

    ordinary_failures = [
        x
        for x in errors
        if (
            x["pair_id"] not in HISTORICAL_EXCEPTION_IDS
            and x["error"] > HISTORICAL_MARGIN_TOL
        )
    ]

    if ordinary_failures:
        worst = sorted(
            ordinary_failures,
            key=lambda x: x["error"],
            reverse=True,
        )[:10]
        raise ValueError(
            "Full-178 historical-margin gate failed outside frozen exceptions: "
            + json.dumps(worst, indent=2)
        )

    observed_exceptions = {
        x["pair_id"]
        for x in errors
        if x["error"] > HISTORICAL_MARGIN_TOL
    }

    if observed_exceptions != HISTORICAL_EXCEPTION_IDS:
        raise ValueError(
            "Observed >0.05 pair set differs from frozen diagnostic exceptions: "
            + json.dumps(sorted(observed_exceptions))
        )

    max_reference_error = max(
        float(x["diagnostic_reference_error"] or 0.0)
        for x in errors
    )

    return {
        "passed": True,
        "pairs": len(errors),
        "tolerance": HISTORICAL_MARGIN_TOL,
        "exception_pair_ids": sorted(HISTORICAL_EXCEPTION_IDS),
        "exception_reference_tolerance": EXCEPTION_REFERENCE_TOL,
        "compatibility_diagnostic_sha256": compat_reference["sha256"],
        "max_historical_margin_error": maximum,
        "max_exception_reference_error": max_reference_error,
        "errors": errors,
    }


def run_self_gate(
    engine,
    pair,
    registry,
    groups,
    heads,
    full_z_layers,
    compat_reference,
):
    spec, clean, corr, historical_margin_error, compatibility = capture_pair(
        engine, pair, heads, full_z_layers, compat_reference
    )
    max_endpoint = 0.0
    max_clamp = 0.0
    tested = 0

    for label, recipient, ids in [
        ("clean_self", clean, spec["clean"]),
        ("corrupt_self", corr, spec["corr"]),
    ]:
        baseline = recipient["output"]
        for key, configs in groups.items():
            cfg0 = configs[0]
            positions = pair["masks"][cfg0["site"]]
            hybrid = engine.hybrid_for_config(
                ids=ids,
                config=cfg0,
                positions=positions,
                donor=recipient,
                recipient=recipient,
                heads=heads,
                g=spec["g"],
                d=spec["d"],
            )
            clamp = base.clamp_error(hybrid, recipient, cfg0, engine)
            if clamp is not None:
                max_clamp = max(max_clamp, float(clamp))

            for cfg in configs:
                result = engine.endpoint_for_config(
                    ids=ids,
                    config=cfg,
                    positions=positions,
                    hybrid=hybrid,
                    g=spec["g"],
                    d=spec["d"],
                    n=spec["n"],
                )
                err = max(
                    abs(float(result["clean_logit"]) - float(baseline["clean_logit"])),
                    abs(float(result["corr_logit"]) - float(baseline["corr_logit"])),
                    abs(float(result["margin"]) - float(baseline["margin"])),
                )
                max_endpoint = max(max_endpoint, err)
                tested += 1
            del hybrid

    base.require_le(max_endpoint, SELF_TOL, "extension self-donor endpoint error")
    base.require_le(max_clamp, 0.0, "extension non-live clamp error")
    return {
        "passed": True,
        "pair_id": pair["id"],
        "directions": ["clean_self", "corrupt_self"],
        "configs_per_direction": EXPECTED_CONFIGS,
        "hybrid_groups_per_direction": len(groups),
        "endpoint_tests": tested,
        "max_self_endpoint_error": max_endpoint,
        "max_nonlive_clamp_error": max_clamp,
        "historical_margin_error": historical_margin_error,
    }


def run_science(args):
    x = verify_extension_inputs(
        args.extension_inputs,
        args.frozen_inputs,
        args.smoke_report,
        args.analysis_dir,
    )
    pairs = x["pairs"]
    registry = x["registry"]
    groups = x["groups"]
    roles = x["roles"]
    priority = x["priority"]
    plan = x["plan"]

    import unittest
    from test_s43 import TinyS43

    tiny = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TinyS43)
    )
    if not tiny.wasSuccessful() or tiny.skipped or tiny.testsRun != 8:
        raise RuntimeError("Mandatory S43Engine tiny gate failed")

    from stage1_scan import load_model
    from s43_engine import S43Engine

    print("\nLoading pinned OLMo-2 model...", flush=True)
    t, tok, model, lock = load_model()
    if lock["revision"] != plan["model"]["revision"]:
        raise ValueError("Loaded model revision differs from frozen plan")
    if lock["model_id"] != plan["model"]["model_id"]:
        raise ValueError("Loaded model ID differs from frozen plan")

    engine = S43Engine(t, tok, model)

    heads = set()
    for cfg in registry:
        heads.add(int(cfg["source_id"]))
        if cfg.get("receiver_id") is not None:
            heads.add(int(cfg["receiver_id"]))
        heads.update(int(h) for h in cfg.get("live_head_ids", []))
    heads = sorted(heads)
    full_z_layers = sorted({
        int(h) // engine.H
        for cfg in registry
        for h in cfg.get("live_head_ids", [])
    })
    if full_z_layers != [16, 18]:
        raise ValueError(f"Unexpected selective-head layers: {full_z_layers}")

    runner_sha = sha256_path(Path(__file__))
    engine_sha = sha256_path(ROOT / "s43_engine.py")
    test_sha = sha256_path(ROOT / "test_s43.py")
    helper_sha = sha256_path(ROOT / "s43_run_v2.py")
    extension_manifest_sha = sha256_path(Path(args.extension_inputs) / "manifest.json")

    compat_reference = load_compatibility_reference(
        args.compat_diagnostic,
        model_id=plan["model"]["model_id"],
        revision=plan["model"]["revision"],
    )

    current_gpu = t.cuda.get_device_name(0)

    # The frozen compatibility reference was produced by the
    # baseline-only A100-SXM4-40GB diagnostic. The current
    # runtime is NOT accepted or rejected by GPU name.
    #
    # Instead, the family-034 exception prompts must reproduce
    # that frozen numerical reference within 1e-6. Therefore a
    # different accelerator can proceed only if it numerically
    # agrees with the already-frozen reference.
    identity = {
        "stage": "S4.3_extension_v3_family034_compat_hardware_independent",
        "model_id": plan["model"]["model_id"],
        "model_revision": plan["model"]["revision"],
        "extension_manifest_sha256": extension_manifest_sha,
        "runner_sha256": runner_sha,
        "s43_engine_sha256": engine_sha,
        "test_s43_sha256": test_sha,
        "s43_run_v2_helper_sha256": helper_sha,
        "population": "178 discovery pairs / 89 discovery families",
        "selected_contrasts": EXPECTED_SELECTED,
        "direct_prerequisites": EXPECTED_PREREQS,
        "configs_per_direction": EXPECTED_CONFIGS,
        "directions": list(EXPECTED_DIRECTIONS),
        "records_per_pair": EXPECTED_RECORDS_PER_PAIR,
        "heldout": "sealed",
        "historical_compatibility": (
            "strict <=0.05 historical clean-margin error for 176/178 pairs; "
            "only 034/direct/base/0/order0 and order1 use the frozen baseline-only "
            "diagnostic numerical reference at <=1e-6; current runtime GPU is "
            "recorded but not pre-whitelisted; no global tolerance change"
        ),
        "compatibility_reference_origin_gpu":
            "NVIDIA A100-SXM4-40GB",
        "historical_compatibility_exception_pair_ids":
            sorted(HISTORICAL_EXCEPTION_IDS),
        "historical_compatibility_exception_reference_tolerance":
            EXCEPTION_REFERENCE_TOL,
        "compatibility_diagnostic_sha256":
            compat_reference["sha256"],
        "compatibility_diagnostic_path":
            compat_reference["path"],
        "runtime_gpu":
            current_gpu,
    }
    identity_hash = digest_object(identity)

    out = Path(args.out)
    manifest_path = out / "run_manifest.json"

    if out.exists():
        if not args.resume:
            raise FileExistsError(
                "Output directory already exists. Use --resume only for the identical run:\n"
                + str(out)
            )
        if not manifest_path.exists():
            raise ValueError("Cannot resume output without run_manifest.json")
        old = read_json(manifest_path)
        if old["identity"] != identity:
            raise ValueError("Cannot resume after code/input/model identity changed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json_atomic(
            manifest_path,
            {
                "identity": identity,
                "identity_hash": identity_hash,
                "created": time.time(),
                "extension_inputs": str(Path(args.extension_inputs).resolve()),
                "frozen_inputs": str(Path(args.frozen_inputs).resolve()),
                "analysis_dir": str(Path(args.analysis_dir).resolve()),
                "smoke_report": str(Path(args.smoke_report).resolve()),
            },
        )

    chunks_dir = out / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        out / "tiny_gate.json",
        {"passed": True, "tests": tiny.testsRun, "test_s43_sha256": test_sha},
    )

    stop_requested = [False]

    def request_stop(signum, frame):
        stop_requested[0] = True
        print("\nStop requested; current unpublished pair will rerun on resume.", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    baseline_gate_path = out / "baseline_gate_178.json"
    if baseline_gate_path.exists():
        gate = read_json(baseline_gate_path)
        if not gate.get("passed") or gate.get("identity_hash") != identity_hash:
            raise ValueError("Invalid existing full-178 baseline gate")
        print("\n✓ Existing full-178 historical-margin gate verified.", flush=True)
    else:
        print("\nRunning full-178 historical-margin compatibility gate...", flush=True)
        start = time.time()
        gate = run_baseline_gate(engine, pairs, compat_reference)
        gate["identity_hash"] = identity_hash
        gate["seconds"] = time.time() - start
        write_json_atomic(baseline_gate_path, gate)
        print(
            "✓ Full-178 amended margin gate passed: "
            f"max historical error={gate['max_historical_margin_error']:.8f}, "
            f"max exception-reference error="
            f"{gate['max_exception_reference_error']:.8f} "
            f"({gate['seconds']:.1f}s)",
            flush=True,
        )

    self_gate_path = out / "self_gate.json"
    if self_gate_path.exists():
        gate = read_json(self_gate_path)
        if not gate.get("passed") or gate.get("identity_hash") != identity_hash:
            raise ValueError("Invalid existing extension self gate")
        print("✓ Existing bidirectional self-donor gate verified.", flush=True)
    else:
        print("Running bidirectional extension self-donor gate...", flush=True)
        start = time.time()
        gate = run_self_gate(
            engine,
            pairs[0],
            registry,
            groups,
            heads,
            full_z_layers,
            compat_reference,
        )
        gate["identity_hash"] = identity_hash
        gate["seconds"] = time.time() - start
        write_json_atomic(self_gate_path, gate)
        print(
            "✓ Bidirectional self gate passed: "
            f"max endpoint={gate['max_self_endpoint_error']:.8f}, "
            f"max clamp={gate['max_nonlive_clamp_error']:.8f} "
            f"({gate['seconds']:.1f}s)",
            flush=True,
        )

    done = sum(
        valid_existing_chunk(chunks_dir, pair, identity_hash)
        for pair in pairs
    )
    print(f"\nExisting complete pair chunks: {done}/{EXPECTED_PAIRS}", flush=True)

    def write_state(status, last_pair=None, seconds=None):
        payload = {
            "status": status,
            "completed_pairs": done,
            "expected_pairs": EXPECTED_PAIRS,
            "completed_records": done * EXPECTED_RECORDS_PER_PAIR,
            "expected_records": EXPECTED_TOTAL_RECORDS,
            "identity_hash": identity_hash,
            "updated": time.time(),
        }
        if last_pair is not None:
            payload["last_pair"] = last_pair
        if seconds is not None:
            payload["last_pair_seconds"] = seconds
        write_json_atomic(out / "state.json", payload)

    write_state("running")
    new_pairs = 0

    try:
        for pair_index, pair in enumerate(pairs, start=1):
            if valid_existing_chunk(chunks_dir, pair, identity_hash):
                continue
            if stop_requested[0]:
                break
            if args.max_new_pairs > 0 and new_pairs >= args.max_new_pairs:
                break

            print("\n" + "=" * 72, flush=True)
            print(f"Pair {pair_index}/{EXPECTED_PAIRS}: {pair['id']}", flush=True)
            start = time.time()
            data = measure_pair(
                engine=engine,
                pair=pair,
                registry=registry,
                groups=groups,
                heads=heads,
                full_z_layers=full_z_layers,
                roles=roles,
                priority=priority,
                compat_reference=compat_reference,
                stop_requested=stop_requested,
            )

            if stop_requested[0]:
                print(
                    "Stop requested before checkpoint publication; pair will rerun on resume.",
                    flush=True,
                )
                break

            seconds = time.time() - start
            data["seconds"] = seconds
            data["identity_hash"] = identity_hash
            chunk, ok = pair_chunk_paths(chunks_dir, pair)
            write_json_atomic(chunk, data)
            chunk_sha = sha256_path(chunk)
            write_json_atomic(
                ok,
                {
                    "pair_id": pair["id"],
                    "identity_hash": identity_hash,
                    "sha256": chunk_sha,
                    "records": EXPECTED_RECORDS_PER_PAIR,
                    "seconds": seconds,
                },
            )
            if not valid_existing_chunk(chunks_dir, pair, identity_hash):
                raise RuntimeError("Fresh extension checkpoint failed verification")

            done += 1
            new_pairs += 1
            write_state("running", pair["id"], seconds)
            print(
                f"✓ Saved pair {done}/{EXPECTED_PAIRS} ({seconds:.1f}s; "
                f"{EXPECTED_RECORDS_PER_PAIR} records)",
                flush=True,
            )
            del data
            gc.collect()
            engine.sync()

    except InterruptedError:
        stop_requested[0] = True

    done = sum(
        valid_existing_chunk(chunks_dir, pair, identity_hash)
        for pair in pairs
    )

    if done == EXPECTED_PAIRS:
        write_json_atomic(
            out / "done.json",
            {
                "complete": True,
                "pairs": EXPECTED_PAIRS,
                "families": EXPECTED_FAMILIES,
                "records": EXPECTED_TOTAL_RECORDS,
                "directions": list(EXPECTED_DIRECTIONS),
                "identity_hash": identity_hash,
                "completed": time.time(),
            },
        )
        write_state("complete")
        print("\n" + "=" * 72)
        print("S4.3 EXTENSION RUN COMPLETE")
        print("=" * 72)
        print(f"Pairs      : {EXPECTED_PAIRS} / {EXPECTED_PAIRS}")
        print(f"Families   : {EXPECTED_FAMILIES} / {EXPECTED_FAMILIES}")
        print(f"Records    : {EXPECTED_TOTAL_RECORDS:,} / {EXPECTED_TOTAL_RECORDS:,}")
        print("Directions : noise + restore")
        print("Held-out   : sealed")
    else:
        status = "interrupted" if stop_requested[0] else "paused"
        write_state(status)
        print("\nRun paused safely.")
        print(f"Complete pairs: {done}/{EXPECTED_PAIRS}")
        print("Resume with exactly the same command plus --resume.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-inputs", required=True)
    parser.add_argument("--frozen-inputs", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--smoke-report", required=True)
    parser.add_argument("--compat-diagnostic", required=True)
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-pairs", type=int, default=0)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    if args.audit:
        audit(
            args.extension_inputs,
            args.frozen_inputs,
            args.smoke_report,
            args.analysis_dir,
        )
        return

    if not args.out:
        parser.error("--out is required unless --audit is used")
    if args.max_new_pairs < 0:
        parser.error("--max-new-pairs must be >= 0")
    run_science(args)


if __name__ == "__main__":
    main()
