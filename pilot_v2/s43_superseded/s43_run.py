"""Restartable Stage 4.3 initial-panel runner.

Initial S4.3 scope
------------------
Population:
    20 common discovery families x 2 orders = 40 pairs.

Registry:
    249 main configurations
    56 additional matched direct comparators
    305 total scientific configurations.

Direction:
    clean recipient <- corrupt donor (noising) only.

The initial run does NOT:
    - use held-out families,
    - perform a broad search,
    - perform an automatic second round,
    - run reverse restoration,
    - select extension routes.

Reverse restoration and full-178 extension are later steps, after applying
the predeclared retention rule and extension cap.
"""

import argparse
import collections
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parent


# ============================================================
# Basic I/O / hashing
# ============================================================

def sha256_path(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def canonical_bytes(obj):
    return (
        json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def digest_object(obj):
    return hashlib.sha256(
        canonical_bytes(obj)
    ).hexdigest()


def read_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def read_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def read_jsonl_gz(path):
    rows = []

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_json_atomic(path, obj):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    os.replace(
        tmp,
        path,
    )


# ============================================================
# Head / registry helpers
# ============================================================

def hid(name):
    """L17H1 -> 545."""

    if (
        not name.startswith("L")
        or "H" not in name
    ):
        raise ValueError(
            f"Bad head name: {name}"
        )

    layer_text, head_text = (
        name[1:].split("H")
    )

    layer = int(layer_text)
    head = int(head_text)

    if not (
        0 <= layer < 32
        and 0 <= head < 32
    ):
        raise ValueError(
            f"Head outside OLMo2-7B: {name}"
        )

    return layer * 32 + head


def runtime_config(cfg):
    """Add execution-only numeric IDs without modifying frozen registry."""

    x = dict(cfg)

    x["live_head_ids"] = [
        hid(name)
        for name in x.get(
            "live_heads",
            [],
        )
    ]

    return x


def hybrid_key(cfg):
    """Everything that determines the source-replaced hybrid.

    Receiver and endpoint channel are deliberately excluded:
    one hybrid can be reused by several receiver endpoints.
    """

    value = {
        "source_id":
            int(cfg["source_id"]),

        "site":
            cfg["site"],

        "live_branches":
            sorted(
                [
                    [int(li), str(kind)]
                    for li, kind
                    in cfg.get(
                        "live_branches",
                        [],
                    )
                ]
            ),

        "live_head_ids":
            sorted(
                int(h)
                for h in cfg.get(
                    "live_head_ids",
                    [],
                )
            ),
    }

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def group_registry(registry):
    groups = collections.defaultdict(
        list
    )

    for cfg in registry:
        groups[
            hybrid_key(cfg)
        ].append(cfg)

    return dict(
        sorted(
            groups.items(),
            key=lambda x: x[0],
        )
    )


# ============================================================
# Frozen input verification
# ============================================================

def verify_frozen_bundle(
    inputs,
    smoke_report_path,
):
    inputs = Path(inputs)

    manifest_path = (
        inputs / "manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = read_json(
        manifest_path
    )

    # --------------------------------------------------------
    # Every file listed in the frozen manifest must still have
    # exactly the frozen SHA-256.
    # --------------------------------------------------------

    for name, expected in (
        manifest["files"].items()
    ):
        path = inputs / name

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        actual = sha256_path(
            path
        )

        if actual != expected:
            raise ValueError(
                "Frozen S4.3 input changed:\n"
                f"  file: {name}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

    plan = read_json(
        inputs / "plan.json"
    )

    pairs = read_jsonl_gz(
        inputs / "pairs.jsonl.gz"
    )

    registry = [
        runtime_config(x)
        for x in read_jsonl(
            inputs
            / "scientific_registry.jsonl"
        )
    ]

    # --------------------------------------------------------
    # Frozen population assertions
    # --------------------------------------------------------

    if len(pairs) != 178:
        raise ValueError(
            f"Expected 178 discovery pairs; got {len(pairs)}"
        )

    families = {
        p["family"]
        for p in pairs
    }

    if len(families) != 89:
        raise ValueError(
            f"Expected 89 discovery families; got {len(families)}"
        )

    if any(
        p["clean"]["split"] != "discovery"
        or p["corr"]["split"] != "discovery"
        for p in pairs
    ):
        raise ValueError(
            "Held-out leakage in frozen pair population"
        )

    common = sorted(
        [
            p
            for p in pairs
            if p["exact_all"]
        ],
        key=lambda p: (
            p["family"],
            p["order"],
        ),
    )

    if len(common) != 40:
        raise ValueError(
            f"Expected 40 common pairs; got {len(common)}"
        )

    common_families = {
        p["family"]
        for p in common
    }

    if len(common_families) != 20:
        raise ValueError(
            "Expected 20 common families"
        )

    if any(
        sum(
            q["family"] == family
            for q in common
        ) != 2
        for family in common_families
    ):
        raise ValueError(
            "Common panel is not 20 families x 2 orders"
        )

    # --------------------------------------------------------
    # Registry assertions
    # --------------------------------------------------------

    if len(registry) != 305:
        raise ValueError(
            f"Expected 305 configs; got {len(registry)}"
        )

    ids = [
        cfg["id"]
        for cfg in registry
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate S4.3 config IDs"
        )

    if (
        plan["population"]["heldout"]
        != "sealed"
    ):
        raise ValueError(
            "Frozen plan does not seal held-out families"
        )

    if (
        plan["registry"]["main"]
        != 249
        or plan["registry"][
            "additional_direct_comparators"
        ]
        != 56
        or plan["registry"][
            "scientific_total"
        ]
        != 305
    ):
        raise ValueError(
            "Frozen plan registry counts changed"
        )

    # --------------------------------------------------------
    # Verify successful real-model smoke gate against current
    # code and frozen bundle.
    # --------------------------------------------------------

    smoke_report_path = Path(
        smoke_report_path
    )

    if not smoke_report_path.exists():
        raise FileNotFoundError(
            "Real-model S4.3 smoke report missing:\n"
            + str(smoke_report_path)
        )

    smoke = read_json(
        smoke_report_path
    )

    if smoke.get("status") != "passed":
        raise ValueError(
            "Real-model S4.3 smoke gate did not pass"
        )

    frozen_manifest_sha = (
        sha256_path(
            manifest_path
        )
    )

    if (
        smoke[
            "frozen_input_manifest_sha256"
        ]
        != frozen_manifest_sha
    ):
        raise ValueError(
            "Smoke gate used a different frozen S4.3 bundle"
        )

    engine_path = (
        ROOT / "s43_engine.py"
    )

    test_path = (
        ROOT / "test_s43.py"
    )

    if (
        smoke["s43_engine_sha256"]
        != sha256_path(engine_path)
    ):
        raise ValueError(
            "s43_engine.py changed after real-model smoke gate"
        )

    if (
        smoke["test_s43_sha256"]
        != sha256_path(test_path)
    ):
        raise ValueError(
            "test_s43.py changed after real-model smoke gate"
        )

    expected_revision = (
        plan["model"]["revision"]
    )

    if (
        smoke["model"]["revision"]
        != expected_revision
    ):
        raise ValueError(
            "Smoke report model revision mismatch"
        )

    return (
        manifest,
        plan,
        pairs,
        common,
        registry,
        smoke,
    )


# ============================================================
# Execution-plan audit
# ============================================================

def audit(
    inputs,
    smoke_report,
):
    (
        manifest,
        plan,
        pairs,
        common,
        registry,
        smoke,
    ) = verify_frozen_bundle(
        inputs,
        smoke_report,
    )

    groups = group_registry(
        registry
    )

    # --------------------------------------------------------
    # Exact deduplicated execution counts
    # --------------------------------------------------------

    config_count = len(
        registry
    )

    pair_count = len(
        common
    )

    record_count = (
        config_count
        * pair_count
    )

    hybrid_count = len(
        groups
    )

    bypass_count = sum(
        cfg["channel"] == "bypass"
        for cfg in registry
    )

    endpoint_count = (
        config_count
        - bypass_count
    )

    # Each scientific pair:
    #
    #   1 clean intact capture
    #   1 corrupt intact capture
    #   N unique hybrid captures
    #   one fresh run for each non-bypass endpoint
    #
    forwards_per_pair = (
        2
        + hybrid_count
        + endpoint_count
    )

    total_science_forwards = (
        forwards_per_pair
        * pair_count
    )

    # Full self-donor gate:
    #
    #   1 intact clean capture
    #   hybrid_count self hybrids
    #   endpoint_count fresh endpoints
    #
    self_gate_forwards = (
        1
        + hybrid_count
        + endpoint_count
    )

    # --------------------------------------------------------
    # Expected exact counts from frozen registry
    # --------------------------------------------------------

    if config_count != 305:
        raise ValueError(
            f"Registry count drift: {config_count}"
        )

    if pair_count != 40:
        raise ValueError(
            f"Common pair count drift: {pair_count}"
        )

    if record_count != 12200:
        raise ValueError(
            f"Expected 12,200 records; got {record_count}"
        )

    if hybrid_count != 53:
        raise ValueError(
            "Hybrid deduplication count changed: "
            f"expected 53, got {hybrid_count}"
        )

    if bypass_count != 15:
        raise ValueError(
            f"Expected 15 bypass configs; got {bypass_count}"
        )

    if endpoint_count != 290:
        raise ValueError(
            f"Expected 290 endpoint runs; got {endpoint_count}"
        )

    # --------------------------------------------------------
    # Hybrid groups by source — useful independent audit.
    # --------------------------------------------------------

    source_groups = (
        collections.Counter()
    )

    for cfgs in groups.values():
        source_groups[
            cfgs[0]["source"]
        ] += 1

    expected_source_groups = {
        "L13H18": 18,
        "L8H15": 8,
        "L16H31": 5,
        "L20H7": 5,
        "L26H23": 5,
        "L15H25": 8,
        "L17H1": 4,
    }

    if dict(source_groups) != (
        expected_source_groups
    ):
        raise ValueError(
            "Unexpected hybrid-group decomposition:\n"
            f"actual={dict(source_groups)}\n"
            f"expected={expected_source_groups}"
        )

    # --------------------------------------------------------
    # Required heads
    # --------------------------------------------------------

    heads = set()

    for cfg in registry:
        heads.add(
            int(cfg["source_id"])
        )

        if cfg["receiver_id"] is not None:
            heads.add(
                int(cfg["receiver_id"])
            )

        heads.update(
            cfg["live_head_ids"]
        )

    heads = sorted(
        heads
    )

    full_z_layers = sorted({
        h // 32
        for cfg in registry
        for h in cfg[
            "live_head_ids"
        ]
    })

    if full_z_layers != [16, 18]:
        raise ValueError(
            "Unexpected selective-head layers: "
            + str(full_z_layers)
        )

    # --------------------------------------------------------
    # Display audit
    # --------------------------------------------------------

    print("=" * 76)
    print("S4.3 INITIAL PRODUCTION RUN — DRY AUDIT")
    print("=" * 76)

    print("\nFrozen input:")
    print(
        "  manifest SHA-256 :",
        sha256_path(
            Path(inputs)
            / "manifest.json"
        ),
    )

    print(
        "  registry SHA-256 :",
        manifest["files"][
            "scientific_registry.jsonl"
        ],
    )

    print(
        "  model revision    :",
        plan["model"]["revision"],
    )

    print(
        "  smoke gate        : PASSED"
    )

    print("\nPopulation:")
    print(
        f"  discovery pairs   : {len(pairs)}"
    )

    print(
        f"  discovery families: "
        f"{len({p['family'] for p in pairs})}"
    )

    print(
        f"  initial pairs     : {pair_count}"
    )

    print(
        f"  initial families  : "
        f"{len({p['family'] for p in common})}"
    )

    print(
        "  held-out          : SEALED"
    )

    print("\nRegistry:")
    print(
        f"  configs / pair    : {config_count}"
    )

    print(
        f"  main              : "
        f"{plan['registry']['main']}"
    )

    print(
        "  extra comparators : "
        f"{plan['registry']['additional_direct_comparators']}"
    )

    print(
        f"  total records     : {record_count:,}"
    )

    print("\nHybrid reuse:")
    print(
        f"  unique hybrids / pair : {hybrid_count}"
    )

    for source in [
        "L13H18",
        "L8H15",
        "L16H31",
        "L20H7",
        "L26H23",
        "L15H25",
        "L17H1",
    ]:
        print(
            f"    {source:8s}: "
            f"{source_groups[source]}"
        )

    print("\nEndpoint execution:")
    print(
        f"  bypass configs / pair : {bypass_count}"
    )

    print(
        f"  fresh endpoints / pair: {endpoint_count}"
    )

    print("\nForward-pass estimate:")
    print(
        "  intact captures / pair: 2"
    )

    print(
        f"  hybrid runs / pair     : {hybrid_count}"
    )

    print(
        f"  endpoint runs / pair   : {endpoint_count}"
    )

    print(
        f"  total / pair           : {forwards_per_pair}"
    )

    print(
        f"  40-pair science total  : "
        f"{total_science_forwards:,}"
    )

    print(
        f"  one-time self gate     : "
        f"{self_gate_forwards}"
    )

    print("\nSelective chain layers:")
    print(
        " ", full_z_layers
    )

    print("\nCaptured heads:")
    for h in heads:
        print(
            f"  {h:4d} = "
            f"L{h // 32}H{h % 32}"
        )

    print("\nDirection:")
    print(
        "  clean recipient <- corrupt donor"
    )

    print(
        "  initial run = NOISING ONLY"
    )

    print("\nExtension:")
    print(
        "  reverse restoration: NOT in initial run"
    )

    print(
        "  full-178 extension : NOT in initial run"
    )

    print(
        "  max later retained contrasts: "
        f"{plan['registry']['extension_max_route_contrasts']}"
    )

    print("\n✓ Frozen bundle verified.")
    print("✓ Real-model smoke provenance verified.")
    print("✓ 40-pair panel verified.")
    print("✓ 305-config registry verified.")
    print("✓ 53 hybrid groups verified.")
    print("✓ 12,200 expected records verified.")
    print("✓ Held-out families remain sealed.")
    print(
        "\nAUDIT ONLY — no scientific measurements were run."
    )

    return {
        "pairs": pair_count,
        "configs": config_count,
        "records": record_count,
        "hybrids": hybrid_count,
        "bypass": bypass_count,
        "endpoints": endpoint_count,
        "forwards_per_pair":
            forwards_per_pair,
        "science_forwards":
            total_science_forwards,
        "self_gate_forwards":
            self_gate_forwards,
    }


# ============================================================
# Runtime checks used by the actual measurement mode
# ============================================================

def output_error(a, b):
    return max(
        abs(
            a["clean_logit"]
            - b["clean_logit"]
        ),
        abs(
            a["corr_logit"]
            - b["corr_logit"]
        ),
    )


def require_le(
    value,
    limit,
    label,
):
    if not math.isfinite(
        float(value)
    ):
        raise ValueError(
            f"{label}: nonfinite"
        )

    if float(value) > limit:
        raise ValueError(
            f"{label}: "
            f"{float(value):.8f} > {limit}"
        )

    return float(value)


def checked_spec(
    engine,
    pair,
):
    spec = engine.encode(
        pair
    )

    n = spec["n"]

    if (
        spec["clean"][0, :n].tolist()
        != pair["original_ids"]["clean"]
    ):
        raise ValueError(
            "Clean token IDs changed: "
            + pair["id"]
        )

    if (
        spec["corr"][0, :n].tolist()
        != pair["original_ids"]["corr"]
    ):
        raise ValueError(
            "Corrupt token IDs changed: "
            + pair["id"]
        )

    if (
        spec["shared_prefix"]
        != pair["shared_prefix"]
    ):
        raise ValueError(
            "Shared answer prefix changed: "
            + pair["id"]
        )

    return spec


def clamp_error(
    hybrid,
    recipient,
    cfg,
    engine,
):
    live_heads = cfg.get(
        "live_head_ids",
        [],
    )

    if not live_heads:
        return None

    by_layer = (
        collections.defaultdict(
            set
        )
    )

    for h in live_heads:
        by_layer[
            h // engine.H
        ].add(
            h % engine.H
        )

    errors = []

    for li, live_local in (
        by_layer.items()
    ):
        mixed = (
            hybrid["full_z"][li]
            .reshape(
                -1,
                engine.H,
                engine.DH,
            )
        )

        intact = (
            recipient["full_z"][li]
            .reshape(
                -1,
                engine.H,
                engine.DH,
            )
        )

        frozen_local = [
            hi
            for hi in range(
                engine.H
            )
            if hi not in live_local
        ]

        if not frozen_local:
            continue

        err = float(
            (
                mixed[
                    :,
                    frozen_local,
                    :
                ]
                - intact[
                    :,
                    frozen_local,
                    :
                ]
            )
            .abs()
            .max()
        )

        # This clamp is an explicit tensor assignment.
        # The smoke gate demonstrated exact equality.
        require_le(
            err,
            0.0,
            (
                "non-live intermediate "
                f"head clamp L{li}"
            ),
        )

        errors.append(
            err
        )

    return (
        max(errors)
        if errors
        else 0.0
    )


# ============================================================
# One-time full 305-config self-donor gate
# ============================================================

def run_self_gate(
    engine,
    pair,
    registry,
    groups,
    heads,
    full_z_layers,
):
    spec = checked_spec(
        engine,
        pair,
    )

    g = spec["g"]
    d = spec["d"]
    n = spec["n"]

    recipient = (
        engine.capture_s43(
            spec["clean"],
            heads,
            g,
            d,
            full_z_layers=
                full_z_layers,
        )
    )

    historical_error = max(
        abs(
            recipient[
                "output"
            ]["clean_logit"]
            - pair["clean_logits"][0]
        ),
        abs(
            recipient[
                "output"
            ]["corr_logit"]
            - pair["clean_logits"][1]
        ),
    )

    require_le(
        historical_error,
        0.05,
        "self-gate historical logits",
    )

    require_le(
        recipient[
            "reconstruction_error"
        ],
        0.02,
        "self-gate SDPA reconstruction",
    )

    rows = []

    max_self_error = 0.0
    max_clamp_error = 0.0

    for group_index, (
        key,
        configs,
    ) in enumerate(
        groups.items(),
        start=1,
    ):
        cfg0 = configs[0]

        positions = pair[
            "masks"
        ][cfg0["site"]]

        hybrid = (
            engine.hybrid_for_config(
                ids=spec["clean"],
                config=cfg0,
                positions=positions,
                donor=recipient,
                recipient=recipient,
                heads=heads,
                g=g,
                d=d,
            )
        )

        clamp = clamp_error(
            hybrid,
            recipient,
            cfg0,
            engine,
        )

        if clamp is not None:
            max_clamp_error = max(
                max_clamp_error,
                clamp,
            )

        for cfg in configs:
            result = (
                engine.endpoint_for_config(
                    ids=spec["clean"],
                    config=cfg,
                    positions=positions,
                    hybrid=hybrid,
                    g=g,
                    d=d,
                    n=n,
                )
            )

            error = output_error(
                result,
                recipient["output"],
            )

            require_le(
                error,
                0.001,
                (
                    "self-donor endpoint "
                    + cfg["id"]
                ),
            )

            max_self_error = max(
                max_self_error,
                error,
            )

            rows.append({
                "config_id":
                    cfg["id"],

                "self_endpoint_error":
                    float(error),

                "clamp_error":
                    clamp,
            })

        del hybrid

        if (
            group_index % 10 == 0
            or group_index
            == len(groups)
        ):
            print(
                "  self gate: "
                f"{group_index}/"
                f"{len(groups)} "
                "hybrid groups",
                flush=True,
            )

    if len(rows) != 305:
        raise ValueError(
            "Self gate did not cover "
            "all 305 configurations"
        )

    return {
        "passed": True,
        "pair_id": pair["id"],
        "configs": len(rows),
        "hybrid_groups":
            len(groups),

        "historical_logit_error":
            historical_error,

        "sdpa_reconstruction_error":
            recipient[
                "reconstruction_error"
            ],

        "max_self_endpoint_error":
            max_self_error,

        "max_nonlive_clamp_error":
            max_clamp_error,

        "rows": rows,
    }


# ============================================================
# One scientific pair
# ============================================================

def measure_pair(
    engine,
    pair,
    registry,
    groups,
    heads,
    full_z_layers,
    stop_requested,
):
    spec = checked_spec(
        engine,
        pair,
    )

    g = spec["g"]
    d = spec["d"]
    n = spec["n"]

    # --------------------------------------------------------
    # Intact clean/corrupt captures.
    # --------------------------------------------------------

    clean = (
        engine.capture_s43(
            spec["clean"],
            heads,
            g,
            d,
            full_z_layers=
                full_z_layers,
        )
    )

    corr = (
        engine.capture_s43(
            spec["corr"],
            heads,
            g,
            d,
            full_z_layers=
                full_z_layers,
        )
    )

    historical_error = max(
        abs(
            clean[
                "output"
            ]["clean_logit"]
            - pair["clean_logits"][0]
        ),
        abs(
            clean[
                "output"
            ]["corr_logit"]
            - pair["clean_logits"][1]
        ),
    )

    require_le(
        historical_error,
        0.05,
        (
            "historical clean logits "
            + pair["id"]
        ),
    )

    require_le(
        clean[
            "reconstruction_error"
        ],
        0.02,
        (
            "clean SDPA reconstruction "
            + pair["id"]
        ),
    )

    require_le(
        corr[
            "reconstruction_error"
        ],
        0.02,
        (
            "corrupt SDPA reconstruction "
            + pair["id"]
        ),
    )

    baseline = clean[
        "output"
    ]

    rows = []

    max_clamp = 0.0

    # --------------------------------------------------------
    # One source-replaced hybrid per unique hybrid condition.
    # --------------------------------------------------------

    for group_index, (
        key,
        configs,
    ) in enumerate(
        groups.items(),
        start=1,
    ):
        if stop_requested[0]:
            raise InterruptedError(
                "Stop requested"
            )

        cfg0 = configs[0]

        positions = pair[
            "masks"
        ][cfg0["site"]]

        hybrid = (
            engine.hybrid_for_config(
                ids=spec["clean"],
                config=cfg0,
                positions=positions,
                donor=corr,
                recipient=clean,
                heads=heads,
                g=g,
                d=d,
            )
        )

        clamp = clamp_error(
            hybrid,
            clean,
            cfg0,
            engine,
        )

        if clamp is not None:
            max_clamp = max(
                max_clamp,
                clamp,
            )

        group_id = (
            hashlib.sha256(
                key.encode(
                    "utf-8"
                )
            )
            .hexdigest()[:16]
        )

        # ----------------------------------------------------
        # Reuse the same hybrid for all receiver endpoints
        # belonging to this source/release condition.
        # ----------------------------------------------------

        for cfg in configs:
            result = (
                engine.endpoint_for_config(
                    ids=spec["clean"],
                    config=cfg,
                    positions=positions,
                    hybrid=hybrid,
                    g=g,
                    d=d,
                    n=n,
                )
            )

            for value in result.values():
                if not math.isfinite(
                    float(value)
                ):
                    raise ValueError(
                        "Nonfinite endpoint "
                        + cfg["id"]
                    )

            # Initial S4.3 is NOISING:
            #
            # effect = M(clean)
            #        - M(clean with corrupt donor route)
            effect = (
                baseline["margin"]
                - result["margin"]
            )

            channel_norm = (
                engine.channel_norm_for_config(
                    config=cfg,
                    positions=positions,
                    hybrid=hybrid,
                    recipient=clean,
                    n=n,
                )
            )

            if (
                channel_norm is not None
                and not math.isfinite(
                    float(channel_norm)
                )
            ):
                raise ValueError(
                    "Nonfinite channel norm "
                    + cfg["id"]
                )

            rows.append({
                "pair_id":
                    pair["id"],

                "family":
                    pair["family"],

                "order":
                    pair["order"],

                "query_first":
                    pair["masks"][
                        "_query_first"
                    ],

                "config_id":
                    cfg["id"],

                "config":
                    cfg,

                "direction":
                    "noise",

                "hybrid_group":
                    group_id,

                "source_positions":
                    positions,

                "receiver_row":
                    n - 1,

                "shared_prefix":
                    pair[
                        "shared_prefix"
                    ],

                "baseline":
                    baseline,

                "result":
                    result,

                "effect":
                    float(effect),

                "channel_norm":
                    (
                        None
                        if channel_norm
                        is None
                        else float(
                            channel_norm
                        )
                    ),

                "nonlive_clamp_error":
                    clamp,

                "historical_logit_error":
                    historical_error,

                "clean_sdpa_error":
                    clean[
                        "reconstruction_error"
                    ],

                "corrupt_sdpa_error":
                    corr[
                        "reconstruction_error"
                    ],

                "freeze_semantics": (
                    "post-normalization branches "
                    ">= source frozen except source "
                    "attention and named releases"
                ),
            })

        del hybrid

        if (
            group_index % 10 == 0
            or group_index
            == len(groups)
        ):
            print(
                f"    hybrid groups "
                f"{group_index}/"
                f"{len(groups)}",
                flush=True,
            )

    if len(rows) != 305:
        raise ValueError(
            f"{pair['id']}: "
            f"expected 305 rows, "
            f"got {len(rows)}"
        )

    ids = [
        row["config_id"]
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate config measurement "
            + pair["id"]
        )

    return {
        "pair_id":
            pair["id"],

        "family":
            pair["family"],

        "order":
            pair["order"],

        "historical_logit_error":
            historical_error,

        "clean_sdpa_error":
            clean[
                "reconstruction_error"
            ],

        "corrupt_sdpa_error":
            corr[
                "reconstruction_error"
            ],

        "max_nonlive_clamp_error":
            max_clamp,

        "records":
            rows,
    }


# ============================================================
# Checkpoint validation
# ============================================================

def pair_chunk_paths(
    chunks_dir,
    pair,
):
    base = (
        f"f{pair['family']:03d}"
        f"_o{pair['order']}"
    )

    chunk = (
        chunks_dir
        / f"{base}.json"
    )

    ok = (
        chunks_dir
        / f"{base}.ok"
    )

    return chunk, ok


def valid_existing_chunk(
    chunks_dir,
    pair,
    identity_hash,
):
    chunk, ok = pair_chunk_paths(
        chunks_dir,
        pair,
    )

    if (
        not chunk.exists()
        and not ok.exists()
    ):
        return False

    if (
        not chunk.exists()
        or not ok.exists()
    ):
        raise ValueError(
            "Incomplete checkpoint pair: "
            + pair["id"]
        )

    meta = read_json(
        ok
    )

    if (
        meta["identity_hash"]
        != identity_hash
    ):
        raise ValueError(
            "Checkpoint belongs to "
            "different run identity: "
            + pair["id"]
        )

    actual = sha256_path(
        chunk
    )

    if (
        actual
        != meta["sha256"]
    ):
        raise ValueError(
            "Checkpoint hash mismatch: "
            + pair["id"]
        )

    data = read_json(
        chunk
    )

    if (
        data["pair_id"]
        != pair["id"]
    ):
        raise ValueError(
            "Checkpoint pair ID mismatch"
        )

    records = data[
        "records"
    ]

    if len(records) != 305:
        raise ValueError(
            "Checkpoint does not contain "
            "305 records: "
            + pair["id"]
        )

    config_ids = [
        x["config_id"]
        for x in records
    ]

    if (
        len(config_ids)
        != len(set(config_ids))
    ):
        raise ValueError(
            "Duplicate checkpoint config IDs"
        )

    return True


# ============================================================
# Production execution
# ============================================================

def run_science(args):
    (
        input_manifest,
        plan,
        all_pairs,
        pairs,
        registry,
        smoke,
    ) = verify_frozen_bundle(
        args.inputs,
        args.smoke_report,
    )

    groups = group_registry(
        registry
    )

    if len(groups) != 53:
        raise ValueError(
            "Expected 53 hybrid groups"
        )

    # --------------------------------------------------------
    # Tiny implementation gate must still pass in the runner's
    # actual Python environment before pretrained weights load.
    # --------------------------------------------------------

    import unittest

    from test_s43 import TinyS43

    tiny = (
        unittest.TextTestRunner(
            verbosity=1
        )
        .run(
            unittest
            .defaultTestLoader
            .loadTestsFromTestCase(
                TinyS43
            )
        )
    )

    if (
        not tiny.wasSuccessful()
        or tiny.skipped
        or tiny.testsRun != 8
    ):
        raise RuntimeError(
            "Mandatory S43Engine tiny gate failed"
        )

    # --------------------------------------------------------
    # Load pinned pretrained model.
    # --------------------------------------------------------

    from stage1_scan import (
        load_model,
    )

    from s43_engine import (
        S43Engine,
    )

    print(
        "\nLoading pinned OLMo-2 model...",
        flush=True,
    )

    t, tok, model, lock = (
        load_model()
    )

    if (
        lock["revision"]
        != plan["model"]["revision"]
    ):
        raise ValueError(
            "Loaded model revision differs "
            "from frozen S4.3 plan"
        )

    if (
        lock["model_id"]
        != plan["model"]["model_id"]
    ):
        raise ValueError(
            "Loaded model ID differs "
            "from frozen S4.3 plan"
        )

    engine = S43Engine(
        t,
        tok,
        model,
    )

    # --------------------------------------------------------
    # Capture every source, receiver and selective live head.
    # --------------------------------------------------------

    heads = set()

    for cfg in registry:
        heads.add(
            int(cfg["source_id"])
        )

        if cfg["receiver_id"] is not None:
            heads.add(
                int(cfg["receiver_id"])
            )

        heads.update(
            cfg[
                "live_head_ids"
            ]
        )

    heads = sorted(
        heads
    )

    full_z_layers = sorted({
        h // 32
        for cfg in registry
        for h in cfg[
            "live_head_ids"
        ]
    })

    if full_z_layers != [16, 18]:
        raise ValueError(
            "Unexpected selective-head layers"
        )

    # --------------------------------------------------------
    # Immutable run identity.
    # --------------------------------------------------------

    runner_sha = sha256_path(
        Path(__file__)
    )

    engine_sha = sha256_path(
        ROOT / "s43_engine.py"
    )

    test_sha = sha256_path(
        ROOT / "test_s43.py"
    )

    input_manifest_sha = (
        sha256_path(
            Path(args.inputs)
            / "manifest.json"
        )
    )

    identity = {
        "stage":
            "S4.3_initial_v1",

        "model_id":
            plan["model"][
                "model_id"
            ],

        "model_revision":
            plan["model"][
                "revision"
            ],

        "input_manifest_sha256":
            input_manifest_sha,

        "scientific_registry_sha256":
            input_manifest[
                "files"
            ][
                "scientific_registry.jsonl"
            ],

        "runner_sha256":
            runner_sha,

        "s43_engine_sha256":
            engine_sha,

        "test_s43_sha256":
            test_sha,

        "population":
            "40 common discovery pairs",

        "direction":
            "noise",

        "configs":
            305,

        "hybrid_groups":
            53,

        "heldout":
            "sealed",
    }

    identity_hash = (
        digest_object(
            identity
        )
    )

    # --------------------------------------------------------
    # Output initialization / resume.
    # --------------------------------------------------------

    out = Path(
        args.out
    )

    manifest_path = (
        out
        / "run_manifest.json"
    )

    if out.exists():
        if not args.resume:
            raise FileExistsError(
                "Output directory already exists. "
                "Use --resume only for the identical run:\n"
                + str(out)
            )

        if not manifest_path.exists():
            raise ValueError(
                "Cannot resume output without "
                "run_manifest.json"
            )

        old_manifest = read_json(
            manifest_path
        )

        if (
            old_manifest[
                "identity"
            ]
            != identity
        ):
            raise ValueError(
                "Cannot resume after code/input/"
                "model identity changed"
            )

    else:
        out.mkdir(
            parents=True,
            exist_ok=False,
        )

        write_json_atomic(
            manifest_path,
            {
                "identity":
                    identity,

                "identity_hash":
                    identity_hash,

                "created":
                    time.time(),

                "inputs":
                    str(
                        Path(
                            args.inputs
                        ).resolve()
                    ),

                "smoke_report":
                    str(
                        Path(
                            args.smoke_report
                        ).resolve()
                    ),
            },
        )

    chunks_dir = (
        out / "chunks"
    )

    chunks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json_atomic(
        out / "tiny_gate.json",
        {
            "passed": True,
            "tests": tiny.testsRun,
            "test_s43_sha256":
                test_sha,
        },
    )

    # --------------------------------------------------------
    # Graceful stop request.
    # Current pair is simply rerun on resume.
    # --------------------------------------------------------

    stop_requested = [False]

    def request_stop(
        signum,
        frame,
    ):
        stop_requested[0] = True

        print(
            "\nStop requested; "
            "finishing current safe boundary...",
            flush=True,
        )

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    signal.signal(
        signal.SIGINT,
        request_stop,
    )

    # --------------------------------------------------------
    # Mandatory 305-config self-donor gate once per run.
    # --------------------------------------------------------

    self_gate_path = (
        out / "self_gate.json"
    )

    if self_gate_path.exists():
        gate = read_json(
            self_gate_path
        )

        if (
            not gate.get("passed")
            or gate[
                "identity_hash"
            ]
            != identity_hash
        ):
            raise ValueError(
                "Invalid existing self gate"
            )

        print(
            "\n✓ Existing full self-donor "
            "gate verified.",
            flush=True,
        )

    else:
        print(
            "\nRunning one-time full "
            "305-config self-donor gate...",
            flush=True,
        )

        gate_start = time.time()

        gate = run_self_gate(
            engine=engine,
            pair=pairs[0],
            registry=registry,
            groups=groups,
            heads=heads,
            full_z_layers=
                full_z_layers,
        )

        gate[
            "identity_hash"
        ] = identity_hash

        gate[
            "seconds"
        ] = (
            time.time()
            - gate_start
        )

        write_json_atomic(
            self_gate_path,
            gate,
        )

        print(
            "✓ Full self-donor gate passed "
            f"in {gate['seconds']:.1f}s.",
            flush=True,
        )

    # --------------------------------------------------------
    # Resume accounting.
    # --------------------------------------------------------

    done = 0

    for pair in pairs:
        if valid_existing_chunk(
            chunks_dir,
            pair,
            identity_hash,
        ):
            done += 1

    print(
        f"\nExisting complete pair chunks: "
        f"{done}/40",
        flush=True,
    )

    write_json_atomic(
        out / "state.json",
        {
            "status":
                "running",

            "completed_pairs":
                done,

            "expected_pairs":
                40,

            "completed_records":
                done * 305,

            "expected_records":
                12200,

            "identity_hash":
                identity_hash,

            "updated":
                time.time(),
        },
    )

    new_pairs = 0

    # --------------------------------------------------------
    # Scientific initial-panel run.
    # --------------------------------------------------------

    try:
        for pair_index, pair in enumerate(
            pairs,
            start=1,
        ):
            if valid_existing_chunk(
                chunks_dir,
                pair,
                identity_hash,
            ):
                continue

            if stop_requested[0]:
                break

            if (
                args.max_new_pairs > 0
                and new_pairs
                >= args.max_new_pairs
            ):
                break

            print(
                "\n"
                + "=" * 72,
                flush=True,
            )

            print(
                f"Pair {pair_index}/40: "
                f"{pair['id']}",
                flush=True,
            )

            start = time.time()

            data = measure_pair(
                engine=engine,
                pair=pair,
                registry=registry,
                groups=groups,
                heads=heads,
                full_z_layers=
                    full_z_layers,
                stop_requested=
                    stop_requested,
            )

            if stop_requested[0]:
                print(
                    "Stop requested before "
                    "checkpoint publication; "
                    "pair will rerun on resume.",
                    flush=True,
                )

                break

            seconds = (
                time.time()
                - start
            )

            data[
                "seconds"
            ] = seconds

            data[
                "identity_hash"
            ] = identity_hash

            chunk, ok = (
                pair_chunk_paths(
                    chunks_dir,
                    pair,
                )
            )

            write_json_atomic(
                chunk,
                data,
            )

            chunk_sha = (
                sha256_path(
                    chunk
                )
            )

            write_json_atomic(
                ok,
                {
                    "pair_id":
                        pair["id"],

                    "identity_hash":
                        identity_hash,

                    "sha256":
                        chunk_sha,

                    "records":
                        305,

                    "seconds":
                        seconds,
                },
            )

            # Verify immediately after publication.
            if not valid_existing_chunk(
                chunks_dir,
                pair,
                identity_hash,
            ):
                raise RuntimeError(
                    "Fresh checkpoint failed "
                    "verification"
                )

            done += 1
            new_pairs += 1

            write_json_atomic(
                out / "state.json",
                {
                    "status":
                        "running",

                    "completed_pairs":
                        done,

                    "expected_pairs":
                        40,

                    "completed_records":
                        done * 305,

                    "expected_records":
                        12200,

                    "last_pair":
                        pair["id"],

                    "last_pair_seconds":
                        seconds,

                    "identity_hash":
                        identity_hash,

                    "updated":
                        time.time(),
                },
            )

            print(
                f"✓ Saved pair "
                f"{done}/40 "
                f"({seconds:.1f}s)",
                flush=True,
            )

            del data

            gc.collect()

            engine.sync()

    except InterruptedError:
        stop_requested[0] = True

    # --------------------------------------------------------
    # Final checkpoint recount.
    # --------------------------------------------------------

    done = sum(
        valid_existing_chunk(
            chunks_dir,
            pair,
            identity_hash,
        )
        for pair in pairs
    )

    if done == 40:
        total_records = (
            done * 305
        )

        if total_records != 12200:
            raise RuntimeError(
                "Unexpected final record count"
            )

        write_json_atomic(
            out / "done.json",
            {
                "complete": True,
                "pairs": 40,
                "records": 12200,
                "identity_hash":
                    identity_hash,
                "completed":
                    time.time(),
            },
        )

        write_json_atomic(
            out / "state.json",
            {
                "status":
                    "complete",

                "completed_pairs":
                    40,

                "expected_pairs":
                    40,

                "completed_records":
                    12200,

                "expected_records":
                    12200,

                "identity_hash":
                    identity_hash,

                "updated":
                    time.time(),
            },
        )

        print(
            "\n"
            + "=" * 72
        )

        print(
            "S4.3 INITIAL RUN COMPLETE"
        )

        print(
            "=" * 72
        )

        print(
            "Pairs   : 40 / 40"
        )

        print(
            "Records : 12,200 / 12,200"
        )

        print(
            "Direction: noise only"
        )

        print(
            "Held-out : sealed"
        )

    else:
        status = (
            "interrupted"
            if stop_requested[0]
            else "paused"
        )

        write_json_atomic(
            out / "state.json",
            {
                "status":
                    status,

                "completed_pairs":
                    done,

                "expected_pairs":
                    40,

                "completed_records":
                    done * 305,

                "expected_records":
                    12200,

                "identity_hash":
                    identity_hash,

                "updated":
                    time.time(),
            },
        )

        print(
            "\nRun paused safely."
        )

        print(
            f"Complete pairs: {done}/40"
        )

        print(
            "Resume with exactly the same "
            "command plus --resume."
        )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help=(
            "Frozen stage4_s43_inputs_v1 "
            "directory"
        ),
    )

    parser.add_argument(
        "--smoke-report",
        type=Path,
        required=True,
        help=(
            "Passed real-model "
            "smoke_report.json"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Production run output directory"
        ),
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Verify execution plan without "
            "loading the pretrained model"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the identical hashed run"
        ),
    )

    parser.add_argument(
        "--max-new-pairs",
        type=int,
        default=0,
        help=(
            "Optional Colab session limit. "
            "0 means all remaining pairs. "
            "A positive value processes only "
            "that many new pairs, then pauses "
            "safely."
        ),
    )

    args = parser.parse_args()

    if args.max_new_pairs < 0:
        raise ValueError(
            "--max-new-pairs must be >= 0"
        )

    if args.audit_only:
        audit(
            args.inputs,
            args.smoke_report,
        )

        return

    run_science(
        args
    )


if __name__ == "__main__":
    main()
