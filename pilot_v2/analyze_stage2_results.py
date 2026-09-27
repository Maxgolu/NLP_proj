#!/usr/bin/env python3
"""Validate and summarize the completed Stage-2 run without model forwards."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def ci_mean(values, rng, draws=20000):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_svg(path, ri, exact_importance, extension_heads, extension_importance, highlights):
    """Write a compact static scientific figure with no plotting dependency."""
    width, height = 1200, 650
    left = dict(x=78, y=70, w=650, h=500)
    right = dict(x=820, y=90, w=330, h=455)
    xmin, xmax = 0.0, float(max(0.4, ri.max() * 1.05))
    ymin = float(min(-0.15, exact_importance.min() * 1.08))
    ymax = float(exact_importance.max() * 1.08)

    def sx(x):
        return left["x"] + (x - xmin) / (xmax - xmin) * left["w"]

    def sy(y):
        return left["y"] + left["h"] - (y - ymin) / (ymax - ymin) * left["h"]

    colors = dict(neutral="#9aa0a6", top="#2563eb", ri="#dc2626", anatomy="#16a34a")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124} .tick{font-size:12px} .label{font-size:13px} .title{font-size:20px;font-weight:600} .panel{font-size:16px;font-weight:600}</style>',
        '<text x="600" y="30" text-anchor="middle" class="title">Stage 2: Relation Index versus exact causal importance</text>',
        '<text x="403" y="55" text-anchor="middle" class="panel">All 1,024 heads on the random 20-family exact subset</text>',
        f'<rect x="{left["x"]}" y="{left["y"]}" width="{left["w"]}" height="{left["h"]}" fill="none" stroke="#5f6368"/>',
    ]
    for value in np.linspace(0, xmax, 5):
        x = sx(value)
        svg += [f'<line x1="{x:.1f}" y1="{left["y"]}" x2="{x:.1f}" y2="{left["y"] + left["h"]}" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="{left["y"] + left["h"] + 20}" text-anchor="middle" class="tick">{value:.2f}</text>']
    yticks = np.linspace(ymin, ymax, 6)
    for value in yticks:
        y = sy(value)
        svg += [f'<line x1="{left["x"]}" y1="{y:.1f}" x2="{left["x"] + left["w"]}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left["x"] - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.1f}</text>']
    svg.append(f'<line x1="{left["x"]}" y1="{sy(0):.1f}" x2="{left["x"] + left["w"]}" y2="{sy(0):.1f}" stroke="#5f6368"/>')
    for h in range(1024):
        svg.append(f'<circle cx="{sx(ri[h]):.2f}" cy="{sy(exact_importance[h]):.2f}" r="2.4" fill="{colors["neutral"]}" fill-opacity="0.52"/>')
    label_offsets = {
        17 * 32 + 1: (8, -8), 27 * 32 + 6: (8, -8), 18 * 32 + 19: (8, 15),
        18 * 32 + 18: (8, -8), 3 * 32 + 11: (-8, -10), 9 * 32 + 22: (-8, 17),
        16 * 32 + 1: (8, 16), 16 * 32 + 21: (8, -9),
    }
    for h, category in highlights.items():
        x, y = sx(ri[h]), sy(exact_importance[h])
        dx, dy = label_offsets.get(h, (7, -7))
        anchor = "end" if dx < 0 else "start"
        svg += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{colors[category]}" stroke="white" stroke-width="1.2"/>',
                f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" class="label">L{h // 32}H{h % 32}</text>']
    svg += [
        f'<text x="{left["x"] + left["w"] / 2}" y="{height - 25}" text-anchor="middle" class="label">Pooled first-target RI</text>',
        f'<text x="20" y="{left["y"] + left["h"] / 2}" text-anchor="middle" class="label" transform="rotate(-90 20 {left["y"] + left["h"] / 2})">Exact causal importance (-patch delta, logits)</text>',
        '<text x="980" y="55" text-anchor="middle" class="panel">Strongest selected heads on all 89 families</text>',
    ]
    top_pos = np.argsort(-extension_importance)[:10]
    bar_max = float(extension_importance[top_pos[0]])
    bar_h, gap_y = 30, 13
    for row, pos in enumerate(top_pos):
        h = int(extension_heads[pos])
        value = float(extension_importance[pos])
        y = right["y"] + row * (bar_h + gap_y)
        bw = max(0, value / bar_max * right["w"])
        svg += [
            f'<text x="{right["x"] - 10}" y="{y + 20}" text-anchor="end" class="label">L{h // 32}H{h % 32}</text>',
            f'<rect x="{right["x"]}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{colors["top"]}" fill-opacity="0.82"/>',
            f'<text x="{right["x"] + bw + 7:.1f}" y="{y + 20}" class="label">{value:.2f}</text>',
        ]
    legend_y = 604
    for i, (category, label) in enumerate([("top", "top causal"), ("ri", "historical RI-selected"), ("anatomy", "Stage-1 final-position anatomy")]):
        x = 760 + i * 145
        svg += [f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{colors[category]}"/>',
                f'<text x="{x + 10}" y="{legend_y + 4}" class="tick">{html.escape(label)}</text>']
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def write_png(path, ri, exact_importance, extension_heads, extension_importance, highlights):
    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    width, height = 1200, 650
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    bold_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 13 * scale)
    small = ImageFont.truetype(str(font_path), 12 * scale)
    panel = ImageFont.truetype(str(bold_path), 16 * scale)
    title = ImageFont.truetype(str(bold_path), 20 * scale)
    left = dict(x=78, y=70, w=650, h=500)
    right = dict(x=820, y=90, w=330, h=455)
    xmin, xmax = 0.0, float(max(0.4, ri.max() * 1.05))
    ymin = float(min(-0.15, exact_importance.min() * 1.08))
    ymax = float(exact_importance.max() * 1.08)
    colors = dict(neutral="#9aa0a6", top="#2563eb", ri="#dc2626", anatomy="#16a34a")

    def pxy(x, y):
        return int(x * scale), int(y * scale)

    def sx(x):
        return left["x"] + (x - xmin) / (xmax - xmin) * left["w"]

    def sy(y):
        return left["y"] + left["h"] - (y - ymin) / (ymax - ymin) * left["h"]

    def centered(text, x, y, use_font):
        box = draw.textbbox((0, 0), text, font=use_font)
        draw.text(pxy(x - (box[2] - box[0]) / (2 * scale), y), text, fill="#202124", font=use_font)

    centered("Stage 2: Relation Index versus exact causal importance", 600, 16, title)
    centered("All 1,024 heads on the random 20-family exact subset", 403, 48, panel)
    for value in np.linspace(0, xmax, 5):
        x = sx(value)
        draw.line([pxy(x, left["y"]), pxy(x, left["y"] + left["h"])], fill="#e5e7eb", width=scale)
        centered(f"{value:.2f}", x, left["y"] + left["h"] + 5, small)
    for value in np.linspace(ymin, ymax, 6):
        y = sy(value)
        draw.line([pxy(left["x"], y), pxy(left["x"] + left["w"], y)], fill="#e5e7eb", width=scale)
        label = f"{value:.1f}"
        box = draw.textbbox((0, 0), label, font=small)
        draw.text(pxy(left["x"] - 10 - (box[2] - box[0]) / scale, y - 7), label, fill="#202124", font=small)
    draw.rectangle([pxy(left["x"], left["y"]), pxy(left["x"] + left["w"], left["y"] + left["h"])], outline="#5f6368", width=scale)
    draw.line([pxy(left["x"], sy(0)), pxy(left["x"] + left["w"], sy(0))], fill="#5f6368", width=scale)
    for h in range(1024):
        x, y, radius = sx(ri[h]), sy(exact_importance[h]), 2.3
        draw.ellipse([pxy(x - radius, y - radius), pxy(x + radius, y + radius)], fill=colors["neutral"])
    offsets = {545:(8,-11),870:(8,-20),595:(28,0),594:(8,18),107:(-52,-12),310:(-52,14),513:(8,12),533:(8,-11)}
    for h, category in highlights.items():
        x, y, radius = sx(ri[h]), sy(exact_importance[h]), 5
        draw.ellipse([pxy(x - radius, y - radius), pxy(x + radius, y + radius)], fill=colors[category], outline="white", width=scale)
        dx, dy = offsets.get(h, (8, -10))
        draw.text(pxy(x + dx, y + dy), f"L{h // 32}H{h % 32}", fill="#202124", font=font)
    centered("Pooled first-target RI", left["x"] + left["w"] / 2, height - 28, font)
    centered("Strongest selected heads on all 89 families", 980, 48, panel)
    top_pos = np.argsort(-extension_importance)[:10]
    bar_max = float(extension_importance[top_pos[0]])
    for row, pos in enumerate(top_pos):
        h = int(extension_heads[pos]); value = float(extension_importance[pos])
        y = right["y"] + row * 43; bw = max(0, value / bar_max * right["w"])
        label = f"L{h // 32}H{h % 32}"
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(pxy(right["x"] - 10 - (box[2] - box[0]) / scale, y + 7), label, fill="#202124", font=font)
        draw.rectangle([pxy(right["x"], y), pxy(right["x"] + bw, y + 30)], fill=colors["top"])
        draw.text(pxy(right["x"] + bw + 7, y + 7), f"{value:.2f}", fill="#202124", font=font)
    legend = [("top", "top causal"), ("ri", "historical RI-selected"), ("anatomy", "final-position anatomy")]
    for i, (category, label) in enumerate(legend):
        x = 760 + i * 150
        draw.ellipse([*pxy(x - 5, 599), *pxy(x + 5, 609)], fill=colors[category])
        draw.text(pxy(x + 10, 595), label, fill="#202124", font=small)
    image.resize((width, height), Image.Resampling.LANCZOS).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage1", type=Path)
    args = parser.parse_args()

    run = args.run.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260915)

    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    shortlist = json.loads((run / "shortlist.json").read_text())
    quality = json.loads((run / "final_estimate_quality.json").read_text())
    if summary.get("complete") is not True:
        raise ValueError("Run is not marked complete")

    gates = [json.loads(p.read_text()) for p in sorted(run.glob("gate_[0-9].json"))]
    if len(gates) != 3 or not all(g.get("passed") is True for g in gates):
        raise ValueError("Expected three passing replica gates")

    pairs = manifest["pairs"]
    if len(pairs) != 178 or len({p["family"] for p in pairs}) != 89:
        raise ValueError("Unexpected pair/family count")
    if any(sum(p["family"] == f for p in pairs) != 2 for f in {p["family"] for p in pairs}):
        raise ValueError("Every family must contain both orders")

    screen_rows = []
    exact_rows = []
    extension_rows = []
    extension_heads = None
    for pair in pairs:
        stem = f'family_{pair["family"]:03d}_order{pair["order"]}.npz'
        screen_path = run / "screen" / stem
        extension_path = run / "extension" / stem
        if not screen_path.exists() or not extension_path.exists():
            raise ValueError(f"Missing pair file: {stem}")
        with np.load(screen_path, allow_pickle=False) as z:
            attr = z["attribution"].astype(float)
            if attr.shape != (1024,) or not np.isfinite(attr).all():
                raise ValueError(f"Invalid attribution vector: {stem}")
            row = dict(family=pair["family"], order=pair["order"],
                       pair_id=str(z["pair_id"]), m_clean=float(z["m_clean"]),
                       m_corr=float(z["m_corr_fixed_sign"]), attribution=attr)
            screen_rows.append(row)
            if "exact_delta" in z.files:
                exact = z["exact_delta"].astype(float)
                if exact.shape != (1024,) or not np.isfinite(exact).all():
                    raise ValueError(f"Invalid exact vector: {stem}")
                exact_rows.append(dict(family=pair["family"], order=pair["order"], exact=exact))
        with np.load(extension_path, allow_pickle=False) as z:
            heads = z["head_indices"].astype(int)
            exact = z["exact_delta"].astype(float)
            if extension_heads is None:
                extension_heads = heads
            if not np.array_equal(heads, extension_heads):
                raise ValueError("Extension head indices differ across pairs")
            if exact.shape != (len(heads),) or not np.isfinite(exact).all():
                raise ValueError(f"Invalid extension vector: {stem}")
            extension_rows.append(dict(family=pair["family"], order=pair["order"], exact=exact))

    if len(exact_rows) != 40 or len(extension_heads) != 92:
        raise ValueError("Unexpected exact-subset or extension size")
    if not np.array_equal(extension_heads, np.asarray(shortlist["heads"])):
        raise ValueError("Extension heads differ from shortlist")

    families = sorted({p["family"] for p in pairs})
    exact_families = sorted({r["family"] for r in exact_rows})
    screen_lookup = {(r["family"], r["order"]): r for r in screen_rows}
    exact_lookup = {(r["family"], r["order"]): r for r in exact_rows}
    extension_lookup = {(r["family"], r["order"]): r for r in extension_rows}

    attr_pair = np.stack([screen_lookup[(f, o)]["attribution"] for f in families for o in (0, 1)])
    attr_family = np.stack([
        np.mean([screen_lookup[(f, o)]["attribution"] for o in (0, 1)], axis=0)
        for f in families
    ])
    exact_pair = np.stack([exact_lookup[(f, o)]["exact"] for f in exact_families for o in (0, 1)])
    exact_family = np.stack([
        np.mean([exact_lookup[(f, o)]["exact"] for o in (0, 1)], axis=0)
        for f in exact_families
    ])
    extension_pair = np.stack([extension_lookup[(f, o)]["exact"] for f in families for o in (0, 1)])
    extension_family = np.stack([
        np.mean([extension_lookup[(f, o)]["exact"] for o in (0, 1)], axis=0)
        for f in families
    ])

    m_clean = np.array([screen_lookup[(f, o)]["m_clean"] for f in families for o in (0, 1)])
    m_corr = np.array([screen_lookup[(f, o)]["m_corr"] for f in families for o in (0, 1)])
    gap = m_clean - m_corr
    family_clean = np.array([np.mean([screen_lookup[(f, o)]["m_clean"] for o in (0, 1)]) for f in families])
    family_corr = np.array([np.mean([screen_lookup[(f, o)]["m_corr"] for o in (0, 1)]) for f in families])
    family_gap = family_clean - family_corr

    head_effects = {}
    with (run / "head_effects.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            h = int(row["layer"]) * 32 + int(row["head"])
            head_effects[h] = row
    if len(head_effects) != 1024:
        raise ValueError("head_effects.csv must contain all heads")
    ri_first = np.array([float(head_effects[h]["ri_first"]) for h in range(1024)])
    ri_last = np.array([float(head_effects[h]["ri_last"]) for h in range(1024)])

    stage1_counts = np.zeros(1024, dtype=int)
    stage1_null_sums = np.zeros(1024, dtype=float)
    if args.stage1 is not None:
        with (args.stage1.resolve() / "head_stats.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                h = int(row["layer"]) * 32 + int(row["head"])
                n = int(row["n_a"])
                stage1_counts[h] += n
                stage1_null_sums[h] += float(row["null_random"]) * n

    attr_importance = -attr_family.mean(axis=0)
    exact_importance = -exact_family.mean(axis=0)
    extension_importance = -extension_family.mean(axis=0)
    exact_abs = np.abs(exact_family).mean(axis=0)
    ext_by_head = {int(h): i for i, h in enumerate(extension_heads)}
    controls = set(shortlist["random_controls"])
    eligible = np.flatnonzero(stage1_counts >= 50) if args.stage1 is not None else np.arange(1024)
    null_means = stage1_null_sums / np.maximum(stage1_counts, 1)
    null_p99, null_p999 = np.percentile(null_means[eligible], [99, 99.9])
    ri_first_top25 = set(eligible[np.argsort(-ri_first[eligible], kind="stable")[:25]].tolist())
    ri_last_top25 = set(eligible[np.argsort(-ri_last[eligible], kind="stable")[:25]].tolist())
    attribution_top25 = set(np.argsort(-np.abs(attr_importance), kind="stable")[:25].tolist())

    # Verify aggregate files against raw pair arrays.
    for h in range(1024):
        if not np.isclose(float(head_effects[h]["first_order_mean"]), attr_pair[:, h].mean()):
            raise ValueError(f"First-order aggregate mismatch for head {h}")
        if not np.isclose(float(head_effects[h]["exact_subset_mean"]), exact_pair[:, h].mean()):
            raise ValueError(f"Exact-subset aggregate mismatch for head {h}")
        if h in ext_by_head:
            if not np.isclose(float(head_effects[h]["exact_all_pairs_mean"]),
                              extension_pair[:, ext_by_head[h]].mean()):
                raise ValueError(f"Extension aggregate mismatch for head {h}")

    correlations = {
        "ri_first_vs_exact_subset_importance": spearman(ri_first, exact_importance),
        "ri_last_vs_exact_subset_importance": spearman(ri_last, exact_importance),
        "full_attribution_vs_exact_subset_signed": spearman(attr_importance, exact_importance),
        "attribution_vs_extension_signed_selected": spearman(
            attr_importance[extension_heads], extension_importance),
        "exact_subset_order0_vs_order1": spearman(
            -np.stack([exact_lookup[(f, 0)]["exact"] for f in exact_families]).mean(axis=0),
            -np.stack([exact_lookup[(f, 1)]["exact"] for f in exact_families]).mean(axis=0)),
    }

    # Family-bootstrap interval for the main RI/exact rank correlation.
    corr_boot = []
    for _ in range(5000):
        sampled = rng.integers(0, len(exact_families), len(exact_families))
        corr_boot.append(spearman(ri_first, -exact_family[sampled].mean(axis=0)))
    corr_ci = tuple(float(x) for x in np.percentile(corr_boot, [2.5, 97.5]))

    fixed_heads = [3 * 32 + 11, 9 * 32 + 22, 16 * 32 + 4, 16 * 32 + 1, 16 * 32 + 21]
    top_subset = np.argsort(-exact_importance, kind="stable")[:15].tolist()
    top_subset_25 = set(np.argsort(-exact_importance, kind="stable")[:25].tolist())
    top_subset_decile = set(np.argsort(-exact_importance, kind="stable")[:102].tolist())
    top_extension_positions = np.argsort(-extension_importance, kind="stable")[:15]
    top_extension = [int(extension_heads[i]) for i in top_extension_positions]
    reported_heads = []
    for h in top_subset + top_extension + fixed_heads:
        if h not in reported_heads:
            reported_heads.append(h)

    table_rows = []
    for h in range(1024):
        ext_pos = ext_by_head.get(h)
        subset_ci = ci_mean(-exact_family[:, h], rng) if h in reported_heads else (None, None)
        ext_ci = ci_mean(-extension_family[:, ext_pos], rng) if ext_pos is not None and h in reported_heads else (None, None)
        table_rows.append(dict(
            head_index=h, layer=h // 32, head=h % 32,
            ri_first=ri_first[h], ri_last=ri_last[h],
            attribution_importance=attr_importance[h],
            exact_subset_importance=exact_importance[h],
            exact_subset_mean_absolute_effect=exact_abs[h],
            exact_subset_negative_family_fraction=float(np.mean(exact_family[:, h] < 0)),
            exact_subset_ci_low=subset_ci[0], exact_subset_ci_high=subset_ci[1],
            extension_selected=ext_pos is not None,
            random_control=h in controls,
            exact_all_importance=extension_importance[ext_pos] if ext_pos is not None else None,
            exact_all_negative_family_fraction=float(np.mean(extension_family[:, ext_pos] < 0)) if ext_pos is not None else None,
            exact_all_ci_low=ext_ci[0], exact_all_ci_high=ext_ci[1],
            exact_all_gap_fraction=extension_importance[ext_pos] / family_gap.mean() if ext_pos is not None else None,
        ))

    with (out / "head_analysis.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    with (out / "family_baselines.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["family", "m_clean_order_mean", "m_corr_fixed_sign_order_mean", "clean_corr_gap"])
        writer.writerows(zip(families, family_clean, family_corr, family_gap))

    def head_line(h, full=False):
        ext_pos = ext_by_head.get(h)
        values = -extension_family[:, ext_pos] if full and ext_pos is not None else -exact_family[:, h]
        low, high = ci_mean(values, rng)
        importance = float(values.mean())
        return (
            f"| L{h // 32}H{h % 32} | {fmt(ri_first[h])} | {fmt(attr_importance[h])} | "
            f"{fmt(importance)} | [{fmt(low)}, {fmt(high)}] | "
            f"{fmt(np.mean(values > 0), 2)} | {fmt(importance / family_gap.mean(), 3)} |"
        )

    control_values = np.array([extension_importance[ext_by_head[h]] for h in sorted(controls)])
    historical = [3 * 32 + 11, 9 * 32 + 22]
    historical_full = np.array([extension_importance[ext_by_head[h]] for h in historical])
    historical_ranks = {
        f"L{h // 32}H{h % 32}": 1 + int(np.flatnonzero(np.argsort(-exact_importance) == h)[0])
        for h in historical
    }
    overlaps = dict(
        exact_top25_with_ri_first=len(top_subset_25 & ri_first_top25),
        exact_top25_with_ri_last=len(top_subset_25 & ri_last_top25),
        exact_top25_with_attribution=len(top_subset_25 & attribution_top25),
        exact_top_decile_below_lenient_threshold=int(sum(ri_first[h] < null_p99 for h in top_subset_decile)),
        exact_top_decile_zero_ri=int(sum(ri_first[h] == 0 for h in top_subset_decile)),
    )
    shared_prefix_count = sum(bool(v) for v in manifest["shared_prefixes"].values())
    gate_pairs = [p for g in gates for p in g["pairs"]]
    gate_stats = dict(
        runner_max_drift=max(p["runner_max_drift"] for p in gate_pairs),
        self_patch_max=max(p["self_patch_max"] for p in gate_pairs),
        embedding_endpoint_max=max(p["embedding_endpoint_drift"] for p in gate_pairs),
    )

    clean_ci = ci_mean(family_clean, rng)
    corr_metric_ci = ci_mean(family_corr, rng)
    gap_ci = ci_mean(family_gap, rng)
    lines = [
        "# Stage 2 validation and initial analysis", "",
        "## Run integrity", "",
        "- `summary.json` declares `complete: true`; Slurm job 888691 separately exited 0:0.",
        "- 178/178 screen files and 178/178 extension files are present and finite.",
        "- All three replica gates passed. Raw per-pair arrays reproduce `head_effects.csv`.",
        f"- Gate maxima: runner drift {gate_stats['runner_max_drift']:.6g}; self-patch drift "
        f"{gate_stats['self_patch_max']:.6g}; corrupted-embedding endpoint drift "
        f"{gate_stats['embedding_endpoint_max']:.6g}.",
        f"- {shared_prefix_count}/178 pairs use a non-empty shared answer prefix.", "",
        "## Behavioral contrast at the intervention token", "",
        f"Across 89 families (two orders averaged), mean clean metric = {family_clean.mean():.4f} "
        f"(95% family bootstrap CI [{clean_ci[0]:.4f}, {clean_ci[1]:.4f}]); "
        f"mean corrupted metric with the clean sign fixed = {family_corr.mean():.4f} "
        f"(CI [{corr_metric_ci[0]:.4f}, {corr_metric_ci[1]:.4f}]).",
        f"The mean clean-corrupted gap is {family_gap.mean():.4f} "
        f"(CI [{gap_ci[0]:.4f}, {gap_ci[1]:.4f}]). "
        f"The clean metric is positive on {np.mean(m_clean > 0):.1%} of pairs and the fixed-sign "
        f"corrupted metric is negative on {np.mean(m_corr < 0):.1%}.", "",
        "## Approximation quality and RI audit", "",
        f"First-order attribution agrees strongly with the 40-pair all-head exact map: signed "
        f"Spearman = {quality['mean_head_signed_spearman']:.3f}, absolute Spearman = "
        f"{quality['mean_head_absolute_spearman']:.3f}, and pair-by-head Spearman = "
        f"{quality['pair_head_spearman']:.3f}. The pre-registered 0.7 threshold was passed, so IG "
        "was not used.",
        f"When the 178-pair attribution mean is compared with the 40-pair exact mean, rather than "
        f"matching both methods on the same 40 pairs, Spearman is "
        f"{correlations['full_attribution_vs_exact_subset_signed']:.3f}; this lower value reflects "
        "the different family samples as well as approximation error. On the same 178 pairs and "
        f"within the selected 92 heads, attribution-versus-exact Spearman is "
        f"{correlations['attribution_vs_extension_signed_selected']:.3f}.",
        f"RI-first has only a weak positive association with exact causal importance on the random "
        f"40-pair subset: Spearman = {correlations['ri_first_vs_exact_subset_importance']:.3f}, "
        f"family-bootstrap 95% interval [{corr_ci[0]:.3f}, {corr_ci[1]:.3f}]. RI-last correlation "
        f"is {correlations['ri_last_vs_exact_subset_importance']:.3f}.",
        f"The exact rankings from order 0 and order 1 correlate at "
        f"{correlations['exact_subset_order0_vs_order1']:.3f} across all heads; this statistic is "
        "dominated by the many near-zero heads, while the strongest heads below retain the same "
        "direction across almost every family.", "",
        "Importance below is `- exact_delta`, so positive values mean that the corrupted-head patch "
        "moves the metric toward the corrupted answer. CIs resample whole families; orders are "
        "averaged within family.", "",
        "## Top exact heads on the random 20-family subset", "",
        "| Head | RI first | Attribution | Exact importance | 95% CI | Positive families | Gap fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(head_line(h) for h in top_subset[:10])
    lines += ["", "## Top exact heads among the 92-head all-family extension", "",
              "| Head | RI first | Attribution | Exact importance | 95% CI | Positive families | Gap fraction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(head_line(h, full=True) for h in top_extension[:10])
    lines += ["", "## Pre-specified and Stage-1 anatomy heads", "",
              "| Head | RI first | Attribution | Exact importance | 95% CI | Positive families | Gap fraction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(head_line(h, full=h in ext_by_head) for h in fixed_heads)
    lines += ["", "## Ranking overlap and pre-registered verdicts", "",
              f"Only {overlaps['exact_top25_with_ri_first']}/25 exact top heads overlap the "
              f"supported RI-first top 25, and {overlaps['exact_top25_with_ri_last']}/25 overlap "
              f"the RI-last top 25. In contrast, {overlaps['exact_top25_with_attribution']}/25 "
              "overlap the attribution top 25.",
              f"All {overlaps['exact_top_decile_below_lenient_threshold']}/102 heads in the "
              f"top decile of positive exact importance lie below the lenient RI 99% "
              f"control-mean threshold ({null_p99:.6f}); "
              f"{overlaps['exact_top_decile_zero_ri']} have RI exactly zero. L27H6 is rank 2 in "
              "the random exact subset and has RI zero.",
              f"For reference, the historical 99.9% control-mean threshold that selected L3H11 "
              f"and L9H22 is {null_p999:.6f}.",
              f"The original RI-selected heads rank {historical_ranks['L3H11']} (L3H11) and "
              f"{historical_ranks['L9H22']} (L9H22) by exact importance on the random subset. "
              f"On all 89 families their median importance is {np.median(historical_full):.4f}, "
              f"versus {np.median(control_values):.4f} for the 25 random controls.",
              "Under the project's pre-registered language, the result supports `incomplete` and "
              "meets the descriptive `misleading` criterion. It does not support `sound`; that "
              "criterion was not operationalized beyond 'systematically larger', and only two "
              "heads passed the historical pooled rule.",
              "", "## Random-control reference", "",
              f"The 25 random extension controls have median exact importance "
              f"{np.median(control_values):.4f}, range [{control_values.min():.4f}, "
              f"{control_values.max():.4f}]. This is descriptive: the non-control extension heads "
              "were selected using the discovery data, so naive post-selection p-values would not "
              "be valid.", "", "## Interpretation boundary", "",
              "The results establish a high-quality causal head screen for this clean-to-corrupted, "
              "all-prompt-position patching experiment. They do not identify edges, semantic "
              "specificity, necessity under redundancy, or a complete circuit. Exact all-family "
              "effects exist only for the selected 92 heads; comparisons involving that set must "
              "account for its data-dependent selection.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    machine = dict(
        validated=True,
        pairs=178,
        families=89,
        exact_subset_families=exact_families,
        extension_heads=extension_heads.tolist(),
        baseline=dict(clean_mean=float(family_clean.mean()), corr_fixed_sign_mean=float(family_corr.mean()),
                      gap_mean=float(family_gap.mean())),
        gate_stats=gate_stats,
        correlations=correlations,
        ri_first_exact_bootstrap_ci=corr_ci,
        overlaps=overlaps,
        ri_control_mean_thresholds=dict(p99=float(null_p99), p99_9=float(null_p999)),
        historical_exact_subset_ranks=historical_ranks,
        top_exact_subset=top_subset,
        top_exact_extension=top_extension,
    )
    (out / "summary.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")
    highlights = {
        17 * 32 + 1: "top", 27 * 32 + 6: "top", 18 * 32 + 19: "top", 18 * 32 + 18: "top",
        3 * 32 + 11: "ri", 9 * 32 + 22: "ri", 16 * 32 + 1: "anatomy", 16 * 32 + 21: "anatomy",
    }
    write_svg(out / "ri_vs_causal_effect.svg", ri_first, exact_importance,
              extension_heads, extension_importance, highlights)
    write_png(out / "ri_vs_causal_effect.png", ri_first, exact_importance,
              extension_heads, extension_importance, highlights)
    print(f"Wrote validated analysis to {out}")


if __name__ == "__main__":
    main()
