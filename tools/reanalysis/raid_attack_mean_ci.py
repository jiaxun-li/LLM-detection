"""Joint, domain-stratified source bootstrap of frozen all-attack TPR changes.

CPU only: stream existing packs, never fit or score with a language model.
Requires a NEW output directory. Accepted source results remain read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DETECTORS = ("log_likelihood", "rank", "log_rank", "lrr", "entropy", "entropy_gap", "binocular_origin")
ATTACKS = ("alternative_spelling", "article_deletion", "homoglyph", "insert_paragraphs", "number", "paraphrase", "perplexity_misspelling", "synonym", "upper_lower", "whitespace", "zero_width_space")
SOURCE_ID = "raid-full-excluded500-bootstrap500-provisional-20260824T132810Z"
REVISION_ID = "raid-origin-anchored-v41-20260907T044156Z"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


def joint_bootstrap(hits, domains, repetitions=2000, seed=481516):
    """hits: source x attack x detector x (raw, full, eligible).

    Every source has all eleven attacks, so mean(source attack-mean) equals
    the equal-attack mean. The SAME source multiplicities apply to all attacks,
    methods and scopes. Domain sample sizes remain fixed.
    """
    hits = np.asarray(hits)
    if hits.ndim != 4 or hits.shape[1] != 11 or hits.shape[3] != 3:
        raise ValueError("Expected complete source x 11 attacks x detector x 3 rules")
    if not np.isin(hits, [0, 1]).all() or len(domains) != len(hits) or len(hits) == 0:
        raise ValueError("Invalid source hits/domains")
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions required")
    return bootstrap_source_means(hits.mean(axis=1), domains, repetitions, seed)


def bootstrap_source_means(per_source, domains, repetitions=2000, seed=481516):
    """Paired domain-stratified bootstrap; one observation per source.

    For FPR, pass each human's three binary decisions directly (not replicated
    once per attack). For attack TPR, pass each source's eleven-attack mean.
    """
    per_source = np.asarray(per_source, dtype=float)
    if (per_source.ndim != 3 or per_source.shape[2] != 3 or len(per_source) == 0
            or len(domains) != len(per_source) or repetitions < 2
            or not np.isfinite(per_source).all()
            or np.any((per_source < 0) | (per_source > 1))):
        raise ValueError("Invalid per-source decisions/means")
    delta = per_source[:, :, 1:] - per_source[:, :, :1]
    groups = [np.flatnonzero(np.asarray(domains) == d) for d in sorted(set(domains))]
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, per_source.shape[1], 2))
    for rep in range(repetitions):
        total = np.zeros((per_source.shape[1], 2))
        for indices in groups:
            w = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)))
            total += np.einsum("i,ijk->jk", w, delta[indices])
        samples[rep] = total / len(per_source)
    return per_source.mean(axis=0), np.quantile(samples, [.025, .975], axis=0), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=481516)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError("Output must be a new directory; existing results are never overwritten")
    if args.repetitions < 2:
        raise ValueError("At least two repetitions required")
    # Import the existing frozen-score implementation, never the fitting runner.
    from RAID.raid_evaluation import source, domain, split, condition, oriented_document_score

    root = args.workspace.resolve()
    results = root / "results/raid" / REVISION_ID
    source_dir = root / "runs/raid" / SOURCE_ID
    origin_dir = root / "runs/raid_origin" / REVISION_ID
    marker = json.loads((results / "revision.complete.json").read_text())
    manifest = json.loads((results / "revision_manifest.json").read_text())
    if marker.get("validation_status") != "pass" or manifest.get("completion_status") != "complete":
        raise ValueError("Accepted revision must be complete and validated")
    if manifest.get("implementation_version") != "official-score-anchored-clipping-v4.1":
        raise ValueError("Unexpected accepted implementation")
    if manifest.get("source_run_id") != SOURCE_ID or manifest.get("revision_id") != REVISION_ID:
        raise ValueError("Unexpected source/revision identity")
    small_inputs = [results / name for name in ("frozen_specs.json", "attack_summary.csv")]
    for path in small_inputs:
        if digest(path) != marker["artifacts"][path.name]:
            raise ValueError(f"Accepted artifact hash mismatch: {path}")
    frozen = json.loads(small_inputs[0].read_text())["detectors"]
    with small_inputs[1].open(newline="") as handle:
        reference = {(r["detector"], r["condition"]): r for r in csv.DictReader(handle)}
    packs = [(source_dir / "falcon_scores.jsonl", DETECTORS[:-1], 167323)]
    for i in range(4):
        path = origin_dir / f"part-{i:05d}-of-00004.jsonl"
        shard = json.loads(path.with_suffix(".complete.json").read_text())
        if path.stat().st_size != shard["output"]["size"]:
            raise ValueError(f"Origin pack size mismatch: {path}")
        packs.append((path, ("binocular_origin",), shard["score_rows"]))
    # Verify unchanged inputs throughout the run and record full pack hashes.
    inputs = small_inputs + [results / "revision_manifest.json", results / "revision.complete.json"] + [p for p, _, _ in packs]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    records = {d: {} for d in DETECTORS}
    domains = {}
    hashes = {str(p): digest(p) for p in inputs[:4]}
    for path, detectors, expected_rows in packs:
        h = hashlib.sha256()
        count = 0
        print(f"Reading {path.name}", flush=True)
        with path.open("rb") as handle:
            for line in handle:
                h.update(line)
                row = json.loads(line)
                count += 1
                if split(row) != "test":
                    continue
                sid, dom, cond = source(row), domain(row), condition(row)
                if cond not in (*ATTACKS, "none", "human"):
                    raise ValueError(f"Unexpected condition {cond}")
                if sid in domains and domains[sid] != dom:
                    raise ValueError("Source crosses domains")
                domains[sid] = dom
                for detector in detectors:
                    key = (sid, cond)
                    if key in records[detector]:
                        raise ValueError(f"Duplicate source/condition: {detector} {key}")
                    spec = frozen[detector]
                    t = spec["classification_thresholds"][dom]
                    et = spec["eligible_universal_classification_thresholds"][dom]
                    if t["raw"] != et["raw"]:
                        raise ValueError("Raw thresholds disagree between scopes")
                    scores = [oriented_document_score(row, detector, spec["direction"], s) for s in
                              ({}, spec["clipping_specification"], spec["eligible_universal_clipping_specification"])]
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite reconstructed score")
                    records[detector][key] = tuple(int(v >= threshold) for v, threshold in
                                                  zip(scores, (t["raw"], t["clipped"], et["clipped"])))
                if count % 10000 == 0:
                    print(f"  rows read: {count}", flush=True)
        if count != expected_rows:
            raise ValueError(f"Pack row count mismatch: {path}: {count}")
        hashes[str(path)] = {"size": path.stat().st_size, "sha256": h.hexdigest()}
    ids = sorted(domains)
    if len(ids) != 5149:
        raise ValueError(f"Expected 5149 test sources, got {len(ids)}")
    expected_keys = {(sid, c) for sid in ids for c in (*ATTACKS, "none", "human")}
    for d in DETECTORS:
        if set(records[d]) != expected_keys:
            raise ValueError(f"Incomplete test source families: {d}")
        for c in (*ATTACKS, "none", "human"):
            point = np.mean([records[d][sid, c] for sid in ids], axis=0)
            ref = reference[d, "none" if c == "human" else c]
            suffix = "test_fpr" if c == "human" else "tpr"
            expected = [float(ref[p + suffix]) for p in ("raw_", "clipped_", "eligible_universal_clipped_")]
            if not np.allclose(point, expected, rtol=0, atol=1e-12):
                raise ValueError(f"Frozen point replay disagrees: {d} {c}: {point} != {expected}")
    print("All per-attack TPRs and human FPRs match accepted results: PASS", flush=True)
    hits = np.asarray([[[records[d][sid, c] for d in DETECTORS] for c in ATTACKS] for sid in ids], dtype=np.uint8)
    point, intervals, samples = joint_bootstrap(hits, [domains[sid] for sid in ids], args.repetitions, args.seed)
    human_hits = np.asarray([[records[d][sid, "human"] for d in DETECTORS] for sid in ids], dtype=np.uint8)
    fpoint, fintervals, fsamples = bootstrap_source_means(
        human_hits, [domains[sid] for sid in ids], args.repetitions, args.seed)
    if before != {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}:
        raise ValueError("Input changed during analysis")
    output_rows = []
    for i, detector in enumerate(DETECTORS):
        for j, scope in enumerate(("full", "eligible")):
            output_rows.append({"detector": detector, "scope": scope, "raw_tpr": float(point[i, 0]),
                               "clipped_tpr": float(point[i, j+1]), "paired_tpr_difference": float(point[i, j+1]-point[i, 0]),
                               "ci_low": float(intervals[0, i, j]), "ci_high": float(intervals[1, i, j])})
    out.mkdir(parents=True, exist_ok=False)
    fpr_rows = []
    for i, detector in enumerate(DETECTORS):
        for j, scope in enumerate(("full", "eligible")):
            fpr_rows.append({"detector": detector, "scope": scope,
                "raw_fpr": float(fpoint[i, 0]), "clipped_fpr": float(fpoint[i, j+1]),
                "paired_fpr_difference": float(fpoint[i, j+1]-fpoint[i, 0]),
                "ci_low": float(fintervals[0, i, j]), "ci_high": float(fintervals[1, i, j])})
    with (out / "paired_fpr_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fpr_rows[0]))
        writer.writeheader()
        writer.writerows(fpr_rows)
    with (out / "all_attack_mean_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    np.savez_compressed(out / "joint_bootstrap.npz", changes=samples, fpr_changes=fsamples, detectors=DETECTORS, scopes=("full", "eligible"))
    np.savez_compressed(out / "test_decisions.npz", hits=hits, human_hits=human_hits, source_ids=ids,
                        domains=[domains[sid] for sid in ids], detectors=DETECTORS, attacks=ATTACKS)
    (out / "analysis.complete.json").write_text(json.dumps({"status": "complete", "revision_id": REVISION_ID,
        "bootstrap_repetitions": args.repetitions, "seed": args.seed, "sources": len(ids), "attacks": ATTACKS,
        "procedure": "domain-stratified paired source bootstrap; equal attack weights; all fits fixed; pointwise 95% percentile intervals",
        "source_hashes": hashes, "script_sha256": digest(Path(__file__)), "point_replay": "pass",
        "implementation_hashes": {name: digest(ROOT / name) for name in
            ("RAID/raid_evaluation.py", "experiment_core/detectors/detector_revision.py")},
        "output_hashes": {name: digest(out / name) for name in
            ("all_attack_mean_ci.csv", "paired_fpr_ci.csv", "joint_bootstrap.npz", "test_decisions.npz")}}, indent=2))
    print("\nALL-ATTACK MEAN TPR CHANGE: percentage points [95% CI]", flush=True)
    for r in output_rows:
        print(f"{r['detector']:18} {r['scope']:8} {100*r['paired_tpr_difference']:+.3f} [{100*r['ci_low']:+.3f}, {100*r['ci_high']:+.3f}]")
    print("\nHUMAN FPR CHANGE: percentage points [95% CI]; positive = more false positives", flush=True)
    for r in fpr_rows:
        print(f"{r['detector']:18} {r['scope']:8} {100*r['paired_fpr_difference']:+.3f} [{100*r['ci_low']:+.3f}, {100*r['ci_high']:+.3f}]")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
