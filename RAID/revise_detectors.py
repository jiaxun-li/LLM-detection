#!/usr/bin/env python3
"""Preservation-safe RAID origin parity gate, scoring shards, and reanalysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_detection.io import atomic_write_json, iter_jsonl
from llm_detection.detector_revision import REVISION, IMPLEMENTATION_VERSION
from llm_detection.evaluation import REVISED_METHODS
from llm_detection.revision_artifacts import RAID_ARTIFACTS, finish_revision, resume_completed_revision


def identifier(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in value) or value in {".", ".."}:
        raise ValueError("run IDs must be single safe directory names")
    return value


def fingerprint(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def implementation_fingerprint():
    names=("RAID/binocular_origin.py","RAID/revise_detectors.py","RAID/raid_evaluation.py",
           "llm_detection/detector_revision.py","llm_detection/evaluation.py",
           "llm_detection/scoring.py","RAID/raid_scoring.py",
           "scripts/reevaluate_detector_revision.py", "llm_detection/revision_artifacts.py",
           "RAID/revision_gate.py")
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scorer_config(source):
    first = next(iter_jsonl(source / "binoculars_scores.jsonl"))
    observer = first["binoculars_observer_revision"]
    performer = first["binoculars_performer_revision"]
    # Require resolved immutable revisions, never silently fall back to main.
    for value in (observer, performer):
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("source pair must record resolved model commits")
    return {"observer_revision": observer, "performer_revision": performer,
        "tokenizer_revision": first["binoculars_tokenizer_revision"],
        "performer_tokenizer_revision": first["binoculars_performer_tokenizer_revision"],
        "dtype": "bf16", "max_tokens": 512,
        "observer_device": "cuda:0", "performer_device": "cuda:1",
        "trust_remote_code": False}


def compact_rows(path):
    import numpy as np
    from llm_detection.evaluation import EVALUATION_TOKEN_FEATURES
    for row in iter_jsonl(path, tolerate_partial_last_line=False):
        yield {**{k:v for k,v in row.items() if k not in {"text","prompt","token_features","document_features"}},
            "token_features": {k:np.asarray(v,dtype=float) for k,v in row["token_features"].items()
                               if k in EVALUATION_TOKEN_FEATURES}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("."))
    p.add_argument("--source-run-id", required=True, type=identifier)
    p.add_argument("--revision-id", required=True, type=identifier)
    p.add_argument("--stage", choices=("gate","score","evaluate"), required=True)
    p.add_argument("--reference-metrics", type=Path)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = p.parse_args()
    if args.source_run_id == args.revision_id:
        raise ValueError("use a NEW revision ID")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.bootstrap_repetitions < 0:
        raise ValueError("invalid shard or bootstrap count")
    root = args.workspace.resolve()
    source = root / "runs/raid" / args.source_run_id
    destination = root / "runs/raid_origin" / args.revision_id
    results = root / "results/raid" / args.revision_id
    source_manifest = read_json(source / "manifest.json")
    if source_manifest.get("completion_status") != "complete":
        raise ValueError("source run must be complete")
    config = scorer_config(source)
    cache_path = root/"results/raid"/args.source_run_id/"contamination_records.csv"
    identity = {"revision": REVISION, "implementation_version": IMPLEMENTATION_VERSION,
        "source_run_id": args.source_run_id,
        "revision_id": args.revision_id, "num_shards": args.num_shards,
        "scorer_config": config,"implementation_sha256":implementation_fingerprint(),
        "source_exclusions":source_manifest.get("source_exclusions"),
        "contamination_cache":fingerprint(cache_path),
        "prepared_shards":{str(i):fingerprint(source/"data_shards"/f"part-{i:05d}-of-{args.num_shards:05d}.jsonl")
                           for i in range(args.num_shards)},
        "source_artifacts": {name:fingerprint(source/name) for name in (
            "manifest.json","data.jsonl","falcon_scores.jsonl","binoculars_scores.jsonl")}}
    destination.mkdir(parents=True, exist_ok=True)
    gate_path = destination / "gate.complete.json"
    if args.stage == "gate":
        if gate_path.exists():
            raise ValueError("gate already exists; use a new revision ID to change it")
        if args.reference_metrics is None:
            raise ValueError("gate needs the reviewed upstream binoculars/metrics.py")
        for index in range(args.num_shards):
            if not (source/"data_shards"/f"part-{index:05d}-of-{args.num_shards:05d}.jsonl").is_file():
                raise ValueError("requested shard layout is not present in source run")
        spec = importlib.util.spec_from_file_location("upstream_binoculars_metrics", args.reference_metrics)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        from RAID.revision_gate import load_gate_tokenizer, preflight_inputs, run_pipeline_probe
        preflight, selected = preflight_inputs(source/"data.jsonl",
            [source/"data_shards"/f"part-{i:05d}-of-{args.num_shards:05d}.jsonl" for i in range(args.num_shards)],
            load_gate_tokenizer(config), destination/"tokenizer_preflight.json")
        from RAID.binocular_origin import RAIDBinocularsOriginScorer
        scorer = RAIDBinocularsOriginScorer(config, reference_metrics=reference)
        pipeline = run_pipeline_probe(selected,scorer,source,cache_path,destination,
                                      source_manifest["protocol_config"])
        probes = [
            {"text":"This is a long sentence. " * 150},
            {"text":"A\u200b sentence with \u0430 Unicode homoglyph and number 42."}]
        for row in probes: scorer.score_batch([row])
        if (identity["source_artifacts"] != {name:fingerprint(source/name) for name in identity["source_artifacts"]}
                or identity["contamination_cache"] != fingerprint(cache_path)
                or identity["prepared_shards"] != {str(i):fingerprint(source/"data_shards"/f"part-{i:05d}-of-{args.num_shards:05d}.jsonl") for i in range(args.num_shards)}):
            raise ValueError("source inputs changed during gate")
        atomic_write_json(gate_path, {**identity, "status":"pass",
            "tokenizer_preflight":preflight, "pipeline_probe":pipeline,
            "parity_checks":scorer.parity_checks,
            "parity_scope":"same_logits_upstream_mean_components",
            "reference_sha256":hashlib.sha256(args.reference_metrics.read_bytes()).hexdigest()})
        print("RAID origin upstream component gate: PASS", flush=True)
        return
    gate = read_json(gate_path)
    if (gate.get("status") != "pass" or gate.get("parity_checks",0)<15
            or gate.get("tokenizer_preflight",{}).get("status") != "pass"
            or gate.get("pipeline_probe",{}).get("status") != "pass"
            or any(gate.get(k) != v for k,v in identity.items())):
        raise ValueError("passing gate does not match this source/configuration")
    if args.stage == "score":
        from RAID.binocular_origin import RAIDBinocularsOriginScorer
        from RAID.raid_scoring import score_raid_jsonl
        name = f"part-{args.shard_index:05d}-of-{args.num_shards:05d}"
        prepared = source / "data_shards" / (name + ".jsonl")
        # No preparation, hashing of the RAID CSV, or target-model rescoring.
        output = destination / (name + ".jsonl")
        scorer = RAIDBinocularsOriginScorer(config)
        count = score_raid_jsonl(prepared, output, scorer,
            {"batch_size":1,"microbatch_size":1,"checkpoint_interval":100})
        if fingerprint(prepared) != identity["prepared_shards"][str(args.shard_index)]:
            raise ValueError("prepared shard changed during scoring")
        atomic_write_json(destination/(name+".complete.json"),
            {**identity,"score_rows":count,"input":fingerprint(prepared),"output":fingerprint(output)})
        return
    if (results/"revision_manifest.json").exists():
        previous=read_json(results/"revision_manifest.json")
        if any(previous.get(k)!=v for k,v in identity.items()) or previous["protocol_config"]["evaluation"]["bootstrap_repetitions"]!=args.bootstrap_repetitions:
            raise ValueError("result directory belongs to a different revision/configuration")
        if resume_completed_revision(results,previous,RAID_ARTIFACTS):
            print("RAID revision already complete");return
    elif results.exists() and any(p.name != "revision_manifest.json.tmp" for p in results.iterdir()):
        raise ValueError("nonempty result directory has no revision identity")
    from RAID.raid_evaluation import evaluate_raid, merge_score_rows, write_evaluation_artifacts
    origin = []
    for index in range(args.num_shards):
        name = f"part-{index:05d}-of-{args.num_shards:05d}"
        marker = read_json(destination/(name+".complete.json"))
        path = destination/(name+".jsonl")
        prepared = source/"data_shards"/(name+".jsonl")
        if any(marker.get(k)!=v for k,v in identity.items()) or marker["output"]!=fingerprint(path) or marker["input"]!=fingerprint(prepared):
            raise ValueError("shard identity/provenance mismatch")
        part = list(compact_rows(path))
        if len(part)!=marker["score_rows"]:raise ValueError("shard count mismatch")
        origin.extend(part)
    rows = merge_score_rows(list(compact_rows(source/"falcon_scores.jsonl")),
                           list(compact_rows(source/"binoculars_scores.jsonl")))
    if len(origin)!=len(rows):raise ValueError("origin row count mismatch")
    rows = merge_score_rows(rows, origin)
    del origin
    with cache_path.open(encoding="utf-8",newline="") as handle:
        cache = list(csv.DictReader(handle))
    config = source_manifest["protocol_config"]
    config["evaluation"] = {**config["evaluation"], "detector_revision":REVISION,
        "bootstrap_repetitions":args.bootstrap_repetitions}
    results.mkdir(parents=True,exist_ok=True)
    atomic_write_json(results/"revision_manifest.json", {**identity,"completion_status":"running",
        "protocol_config":config,"contamination_cache":fingerprint(cache_path),
        "binocular_origin_context":"output_only_official_512",
        "clipped_origin_reduction":"bf16_capped_tokens_sum_division_over_saved_official_denominator"})
    result = evaluate_raid(rows, config, contamination_records=cache)
    write_evaluation_artifacts(result, results)
    counts = result.validation_counts
    if counts["detectors"] != REVISED_METHODS or counts["universal_specs"] != 16 or counts["rate_adaptive_specs"] != 32 or len(result.binoculars_sanity)!=7:
        raise ValueError("revised eight-detector contract failed")
    from RAID.raid_plot import plot_raid
    plot_raid(results)
    if (identity["source_artifacts"]!={name:fingerprint(source/name) for name in identity["source_artifacts"]}
            or identity["contamination_cache"] != fingerprint(cache_path)):
        raise ValueError("source artifacts changed during evaluation")
    report = {**counts,"validation_status":"pass","detector_revision":REVISION}
    manifest = read_json(results/"revision_manifest.json")
    finish_revision(results,manifest,report,RAID_ARTIFACTS)
    print("RAID eight-detector revision: PASS",flush=True)


if __name__ == "__main__": main()
