"""Full-input tokenizer preflight and bounded real scoring/serialization gate."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from experiment_core.infrastructure.io import AppendSafeJsonlWriter, atomic_write_json, iter_jsonl
from RAID.raid_data import raid_row_key
from RAID.raid_evaluation import (condition, domain, human, source, split,
    stable_row_key, merge_score_rows, evaluate_raid, write_evaluation_artifacts, _validate)


def load_gate_tokenizer(config):
    from experiment_core.detectors.scoring import _transformers
    _, auto_tokenizer = _transformers()
    tokenizer = auto_tokenizer.from_pretrained(
        "tiiuae/falcon-7b", revision=config["tokenizer_revision"],
        trust_remote_code=False, use_fast=True)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("official tokenizer requires a pad/EOS ID")
    return tokenizer


def _record_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).digest()


def preflight_inputs(data_path, shard_paths, tokenizer, report_path):
    """Tokenize all inputs without inference; do not alter/drop invalid texts."""
    fingerprints = {}
    chosen_sources = {}
    selected = []
    families = {}
    source_groups = {}
    bad = []
    invalid_count = 0
    total = 0
    batch = []

    def check_batch(values):
        nonlocal invalid_count
        enc = tokenizer([r["text"] for r in values], padding=False, truncation=True,
                        max_length=512, return_token_type_ids=False)
        if len(enc["input_ids"]) != len(values):
            raise ValueError("tokenizer preflight returned a different row count")
        for row, ids in zip(values, enc["input_ids"]):
            if len(ids) < 2 or not any(i != tokenizer.pad_token_id for i in ids):
                invalid_count += 1
                if len(bad) < 20:
                    bad.append({"key":list(raid_row_key(row)), "input_tokens":len(ids)})

    for row in iter_jsonl(data_path, tolerate_partial_last_line=False):
        key = raid_row_key(row)
        if key in fingerprints:
            raise ValueError(f"duplicate prepared key: {key}")
        fingerprints[key] = _record_digest(row)
        sid = source(row)
        group = (split(row), domain(row))
        if sid in source_groups and source_groups[sid] != group:
            raise ValueError(f"source split/domain leakage: {sid}")
        source_groups[sid] = group
        families.setdefault(sid, Counter())["human" if human(row) else condition(row)] += 1
        chosen_sources.setdefault(group, sid)
        if chosen_sources[group] == sid:
            selected.append(row)
        batch.append(row)
        total += 1
        if len(batch) == 64:
            check_batch(batch)
            batch = []
        if total % 10000 == 0:
            print(f"origin tokenizer preflight: {total} rows", flush=True)
    if batch:
        check_batch(batch)
    # Validate source families as well as the three independent splits.
    _validate(selected)
    expected_family = next(iter(families.values()))
    if any(value != expected_family for value in families.values()):
        raise ValueError("incomplete source family in full prepared data")
    seen = set()
    for shard in shard_paths:
        for row in iter_jsonl(shard, tolerate_partial_last_line=False):
            key = raid_row_key(row)
            if key in seen or fingerprints.get(key) != _record_digest(row):
                raise ValueError(f"prepared shard duplicate/content mismatch: {key}")
            seen.add(key)
    if seen != set(fingerprints):
        raise ValueError("prepared shards do not cover the source data exactly")
    report = {"status":"pass" if invalid_count == 0 else "fail", "rows":total,
              "sources":len(families), "unscorable_rows":invalid_count,
              "unscorable_examples":bad, "shard_content_verified":True,
              "probe_sources":len(chosen_sources), "probe_rows":len(selected)}
    atomic_write_json(report_path, report)
    if invalid_count:
        raise ValueError(f"{invalid_count} unscorable official windows; see {report_path}; no texts changed")
    print(f"origin tokenizer preflight: PASS ({total} rows)", flush=True)
    return report, selected


def run_pipeline_probe(selected, scorer, source_dir, cache_path, destination, config):
    """Exercise the real writer, no-op resume, saved-pack join and evaluator.

    One complete source per split/domain, selected before scoring. Results are
    engineering diagnostics only; they never supply full-run bounds/thresholds.
    """
    from RAID.raid_scoring import score_raid_jsonl
    from RAID.revise_detectors import compact_rows
    from experiment_core.detectors.detector_revision import REVISION
    from experiment_core.infrastructure.revision_artifacts import artifact_digest
    from experiment_core.analysis.evaluation import REVISED_METHODS

    with tempfile.TemporaryDirectory(prefix="gate-probe-", dir=destination) as temp:
        root = Path(temp)
        prepared, scored = root/"data.jsonl", root/"origin.jsonl"
        with AppendSafeJsonlWriter(prepared) as writer:
            for row in selected:
                writer.write(row)
        options = {"batch_size":1, "microbatch_size":1, "checkpoint_interval":10}
        count = score_raid_jsonl(prepared, scored, scorer, options)
        before, calls = artifact_digest(scored), scorer.parity_checks
        resumed = score_raid_jsonl(prepared, scored, scorer, options)
        if count != len(selected) or resumed != count or before != artifact_digest(scored) or calls != scorer.parity_checks:
            raise ValueError("gate score-pack write/resume check failed")
        keys = {stable_row_key(r) for r in selected}
        packs = []
        for name in ("falcon_scores.jsonl", "binoculars_scores.jsonl"):
            packs.append([r for r in compact_rows(Path(source_dir)/name) if stable_row_key(r) in keys])
        if any(len(pack) != count for pack in packs):
            raise ValueError("gate is missing saved Falcon/gap rows")
        rows = merge_score_rows(merge_score_rows(*packs), list(compact_rows(scored)))
        sources = {source(r) for r in selected}
        with Path(cache_path).open(encoding="utf-8", newline="") as handle:
            cache = [r for r in csv.DictReader(handle) if str(r["source_id"]) in sources]
        cfg = {**config, "evaluation":{**config["evaluation"],
               "detector_revision":REVISION, "bootstrap_repetitions":2}}
        result = evaluate_raid(rows, cfg, contamination_records=cache)
        if result.validation_counts["detectors"] != REVISED_METHODS:
            raise ValueError("gate eight-detector evaluation failed")
        paths = write_evaluation_artifacts(result, root/"results")
        for path in paths.values():
            artifact_digest(path)
        from RAID.raid_plot import plot_raid
        for path in plot_raid(root/"results"):
            artifact_digest(path)
        return {"status":"pass", "score_rows":count, "resume_unchanged":True,
                "bootstrap_repetitions":2, "counts":result.validation_counts}
