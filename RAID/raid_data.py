"""Streaming preparation helpers for the RAID external benchmark.

The public RAID release uses these columns::

    id, adv_source_id, source_id, model, decoding, repetition_penalty,
    attack, domain, title, prompt, generation

An unattacked machine row has ``attack == "none"``.  Each attacked row points
to that row through ``adv_source_id``; every machine row points to its human
source through ``source_id``.  This module deliberately consumes local data by
default.  Downloading is delegated to the optional ``raid-bench`` package and
only happens when a caller explicitly omits a local path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from llm_detection.data import stable_int
from llm_detection.io import AppendSafeJsonlWriter, atomic_write_json, iter_jsonl


RAID_ATTACKS = (
    "none",
    "homoglyph",
    "number",
    "article_deletion",
    "insert_paragraphs",
    "perplexity_misspelling",
    "upper_lower",
    "whitespace",
    "zero_width_space",
    "synonym",
    "paraphrase",
    "alternative_spelling",
)
RAID_ADVERSARIAL_ATTACKS = RAID_ATTACKS[1:]
RAID_NATIVE_ATTACK_RATES = {
    "none": 0.0,
    "alternative_spelling": 1.0,
    "article_deletion": 0.5,
    "homoglyph": 1.0,
    "insert_paragraphs": 0.5,
    "number": 0.5,
    "paraphrase": 1.0,
    "perplexity_misspelling": 0.2,
    "synonym": 0.5,
    "upper_lower": 0.05,
    "whitespace": 0.2,
    "zero_width_space": 1.0,
}
RAID_DOMAINS = (
    "abstracts",
    "books",
    "news",
    "poetry",
    "recipes",
    "reddit",
    "reviews",
    "wiki",
)
SPLIT_ORDER = ("clipping_tuning", "calibration", "test")
DEFAULT_SPLIT_FRACTIONS = {
    "clipping_tuning": 0.4,
    "calibration": 0.2,
    "test": 0.4,
}


def _allow_large_csv_fields() -> None:
    """Raise the process-wide CSV field limit to the platform maximum.

    RAID generations can exceed Python's conservative 128 KiB default.  The
    accepted maximum is platform-dependent, so reduce ``sys.maxsize`` only if
    the C CSV parser reports that the value itself is too large.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _string(value: Any, field: str, *, required: bool = True) -> str | None:
    if _missing(value):
        if required:
            raise ValueError(f"RAID field {field!r} is missing")
        return None
    result = str(value).strip()
    if required and not result:
        raise ValueError(f"RAID field {field!r} is empty")
    return result or None


def _optional_float(record: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = record.get(name)
        if _missing(value) or value == "":
            continue
        try:
            if isinstance(value, str) and value.strip().endswith("%"):
                return float(value.strip()[:-1]) / 100.0
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"RAID field {name!r} must be numeric") from None
    return None


def normalize_raid_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one official RAID record without changing its scored text."""
    raid_id = _string(record.get("id"), "id")
    source_id = _string(record.get("source_id"), "source_id")
    model = _string(record.get("model"), "model")
    attack = (_string(record.get("attack", "none"), "attack") or "none").lower()
    if attack not in RAID_ATTACKS:
        raise ValueError(f"unknown RAID attack {attack!r}")
    domain = (_string(record.get("domain"), "domain") or "").lower()
    if domain not in RAID_DOMAINS:
        raise ValueError(
            f"RAID core English protocol does not accept domain {domain!r}"
        )
    generation = record.get("generation")
    if not isinstance(generation, str) or not generation.strip():
        raise ValueError("RAID field 'generation' must be a non-empty string")

    is_human = model.lower() == "human"
    adv_source_id = _string(record.get("adv_source_id"), "adv_source_id", required=False)
    if attack == "none":
        # The release normally repeats the row id here.  Normalizing missing
        # values makes older/local exports interoperable without weakening the
        # attacked-row linkage requirement.
        adv_source_id = adv_source_id or raid_id
    elif not adv_source_id:
        raise ValueError("an attacked RAID row is missing adv_source_id")
    if is_human and attack != "none":
        # Attacked human controls exist in some exports but are intentionally
        # outside this study's human-once protocol.
        record_kind = "attacked_human"
    elif is_human:
        record_kind = "human"
    elif attack == "none":
        record_kind = "machine_clean"
    else:
        record_kind = "machine_attack"

    native_attack_rate = _optional_float(
        record,
        ("native_attack_rate", "attack_rate", "theta", "attack_percentage"),
    )
    native_attack_rate_source = "release_column"
    if native_attack_rate is None:
        native_attack_rate = RAID_NATIVE_ATTACK_RATES[attack]
        native_attack_rate_source = "raid_paper_attack_default"
    if not 0.0 <= native_attack_rate <= 1.0:
        raise ValueError("RAID native attack rate must lie in [0, 1]")
    return {
        "raid_id": raid_id,
        "raid_adv_source_id": adv_source_id,
        "source_id": source_id,
        "model": model.lower(),
        "decoding": _string(record.get("decoding"), "decoding", required=False),
        "repetition_penalty": _string(
            record.get("repetition_penalty"), "repetition_penalty", required=False
        ),
        "attack": attack,
        "domain": domain,
        "title": _string(record.get("title"), "title", required=False),
        "prompt": _string(record.get("prompt"), "prompt", required=False),
        # Do not strip or otherwise normalize the text: zero-width/whitespace
        # attacks would be damaged and scoring would no longer match RAID.
        "generation": generation,
        "native_attack_rate": native_attack_rate,
        "native_attack_rate_source": native_attack_rate_source,
        "record_kind": record_kind,
    }


def iter_raid_records(source: Any) -> Iterator[dict[str, Any]]:
    """Yield raw records from local CSV/JSONL, a dataframe, or an iterable.

    This function never downloads data.  It permits unit tests and Delta jobs
    to use an explicit local RAID export and also accepts Hugging Face dataset
    rows or pandas-like records supplied by a caller.
    """
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            if suffix == ".jsonl":
                yield from iter_jsonl(path)
                return
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, list):
                raise ValueError("a RAID .json input must contain a list of records")
            for row in value:
                yield dict(row)
            return
        if suffix in {".csv", ".tsv"}:
            _allow_large_csv_fields()
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
                for row in reader:
                    yield dict(row)
            return
        raise ValueError(f"unsupported RAID input format: {path}")

    if hasattr(source, "to_dict"):
        try:
            records = source.to_dict(orient="records")
        except TypeError:
            records = source.to_dict("records")
        for row in records:
            yield dict(row)
        return
    for row in source:
        yield dict(row)


def load_raid_records(config: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Resolve a config-driven local RAID source or explicit raid-bench load.

    ``dataset.local_path`` (or ``dataset.path``) always wins.  Only the absence
    of either path permits the installed ``raid-bench`` loader to download or
    reuse its cache.
    """
    dataset = config.get("dataset", config)
    local_path = dataset.get("local_path") or dataset.get("path")
    if local_path:
        return iter_raid_records(local_path)
    try:
        from raid.utils import load_data  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "No local RAID path was configured and raid-bench is not installed"
        ) from exc
    return load_data(
        split=str(dataset.get("split", "train")),
        include_adversarial=True,
        fp=dataset.get("cache_path"),
    )


def _canonical_payload(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def index_raid_records(
    source: Any,
    index_path: str | Path,
    *,
    reset: bool = False,
    commit_interval: int = 10_000,
) -> dict[str, int]:
    """Stream RAID rows into a restartable SQLite relationship index."""
    target = Path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if reset and target.exists():
        target.unlink()
    connection = sqlite3.connect(target)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            raid_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            adv_source_id TEXT NOT NULL,
            model TEXT NOT NULL,
            attack TEXT NOT NULL,
            domain TEXT NOT NULL,
            record_kind TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    counts = {"input_rows": 0, "inserted_rows": 0, "duplicate_rows": 0}
    pending: list[tuple[str, str, str, str, str, str, str, str]] = []

    def flush() -> None:
        if not pending:
            return
        before = connection.total_changes
        connection.executemany(
            "INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            pending,
        )
        inserted = connection.total_changes - before
        counts["inserted_rows"] += inserted
        if inserted != len(pending):
            for values in pending:
                existing = connection.execute(
                    "SELECT payload FROM records WHERE raid_id = ?", (values[0],)
                ).fetchone()
                if existing is None or existing[0] != values[-1]:
                    raise ValueError(
                        f"conflicting duplicate RAID id {values[0]!r}"
                    )
            counts["duplicate_rows"] += len(pending) - inserted
        pending.clear()
        connection.commit()

    try:
        for raw in iter_raid_records(source):
            row = normalize_raid_record(raw)
            counts["input_rows"] += 1
            payload = _canonical_payload(row)
            pending.append(
                (
                    row["raid_id"],
                    row["source_id"],
                    row["raid_adv_source_id"],
                    row["model"],
                    row["attack"],
                    row["domain"],
                    row["record_kind"],
                    payload,
                )
            )
            if len(pending) >= max(1, int(commit_interval)):
                flush()
        flush()
        # Building secondary indexes once after the bulk load is substantially
        # faster than maintaining both trees for every one of millions of rows.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS records_source_idx ON records(source_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS records_family_idx ON records(adv_source_id, attack)"
        )
        connection.commit()
        counts["indexed_rows"] = int(
            connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        )
        return counts
    finally:
        connection.close()


def inspect_raid_index(index_path: str | Path) -> dict[str, int]:
    """Validate the reusable index schema and return its stable row counts."""
    target = Path(index_path)
    if not target.is_file():
        raise FileNotFoundError(f"RAID index is missing: {target}")
    wal = target.with_name(target.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"RAID index has a nonempty WAL and may still be active: {wal}")
    connection = sqlite3.connect(f"file:{target.resolve().as_posix()}?mode=ro", uri=True)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(records)")
        }
        expected = {
            "raid_id",
            "source_id",
            "adv_source_id",
            "model",
            "attack",
            "domain",
            "record_kind",
            "payload",
        }
        if columns != expected:
            raise ValueError("RAID reusable index has an incompatible records schema")
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(records)")}
        if not {"records_source_idx", "records_family_idx"}.issubset(indexes):
            raise ValueError("RAID reusable index is missing required relationship indexes")
        rows = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        sources = int(
            connection.execute("SELECT COUNT(DISTINCT source_id) FROM records").fetchone()[0]
        )
        if rows < 1 or sources < 1:
            raise ValueError("RAID reusable index is empty")
        return {"indexed_rows": rows, "indexed_sources": sources}
    finally:
        connection.close()


def _split_assignments(
    source_rows: Sequence[dict[str, Any]],
    split_seed: int,
    fractions: Mapping[str, float],
) -> dict[str, str]:
    if set(fractions) != set(SPLIT_ORDER):
        raise ValueError(f"split fractions must define {SPLIT_ORDER}")
    total_fraction = sum(float(fractions[name]) for name in SPLIT_ORDER)
    if not math.isclose(total_fraction, 1.0, abs_tol=1e-12):
        raise ValueError("split fractions must sum to one")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        grouped.setdefault(str(row["domain"]), []).append(row)
    ordered_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain, rows in sorted(grouped.items()):
        ordered_by_domain[domain] = sorted(
            rows,
            key=lambda row: (
                stable_int(split_seed, domain, row["source_id"]),
                row["source_id"],
            ),
        )

    total = len(source_rows)
    target_counts = {
        "clipping_tuning": int(math.floor(total * float(fractions["clipping_tuning"]))),
        "calibration": int(math.floor(total * float(fractions["calibration"]))),
    }
    target_counts["test"] = total - sum(target_counts.values())
    capacities = {domain: len(rows) for domain, rows in ordered_by_domain.items()}
    original_counts = dict(capacities)
    allocated: dict[str, dict[str, int]] = {}
    for split in SPLIT_ORDER[:-1]:
        target = target_counts[split]
        ideals = {
            domain: original_counts[domain] * float(fractions[split])
            for domain in capacities
        }
        counts = {
            domain: min(capacities[domain], int(math.floor(ideals[domain])))
            for domain in capacities
        }
        remaining = target - sum(counts.values())
        while remaining > 0:
            candidates = [domain for domain in capacities if counts[domain] < capacities[domain]]
            if not candidates:
                raise AssertionError("unable to allocate stratified split")
            domain = min(
                candidates,
                key=lambda name: (
                    -(ideals[name] - counts[name]),
                    stable_int(split_seed, split, name),
                    name,
                ),
            )
            counts[domain] += 1
            remaining -= 1
        allocated[split] = counts
        capacities = {
            domain: capacities[domain] - counts[domain] for domain in capacities
        }
    allocated["test"] = capacities

    assigned: dict[str, str] = {}
    for domain, rows in ordered_by_domain.items():
        offset = 0
        for split in SPLIT_ORDER:
            count = allocated[split][domain]
            for row in rows[offset : offset + count]:
                assigned[str(row["source_id"])] = split
            offset += count
        if offset != len(rows):
            raise AssertionError("stratified split did not consume a domain")
    if {split: list(assigned.values()).count(split) for split in SPLIT_ORDER} != target_counts:
        raise AssertionError("stratified split does not match global target counts")
    return assigned


def _round_robin_limit(
    rows: Sequence[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Keep a bounded deterministic sample without dropping small domains first."""
    if limit >= len(rows):
        return list(rows)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(str(row["domain"]), []).append(row)
    for domain in by_domain:
        by_domain[domain].sort(
            key=lambda row: (stable_int(seed, row["source_id"]), row["source_id"])
        )
    result: list[dict[str, Any]] = []
    domains = sorted(by_domain)
    while len(result) < limit:
        progressed = False
        for domain in domains:
            if by_domain[domain] and len(result) < limit:
                result.append(by_domain[domain].pop(0))
                progressed = True
        if not progressed:
            break
    return result


def select_complete_families(
    index_path: str | Path,
    *,
    selection_seed: int,
    split_seed: int,
    split_fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    limit_sources: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select one uniformly random complete family using one grouped SQL scan.

    The previous implementation issued one query per source and then one query
    per candidate generation.  On the public RAID index this became hundreds
    of thousands of random queries.  This implementation streams rows in
    ``source_id`` order and builds each source family in memory exactly once.
    A bounded debug run stops after it has enough complete sources in every
    core domain; full runs consume the complete index.
    """
    connection = sqlite3.connect(index_path)
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    incomplete_family_count = 0
    scanned_sources = 0
    bounded_scan_complete = limit_sources is None
    if limit_sources is not None and int(limit_sources) < 1:
        raise ValueError("limit_sources must be positive")
    per_domain_target = (
        None
        if limit_sources is None
        else int(math.ceil(int(limit_sources) / len(RAID_DOMAINS)))
    )

    def add_source(source_id: str, payloads: Iterable[str]) -> None:
        nonlocal incomplete_family_count, scanned_sources
        scanned_sources += 1
        records = [json.loads(payload) for payload in payloads]
        humans = [row for row in records if row["record_kind"] == "human"]
        if len(humans) != 1:
            exclusions.append(
                {
                    "source_id": source_id,
                    "reason": "missing_human" if not humans else "ambiguous_human",
                    "human_rows": len(humans),
                }
            )
            return
        human = humans[0]
        clean_rows = sorted(
            (row for row in records if row["record_kind"] == "machine_clean"),
            key=lambda row: row["raid_id"],
        )
        attacks_by_clean: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in records:
            if row["record_kind"] == "machine_attack":
                attacks_by_clean.setdefault(row["raid_adv_source_id"], {}).setdefault(
                    row["attack"], []
                ).append(row)
        complete: list[dict[str, Any]] = []
        source_incomplete = 0
        metadata_fields = ("model", "decoding", "repetition_penalty", "domain")
        for clean in clean_rows:
            attacks = attacks_by_clean.get(clean["raid_id"], {})
            metadata_match = clean["domain"] == human["domain"] and all(
                all(attack_row[field] == clean[field] for field in metadata_fields)
                for attack_rows in attacks.values()
                for attack_row in attack_rows
            )
            if metadata_match and all(
                len(attacks.get(name, [])) == 1
                for name in RAID_ADVERSARIAL_ATTACKS
            ):
                complete.append(
                    {
                        "human": human,
                        "clean": clean,
                        "attacks": {
                            name: attacks[name][0]
                            for name in RAID_ADVERSARIAL_ATTACKS
                        },
                    }
                )
            else:
                source_incomplete += 1
        incomplete_family_count += source_incomplete
        if not complete:
            exclusions.append(
                {
                    "source_id": source_id,
                    "domain": human["domain"],
                    "reason": "no_complete_machine_family",
                    "candidate_families": len(clean_rows),
                    "incomplete_families": source_incomplete,
                }
            )
            return
        selected_index = random.Random(
            stable_int(selection_seed, source_id)
        ).randrange(len(complete))
        family = complete[selected_index]
        selected.append(
            {
                "source_id": source_id,
                "domain": human["domain"],
                "human": human,
                "clean": family["clean"],
                "attacks": family["attacks"],
                "complete_family_count": len(complete),
                "incomplete_family_count": source_incomplete,
                "selection_seed": int(selection_seed),
            }
        )

    try:
        cursor = connection.execute(
            """
            SELECT source_id, payload
            FROM records INDEXED BY records_source_idx
            WHERE record_kind IN ('human', 'machine_clean', 'machine_attack')
            ORDER BY source_id
            """
        )
        for source_id, rows in groupby(cursor, key=lambda item: item[0]):
            add_source(str(source_id), (item[1] for item in rows))
            if per_domain_target is not None:
                domain_counts = Counter(row["domain"] for row in selected)
                if all(domain_counts[name] >= per_domain_target for name in RAID_DOMAINS):
                    bounded_scan_complete = False
                    break
    finally:
        connection.close()

    if limit_sources is not None:
        selected = _round_robin_limit(selected, int(limit_sources), selection_seed)
    assignments = _split_assignments(selected, split_seed, split_fractions)
    for source in selected:
        source["split"] = assignments[source["source_id"]]
        source["split_seed"] = int(split_seed)

    split_domain_counts: dict[str, dict[str, int]] = {}
    for source in selected:
        split_domain_counts.setdefault(source["split"], {}).setdefault(source["domain"], 0)
        split_domain_counts[source["split"]][source["domain"]] += 1
    connection = sqlite3.connect(index_path)
    try:
        indexed_source_count = int(
            connection.execute("SELECT COUNT(DISTINCT source_id) FROM records").fetchone()[0]
        )
    finally:
        connection.close()
    report = {
        "indexed_sources": indexed_source_count,
        "available_complete_sources": (
            indexed_source_count - len(exclusions)
            if bounded_scan_complete
            else None
        ),
        "selection_scan_complete": bounded_scan_complete,
        "selection_scanned_sources": scanned_sources,
        "selected_sources": len(selected),
        "excluded_sources": len(exclusions),
        "incomplete_families": incomplete_family_count,
        "split_domain_counts": split_domain_counts,
    }
    return selected, exclusions, report


def _stable_row_id(parts: Sequence[Any]) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raid_row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Stable unique key for one prepared human or machine-condition row."""
    source_id = str(row["source_id"])
    label = str(row["label"])
    if label == "human":
        return source_id, "human", "human"
    return source_id, str(row["base_generation_id"]), str(row["attack"])


def _prepared_row(
    source: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    label: str,
    attack: str,
    dataset_revision: str | None,
    dataset_fingerprint: str | None,
) -> dict[str, Any]:
    clean = source["clean"]
    key = (
        str(source["source_id"]),
        "human" if label == "human" else str(clean["raid_id"]),
        "human" if label == "human" else attack,
    )
    return {
        "row_id": _stable_row_id(key),
        "dataset": "raid",
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "sample_id": str(source["source_id"]),
        "source_id": str(source["source_id"]),
        "split": str(source["split"]),
        "label": label,
        "machine_origin": label == "llm",
        "text": record["generation"],
        "attack": attack,
        "raid_attack": record["attack"],
        "contamination_mode": attack,
        "realized_contamination_rate": (
            None if label == "human" or attack != "none" else 0.0
        ),
        "base_generation_id": None if label == "human" else clean["raid_id"],
        "raid_id": record["raid_id"],
        "raid_adv_source_id": record["raid_adv_source_id"],
        "model": record["model"],
        "source_generator_model": None if label == "human" else clean["model"],
        "decoding": None if label == "human" else clean["decoding"],
        "repetition_penalty": None if label == "human" else clean["repetition_penalty"],
        "domain": source["domain"],
        "title": record["title"],
        "prompt": record["prompt"],
        "native_attack_rate": None if label == "human" else record["native_attack_rate"],
        "native_attack_rate_source": (
            None if label == "human" else record["native_attack_rate_source"]
        ),
        "selection_seed": int(source["selection_seed"]),
        "split_seed": int(source["split_seed"]),
    }


def selected_source_manifest_row(source: Mapping[str, Any]) -> dict[str, Any]:
    clean = source["clean"]
    return {
        "source_id": source["source_id"],
        "domain": source["domain"],
        "split": source["split"],
        "human_raid_id": source["human"]["raid_id"],
        "base_generation_id": clean["raid_id"],
        "source_generator_model": clean["model"],
        "decoding": clean["decoding"],
        "repetition_penalty": clean["repetition_penalty"],
        "complete_family_count": source["complete_family_count"],
        "incomplete_family_count": source["incomplete_family_count"],
        "selection_seed": source["selection_seed"],
        "split_seed": source["split_seed"],
        "attack_raid_ids": {
            attack: source["attacks"][attack]["raid_id"]
            for attack in RAID_ADVERSARIAL_ATTACKS
        },
    }


def prepared_rows(
    source: Mapping[str, Any],
    *,
    dataset_revision: str | None = None,
    dataset_fingerprint: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one human row and the selected twelve-condition machine family."""
    yield _prepared_row(
        source,
        source["human"],
        label="human",
        attack="human",
        dataset_revision=dataset_revision,
        dataset_fingerprint=dataset_fingerprint,
    )
    yield _prepared_row(
        source,
        source["clean"],
        label="llm",
        attack="none",
        dataset_revision=dataset_revision,
        dataset_fingerprint=dataset_fingerprint,
    )
    for attack in RAID_ADVERSARIAL_ATTACKS:
        yield _prepared_row(
            source,
            source["attacks"][attack],
            label="llm",
            attack=attack,
            dataset_revision=dataset_revision,
            dataset_fingerprint=dataset_fingerprint,
        )


def _append_unique_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    key_fn: Any,
) -> int:
    target = Path(path)
    completed = set()
    if target.exists():
        for row in iter_jsonl(target):
            key = key_fn(row)
            if key in completed:
                raise ValueError(f"duplicate completed key {key!r} in {target}")
            completed.add(key)
    with AppendSafeJsonlWriter(target) as writer:
        for row in rows:
            key = key_fn(row)
            if key not in completed:
                writer.write(row)
                completed.add(key)
        writer.checkpoint()
    return len(completed)


def prepare_raid_data(
    source: Any,
    output_path: str | Path,
    selected_sources_path: str | Path,
    exclusions_path: str | Path,
    index_path: str | Path,
    *,
    selection_seed: int,
    split_seed: int,
    split_fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    limit_sources: int | None = None,
    dataset_revision: str | None = None,
    dataset_fingerprint: str | None = None,
    reset_index: bool = False,
    reuse_index: bool = False,
) -> dict[str, Any]:
    """Index, sample, split, and append the frozen RAID one-to-one dataset."""
    if reuse_index and reset_index:
        raise ValueError("cannot reset a reusable RAID index")
    index_report = (
        {**inspect_raid_index(index_path), "index_reused": True}
        if reuse_index
        else {
            **index_raid_records(source, index_path, reset=reset_index),
            "index_reused": False,
        }
    )
    selected, exclusions, selection_report = select_complete_families(
        index_path,
        selection_seed=selection_seed,
        split_seed=split_seed,
        split_fractions=split_fractions,
        limit_sources=limit_sources,
    )
    manifest_rows = [selected_source_manifest_row(source_row) for source_row in selected]
    manifest_count = _append_unique_rows(
        selected_sources_path, manifest_rows, lambda row: str(row["source_id"])
    )
    output_count = _append_unique_rows(
        output_path,
        (
            row
            for source_row in selected
            for row in prepared_rows(
                source_row,
                dataset_revision=dataset_revision,
                dataset_fingerprint=dataset_fingerprint,
            )
        ),
        raid_row_key,
    )
    atomic_write_json(exclusions_path, exclusions)
    expected_rows = len(selected) * (1 + len(RAID_ATTACKS))
    if manifest_count != len(selected) or output_count != expected_rows:
        raise ValueError(
            "prepared RAID artifacts failed source/row count validation: "
            f"manifest={manifest_count}/{len(selected)}, rows={output_count}/{expected_rows}"
        )
    return {
        **index_report,
        **selection_report,
        "prepared_rows": output_count,
        "rows_per_source": 1 + len(RAID_ATTACKS),
        "selected_sources_path": str(selected_sources_path),
        "data_path": str(output_path),
        "exclusions_path": str(exclusions_path),
        "index_path": str(index_path),
    }


@dataclass(frozen=True)
class EditCounts:
    distance: int
    substitutions: int
    deletions: int
    insertions: int

    def as_dict(self) -> dict[str, int]:
        return {
            "levenshtein_distance": self.distance,
            "substitution_count": self.substitutions,
            "deletion_count": self.deletions,
            "insertion_count": self.insertions,
        }


def _trim_shared_ends(
    clean_ids: Sequence[int], attacked_ids: Sequence[int]
) -> tuple[list[int], list[int]]:
    clean = list(clean_ids)
    attacked = list(attacked_ids)
    prefix = 0
    while prefix < len(clean) and prefix < len(attacked) and clean[prefix] == attacked[prefix]:
        prefix += 1
    clean_end = len(clean)
    attacked_end = len(attacked)
    while (
        clean_end > prefix
        and attacked_end > prefix
        and clean[clean_end - 1] == attacked[attacked_end - 1]
    ):
        clean_end -= 1
        attacked_end -= 1
    return clean[prefix:clean_end], attacked[prefix:attacked_end]


def levenshtein_edit_counts(
    clean_ids: Sequence[int], attacked_ids: Sequence[int]
) -> EditCounts:
    """Return a deterministic minimum token edit script's S/D/I counts.

    RapidFuzz's standard Levenshtein edit operations are used when available.
    A dependency-free Wagner--Fischer implementation is retained for bounded
    tests and small edits.  Shared prefix/suffix trimming makes sparse RAID
    attacks cheap in either implementation.
    """
    clean, attacked = _trim_shared_ends(clean_ids, attacked_ids)
    if not clean:
        return EditCounts(len(attacked), 0, 0, len(attacked))
    if not attacked:
        return EditCounts(len(clean), 0, len(clean), 0)
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore[import-not-found]

        operations = Levenshtein.editops(clean, attacked)
        substitutions = sum(operation.tag == "replace" for operation in operations)
        deletions = sum(operation.tag == "delete" for operation in operations)
        insertions = sum(operation.tag == "insert" for operation in operations)
        return EditCounts(len(operations), substitutions, deletions, insertions)
    except ImportError:
        pass

    cells = (len(clean) + 1) * (len(attacked) + 1)
    if cells > 4_000_000:
        raise RuntimeError(
            "exact token edit backtracing for this long pair requires rapidfuzz"
        )
    matrix = [[0] * (len(attacked) + 1) for _ in range(len(clean) + 1)]
    for i in range(len(clean) + 1):
        matrix[i][0] = i
    for j in range(len(attacked) + 1):
        matrix[0][j] = j
    for i in range(1, len(clean) + 1):
        clean_token = clean[i - 1]
        for j in range(1, len(attacked) + 1):
            substitution_cost = 0 if clean_token == attacked[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j - 1] + substitution_cost,
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
            )
    i, j = len(clean), len(attacked)
    substitutions = deletions = insertions = 0
    while i or j:
        if i and j and clean[i - 1] == attacked[j - 1] and matrix[i][j] == matrix[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i and j and matrix[i][j] == matrix[i - 1][j - 1] + 1:
            # Deterministic tie policy: substitution, then deletion, then insertion.
            substitutions += 1
            i -= 1
            j -= 1
        elif i and matrix[i][j] == matrix[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    distance = matrix[-1][-1]
    if distance != substitutions + deletions + insertions:
        raise AssertionError("invalid Levenshtein backtrace")
    return EditCounts(distance, substitutions, deletions, insertions)


def measure_realized_contamination(
    clean_scored_token_ids: Sequence[int],
    attacked_scored_token_ids: Sequence[int],
    *,
    clean_full_token_ids: Sequence[int] | None = None,
    attacked_full_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Measure realized modification on exact scored IDs and audit truncation."""
    clean_scored = list(clean_scored_token_ids)
    attacked_scored = list(attacked_scored_token_ids)
    scored = levenshtein_edit_counts(clean_scored, attacked_scored)
    denominator = max(len(clean_scored), len(attacked_scored))
    result: dict[str, Any] = {
        **scored.as_dict(),
        "clean_scored_token_count": len(clean_scored),
        "attacked_scored_token_count": len(attacked_scored),
        "realized_contamination_rate": (
            float(scored.distance / denominator) if denominator else 0.0
        ),
    }
    if (clean_full_token_ids is None) != (attacked_full_token_ids is None):
        raise ValueError("clean and attacked full token ids must be supplied together")
    if clean_full_token_ids is None:
        result.update(
            {
                "full_text_edit_audited": False,
                "clean_full_token_count": None,
                "attacked_full_token_count": None,
                "full_levenshtein_distance": None,
                "full_substitution_count": None,
                "full_deletion_count": None,
                "full_insertion_count": None,
                "truncation_applied_clean": None,
                "truncation_applied_attacked": None,
                "truncation_hidden_edit_count": None,
                "truncation_hid_edits": None,
                "all_full_text_edits_hidden": None,
            }
        )
        return result

    clean_full = list(clean_full_token_ids)
    attacked_full = list(attacked_full_token_ids or [])
    full = levenshtein_edit_counts(clean_full, attacked_full)
    hidden_edit_count = max(0, full.distance - scored.distance)
    result.update(
        {
            "full_text_edit_audited": True,
            "clean_full_token_count": len(clean_full),
            "attacked_full_token_count": len(attacked_full),
            "full_levenshtein_distance": full.distance,
            "full_substitution_count": full.substitutions,
            "full_deletion_count": full.deletions,
            "full_insertion_count": full.insertions,
            "truncation_applied_clean": len(clean_scored) < len(clean_full),
            "truncation_applied_attacked": len(attacked_scored) < len(attacked_full),
            "truncation_hidden_edit_count": hidden_edit_count,
            "truncation_hid_edits": hidden_edit_count > 0,
            "all_full_text_edits_hidden": full.distance > 0 and scored.distance == 0,
        }
    )
    return result
