from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from llm_detection.io import iter_jsonl
from RAID.raid_data import (
    DEFAULT_SPLIT_FRACTIONS,
    RAID_ADVERSARIAL_ATTACKS,
    RAID_ATTACKS,
    index_raid_records,
    iter_raid_records,
    levenshtein_edit_counts,
    measure_realized_contamination,
    normalize_raid_record,
    prepare_raid_data,
    raid_row_key,
    select_complete_families,
)


def _row(
    *,
    raid_id: str,
    source_id: str,
    model: str,
    attack: str = "none",
    adv_source_id: str | None = None,
    domain: str = "news",
    generation: str | None = None,
) -> dict[str, object]:
    return {
        "id": raid_id,
        "adv_source_id": adv_source_id if adv_source_id is not None else raid_id,
        "source_id": source_id,
        "model": model,
        "decoding": None if model == "human" else "sampling",
        "repetition_penalty": None if model == "human" else "no",
        "attack": attack,
        "domain": domain,
        "title": f"Title {source_id}",
        "prompt": None if model == "human" else f"Prompt {source_id}",
        "generation": generation or f"Text {raid_id}",
    }


def _source_family(
    source_id: str,
    *,
    domain: str = "news",
    family_names: tuple[str, ...] = ("a",),
    incomplete_family: str | None = None,
) -> list[dict[str, object]]:
    rows = [
        _row(
            raid_id=f"human-{source_id}",
            source_id=source_id,
            model="human",
            domain=domain,
        )
    ]
    for family_name in family_names:
        clean_id = f"machine-{source_id}-{family_name}"
        rows.append(
            _row(
                raid_id=clean_id,
                source_id=source_id,
                model="gpt4" if family_name == "a" else "mistral",
                domain=domain,
            )
        )
        for attack in RAID_ADVERSARIAL_ATTACKS:
            if family_name == incomplete_family and attack == "synonym":
                continue
            rows.append(
                _row(
                    raid_id=f"{clean_id}-{attack}",
                    adv_source_id=clean_id,
                    source_id=source_id,
                    model="gpt4" if family_name == "a" else "mistral",
                    attack=attack,
                    domain=domain,
                )
            )
    return rows


class RaidLoaderTests(unittest.TestCase):
    def test_normalizes_official_schema_without_stripping_attacked_text(self) -> None:
        raw = _row(
            raid_id="attack-id",
            source_id="source-id",
            model="gpt4",
            attack="zero_width_space",
            adv_source_id="clean-id",
            generation="  visible\u200b text  ",
        )
        raw["theta"] = "0.5"
        normalized = normalize_raid_record(raw)
        self.assertEqual(normalized["record_kind"], "machine_attack")
        self.assertEqual(normalized["generation"], "  visible\u200b text  ")
        self.assertEqual(normalized["native_attack_rate"], 0.5)

        raw.pop("theta")
        inferred = normalize_raid_record(raw)
        self.assertEqual(inferred["native_attack_rate"], 1.0)

    def test_reads_local_csv_and_jsonl_without_network(self) -> None:
        records = _source_family("s0")[:2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            csv_path = directory / "raid.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            jsonl_path = directory / "raid.jsonl"
            jsonl_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            self.assertEqual(len(list(iter_raid_records(csv_path))), 2)
            self.assertEqual(len(list(iter_raid_records(jsonl_path))), 2)

    def test_reads_csv_generation_larger_than_python_default_field_limit(self) -> None:
        record = _row(
            raid_id="long-row",
            source_id="long-source",
            model="human",
            generation="x" * 200_000,
        )
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "raid-long.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
                writer.writerow(record)

            loaded = list(iter_raid_records(csv_path))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(loaded[0]["generation"]), 200_000)

    def test_accepts_dataframe_like_records(self) -> None:
        class Frame:
            def to_dict(self, orient: str) -> list[dict[str, object]]:
                if orient != "records":
                    raise AssertionError(orient)
                return _source_family("s0")[:1]

        self.assertEqual(len(list(iter_raid_records(Frame()))), 1)


class RaidSelectionTests(unittest.TestCase):
    def test_selects_one_complete_family_and_reports_incomplete_source(self) -> None:
        records = []
        records.extend(
            _source_family(
                "complete", family_names=("a", "b"), incomplete_family="b"
            )
        )
        records.extend(
            _source_family("excluded", family_names=("b",), incomplete_family="b")
        )
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "raid.sqlite"
            report = index_raid_records(records, index_path)
            self.assertEqual(report["indexed_rows"], len(records))
            selected, exclusions, summary = select_complete_families(
                index_path,
                selection_seed=17,
                split_seed=23,
            )
            self.assertEqual([row["source_id"] for row in selected], ["complete"])
            self.assertEqual(selected[0]["clean"]["raid_id"], "machine-complete-a")
            self.assertEqual(set(selected[0]["attacks"]), set(RAID_ADVERSARIAL_ATTACKS))
            self.assertEqual(exclusions[0]["source_id"], "excluded")
            self.assertEqual(exclusions[0]["reason"], "no_complete_machine_family")
            self.assertEqual(summary["selected_sources"], 1)
            self.assertEqual(summary["excluded_sources"], 1)

    def test_domain_stratified_split_is_deterministic_and_grouped(self) -> None:
        records = []
        for domain in ("news", "wiki"):
            for index in range(10):
                records.extend(_source_family(f"{domain}-{index}", domain=domain))
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "raid.sqlite"
            index_raid_records(records, index_path)
            first, _, first_report = select_complete_families(
                index_path, selection_seed=11, split_seed=12
            )
            second, _, second_report = select_complete_families(
                index_path, selection_seed=11, split_seed=12
            )
            self.assertEqual(
                [(row["source_id"], row["split"]) for row in first],
                [(row["source_id"], row["split"]) for row in second],
            )
            expected = {
                "clipping_tuning": {"news": 4, "wiki": 4},
                "calibration": {"news": 2, "wiki": 2},
                "test": {"news": 4, "wiki": 4},
            }
            self.assertEqual(first_report["split_domain_counts"], expected)
            self.assertEqual(second_report["split_domain_counts"], expected)

    def test_development_sources_are_excluded_before_splitting(self) -> None:
        records = []
        for index in range(6):
            records.extend(_source_family(f"s{index}"))
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "raid.sqlite"
            index_raid_records(records, index_path)
            selected, exclusions, report = select_complete_families(
                index_path,
                selection_seed=11,
                split_seed=12,
                excluded_source_ids={"s1", "s4"},
            )
        self.assertEqual({row["source_id"] for row in selected},
                         {"s0", "s2", "s3", "s5"})
        self.assertEqual(
            {row["source_id"] for row in exclusions
             if row["reason"] == "pilot_development_source"},
            {"s1", "s4"},
        )
        self.assertEqual(report["requested_development_source_exclusions"], 2)
        self.assertEqual(report["applied_development_source_exclusions"], 2)
        self.assertEqual(report["missing_development_source_exclusions"], [])

    def test_full_preparation_is_restart_safe_and_has_thirteen_rows_per_source(self) -> None:
        records = []
        for index in range(5):
            records.extend(_source_family(f"s{index}"))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arguments = dict(
                source=records,
                output_path=directory / "data.jsonl",
                selected_sources_path=directory / "selected.jsonl",
                exclusions_path=directory / "exclusions.json",
                index_path=directory / "index.sqlite",
                selection_seed=101,
                split_seed=202,
            )
            first = prepare_raid_data(**arguments)
            second = prepare_raid_data(**arguments)
            self.assertEqual(first["prepared_rows"], 5 * 13)
            self.assertEqual(second["prepared_rows"], 5 * 13)
            rows = list(iter_jsonl(directory / "data.jsonl"))
            self.assertEqual(len(rows), 65)
            self.assertEqual(len({raid_row_key(row) for row in rows}), 65)
            for source_id in {row["source_id"] for row in rows}:
                family = [row for row in rows if row["source_id"] == source_id]
                self.assertEqual(sum(row["label"] == "human" for row in family), 1)
                self.assertEqual(
                    {row["attack"] for row in family if row["label"] == "llm"},
                    set(RAID_ATTACKS),
                )
                self.assertEqual(len({row["split"] for row in family}), 1)
            self.assertEqual(len(list(iter_jsonl(directory / "selected.jsonl"))), 5)

    def test_completed_index_can_be_reused_without_reading_the_csv_again(self) -> None:
        records = []
        for index in range(5):
            records.extend(_source_family(f"cached-{index}"))

        class MustNotIterate:
            def __iter__(self):
                raise AssertionError("reused index unexpectedly reread the RAID source")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index_path = directory / "cached.sqlite3"
            index_raid_records(records, index_path)
            report = prepare_raid_data(
                MustNotIterate(),
                directory / "data.jsonl",
                directory / "selected.jsonl",
                directory / "excluded.json",
                index_path,
                selection_seed=101,
                split_seed=202,
                reuse_index=True,
                fast_reuse_validation=True,
            )
        self.assertTrue(report["index_reused"])
        self.assertEqual(report["validation_mode"], "schema_and_record_sample")
        self.assertEqual(report["prepared_rows"], 5 * 13)


class TokenEditTests(unittest.TestCase):
    def test_standard_minimum_edit_counts(self) -> None:
        result = levenshtein_edit_counts([10], [20, 30])
        self.assertEqual(result.distance, 2)
        self.assertEqual(result.substitutions, 1)
        self.assertEqual(result.insertions, 1)
        self.assertEqual(result.deletions, 0)

        deletion = levenshtein_edit_counts([1, 2, 3], [1, 3])
        self.assertEqual(deletion.distance, 1)
        self.assertEqual(deletion.deletions, 1)

    def test_realized_rate_uses_exact_scored_window(self) -> None:
        result = measure_realized_contamination([1, 2, 3, 4], [1, 9, 3, 8])
        self.assertEqual(result["levenshtein_distance"], 2)
        self.assertAlmostEqual(result["realized_contamination_rate"], 0.5)
        self.assertFalse(result["full_text_edit_audited"])

    def test_truncation_hidden_edit_audit(self) -> None:
        result = measure_realized_contamination(
            [1, 2, 3],
            [1, 2, 3],
            clean_full_token_ids=[1, 2, 3, 4],
            attacked_full_token_ids=[1, 2, 3, 9],
        )
        self.assertEqual(result["realized_contamination_rate"], 0.0)
        self.assertEqual(result["full_levenshtein_distance"], 1)
        self.assertEqual(result["truncation_hidden_edit_count"], 1)
        self.assertTrue(result["truncation_hid_edits"])
        self.assertTrue(result["all_full_text_edits_hidden"])

    def test_empty_scored_windows_have_zero_rate(self) -> None:
        result = measure_realized_contamination([], [])
        self.assertEqual(result["levenshtein_distance"], 0)
        self.assertEqual(result["realized_contamination_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
