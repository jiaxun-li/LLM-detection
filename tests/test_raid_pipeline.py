from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_detection.io import atomic_write_json, iter_jsonl
from RAID.raid_data import RAID_ADVERSARIAL_ATTACKS
from RAID.raid_pipeline import (
    evaluate_stage,
    load_raid_config,
    merge_score_shards,
    plot_stage,
    score_shard_index,
    score_shard_output_path,
    validate_artifacts,
    write_prepared_shards,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _packs() -> tuple[list[dict], list[dict], list[dict]]:
    prepared: list[dict] = []
    falcon: list[dict] = []
    binoculars: list[dict] = []
    split_ranges = (
        ("clipping_tuning", range(0, 8)),
        ("calibration", range(8, 16)),
        ("test", range(16, 24)),
    )
    domains = ("abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki")
    for split, indices in split_ranges:
        for index in indices:
            source_id = f"source-{index}"
            domain = domains[index % len(domains)]
            conditions = [("human", "human"), ("llm", "none")]
            conditions.extend(("llm", attack) for attack in RAID_ADVERSARIAL_ATTACKS)
            for attack_index, (label, attack) in enumerate(conditions):
                is_machine = label == "llm"
                base_id = None if label == "human" else f"generation-{index}"
                text_ids = [90, 91, 92] if label == "human" else [1, 2, 3]
                if is_machine and attack != "none":
                    text_ids = [1, 20 + attack_index, 3, *([40] * (attack_index // 4))]
                row = {
                    "row_id": f"{source_id}:{label}:{attack}",
                    "dataset": "raid",
                    "sample_id": source_id,
                    "source_id": source_id,
                    "split": split,
                    "label": label,
                    "machine_origin": is_machine,
                    "text": "synthetic test text",
                    "attack": attack,
                    "base_generation_id": base_id,
                    "raid_id": f"raid:{source_id}:{attack}",
                    "domain": domain,
                    "source_generator_model": None if not is_machine else "gpt2",
                    "model": "human" if not is_machine else "gpt2",
                    "decoding": None if not is_machine else "sampling",
                    "repetition_penalty": None if not is_machine else "no",
                    "native_attack_rate": None if not is_machine else (0.0 if attack == "none" else 0.5),
                    "native_attack_rate_source": None if not is_machine else "test",
                }
                prepared.append(row)
                evidence_shift = 1.0 if is_machine else 0.0
                logp = [-3.0 + evidence_shift, -2.5 + evidence_shift, -2.0 + evidence_shift]
                rank = [8.0 - 2 * evidence_shift, 7.0 - 2 * evidence_shift, 6.0 - 2 * evidence_shift]
                log_rank = [2.0794415, 1.9459101, 1.7917595] if not is_machine else [1.7917595, 1.6094379, 1.3862944]
                entropy = [1.0 + 0.2 * evidence_shift] * 3
                common = {
                    **row,
                    "score_row_schema": "raid-score-row-v1",
                    "scoring_max_tokens": 512,
                    "original_num_input_tokens": len(text_ids) + 1,
                    "boundary_token_added": False,
                    "truncated_token_count": 0,
                    "num_scored_tokens": len(text_ids),
                }
                falcon.append(
                    {
                        **common,
                        "scoring_feature_schema": "raid-falcon-token-features-v1",
                        "scoring_context_policy": "raid_output_only_512",
                        "token_features": {
                            "logp": logp[: len(text_ids)] + [logp[-1]] * max(0, len(text_ids) - 3),
                            "nll": [-value for value in (logp[: len(text_ids)] + [logp[-1]] * max(0, len(text_ids) - 3))],
                            "rank": rank[: len(text_ids)] + [rank[-1]] * max(0, len(text_ids) - 3),
                            "log_rank": log_rank[: len(text_ids)] + [log_rank[-1]] * max(0, len(text_ids) - 3),
                            "entropy": entropy[: len(text_ids)] + [entropy[-1]] * max(0, len(text_ids) - 3),
                            "entropy_gap": [1.0] * len(text_ids),
                            "scored_token_ids": text_ids,
                            "full_scored_token_ids": text_ids,
                        },
                        "doc_scores": {},
                    }
                )
                binoculars.append(
                    {
                        **common,
                        "scoring_feature_schema": "raid-binoculars-token-features-v1",
                        "scoring_context_policy": "binoculars_official_output_only_512",
                        "token_features": {
                            "performer_nll": [1.5 + 0.1 * evidence_shift] * len(text_ids),
                            "observer_to_performer_cross_entropy": [1.0] * len(text_ids),
                            "local_gap": [0.5 + 0.1 * evidence_shift] * len(text_ids),
                            "scored_token_ids": text_ids,
                        },
                        "doc_scores": {"binoculars": 1.0},
                    }
                )
    return prepared, falcon, binoculars


class RaidPipelineTests(unittest.TestCase):
    def test_source_shards_are_family_atomic_and_merge_exactly(self):
        prepared, falcon, binoculars = _packs()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            _write_jsonl(run_dir / "data.jsonl", prepared)
            marker = write_prepared_shards(run_dir, 4)
            self.assertEqual(sum(marker["rows_per_shard"]), len(prepared))
            self.assertEqual(sum(marker["sources_per_shard"]), 24)

            source_shards: dict[str, set[int]] = {}
            for index in range(4):
                path = run_dir / "data_shards" / f"part-{index:05d}-of-00004.jsonl"
                for row in iter_jsonl(path):
                    source_shards.setdefault(row["source_id"], set()).add(index)
            self.assertTrue(all(len(values) == 1 for values in source_shards.values()))

            for scorer_name, rows in (("falcon", falcon), ("binoculars", binoculars)):
                by_shard = [[] for _ in range(4)]
                for row in rows:
                    by_shard[score_shard_index(row["source_id"], 4)].append(row)
                for index, shard_rows in enumerate(by_shard):
                    path = score_shard_output_path(run_dir, scorer_name, index, 4)
                    _write_jsonl(path, shard_rows)
                    atomic_write_json(
                        path.with_suffix(".complete.json"),
                        {
                            "scorer": scorer_name,
                            "shard_index": index,
                            "num_shards": 4,
                            "score_rows": len(shard_rows),
                            "resolved": {"test": True},
                        },
                    )

            merged = merge_score_shards(run_dir, 4)
            self.assertEqual(merged["score_summary"]["data_rows"], len(prepared))
            self.assertEqual(
                len((run_dir / "falcon_scores.jsonl").read_text(encoding="utf-8").splitlines()),
                len(prepared),
            )

    def test_cpu_evaluate_plot_validate_contract(self):
        root = Path(__file__).resolve().parents[1]
        config = load_raid_config(root / "RAID" / "config.json")
        prepared, falcon, binoculars = _packs()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run_dir = workspace / "runs" / "raid" / "test-run"
            results_dir = workspace / "results" / "raid" / "test-run"
            run_dir.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            _write_jsonl(run_dir / "data.jsonl", prepared)
            _write_jsonl(run_dir / "falcon_scores.jsonl", falcon)
            _write_jsonl(run_dir / "binoculars_scores.jsonl", binoculars)
            selected = [
                {"source_id": f"source-{index}"} for index in range(24)
            ]
            _write_jsonl(run_dir / "selected_sources.jsonl", selected)
            atomic_write_json(run_dir / "prepare.complete.json", {"prepared_rows": len(prepared)})
            atomic_write_json(run_dir / "score.complete.json", {"data_rows": len(prepared)})

            evaluation = evaluate_stage(
                run_dir,
                results_dir,
                config,
                bootstrap_repetitions=0,
                skip_binoculars=False,
            )
            self.assertEqual(evaluation["evaluation_summary"]["universal_specs"], 7)
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                plots = results_dir / "plots"
                plots.mkdir()
                (plots / "raid_attack_tpr.png").write_bytes(b"test")
                (plots / "raid_contamination_tpr.png").write_bytes(b"test")
                atomic_write_json(results_dir / "plot.complete.json", {"plots": []})
            else:
                plot_stage(results_dir)
            report = validate_artifacts(
                run_dir,
                results_dir,
                config,
                limit_sources=24,
                bootstrap_repetitions=0,
                skip_binoculars=False,
            )
            self.assertEqual(report["validation_status"], "pass")
            self.assertEqual(report["contamination_record_rows"], 24 * 12)


if __name__ == "__main__":
    unittest.main()
