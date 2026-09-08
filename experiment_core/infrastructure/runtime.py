"""Run manifests, environment provenance, and throughput reporting."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from experiment_core.infrastructure.io import atomic_write_json


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        process = subprocess.run(
            ["git", "-c", f"safe.directory={cwd.as_posix()}", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def software_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "vllm",
        "pandas",
        "rapidfuzz",
        "matplotlib",
    ]
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def accelerator_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "gpus": [],
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                info["gpus"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "capability": [properties.major, properties.minor],
                    }
                )
    except ImportError:
        pass
    return info


def build_manifest(
    run_config: dict[str, Any],
    workspace: str | Path,
    inputs: dict[str, str],
    outputs: dict[str, str],
    source_manifest: str,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    dirty = _git(["status", "--porcelain"], root)
    return {
        "run_id": run_config["run_id"],
        "experiment_name": run_config.get("experiment_name"),
        "result_label": run_config.get("result_label", "SCIENTIFIC_RUN"),
        "debug_only": bool(run_config.get("debug_only", False)),
        "creation_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"], root),
        "dirty_worktree": None if dirty is None else bool(dirty),
        "dataset": {
            "name": run_config["dataset"],
            **run_config["datasets"][run_config["dataset"]],
        },
        "source_sample_manifest": str(Path(source_manifest).resolve()),
        "target_model": run_config["target_model"],
        "target_model_revision": run_config.get("target_model_revision"),
        "target_tokenizer_revision": run_config.get("target_tokenizer_revision"),
        "binoculars_models": run_config["models"]["binoculars"],
        "generation": run_config["generation"],
        "random_seeds": {
            "source_selection": run_config["splits"]["selection_seed"],
            "generation": run_config["generation"]["seeds"],
            "corruption": run_config["contamination"]["corruption_seed"],
            "bootstrap": run_config["evaluation"]["bootstrap_seed"],
        },
        "split_sizes": run_config["splits"],
        "contamination": run_config["contamination"],
        "scoring": run_config["scoring"],
        "evaluation": run_config["evaluation"],
        "software_versions": software_versions(),
        "accelerator": accelerator_info(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "inputs": inputs,
        "outputs": outputs,
        "completion_status": "running",
        "completed_stages": [],
    }


def update_manifest(path: str | Path, manifest: dict[str, Any], **changes: Any) -> None:
    manifest.update(changes)
    atomic_write_json(path, manifest)


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class Throughput:
    """Compact progress reporter shared by generation and scoring."""

    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.started = time.monotonic()
        self.examples = 0
        self.tokens = 0

    def update(self, examples: int, tokens: int, peak_gpu_memory: int | None = None) -> None:
        self.examples += examples
        self.tokens += tokens
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.tokens / elapsed
        remaining = max(self.total - self.examples, 0)
        examples_per_second = self.examples / elapsed
        eta = remaining / examples_per_second if examples_per_second else float("inf")
        memory = "n/a" if peak_gpu_memory is None else f"{peak_gpu_memory / 2**30:.2f} GiB"
        print(
            f"{self.label}: examples={self.examples}/{self.total} "
            f"tokens/s={rate:.2f} elapsed_s={elapsed:.1f} eta_s={eta:.1f} "
            f"peak_gpu_memory={memory}",
            flush=True,
        )


def peak_gpu_memory_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return max(
                torch.cuda.max_memory_allocated(index)
                for index in range(torch.cuda.device_count())
            )
    except ImportError:
        pass
    return None
