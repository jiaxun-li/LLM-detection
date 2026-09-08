"""Content-checked completion and recovery for opt-in detector reanalyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiment_core.infrastructure.io import atomic_write_json

PRIMARY_ARTIFACTS = ("metrics.csv", "clipping_rejections.json", "tpr_contamination.png")
RAID_ARTIFACTS = (
    "frozen_specs.json", "metrics.csv", "attack_summary.csv",
    "contamination_summary.csv", "binoculars_sanity.csv",
    "rate_bound_tradeoff_summary.csv", "calibration_summary.csv",
    "contamination_records.csv", "evaluation_counts.json",
    "plots/raid_attack_tpr.png", "plots/raid_contamination_tpr.png",
    "plots/raid_rate_bound_tradeoff.png",
)


def artifact_digest(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"empty revision artifact: {path}")
    return {"size": size, "sha256": digest.hexdigest()}


def _identity(manifest):
    payload = {k:v for k,v in manifest.items() if k != "completion_status"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def finish_revision(directory, manifest, report, required):
    root = Path(directory)
    report = {**report, "manifest_identity_sha256": _identity(manifest),
              "artifacts": {name:artifact_digest(root/name) for name in required}}
    if report.get("validation_status") != "pass":
        raise ValueError("cannot complete an unvalidated revision")
    # Durable report contains all hashes BEFORE either completion indicator.
    atomic_write_json(root/"validation_report.json", report)
    atomic_write_json(root/"revision_manifest.json", {**manifest, "completion_status":"complete"})
    atomic_write_json(root/"revision.complete.json", report)


def resume_completed_revision(directory, manifest, required):
    """Repair interrupted finalization, never accept missing/changed artifacts.

    An unfinished evaluation without a valid final report can be recalculated.
    A claimed complete result with damaged artifacts fails explicitly instead
    of silently overwriting it or falsely reporting success.
    """
    root = Path(directory)
    try:
        report = json.loads((root/"validation_report.json").read_text(encoding="utf-8"))
        valid = (report.get("validation_status") == "pass"
                 and report.get("manifest_identity_sha256") == _identity(manifest)
                 and set(report.get("artifacts", {})) == set(required)
                 and all(report["artifacts"][name] == artifact_digest(root/name) for name in required))
        if not valid:
            raise ValueError("revision artifact hashes or validation identity disagree")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if manifest.get("completion_status") == "complete":
            raise ValueError(f"completed revision failed artifact validation: {exc}") from exc
        return False
    marker = root/"revision.complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved != report:
            raise ValueError("revision completion marker disagrees with validation report")
    if manifest.get("completion_status") != "complete":
        atomic_write_json(root/"revision_manifest.json", {**manifest, "completion_status":"complete"})
    if not marker.exists():
        atomic_write_json(marker, report)
    return True
