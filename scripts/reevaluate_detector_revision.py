#!/usr/bin/env python3
"""Re-evaluate a primary cell using immutable saved, prompt-conditioned scores."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RAID.revise_detectors import identifier, fingerprint, read_json, implementation_fingerprint
from llm_detection.detector_revision import REVISION
from llm_detection.evaluation import REVISED_METHODS, evaluate
from llm_detection.io import atomic_write_json, iter_jsonl


def revised_config(manifest, result_id, repetitions=None):
    config = {k:copy.deepcopy(manifest[k]) for k in ("generation","contamination","evaluation","scoring")}
    dataset = manifest["dataset"]
    config.update(run_id=result_id, dataset=dataset["name"],
                  datasets={dataset["name"]:copy.deepcopy(dataset)}, target_model=manifest["target_model"])
    config["scoring"]["detectors"] = list(REVISED_METHODS)
    config["evaluation"]["detector_revision"] = REVISION
    if repetitions is not None:config["evaluation"]["bootstrap_repetitions"]=repetitions
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("."))
    p.add_argument("--source-run-id",type=identifier,required=True)
    p.add_argument("--revision-id",type=identifier,required=True)
    p.add_argument("--bootstrap-repetitions",type=int)
    args=p.parse_args()
    if args.source_run_id==args.revision_id:raise ValueError("use a NEW result ID")
    if args.bootstrap_repetitions is not None and args.bootstrap_repetitions<1:raise ValueError("primary evaluation requires at least one bootstrap")
    source=args.workspace.resolve()/"runs"/args.source_run_id
    output=args.workspace.resolve()/"results"/args.revision_id
    manifest=read_json(source/"manifest.json")
    if manifest.get("completion_status")!="complete":raise ValueError("source cell must be complete")
    config=revised_config(manifest,args.revision_id,args.bootstrap_repetitions)
    paths={name:source/name for name in ("manifest.json","target_scores.jsonl","binoculars_scores.jsonl")}
    identity={"detector_revision":REVISION,"source_run_id":args.source_run_id,
        "implementation_sha256":implementation_fingerprint(),
        "revision_id":args.revision_id,"configuration":config,
        "source_artifacts":{name:fingerprint(path) for name,path in paths.items()},
        "binocular_origin_context":"conditional_same_saved_continuation_window",
        "binocular_gap":"legacy_exp_mean_gap_unchanged"}
    marker=output/"revision_manifest.json"
    if output.exists():
        previous=read_json(marker)
        if any(previous.get(k)!=v for k,v in identity.items()):raise ValueError("revision resume mismatch")
        if previous.get("completion_status")=="complete":
            if read_json(output/"revision.complete.json").get("validation_status")!="pass":
                raise ValueError("completed revision lacks passing marker")
            print("primary revision already complete");return
    output.mkdir(parents=True,exist_ok=True)
    atomic_write_json(marker,{**identity,"completion_status":"running"})
    # Fail on missing or incompatible conditioning rather than quietly changing it.
    for row in iter_jsonl(paths["binoculars_scores.jsonl"],tolerate_partial_last_line=False):
        policy=row.get("scoring_context_policy", "prompt_conditioned_response_only")
        if policy!="prompt_conditioned_response_only":
            raise ValueError(f"primary pack is not prompt-conditioned: {policy}")
    rows=evaluate(paths["target_scores.jsonl"],output/"metrics.csv",config,
                  paths["binoculars_scores.jsonl"])
    if {r["detector"] for r in rows}!=set(REVISED_METHODS):raise ValueError("missing revised detectors")
    if any(r["direction"]!=1 for r in rows if r["detector"]=="lrr"):
        raise ValueError("LRR direction contract failed")
    from plot_tpr_contamination import read_metrics, plot
    plot(read_metrics(output/"metrics.csv","primary_frozen_mixture"),REVISED_METHODS,
         output/"tpr_contamination.png",True)
    if identity["source_artifacts"]!={name:fingerprint(path) for name,path in paths.items()}:
        raise ValueError("source artifacts changed during evaluation")
    report={"validation_status":"pass","detectors":REVISED_METHODS,"metric_rows":len(rows),
        "bootstrap_repetitions":config["evaluation"]["bootstrap_repetitions"]}
    atomic_write_json(output/"validation_report.json",report)
    atomic_write_json(marker,{**identity,"completion_status":"complete"})
    atomic_write_json(output/"revision.complete.json",report)
    print("primary eight-detector revision: PASS",flush=True)


if __name__=="__main__":main()
