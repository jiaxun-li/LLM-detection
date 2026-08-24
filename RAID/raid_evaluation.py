"""Leakage-safe, CPU-only evaluation for the frozen RAID protocol."""
from __future__ import annotations

import csv, json, math, os, random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from llm_detection.evaluation import ALL_METHODS, actual_fpr, auroc, calibration_threshold, percentile_interval
from llm_detection.io import atomic_write_json
from llm_detection.scoring import EPS
from RAID.raid_data import measure_realized_contamination

DETECTORS = tuple(ALL_METHODS)
QUANTILE_GRID = (0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995)
TARGET_FPR = .05
RATE_ADAPTIVE_CUTPOINTS = (.05, .10, .20, .50)
RATE_ADAPTIVE_BIN_INDICES = (1, 2, 3, 4)
RATE_ADAPTIVE_CLEAN_AUROC_LOSS_BUDGET = .01
PUBLISHED_BINOCULARS_TPR = {"none":.796,"paraphrase":.803,"synonym":.435,
    "perplexity_misspelling":.780,"homoglyph":.377,"whitespace":.701,"article_deletion":.743}
FEATURES = {
 "logp":("logp","log_probability","token_logp"), "rank":("rank","token_rank"),
 "log_rank":("log_rank","token_log_rank"), "entropy":("entropy","token_entropy"),
 "performer_nll":("performer_nll","binoculars_performer_nll"),
 "cross_entropy":("observer_to_performer_cross_entropy","cross_entropy","binoculars_cross_entropy"),
 "token_ids":("scored_token_ids","token_ids"),
 "full_token_ids":("full_scored_token_ids",),
}

@dataclass(frozen=True)
class RaidEvaluationResult:
    frozen_specs: dict[str,Any]; metrics:list[dict[str,Any]]
    attack_summary:list[dict[str,Any]]; contamination_summary:list[dict[str,Any]]
    rate_bound_tradeoff_summary:list[dict[str,Any]]
    contamination_cutpoints:list[float]; binoculars_sanity:list[dict[str,Any]]
    calibration_summary:list[dict[str,Any]]; contamination_records:list[dict[str,Any]]
    validation_counts:dict[str,Any]

def _first(r,names,default=None):
    return next((r[n] for n in names if n in r and r[n] is not None),default)
def source(r): return str(_first(r,("source_id","sample_id","human_source_id")))
def domain(r): return str(_first(r,("domain","dataset_domain","genre")))
def split(r):
    v=str(_first(r,("split","partition"),""))
    return {"tuning":"clipping_tuning","tune":"clipping_tuning"}.get(v,v)
def human(r): return str(_first(r,("label","origin","text_origin"),"")).lower() in {"human","h","0"}
def condition(r):
    if human(r): return "human"
    v=str(_first(r,("attack","condition","attack_name","contamination_mode"),"none")).lower()
    v=v.replace("-","_").replace(" ","_")
    return {"clean":"none","original":"none","unattacked":"none","no_attack":"none",
      "delete_article":"article_deletion","delete_articles":"article_deletion",
      "synonyms":"synonym","homoglyphs":"homoglyph",
      "misspelling":"perplexity_misspelling","misspellings":"perplexity_misspelling",
      "paraphrasing":"paraphrase",
      "whitespace_insertion":"whitespace"}.get(v,v)
def feat(r,name,dtype=float):
    fs=r.get("token_features",r.get("features",{}))
    for alias in FEATURES[name]:
        if alias in fs:
            x=np.asarray(fs[alias],dtype=dtype)
            if x.ndim!=1 or not len(x): raise ValueError(f"empty/invalid feature {alias}")
            return x
    raise KeyError(f"missing token feature {name}")
def local(r,d):
    if d=="log_likelihood": return feat(r,"logp")
    if d=="rank": return feat(r,"rank")
    if d=="log_rank": return feat(r,"log_rank")
    if d=="entropy": return feat(r,"entropy")
    if d=="entropy_gap": return -feat(r,"logp")-feat(r,"entropy")
    if d=="binoculars": return feat(r,"performer_nll")-feat(r,"cross_entropy")
    raise KeyError(d)
def raw_document_score(r,d):
    ds=r.get("doc_scores",{})
    if d in ds: return float(ds[d])
    if d=="lrr": return float((-feat(r,"logp")).mean()/(feat(r,"log_rank").mean()+EPS))
    x=local(r,d); return float(math.exp(x.mean()) if d=="binoculars" else x.mean())
def oriented_document_score(r,d,direction,spec=None):
    spec=spec or {}
    if not spec: return direction*raw_document_score(r,d)
    if d=="lrr":
        n=np.minimum(-feat(r,"logp"),spec["nll_upper"])
        q=np.minimum(feat(r,"log_rank"),spec["log_rank_upper"])
        return direction*float(n.mean()/(q.mean()+EPS))
    m=float(np.maximum(direction*local(r,d),spec["lower"]).mean())
    return direction*math.exp(direction*m) if d=="binoculars" else m

def stable_row_key(r):
    explicit=_first(r,("row_key","stable_row_key"))
    if explicit is not None:return (str(explicit),)
    return (source(r),split(r),"human" if human(r) else "machine",condition(r),
            str(_first(r,("base_generation_id","generation_id","machine_id"),"")))
def merge_score_rows(falcon_rows,binoculars_rows=None):
    out={stable_row_key(r):dict(r) for r in falcon_rows}
    if len(out)!=len(falcon_rows): raise ValueError("duplicate Falcon row key")
    if binoculars_rows is None:return list(out.values())
    b={stable_row_key(r):r for r in binoculars_rows}
    if set(out)!=set(b): raise ValueError("Falcon/Binoculars row-key mismatch")
    for k,br in b.items():
        out[k].setdefault("doc_scores",{}).update({x:y for x,y in br.get("doc_scores",{}).items() if x=="binoculars"})
        names=set(FEATURES["performer_nll"]+FEATURES["cross_entropy"])
        out[k].setdefault("token_features",{}).update({x:y for x,y in br.get("token_features",{}).items() if x in names})
    return list(out.values())

def token_edit_counts(original, attacked):
    """Compatibility wrapper over the shared, prefix-trimmed RapidFuzz path."""
    z=measure_realized_contamination(original,attacked)
    return {"substitutions":z["substitution_count"],"deletions":z["deletion_count"],
      "insertions":z["insertion_count"],"edit_distance":z["levenshtein_distance"],
      "contamination_rate":z["realized_contamination_rate"]}
def attach_contamination_rates(rows):
    none={(split(r),source(r)):r for r in rows if not human(r) and condition(r)=="none"}
    for r in rows:
        if human(r):continue
        base=r if condition(r)=="none" else none.get((split(r),source(r)))
        if base is None: raise ValueError(f"missing none row for {source(r)}")
        kwargs={}
        base_features=base.get("token_features",{}); attacked_features=r.get("token_features",{})
        full_name=FEATURES["full_token_ids"][0]
        if full_name in base_features and full_name in attacked_features:
            kwargs={"clean_full_token_ids":base_features[full_name],
                    "attacked_full_token_ids":attacked_features[full_name]}
        measured=measure_realized_contamination(
          feat(base,"token_ids",np.int64),feat(r,"token_ids",np.int64),**kwargs)
        stats={**measured,"substitutions":measured["substitution_count"],"deletions":measured["deletion_count"],
          "insertions":measured["insertion_count"],"edit_distance":measured["levenshtein_distance"],
          "contamination_rate":measured["realized_contamination_rate"]}
        for k,v in stats.items():
            if r.get(k) is not None and v is not None and not math.isclose(float(r[k]),float(v),rel_tol=0,abs_tol=1e-12):
                raise ValueError(f"saved {k} disagrees with scored-token alignment")
            r[k]=v
    return rows
def rho(r): return float(r.get("contamination_rate",0.0))

def learn_direction(d,h,m):
    return 1 if np.mean([raw_document_score(r,d) for r in m])>=np.mean([raw_document_score(r,d) for r in h]) else -1
def candidate_specifications(d,direction,clean,quantiles=QUANTILE_GRID):
    out=[{}]
    if d=="lrr":
        a=np.concatenate([-feat(r,"logp") for r in clean]); b=np.concatenate([feat(r,"log_rank") for r in clean])
        return out+[{"nll_upper":float(np.quantile(a,q)),"log_rank_upper":float(np.quantile(b,z))} for q in quantiles for z in quantiles]
    a=np.concatenate([direction*local(r,d) for r in clean])
    return out+[{"lower":float(np.quantile(a,1-q))} for q in quantiles]
def select_universal_specification(d,direction,h,clean,attacks,quantiles=QUANTILE_GRID):
    if len(attacks)!=11 or any(not x for x in attacks.values()):raise ValueError("requires 11 nonempty attacks")
    best={}; best_j=-math.inf; diagnostics=[]
    for i,spec in enumerate(candidate_specifications(d,direction,list(h)+list(clean),quantiles)):
        hs=[oriented_document_score(r,d,direction,spec) for r in h]
        ca=auroc(hs,[oriented_document_score(r,d,direction,spec) for r in clean])
        aa={a:auroc(hs,[oriented_document_score(r,d,direction,spec) for r in rs]) for a,rs in sorted(attacks.items())}
        j=.8*float(np.mean(list(aa.values())))+.2*ca
        diagnostics.append({"candidate_index":i,"specification":spec,"objective":j,"clean_auroc":ca,"attack_aurocs":aa})
        if j>best_j+1e-12:best_j=j;best=dict(spec)
    return best,diagnostics

def select_rate_adaptive_specification(
    d,direction,h,clean,bin_rows,universal_spec,min_tuning_rows,
    clean_auroc_loss_budget=RATE_ADAPTIVE_CLEAN_AUROC_LOSS_BUDGET,
    quantiles=QUANTILE_GRID
):
    """Fit one bound by attacked AUROC gain under a clean-loss constraint.

    Attacks represented in the bin are equally weighted. A sparse bin uses the
    already-frozen universal rule; fixed rate boundaries are never moved.
    """
    attack_rows=defaultdict(list)
    for r in bin_rows:attack_rows[condition(r)].append(r)
    attack_rows=dict(sorted(attack_rows.items()))
    counts={name:len(values) for name,values in attack_rows.items()}
    if len(bin_rows)<min_tuning_rows or not attack_rows:
        return dict(universal_spec),[],{
          "fallback_to_universal":True,"fallback_reason":"insufficient_tuning_rows",
          "n_tuning_machine":len(bin_rows),"attack_counts":counts,
          "selection_rule":"max_attack_auroc_gain_subject_to_clean_loss",
          "clean_auroc_loss_budget":clean_auroc_loss_budget,
        }
    candidates=candidate_specifications(d,direction,list(h)+list(clean),quantiles)
    raw_h=[oriented_document_score(r,d,direction,{}) for r in h]
    raw_clean_auc=auroc(raw_h,[oriented_document_score(r,d,direction,{}) for r in clean])
    raw_attack_aucs={name:auroc(raw_h,[oriented_document_score(r,d,direction,{}) for r in values])
      for name,values in attack_rows.items()}
    best={};best_gain=-math.inf;diagnostics=[]
    for i,spec in enumerate(candidates):
        hs=[oriented_document_score(r,d,direction,spec) for r in h]
        clean_auc=auroc(hs,[oriented_document_score(r,d,direction,spec) for r in clean])
        attack_aucs={name:auroc(hs,[oriented_document_score(r,d,direction,spec) for r in values])
          for name,values in attack_rows.items()}
        gains={name:attack_aucs[name]-raw_attack_aucs[name] for name in attack_aucs}
        mean_gain=float(np.mean(list(gains.values())))
        clean_loss=raw_clean_auc-clean_auc
        feasible=clean_loss<=clean_auroc_loss_budget+1e-12
        diagnostics.append({"candidate_index":i,"specification":spec,
          "mean_attack_auroc_gain":mean_gain,"attack_auroc_gains":gains,
          "clean_auroc":clean_auc,"raw_clean_auroc":raw_clean_auc,
          "clean_auroc_loss":clean_loss,"clean_auroc_loss_budget":clean_auroc_loss_budget,
          "constraint_feasible":feasible,"attack_aurocs":attack_aucs,
          "raw_attack_aurocs":raw_attack_aucs})
        if feasible and mean_gain>best_gain+1e-12:best_gain=mean_gain;best=dict(spec)
    return best,diagnostics,{"fallback_to_universal":False,"fallback_reason":None,
      "n_tuning_machine":len(bin_rows),"attack_counts":counts,
      "selection_rule":"max_attack_auroc_gain_subject_to_clean_loss",
      "clean_auroc_loss_budget":clean_auroc_loss_budget}

def freeze_contamination_cutpoints(tuning_rows,requested=None,positive_groups=4,min_count=1):
    """Return the protocol-fixed rate-oracle boundaries.

    ``tuning_rows`` and the legacy sizing arguments remain in the signature so
    older callers fail scientifically rather than silently learning new bins.
    Bin sample sizes are handled when each rate-specific clipping rule is fit.
    """
    del tuning_rows, positive_groups, min_count
    values=RATE_ADAPTIVE_CUTPOINTS if requested is None else requested
    cuts=tuple(float(x) for x in values)
    if cuts!=RATE_ADAPTIVE_CUTPOINTS:
        raise ValueError(
          f"rate-adaptive cutpoints are frozen at {list(RATE_ADAPTIVE_CUTPOINTS)}"
        )
    return list(cuts),"fixed_protocol"
def contamination_bin(x,cuts):
    if x<=0:return 0,"rho=0"
    for i,u in enumerate(cuts,1):
        lo=0 if i==1 else cuts[i-2]
        if x<=u:return i,f"({lo:.6g},{u:.6g}]"
    lo=cuts[-1] if cuts else 0
    return len(cuts)+1,f"({lo:.6g},1]"
def _rows(rows,s,h=None,c=None):
    z=[r for r in rows if split(r)==s]
    if h is not None:z=[r for r in z if human(r)==h]
    if c is not None:z=[r for r in z if condition(r)==c]
    return z
def _thresholds(cal,d,direction,spec,clipping_mode="universal",bin_index=None,bin_label=None):
    by=defaultdict(list)
    for r in cal:by[domain(r)].append(r)
    out={}; summary=[]
    for dom,rs in sorted(by.items()):
        a=[oriented_document_score(r,d,direction) for r in rs]
        b=[oriented_document_score(r,d,direction,spec) for r in rs]
        ta=calibration_threshold(a,TARGET_FPR);tb=calibration_threshold(b,TARGET_FPR)
        out[dom]={"raw":ta,"clipped":tb}
        summary.append({"detector":d,"domain":dom,"clipping_mode":clipping_mode,
          "contamination_bin_index":bin_index,"contamination_bin":bin_label,
          "target_fpr":TARGET_FPR,"n_calibration_human":len(rs),
          "raw_threshold":ta,"raw_calibration_fpr":actual_fpr(a,ta),
          "clipped_threshold":tb,"clipped_calibration_fpr":actual_fpr(b,tb)})
    return out,summary
def _cache(rows,d,direction,spec):
    return ({id(r):oriented_document_score(r,d,direction) for r in rows},
            {id(r):oriented_document_score(r,d,direction,spec) for r in rows})
def _point(h,m,a,b,thresholds):
    ah=[a[id(r)] for r in h];bh=[b[id(r)] for r in h]
    am=[a[id(r)] for r in m];bm=[b[id(r)] for r in m]
    at=float(np.mean([a[id(r)]>=thresholds[domain(r)]["raw"] for r in m])) if m else float("nan")
    bt=float(np.mean([b[id(r)]>=thresholds[domain(r)]["clipped"] for r in m])) if m else float("nan")
    af=float(np.mean([a[id(r)]>=thresholds[domain(r)]["raw"] for r in h]))
    bf=float(np.mean([b[id(r)]>=thresholds[domain(r)]["clipped"] for r in h]))
    aa=auroc(ah,am);ba=auroc(bh,bm)
    return {"raw_tpr":at,"clipped_tpr":bt,"paired_tpr_difference":bt-at,
      "raw_auroc":aa,"clipped_auroc":ba,"paired_auroc_difference":ba-aa,
      "raw_test_fpr":af,"clipped_test_fpr":bf}
def _seed(base,*parts):
    x=int(base)&0xffffffff
    for p in parts:
        for q in str(p).encode():x=((x*16777619)^q)&0xffffffff
    return x
def _bootstrap(h,m,a,b,thresholds,reps,seed):
    if reps<=0:return {}
    ids=sorted(source(r) for r in h); pos={x:i for i,x in enumerate(ids)}
    hi=np.arange(len(ids)); domain_indices=defaultdict(list)
    hm={source(r):r for r in h}
    for sid in ids:domain_indices[domain(hm[sid])].append(pos[sid])
    ms=np.asarray([pos[source(r)] for r in m],dtype=np.int64)
    ah=np.asarray([a[id(hm[x])] for x in ids]);bh=np.asarray([b[id(hm[x])] for x in ids])
    am=np.asarray([a[id(r)] for r in m]);bm=np.asarray([b[id(r)] for r in m])
    ahit=np.asarray([ah[i]>=thresholds[domain(hm[x])]["raw"] for i,x in enumerate(ids)])
    bhit=np.asarray([bh[i]>=thresholds[domain(hm[x])]["clipped"] for i,x in enumerate(ids)])
    amit=np.asarray([am[i]>=thresholds[domain(r)]["raw"] for i,r in enumerate(m)])
    bmit=np.asarray([bm[i]>=thresholds[domain(r)]["clipped"] for i,r in enumerate(m)])
    def auc_plan(hs,mscores):
        order=np.argsort(hs,kind="mergesort"); sh=hs[order]
        return order,np.searchsorted(sh,mscores,"left"),np.searchsorted(sh,mscores,"right")
    plans=(auc_plan(ah,am),auc_plan(bh,bm))
    def weighted_auc(hw,mw,plan):
        order,left,right=plan; cum=np.r_[0,np.cumsum(hw[order])]
        favorable=cum[left]+.5*(cum[right]-cum[left])
        denom=hw.sum()*mw.sum()
        return float(np.dot(mw,favorable)/denom) if denom else float("nan")
    rng=np.random.default_rng(seed); vals=defaultdict(list)
    for _ in range(reps):
        w=np.zeros(len(ids),dtype=np.int64)
        for dom in sorted(domain_indices):
            ix=np.asarray(domain_indices[dom]);w[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
        mw=w[ms]; hden=w.sum();mden=mw.sum()
        af=float(np.dot(w,ahit)/hden);bf=float(np.dot(w,bhit)/hden)
        at=float(np.dot(mw,amit)/mden) if mden else float("nan")
        bt=float(np.dot(mw,bmit)/mden) if mden else float("nan")
        aa=weighted_auc(w,mw,plans[0]);ba=weighted_auc(w,mw,plans[1])
        p={"raw_tpr":at,"clipped_tpr":bt,"paired_tpr_difference":bt-at,
          "raw_auroc":aa,"clipped_auroc":ba,"paired_auroc_difference":ba-aa,
          "raw_test_fpr":af,"clipped_test_fpr":bf}
        for k,v in p.items(): vals[k].append(v)
    return {k:percentile_interval(v) for k,v in vals.items()}
def _add_ci(row,ci):
    for k,(lo,hi) in ci.items():row[k+"_ci_low"]=lo;row[k+"_ci_high"]=hi

def _add_rate_adaptive_pair(row,point,ci):
    """Add the clipped side of a raw/rate-specific paired comparison."""
    for key,value in point.items():
        if key.startswith("clipped_"):
            row["rate_adaptive_"+key]=value
        elif key.startswith("paired_"):
            row["rate_adaptive_"+key]=value
    for key,(lo,hi) in ci.items():
        if key.startswith("clipped_") or key.startswith("paired_"):
            name="rate_adaptive_"+key
            row[name+"_ci_low"]=lo;row[name+"_ci_high"]=hi
def _validate(rows,expected=None):
    if not rows:raise ValueError("no RAID score rows")
    ss=defaultdict(set)
    for r in rows:ss[source(r)].add(split(r))
    if any(len(v)!=1 for v in ss.values()):raise ValueError("source leakage across splits")
    if set().union(*ss.values())!={"clipping_tuning","calibration","test"}:raise ValueError("invalid split set")
    human_domains={s:{domain(r) for r in rows if split(r)==s and human(r)}
                   for s in ("clipping_tuning","calibration","test")}
    if len({tuple(sorted(v)) for v in human_domains.values()})!=1:
        raise ValueError(f"human domains differ across splits: {human_domains}")
    attacks=sorted({condition(r) for r in rows if not human(r)}-{"none"})
    normalized_expected=(
        {condition({"label":"llm","attack":x}) for x in expected}
        if expected is not None else set(attacks)
    )
    if set(attacks)!=normalized_expected:raise ValueError("configured attacks disagree with data")
    if len(attacks)!=11:raise ValueError("RAID protocol requires exactly 11 attacks")
    # Every source must have exactly one human and one machine row per condition.
    grouped=defaultdict(list)
    for r in rows:grouped[source(r)].append(r)
    for sid,sr in grouped.items():
        if sum(human(r) for r in sr)!=1 or Counter(condition(r) for r in sr if not human(r))!=Counter(["none"]+attacks):
            raise ValueError(f"incomplete source family {sid}")
    return attacks,{"score_rows":len(rows),"sources":len(ss),"human_rows":sum(human(r) for r in rows),
      "machine_rows":sum(not human(r) for r in rows),"attacks":attacks,
      "splits":dict(Counter(split(r) for r in rows)),"target_fprs":[TARGET_FPR],"detectors":list(DETECTORS)}
def _tidy(analysis,wide):
    out=[]
    for r in wide:
        aggregations=[("raw","raw","paired"),("clipped","clipped","paired")]
        if "rate_adaptive_clipped_tpr" in r:
            aggregations.append(("rate_adaptive_clipped","rate_adaptive_clipped","rate_adaptive_paired"))
        for agg,prefix,pair_prefix in aggregations:
            z={k:v for k,v in r.items() if not k.startswith(("raw_","clipped_","paired_","rate_adaptive_"))}
            z.update({"analysis":analysis,"aggregation":agg,"target_fpr":TARGET_FPR,
              "tpr":r[prefix+"_tpr"],"tpr_ci_low":r.get(prefix+"_tpr_ci_low"),"tpr_ci_high":r.get(prefix+"_tpr_ci_high"),
              "auroc":r[prefix+"_auroc"],"auroc_ci_low":r.get(prefix+"_auroc_ci_low"),"auroc_ci_high":r.get(prefix+"_auroc_ci_high"),
              "held_out_fpr":r[prefix+"_test_fpr"],"held_out_fpr_ci_low":r.get(prefix+"_test_fpr_ci_low"),
              "held_out_fpr_ci_high":r.get(prefix+"_test_fpr_ci_high"),
              "paired_tpr_difference":r[pair_prefix+"_tpr_difference"],
              "paired_tpr_difference_ci_low":r.get(pair_prefix+"_tpr_difference_ci_low"),
              "paired_tpr_difference_ci_high":r.get(pair_prefix+"_tpr_difference_ci_high")})
            out.append(z)
    return out

def evaluate_raid(falcon_rows,config=None,binoculars_rows=None,published_binoculars=None):
    """Evaluate raw, universal clipped, and fixed-rate-oracle clipped scores."""
    cfg=dict(config or {}); cfg={**cfg,**cfg.get("evaluation",{})}
    if float(cfg.get("target_fpr",TARGET_FPR))!=TARGET_FPR:raise ValueError("only 5% FPR is allowed")
    if tuple(cfg.get("clipping_quantiles",QUANTILE_GRID))!=QUANTILE_GRID:raise ValueError("quantile grid is frozen")
    attack_weight=float(cfg.get("attack_weight",.8));clean_weight=float(cfg.get("clean_weight",.2))
    if not (math.isclose(attack_weight,.8) and math.isclose(clean_weight,.2)):
        raise ValueError("selection weights are frozen at attack=.8 and clean=.2")
    reps=int(cfg.get("bootstrap_repetitions",2000)); bseed=int(cfg.get("bootstrap_seed",481516))
    rows=merge_score_rows(falcon_rows,binoculars_rows)
    configured_attacks=cfg.get("expected_attacks")
    if configured_attacks is None and isinstance(config,Mapping):
        configured_attacks=config.get("dataset",{}).get("required_attacks")
    attacks,counts=_validate(rows,configured_attacks)
    rows=attach_contamination_rates(rows)
    contamination_fields=(
      "source_id","sample_id","split","domain","raid_id","base_generation_id",
      "attack","source_generator_model","model","decoding","repetition_penalty",
      "native_attack_rate","native_attack_rate_source","clean_scored_token_count",
      "attacked_scored_token_count","levenshtein_distance","substitution_count",
      "deletion_count","insertion_count","realized_contamination_rate",
      "full_text_edit_audited","clean_full_token_count","attacked_full_token_count",
      "full_levenshtein_distance","full_substitution_count","full_deletion_count",
      "full_insertion_count","truncation_applied_clean","truncation_applied_attacked",
      "truncation_hidden_edit_count","truncation_hid_edits","all_full_text_edits_hidden")
    contamination_records=[
      {name:r.get(name) for name in contamination_fields}
      for r in rows if not human(r)]
    tune=_rows(rows,"clipping_tuning"); th=_rows(rows,"clipping_tuning",True);tc=_rows(rows,"clipping_tuning",False,"none")
    cal=_rows(rows,"calibration",True);testh=_rows(rows,"test",True);testm=_rows(rows,"test",False)
    requested_cuts=cfg.get("rate_adaptive_cutpoints",cfg.get("contamination_cutpoints"))
    cuts,provenance=freeze_contamination_cutpoints(tune,requested_cuts)
    min_rate_rows=int(cfg.get("rate_adaptive_min_tuning_rows",250))
    if min_rate_rows<1:raise ValueError("rate_adaptive_min_tuning_rows must be positive")
    clean_loss_budget=float(cfg.get("rate_adaptive_clean_auroc_loss_budget",
      RATE_ADAPTIVE_CLEAN_AUROC_LOSS_BUDGET))
    if not math.isclose(clean_loss_budget,RATE_ADAPTIVE_CLEAN_AUROC_LOSS_BUDGET,abs_tol=1e-12):
        raise ValueError("rate-adaptive clean AUROC loss budget is frozen at 0.01")
    ta={x:_rows(rows,"clipping_tuning",False,x) for x in attacks}
    frozen={"protocol":"raid_constrained_fixed_rate_oracle_v3","target_fpr":TARGET_FPR,"quantile_grid":list(QUANTILE_GRID),
      "contamination_cutpoints":cuts,"contamination_cutpoint_provenance":provenance,
      "rate_adaptive_eligible_interval":"0<rho<=0.5",
      "rate_adaptive_excluded_intervals":["rho=0","rho>0.5"],
      "rate_adaptive_min_tuning_rows":min_rate_rows,
      "rate_adaptive_clean_auroc_loss_budget":clean_loss_budget,
      "rate_adaptive_selection_rule":"max_attack_auroc_gain_subject_to_clean_loss",
      "rate_adaptive_clipping":True,"attack_specific_clipping":False,"detectors":{}}
    attack_summary=[];contamination_summary=[];rate_bound_tradeoff_summary=[];calibration_summary=[]
    fallback_specs=0
    test_none=[r for r in testm if condition(r)=="none"]
    for d in DETECTORS:
        direction=learn_direction(d,th,tc);spec,diagnostics=select_universal_specification(d,direction,th,tc,ta)
        thresholds,cs=_thresholds(cal,d,direction,spec,"universal");calibration_summary+=cs
        selected=next(x for x in diagnostics if x["specification"]==spec)
        frozen["detectors"][d]={"direction":direction,"clipping_specification":spec,
          "selection_objective":selected["objective"],"candidate_diagnostics":diagnostics,
          "classification_thresholds":thresholds,"rate_adaptive_bins":{}}
        a,b=_cache(testh+testm,d,direction,spec)
        for record in cs:
            domain_humans=[r for r in testh if domain(r)==record["domain"]]
            record["n_test_human"]=len(domain_humans)
            record["raw_test_fpr"]=actual_fpr(
              [a[id(r)] for r in domain_humans],record["raw_threshold"])
            record["clipped_test_fpr"]=actual_fpr(
              [b[id(r)] for r in domain_humans],record["clipped_threshold"])
        for cond in ["none"]+attacks:
            mr=[r for r in testm if condition(r)==cond]
            z={"detector":d,"condition":cond,"target_fpr":TARGET_FPR,"n_test_human":len(testh),
               "n_test_machine":len(mr),"n_test_sources":len({source(r) for r in mr}),
               "clipping_specification":json.dumps(spec,sort_keys=True),**_point(testh,mr,a,b,thresholds)}
            _add_ci(z,_bootstrap(testh,mr,a,b,thresholds,reps,_seed(bseed,d,"attack",cond)));attack_summary.append(z)
        tuning_bins=defaultdict(list);test_bins=defaultdict(list)
        for r in tune:
            if not human(r) and condition(r)!="none":
                key=contamination_bin(rho(r),cuts)
                if key[0] in RATE_ADAPTIVE_BIN_INDICES:tuning_bins[key].append(r)
        for r in testm:
            key=contamination_bin(rho(r),cuts)
            if key[0] in RATE_ADAPTIVE_BIN_INDICES:test_bins[key].append(r)
        for i in RATE_ADAPTIVE_BIN_INDICES:
            lo=0.0 if i==1 else cuts[i-2];upper=cuts[i-1]
            label=f"({lo:.6g},{upper:.6g}]"
            tuning_machine=tuning_bins.get((i,label),[])
            mr=test_bins.get((i,label),[])
            if not mr:raise ValueError(f"fixed contamination bin {label} has no test rows")
            adaptive_spec,adaptive_diagnostics,adaptive_meta=select_rate_adaptive_specification(
              d,direction,th,tc,tuning_machine,spec,min_rate_rows,clean_loss_budget)
            if adaptive_meta["fallback_to_universal"]:fallback_specs+=1
            adaptive_thresholds,adaptive_calibration=_thresholds(
              cal,d,direction,adaptive_spec,"rate_adaptive",i,label)
            calibration_summary+=adaptive_calibration
            adaptive_selected=(next((x for x in adaptive_diagnostics
              if x["specification"]==adaptive_spec),None))
            frozen["detectors"][d]["rate_adaptive_bins"][str(i)]={
              "contamination_bin":label,"lower_exclusive":lo,"upper_inclusive":upper,
              "clipping_specification":adaptive_spec,
              "selection_mean_attack_auroc_gain":None if adaptive_selected is None else adaptive_selected["mean_attack_auroc_gain"],
              "selection_clean_auroc_loss":None if adaptive_selected is None else adaptive_selected["clean_auroc_loss"],
              "selection_constraint_feasible":None if adaptive_selected is None else adaptive_selected["constraint_feasible"],
              "candidate_diagnostics":adaptive_diagnostics,
              "classification_thresholds":adaptive_thresholds,**adaptive_meta}
            adaptive_raw,adaptive_clipped=_cache(testh+test_none+mr,d,direction,adaptive_spec)
            for record in adaptive_calibration:
                domain_humans=[r for r in testh if domain(r)==record["domain"]]
                record["n_test_human"]=len(domain_humans)
                record["raw_test_fpr"]=actual_fpr(
                  [adaptive_raw[id(r)] for r in domain_humans],record["raw_threshold"])
                record["clipped_test_fpr"]=actual_fpr(
                  [adaptive_clipped[id(r)] for r in domain_humans],record["clipped_threshold"])
            rates=np.asarray([rho(r) for r in mr])
            z={"detector":d,"contamination_bin_index":i,"contamination_bin":label,"target_fpr":TARGET_FPR,
              "n_test_human":len(testh),"n_test_machine":len(mr),"n_test_sources":len({source(r) for r in mr}),
              "n_tuning_machine":len(tuning_machine),
              "rho_median":float(np.median(rates)),"rho_q1":float(np.quantile(rates,.25)),"rho_q3":float(np.quantile(rates,.75)),
              "attack_composition":json.dumps(Counter(condition(r) for r in mr),sort_keys=True),
              "generator_composition":json.dumps(Counter(str(_first(r,("generator","model"),"unknown")) for r in mr),sort_keys=True),
              "decoding_composition":json.dumps(Counter(str(_first(r,("decoding_method","decoding"),"unknown")) for r in mr),sort_keys=True),
              "repetition_penalty_composition":json.dumps(Counter(str(_first(r,("repetition_penalty",),"unknown")) for r in mr),sort_keys=True),
              "clipping_specification":json.dumps(spec,sort_keys=True),
              "rate_adaptive_clipping_specification":json.dumps(adaptive_spec,sort_keys=True),
              "rate_adaptive_fallback_to_universal":adaptive_meta["fallback_to_universal"],
              **_point(testh,mr,a,b,thresholds)}
            _add_ci(z,_bootstrap(testh,mr,a,b,thresholds,reps,_seed(bseed,d,"rho",i)))
            adaptive_point=_point(testh,mr,adaptive_raw,adaptive_clipped,adaptive_thresholds)
            adaptive_ci=_bootstrap(testh,mr,adaptive_raw,adaptive_clipped,adaptive_thresholds,
              reps,_seed(bseed,d,"rho_adaptive",i))
            _add_rate_adaptive_pair(z,adaptive_point,adaptive_ci)
            contamination_summary.append(z)

            tradeoff_groups=[("none","none_counterfactual",test_none),
              ("all_attacked_in_bin","attacked_bin_aggregate",mr)]
            tradeoff_groups.extend((name,"attack_bin",[r for r in mr if condition(r)==name])
              for name in attacks if any(condition(r)==name for r in mr))
            for comparison_condition,scope,comparison_rows in tradeoff_groups:
                tuning_count=(len(tuning_machine) if comparison_condition=="all_attacked_in_bin"
                  else len(tc) if comparison_condition=="none"
                  else int(adaptive_meta["attack_counts"].get(comparison_condition,0)))
                q={"detector":d,"contamination_bin_index":i,"contamination_bin":label,
                  "comparison_condition":comparison_condition,"comparison_scope":scope,
                  "target_fpr":TARGET_FPR,"n_test_human":len(testh),
                  "n_test_machine":len(comparison_rows),"n_tuning_machine":tuning_count,
                  "n_tuning_all_attacks_bin":len(tuning_machine),
                  "rate_adaptive_clipping_specification":json.dumps(adaptive_spec,sort_keys=True),
                  "rate_adaptive_fallback_to_universal":adaptive_meta["fallback_to_universal"],
                  "selection_mean_attack_auroc_gain":None if adaptive_selected is None else adaptive_selected["mean_attack_auroc_gain"],
                  "selection_clean_auroc_loss":None if adaptive_selected is None else adaptive_selected["clean_auroc_loss"],
                  **_point(testh,comparison_rows,adaptive_raw,adaptive_clipped,adaptive_thresholds)}
                _add_ci(q,_bootstrap(testh,comparison_rows,adaptive_raw,adaptive_clipped,
                  adaptive_thresholds,reps,_seed(bseed,d,"rate_tradeoff",i,comparison_condition)))
                rate_bound_tradeoff_summary.append(q)
    configured_published=cfg.get("published_binoculars_tpr",{})
    pub=dict(PUBLISHED_BINOCULARS_TPR)
    pub.update({condition({"label":"llm","attack":k}):float(v) for k,v in configured_published.items()})
    pub.update({condition({"label":"llm","attack":k}):float(v) for k,v in (published_binoculars or {}).items()})
    lookup={r["condition"]:r for r in attack_summary if r["detector"]=="binoculars"}
    sanity=[{"condition":c,"target_fpr":TARGET_FPR,"published_tpr":p,"reproduced_raw_tpr":lookup[c]["raw_tpr"],
      "reproduced_raw_tpr_ci_low":lookup[c].get("raw_tpr_ci_low"),"reproduced_raw_tpr_ci_high":lookup[c].get("raw_tpr_ci_high"),
      "difference_from_published":lookup[c]["raw_tpr"]-p,"reproduced_raw_held_out_fpr":lookup[c]["raw_test_fpr"]}
      for c,p in pub.items() if c in lookup]
    metrics=_tidy("attack",attack_summary)+_tidy("contamination_rate",contamination_summary)
    counts.update({"attack_summary_rows":len(attack_summary),"contamination_summary_rows":len(contamination_summary),
      "rate_bound_tradeoff_rows":len(rate_bound_tradeoff_summary),
      "metrics_rows":len(metrics),"calibration_rows":len(calibration_summary),"binoculars_sanity_rows":len(sanity),
      "bootstrap_repetitions":reps,"contamination_cutpoints":cuts,"universal_specs":7,
      "rate_adaptive_specs":len(DETECTORS)*len(RATE_ADAPTIVE_BIN_INDICES),
      "rate_adaptive_fallback_specs":fallback_specs,"attack_specific_specs":0,
      "contamination_record_rows":len(contamination_records)})
    return RaidEvaluationResult(frozen,metrics,attack_summary,contamination_summary,
      rate_bound_tradeoff_summary,cuts,sanity,calibration_summary,contamination_records,counts)

def _csv(path,rows):
    temporary=path.with_suffix(path.suffix+".tmp")
    if not rows:
        temporary.write_text("",encoding="utf-8");os.replace(temporary,path);return
    fields=list(rows[0]);fields+=sorted(set().union(*(set(r) for r in rows))-set(fields))
    with temporary.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temporary,path)
def write_evaluation_artifacts(result,directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    paths={n:p/f for n,f in {"frozen_specs":"frozen_specs.json","metrics":"metrics.csv","attack_summary":"attack_summary.csv",
      "contamination_summary":"contamination_summary.csv","binoculars_sanity":"binoculars_sanity.csv",
      "rate_bound_tradeoff_summary":"rate_bound_tradeoff_summary.csv",
      "calibration_summary":"calibration_summary.csv","contamination_records":"contamination_records.csv",
      "validation_counts":"evaluation_counts.json"}.items()}
    atomic_write_json(paths["frozen_specs"],result.frozen_specs)
    atomic_write_json(paths["validation_counts"],result.validation_counts)
    for n in ("metrics","attack_summary","contamination_summary","rate_bound_tradeoff_summary",
      "binoculars_sanity","calibration_summary","contamination_records"):_csv(paths[n],getattr(result,n))
    return paths
