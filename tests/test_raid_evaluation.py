import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from RAID.raid_evaluation import (
    DETECTORS,
    QUANTILE_GRID,
    RATE_ADAPTIVE_CUTPOINTS,
    candidate_specifications,
    attach_contamination_rates,
    evaluate_raid,
    merge_score_rows,
    oriented_document_score,
    select_universal_specification,
    token_edit_counts,
    write_evaluation_artifacts,
)

ATTACKS = [
    "alternative_spelling", "article_deletion", "homoglyph", "insert_paragraphs",
    "number", "paraphrase", "perplexity_misspelling", "synonym", "upper_lower",
    "whitespace", "zero_width_space",
]

def row(source, split, label, attack="none", shift=0.0, token_ids=(1,2,3)):
    # Machine evidence is larger for logp/entropy/gap/Binoculars and smaller
    # for ranks. Saved document scores are intentionally omitted so formulas
    # are exercised from the compact token arrays.
    is_machine = label == "machine"
    z = shift + (1.0 if is_machine else 0.0)
    logp = np.array([-3.0+z, -2.5+z, -2.0+z])
    rank = np.array([8.0-z*2, 7.0-z*2, 6.0-z*2])
    entropy = np.array([1.0+z*.2, 1.1+z*.2, 1.2+z*.2])
    performer = np.array([1.5+z*.1, 1.4+z*.1, 1.3+z*.1])
    cross = np.array([1.0, 1.0, 1.0])
    return {
        "row_key": f"{source}:{label}:{attack}",
        "source_id": source, "split": split,
        "domain": "news" if int(source.split("-")[-1]) % 2 else "books",
        "label": label, "attack": attack, "generator": "gpt2",
        "realized_contamination_rate": (
            None if label == "human" or attack != "none" else 0.0
        ),
        "token_features": {
            "logp": logp.tolist(), "rank": rank.tolist(),
            "log_rank": np.log(rank).tolist(), "entropy": entropy.tolist(),
            "performer_nll": performer.tolist(),
            "observer_to_performer_cross_entropy": cross.tolist(),
            "scored_token_ids": list(token_ids),
        },
    }

def protocol_rows():
    rows=[]
    for split, indices in (("clipping_tuning",range(0,4)),("calibration",range(4,8)),("test",range(8,12))):
        for i in indices:
            sid=f"s-{i}"
            base_ids=tuple(range(100))
            rows.append(row(sid,split,"human",token_ids=tuple(range(100,200))))
            rows.append(row(sid,split,"machine",token_ids=base_ids))
            changed_counts=(4,4,8,8,15,15,30,30,4,8,70)
            for j,attack in enumerate(ATTACKS):
                # Populate all four fixed oracle bins.  The final attack is
                # deliberately above .50 and must be excluded from that
                # analysis while remaining in universal attack evaluation.
                changed=changed_counts[j]
                ids=[1000+j]*changed+list(base_ids[changed:])
                rows.append(row(sid,split,"machine",attack,shift=-j*.015,token_ids=ids))
    return rows

class RaidEvaluationTests(unittest.TestCase):
    def test_edit_counts_replacement_one_token_by_two(self):
        got=token_edit_counts([1,2],[1,3,4])
        self.assertEqual((got["substitutions"],got["insertions"],got["deletions"]),(1,1,0))
        self.assertEqual(got["edit_distance"],2)
        self.assertAlmostEqual(got["contamination_rate"],2/3)

    def test_contamination_audits_edits_hidden_by_scoring_truncation(self):
        base=row("s-0","test","machine","none",token_ids=(1,2))
        attacked=row("s-0","test","machine","synonym",token_ids=(1,2))
        base["token_features"]["full_scored_token_ids"]=[1,2,3]
        attacked["token_features"]["full_scored_token_ids"]=[1,2,4]
        attach_contamination_rates([base,attacked])
        self.assertEqual(attacked["realized_contamination_rate"],0.0)
        self.assertTrue(attacked["truncation_hid_edits"])
        self.assertTrue(attacked["all_full_text_edits_hidden"])

    def test_detector_specific_candidate_and_aggregation_contract(self):
        clean=[row("s-0","clipping_tuning","human"),row("s-1","clipping_tuning","machine")]
        self.assertEqual(len(candidate_specifications("log_likelihood",1,clean)),1+len(QUANTILE_GRID))
        self.assertEqual(len(candidate_specifications("lrr",1,clean)),1+len(QUANTILE_GRID)**2)
        r=clean[1]
        lrr=oriented_document_score(r,"lrr",1,{"nll_upper":2.0,"log_rank_upper":1.5})
        expected=np.minimum(-np.asarray(r["token_features"]["logp"]),2).mean()/(np.minimum(np.asarray(r["token_features"]["log_rank"]),1.5).mean()+1e-12)
        self.assertAlmostEqual(lrr,expected)
        gap=np.asarray(r["token_features"]["performer_nll"])-np.asarray(r["token_features"]["observer_to_performer_cross_entropy"])
        lower=.35
        expected_b=np.exp(np.maximum(gap,lower).mean())
        self.assertAlmostEqual(oriented_document_score(r,"binoculars",1,{"lower":lower}),expected_b)

    def test_no_clipping_wins_exact_ties_and_attacks_are_equal_weighted(self):
        h=[row(f"s-{i}","clipping_tuning","human") for i in range(2)]
        m=[row(f"s-{i+2}","clipping_tuning","human") for i in range(2)]
        attacks={a:list(m) for a in ATTACKS}
        spec,diagnostics=select_universal_specification("log_likelihood",1,h,m,attacks)
        self.assertEqual(spec,{})
        self.assertEqual(len(diagnostics),8)
        self.assertTrue(all(len(x["attack_aurocs"])==11 for x in diagnostics))

    def test_pack_merge_uses_stable_keys_not_order(self):
        full=protocol_rows()[:3]
        falcon=[];bino=[]
        for r in full:
            f={**r,"token_features":{k:v for k,v in r["token_features"].items() if k not in ("performer_nll","observer_to_performer_cross_entropy")}}
            b={**r,"token_features":{k:v for k,v in r["token_features"].items() if k in ("performer_nll","observer_to_performer_cross_entropy")}}
            falcon.append(f);bino.append(b)
        merged=merge_score_rows(falcon,list(reversed(bino)))
        self.assertEqual(len(merged),3)
        self.assertIn("performer_nll",merged[0]["token_features"])

    def test_full_protocol_is_deterministic_and_writes_all_artifacts(self):
        cfg={"bootstrap_repetitions":5,"bootstrap_seed":71,
             "rate_adaptive_cutpoints":list(RATE_ADAPTIVE_CUTPOINTS),
             "rate_adaptive_min_tuning_rows":1,"expected_attacks":ATTACKS}
        first=evaluate_raid(protocol_rows(),cfg)
        second=evaluate_raid(protocol_rows(),cfg)
        self.assertEqual(first.metrics,second.metrics)
        self.assertEqual(set(first.frozen_specs["detectors"]),set(DETECTORS))
        self.assertTrue(first.frozen_specs["rate_adaptive_clipping"])
        self.assertEqual(first.validation_counts["rate_adaptive_specs"],28)
        self.assertEqual(first.validation_counts["attack_specific_specs"],0)
        self.assertEqual(len(first.attack_summary),7*12)
        self.assertEqual({r["target_fpr"] for r in first.metrics},{.05})
        self.assertEqual({r["aggregation"] for r in first.metrics},{"raw","clipped","rate_adaptive_clipped"})
        self.assertTrue(all("paired_tpr_difference_ci_low" in r for r in first.attack_summary))
        self.assertEqual(len(first.binoculars_sanity),7)
        self.assertEqual(first.contamination_cutpoints,list(RATE_ADAPTIVE_CUTPOINTS))
        self.assertEqual(len(first.contamination_summary),7*4)
        self.assertEqual(
            {r["contamination_bin_index"] for r in first.contamination_summary},
            {1,2,3,4},
        )
        self.assertTrue(all("rate_adaptive_clipped_tpr" in r for r in first.contamination_summary))
        # Every detector retains exactly one universal specification.
        for detector in DETECTORS:
            specs={r["clipping_specification"] for r in first.attack_summary if r["detector"]==detector}
            specs|={r["clipping_specification"] for r in first.contamination_summary if r["detector"]==detector}
            self.assertEqual(len(specs),1)
        with tempfile.TemporaryDirectory() as tmp:
            paths=write_evaluation_artifacts(first,tmp)
            self.assertEqual(set(paths),{"frozen_specs","metrics","attack_summary","contamination_summary","binoculars_sanity","calibration_summary","contamination_records","validation_counts"})
            self.assertTrue(all(p.exists() for p in paths.values()))
            frozen=json.loads(Path(paths["frozen_specs"]).read_text())
            self.assertEqual(frozen["target_fpr"],.05)

    def test_rejects_source_leakage_and_non_five_percent_protocol(self):
        rows=protocol_rows()
        test_human=next(r for r in rows if r["split"]=="test" and r["label"]=="human")
        test_human["source_id"]="s-0"
        with self.assertRaisesRegex(ValueError,"leakage"):
            evaluate_raid(rows,{"bootstrap_repetitions":0,"expected_attacks":ATTACKS})
        with self.assertRaisesRegex(ValueError,"5%"):
            evaluate_raid(protocol_rows(),{"target_fpr":.01})
        with self.assertRaisesRegex(ValueError,"frozen"):
            evaluate_raid(protocol_rows(),{"rate_adaptive_cutpoints":[.1,.2,.3,.4]})

if __name__=="__main__":
    unittest.main()
