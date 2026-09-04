"""No-download amendment tests; Torch parity tests run in the Delta environment."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from llm_detection.detector_revision import (REVISION, ORIGIN_NLL, ORIGIN_DENOMINATOR,
    constant_clean_scores, origin_score)
from llm_detection.evaluation import (REVISED_METHODS, _candidate_specs,
    detector_raw_score, orientation, oriented_score, tune_clipping_spec, evaluate)
from RAID.raid_evaluation import (learn_direction, merge_score_rows,
    select_universal_specification, select_rate_adaptive_specification, evaluate_raid)


def pair(nll=(1.,3.), ce=(2.,2.)):
    return {"token_features":{"performer_nll":list(nll),
        "observer_to_performer_cross_entropy":list(ce)},
        "doc_scores":{"binoculars":math.exp(np.mean(nll)-np.mean(ce)),"lrr":1.}}


class RevisionTests(unittest.TestCase):
    def test_revision_dtype_reaches_real_loader_without_torch_or_downloads(self):
        from RAID.revise_detectors import scorer_config
        from llm_detection.scoring import _load_model

        source_row={
            "binoculars_observer_revision":"a"*40,
            "binoculars_performer_revision":"b"*40,
            "binoculars_tokenizer_revision":"c"*40,
            "binoculars_performer_tokenizer_revision":"d"*40,
        }
        with patch("RAID.revise_detectors.iter_jsonl",return_value=iter([source_row])):
            config=scorer_config(Path("unused-source"))
        fake_torch=SimpleNamespace(bfloat16=object(),float16=object(),float32=object())
        tokenizer=SimpleNamespace(pad_token_id=0)
        model=Mock()
        model.parameters.return_value=iter([SimpleNamespace(device="cpu")])
        auto_model=Mock();auto_model.from_pretrained.return_value=model
        auto_tokenizer=Mock();auto_tokenizer.from_pretrained.return_value=tokenizer
        with patch("llm_detection.scoring._torch",return_value=fake_torch), \
             patch("llm_detection.scoring._transformers",return_value=(auto_model,auto_tokenizer)):
            _load_model("tiiuae/falcon-7b",config["observer_revision"],
                config["tokenizer_revision"],config["dtype"],"cpu",None,
                trust_remote_code=config["trust_remote_code"])
        self.assertEqual(config["dtype"],"bf16")
        self.assertIs(auto_model.from_pretrained.call_args.kwargs["torch_dtype"],fake_torch.bfloat16)
        self.assertEqual(auto_model.from_pretrained.call_args.kwargs["revision"],"a"*40)
        self.assertEqual(auto_tokenizer.from_pretrained.call_args.kwargs["revision"],"c"*40)

    def test_gap_retained_and_ratio_is_not_exponential_gap(self):
        row=pair((2.,4.),(2.,2.))
        self.assertEqual(detector_raw_score(row,"binocular_gap"),math.e)
        self.assertEqual(detector_raw_score(row,"binocular_origin"),1.5)
        self.assertEqual(oriented_score(row,"binocular_gap",-1,{"lower":-100}),-math.e)

    def test_conditional_origin_caps_only_numerator(self):
        row=pair((1.,9.),(2.,2.)); saved=copy.deepcopy(row)
        self.assertEqual(origin_score(row,{"nll_upper":3}),1.)
        self.assertEqual(oriented_score(row,"binocular_origin",-1,{"nll_upper":3}),-1.)
        self.assertEqual(row,saved)
        self.assertEqual(len(_candidate_specs("binocular_origin",-1,[row],[.8,.9])),3)

    def test_official_components_never_use_legacy_gap_arrays(self):
        row=pair((900.,900.),(800.,800.))
        row["token_features"][ORIGIN_NLL]=[1.,9.]
        row["doc_scores"].update({ORIGIN_DENOMINATOR:2.,"binocular_origin":2.5})
        self.assertEqual(origin_score(row),2.5)
        self.assertEqual(origin_score(row,{"nll_upper":3}),1.)
        row["doc_scores"]["binocular_origin"]=2.5001
        self.assertEqual(origin_score(row,{"nll_upper":100}),2.5001)

    def test_fixed_directions_and_legacy_mode(self):
        h=pair();m=pair();h["doc_scores"]["lrr"]=3
        self.assertEqual(orientation([h],[m],"lrr"),-1)
        self.assertEqual(orientation([h],[m],"lrr",revised=True),1)
        self.assertEqual(learn_direction("lrr",[h],[m],revised=True),1)
        self.assertEqual(orientation([h],[m],"binocular_origin"),-1)

    def test_constant_is_exact_not_nearly_constant(self):
        self.assertTrue(constant_clean_scores([1.],[1.]))
        self.assertFalse(constant_clean_scores([1.],[np.nextafter(1.,2.)]))
        with self.assertRaises(ValueError):constant_clean_scores([float("nan")],[1.])

    def test_primary_rejects_constant_nonempty_candidate_only(self):
        h={"token_features":{"rank":[2.,3.]},"doc_scores":{"rank":2.5}}
        m={"token_features":{"rank":[1.,2.]},"doc_scores":{"rank":1.5}}
        diagnostics=[]
        with patch("llm_detection.evaluation._candidate_specs",return_value=[{}, {"lower":0.}]):
            selected=tune_clipping_spec("rank",-1,[h],[m],[m],[.8],True,diagnostics)
        self.assertEqual(selected,{})
        self.assertEqual(diagnostics[0]["reason"],"constant_clean_tuning_scores")

    def test_raid_universal_and_rate_constant_guard(self):
        h={"token_features":{"rank":[2.,3.]},"doc_scores":{"rank":2.5}}
        m={"label":"machine","attack":"synonym","token_features":{"rank":[1.,2.]},"doc_scores":{"rank":1.5}}
        with patch("RAID.raid_evaluation.candidate_specifications",return_value=[{}, {"lower":0.}]):
            spec,diagnostics=select_universal_specification("rank",-1,[h],[m],{"synonym":[m]},
                require_all_attacks=False,reject_constant=True)
            rate,rd,_=select_rate_adaptive_specification("rank",-1,[h],[m],[m],{},1,reject_constant=True)
        self.assertEqual((spec,rate),({},{}))
        self.assertEqual(diagnostics[1]["reason"],rd[1]["reason"])

    def test_merge_pure_and_duplicate_rejected(self):
        first={"row_key":"a",**pair()};second=copy.deepcopy(first)
        second["doc_scores"]["binocular_origin"]=1.
        merged=merge_score_rows([first],[second])
        self.assertNotIn("binocular_origin",first["doc_scores"])
        self.assertIn("binocular_origin",merged[0]["doc_scores"])
        with self.assertRaises(ValueError):merge_score_rows([first],[second,second])

    def test_raid_eight_detectors_and_cached_rate_parity(self):
        from tests.test_raid_evaluation import protocol_rows
        rows=protocol_rows()
        for r in rows:
            nll=r["token_features"]["performer_nll"]
            r["token_features"][ORIGIN_NLL]=list(nll)
            r["doc_scores"]={ORIGIN_DENOMINATOR:1.,"binocular_origin":float(np.mean(nll))}
        config={"detector_revision":REVISION,"bootstrap_repetitions":0,"rate_adaptive_min_tuning_rows":1}
        result=evaluate_raid(rows,config)
        self.assertEqual(result.validation_counts["detectors"],REVISED_METHODS)
        self.assertEqual(result.validation_counts["universal_specs"],16)
        self.assertEqual(result.validation_counts["rate_adaptive_specs"],32)
        self.assertEqual(len(result.binoculars_sanity),7)
        cached=evaluate_raid(rows,config,contamination_records=result.contamination_records)
        self.assertEqual(result.frozen_specs,cached.frozen_specs)
        del rows[0]["token_features"][ORIGIN_NLL]
        with self.assertRaises(ValueError):evaluate_raid(rows,config)

    def test_primary_eight_detectors_end_to_end(self):
        from llm_detection.config import load_config,resolved_run_config
        from llm_detection.io import append_jsonl
        from tests.test_evaluation import protocol_rows,binoculars_row
        config=resolved_run_config(load_config("configs/smoke.json"),"xsum","test-model","revision-test")
        config["contamination"]["ratios"]=[0.,.1,.5]
        config["evaluation"].update(detector_revision=REVISION,bootstrap_repetitions=2,
            tuning_mixture={"modes":["random","tail"],"ratios":[.1,.5]})
        config["scoring"]["detectors"]=REVISED_METHODS
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for row in protocol_rows(config):
                append_jsonl(root/"target.jsonl",row)
                append_jsonl(root/"pair.jsonl",binoculars_row(row))
            results=evaluate(root/"target.jsonl",root/"metrics.csv",config,root/"pair.jsonl",None)
            self.assertEqual({r["detector"] for r in results},set(REVISED_METHODS))
            self.assertTrue(all(r["direction"]==1 for r in results if r["detector"]=="lrr"))
            self.assertTrue((root/"clipping_rejections.json").exists())

    def test_primary_runner_preserves_sources_and_reuses_completed_result(self):
        from llm_detection.config import load_config,resolved_run_config
        from llm_detection.io import append_jsonl,atomic_write_json
        from tests.test_evaluation import protocol_rows,binoculars_row
        from scripts.reevaluate_detector_revision import main
        config=resolved_run_config(load_config("configs/smoke.json"),"xsum","test-model","source")
        config["contamination"]["ratios"]=[0.,.1,.5]
        config["evaluation"].update(bootstrap_repetitions=2,
            tuning_mixture={"modes":["random","tail"],"ratios":[.1,.5]})
        manifest={**config,"completion_status":"complete","dataset":{"name":"xsum","id":"test"}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"runs/source";source.mkdir(parents=True)
            atomic_write_json(source/"manifest.json",manifest)
            for row in protocol_rows(config):
                append_jsonl(source/"target_scores.jsonl",row)
                append_jsonl(source/"binoculars_scores.jsonl",binoculars_row(row))
            before={p.name:p.read_bytes() for p in source.iterdir()}
            argv=["revision","--workspace",str(root),"--source-run-id","source","--revision-id","new"]
            with patch("sys.argv",argv),patch("plot_tpr_contamination.plot"):
                main()
            result=root/"results/new/metrics.csv";old_result=result.read_bytes()
            with patch("sys.argv",argv),patch("scripts.reevaluate_detector_revision.evaluate") as calculate:
                main();calculate.assert_not_called()
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            self.assertEqual(old_result,result.read_bytes())


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"Torch unavailable; run these tests on Delta before the gate")
class OfficialComponentTests(unittest.TestCase):
    def test_final_position_and_eos_mask(self):
        from RAID.binocular_origin import official_components
        p=torch.tensor([[[1.,2.,3.],[3.,2.,1.],[1.,4.,2.]]])
        q=torch.tensor([[[2.,1.,3.],[4.,1.,2.],[1.,2.,5.]]])
        ids=torch.tensor([[0,1,2]])
        enc={"input_ids":ids,"attention_mask":torch.ones_like(ids)}
        a,b,nll=official_components(p,q,enc,2)
        expected_nll=-torch.log_softmax(q[0,:2],-1)[torch.arange(2),ids[0,1:]]
        expected_ce=-(torch.softmax(p,-1)*torch.log_softmax(q,-1)).sum(-1)
        np.testing.assert_allclose(a,[expected_nll.mean().item()],rtol=1e-6)
        np.testing.assert_allclose(b,[expected_ce[0,:2].mean().item()],rtol=1e-6)
        _,b_all,_=official_components(p,q,enc,99)
        np.testing.assert_allclose(b_all,[expected_ce.mean().item()],rtol=1e-6)
        self.assertNotEqual(float(b[0]),float(b_all[0]))


if __name__=="__main__":unittest.main()
