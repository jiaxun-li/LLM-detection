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
    constant_clean_scores, origin_score, official_nll_mean, round_bfloat16,
    structurally_constant_candidate)
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
        self.assertFalse(config["trust_remote_code"])
        self.assertFalse(auto_model.from_pretrained.call_args.kwargs["trust_remote_code"])
        self.assertFalse(auto_tokenizer.from_pretrained.call_args.kwargs["trust_remote_code"])
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

    def test_bf16_active_cap_cannot_increase_ratio(self):
        row={"token_features":{ORIGIN_NLL:[1.,1.0078125]},
             "doc_scores":{ORIGIN_DENOMINATOR:1.,"binocular_origin":1.}}
        self.assertEqual(official_nll_mean([1.,1.0078125]),1.)
        self.assertLessEqual(origin_score(row,{"nll_upper":1.006}),origin_score(row))
        rng=np.random.default_rng(9)
        for n in (2,3,50,511):
            values=round_bfloat16(rng.uniform(0.,30.,n)).astype(float)
            raw=official_nll_mean(values)
            row["token_features"][ORIGIN_NLL]=values
            row["doc_scores"]["binocular_origin"]=raw
            scores=[origin_score(row,{"nll_upper":float(u)}) for u in np.linspace(0,31,50)]
            self.assertTrue(all(a<=b for a,b in zip(scores,scores[1:])))
            self.assertLessEqual(max(scores),raw)
        self.assertEqual(float(round_bfloat16(1.00390625)),1.)
        self.assertEqual(float(round_bfloat16(1.01171875)),1.015625)

    def test_reported_a100_tensor_division_regression(self):
        # Measured on Delta: torch 2.11.0+cu128, A100-SXM4-40GB.
        # Equal BF16 sums/counts imply equal tensor-division results regardless
        # of how the total is distributed across the token array.
        cases = ((128,776.,6.0625),(255,1544.,6.0625),(256,1552.,6.0625),
                 (257,256.,1.),(257,1552.,6.0625),(257,2704.,10.5625),
                 (259,1568.,6.03125),(341,340.,1.),(511,5344.,10.4375))
        for n,total,expected in cases:
            with self.subTest(n=n,total=total):
                values=np.zeros(n);values[0]=total
                self.assertEqual(official_nll_mean(values),expected)

    def test_fully_saturated_variable_lengths_are_rejected_without_tolerance(self):
        rows=[{"token_features":{"rank":[1000.]*n},"doc_scores":{"rank":1000.}}
              for n in (40,50)]
        spec={"lower":-626.13}
        scores=[oriented_score(r,"rank",-1,spec) for r in rows]
        self.assertNotEqual(*scores)  # The original exact-score guard missed this.
        self.assertTrue(structurally_constant_candidate(rows,"rank",-1,spec))
        self.assertFalse(structurally_constant_candidate(rows,"rank",-1,{}))
        diagnostics=[]
        with patch("llm_detection.evaluation._candidate_specs",return_value=[{},spec]):
            tune_clipping_spec("rank",-1,rows[:1],rows[1:],rows[1:],[.8],True,diagnostics)
        self.assertEqual(diagnostics[0]["reason"],"constant_clean_tuning_scores")
        with patch("RAID.raid_evaluation.candidate_specifications",return_value=[{},spec]):
            _,diag=select_universal_specification("rank",-1,rows[:1],rows[1:],
                {"synonym":rows[1:]},require_all_attacks=False,reject_constant=True)
        self.assertFalse(diag[1]["eligible"])
        rows[0]["token_features"]["rank"][0]=1.
        self.assertFalse(structurally_constant_candidate(rows,"rank",-1,spec))

    def test_saturated_lrr_and_variable_origin_denominator(self):
        rows=[{"token_features":{"logp":[-10.]*n,"log_rank":[8.]*n}} for n in (40,50)]
        self.assertTrue(structurally_constant_candidate(rows,"lrr",1,
            {"nll_upper":3.1,"log_rank_upper":4.2}))
        self.assertFalse(structurally_constant_candidate(rows,"binocular_origin",-1,{"nll_upper":1.}))

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
            def fake_plot(*args):
                Path(args[2]).write_bytes(b"test plot")
            with patch("sys.argv",argv),patch("plot_tpr_contamination.plot",side_effect=fake_plot):
                main()
            result=root/"results/new/metrics.csv";old_result=result.read_bytes()
            with patch("sys.argv",argv),patch("scripts.reevaluate_detector_revision.evaluate") as calculate:
                main();calculate.assert_not_called()
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            self.assertEqual(old_result,result.read_bytes())
            # Recover a lost final marker without repeating scientific analysis.
            (root/"results/new/revision.complete.json").unlink()
            with patch("sys.argv",argv),patch("scripts.reevaluate_detector_revision.evaluate") as calculate:
                main();calculate.assert_not_called()
            result.unlink()
            with patch("sys.argv",argv),self.assertRaisesRegex(ValueError,"artifact validation"):
                main()

    def test_completion_recovery_and_artifact_integrity(self):
        import json
        from llm_detection.revision_artifacts import finish_revision,resume_completed_revision
        from llm_detection.io import atomic_write_json
        for state in ("running","complete"):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/"metrics.csv").write_text("a,b\n1,2\n")
                manifest={"source":"immutable", "completion_status":"running"}
                finish_revision(root,manifest,{"validation_status":"pass"},["metrics.csv"])
                (root/"revision.complete.json").unlink()
                manifest["completion_status"]=state
                atomic_write_json(root/"revision_manifest.json",manifest)
                self.assertTrue(resume_completed_revision(root,manifest,["metrics.csv"]))
                self.assertTrue((root/"revision.complete.json").exists())
                manifest=json.loads((root/"revision_manifest.json").read_text())
                (root/"metrics.csv").write_text("corrupted")
                with self.assertRaisesRegex(ValueError,"artifact validation"):
                    resume_completed_revision(root,manifest,["metrics.csv"])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(resume_completed_revision(tmp,{"completion_status":"running"},["metrics.csv"]))

    def test_full_tokenizer_preflight_and_real_pipeline_probe(self):
        import json
        from RAID.revision_gate import preflight_inputs,run_pipeline_probe
        from llm_detection.io import append_jsonl
        from tests.test_raid_evaluation import protocol_rows
        class Tokenizer:
            pad_token_id=0
            def __call__(self,texts,**kwargs):
                return {"input_ids":[[1] if t=="." else [1,2,3] for t in texts]}
        class Scorer:
            parity_checks=0
            def token_count(self,row):return 3
            def validate_existing_row(self,row):
                if ORIGIN_NLL not in row["token_features"]:raise ValueError("missing origin")
            def score_batch(self,rows):
                self.parity_checks+=len(rows)
                return [{**r,"num_scored_tokens":2,
                    "token_features":{ORIGIN_NLL:[1.,3.]},
                    "doc_scores":{"binocular_origin":1.,ORIGIN_DENOMINATOR:2.}}
                    for r in rows]
        rows=protocol_rows()
        for r in rows:
            r.update(text="a complete sentence",base_generation_id=r["source_id"])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data.jsonl";shard=root/"shard.jsonl"
            original=evaluate_raid(rows,{"bootstrap_repetitions":0})
            from RAID.raid_evaluation import write_evaluation_artifacts
            cache=write_evaluation_artifacts(original,root/"old")['contamination_records']
            for r in rows:
                for name in (data,shard,root/"falcon_scores.jsonl",root/"binoculars_scores.jsonl"):
                    append_jsonl(name,r)
            report,selected=preflight_inputs(data,[shard],Tokenizer(),root/"preflight.json")
            self.assertEqual(report["rows"],len(rows))
            self.assertEqual(report["probe_sources"],6)
            scorer=Scorer()
            def fake_plots(path):
                path=Path(path)/"probe.png";path.write_bytes(b"plot");return [path]
            with patch("RAID.raid_plot.plot_raid",side_effect=fake_plots),patch("RAID.raid_scoring.Throughput"):
                result=run_pipeline_probe(selected,scorer,root,cache,root,{"evaluation":{}})
            self.assertTrue(result["resume_unchanged"])
            self.assertEqual(scorer.parity_checks,len(selected))
            self.assertEqual(result["counts"]["detectors"],REVISED_METHODS)
            # A bad text OUTSIDE the chosen probes must still fail preflight.
            data.unlink();shard.unlink()
            rows[-1]["text"]="."
            for r in rows:
                append_jsonl(data,r);append_jsonl(shard,r)
            with self.assertRaisesRegex(ValueError,"unscorable"):
                preflight_inputs(data,[shard],Tokenizer(),root/"preflight.json")
            self.assertEqual(json.loads((root/"preflight.json").read_text())["unscorable_rows"],1)
            with self.assertRaisesRegex(ValueError,"duplicate/content mismatch"):
                preflight_inputs(data,[shard,shard],Tokenizer(),root/"preflight.json")

    def test_raid_revision_runner_gate_score_evaluate_and_recover(self):
        import json
        from RAID.revise_detectors import main
        from llm_detection.io import append_jsonl,atomic_write_json
        from tests.test_raid_evaluation import protocol_rows
        from llm_detection.revision_artifacts import RAID_ARTIFACTS
        from RAID.raid_evaluation import write_evaluation_artifacts
        class Tokenizer:
            pad_token_id=0
            def __call__(self,texts,**kwargs):return {"input_ids":[[1,2,3] for _ in texts]}
        class Scorer:
            def __init__(self,*args,**kwargs):self.parity_checks=0
            def token_count(self,row):return 3
            def validate_existing_row(self,row):
                if ORIGIN_NLL not in row['token_features']:raise ValueError('missing origin')
            def score_batch(self,rows):
                self.parity_checks+=len(rows)
                return [{**r,"num_scored_tokens":2,
                    "token_features":{ORIGIN_NLL:[1.,3.]},
                    "doc_scores":{"binocular_origin":1.,ORIGIN_DENOMINATOR:2.}}
                    for r in rows]
        def fake_plots(directory):
            paths=[]
            for name in RAID_ARTIFACTS:
                if name.endswith('.png'):
                    path=Path(directory)/name;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'test plot');paths.append(path)
            return paths
        rows=protocol_rows()
        for row in rows:
            row.update(text='a complete sentence',base_generation_id=row['source_id'],
                binoculars_observer_revision='a'*40,binoculars_performer_revision='b'*40,
                binoculars_tokenizer_revision='c'*40,binoculars_performer_tokenizer_revision='d'*40)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'runs/raid/source';src.mkdir(parents=True)
            atomic_write_json(src/'manifest.json',{'completion_status':'complete','protocol_config':{'evaluation':{}}})
            cache=write_evaluation_artifacts(evaluate_raid(rows,{'bootstrap_repetitions':0}),root/'results/raid/source')
            for row in rows:
                for name in ('data.jsonl','data_shards/part-00000-of-00001.jsonl','falcon_scores.jsonl','binoculars_scores.jsonl'):
                    append_jsonl(src/name,row)
            reference=root/'reference.py';reference.write_text('# mocked model/reference boundary\n')
            base=['revision','--workspace',str(root),'--source-run-id','source','--revision-id','new',
                  '--num-shards','1','--bootstrap-repetitions','2']
            def run(stage):
                with patch('sys.argv',base+['--stage',stage,'--reference-metrics',str(reference)]):main()
            before={str(p.relative_to(src)):p.read_bytes() for p in src.rglob('*') if p.is_file()}
            with patch('RAID.revision_gate.load_gate_tokenizer',return_value=Tokenizer()), \
                 patch('RAID.binocular_origin.RAIDBinocularsOriginScorer',Scorer), \
                 patch('RAID.raid_plot.plot_raid',side_effect=fake_plots),patch('RAID.raid_scoring.Throughput'):
                run('gate');run('score');run('evaluate')
                output=root/'results/raid/new'
                report=json.loads((output/'validation_report.json').read_text())
                self.assertEqual(report['detectors'],REVISED_METHODS)
                (output/'revision.complete.json').unlink()
                with patch('RAID.raid_evaluation.evaluate_raid') as evaluate_mock:
                    run('evaluate');evaluate_mock.assert_not_called()
                (output/'metrics.csv').unlink()
                with self.assertRaisesRegex(ValueError,'artifact validation'):run('evaluate')
            self.assertEqual(before,{str(p.relative_to(src)):p.read_bytes() for p in src.rglob('*') if p.is_file()})


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"Torch unavailable; run these tests on Delta before the gate")
class OfficialComponentTests(unittest.TestCase):
    def test_bf16_replay_with_explicit_bf16_tensor_divisor_on_cpu(self):
        generator=torch.Generator().manual_seed(11)
        for n in (2,3,50,220,255,256,257,259,341,511):
            values=(torch.rand(1,n,generator=generator)*30).to(torch.bfloat16)
            for upper in (0.,1.006,3.17,8.1234,100.):
                capped=torch.minimum(values,torch.tensor(upper,dtype=torch.bfloat16))
                # CPU does not implicitly round the int64 tensor count like
                # CUDA does: cast explicitly here; the real CUDA test below
                # uses the untouched int64 mask and the official expression.
                divisor=torch.ones_like(values,dtype=torch.int64).sum(1).to(torch.bfloat16)
                expected=float((capped.sum(1)/divisor).float()[0])
                self.assertEqual(official_nll_mean(values[0].float().numpy(),upper),expected)

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


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),"CUDA required; no model downloads")
class CudaReplayTests(unittest.TestCase):
    def test_official_masked_tensor_reduction_raw_and_capped(self):
        generator=torch.Generator().manual_seed(17)
        for n in (2,128,255,256,257,259,341,511):
            patterns=(torch.ones(1,n),torch.linspace(.1,12,n).reshape(1,n),
                      torch.rand(1,n,generator=generator)*20)
            for pattern in patterns:
                values=pattern.to(dtype=torch.bfloat16,device="cuda")
                mask=torch.ones_like(values,dtype=torch.int64)
                saved=values[0].cpu().float().numpy()
                for upper in (None,0.,1.006,3.17,8.1234,100.):
                    capped=values if upper is None else torch.minimum(
                        values,torch.tensor(upper,dtype=values.dtype,device=values.device))
                    expected=float(((capped*mask).sum(1)/mask.sum(1)).cpu().float()[0])
                    with self.subTest(n=n,upper=upper):
                        self.assertEqual(official_nll_mean(saved,upper),expected)

    def test_replay_from_real_bf16_cross_entropy_kernel(self):
        from RAID.binocular_origin import official_components
        generator=torch.Generator().manual_seed(51)
        for n in (256,257,259,341,511):
            logits=torch.randn(1,n+1,32,generator=generator).to(dtype=torch.bfloat16,device="cuda")
            ids=torch.randint(0,31,(1,n+1),generator=generator).cuda()
            enc={"input_ids":ids,"attention_mask":torch.ones_like(ids)}
            actual,_,nll=official_components(logits,logits,enc,31)
            self.assertEqual(official_nll_mean(nll[0].cpu().float().numpy()),float(actual[0]))


if __name__=="__main__":unittest.main()
