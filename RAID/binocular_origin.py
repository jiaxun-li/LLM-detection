"""Additional output-only ratio scorer; never reinterprets a legacy gap pack.

Raw reductions follow ahans30/Binoculars binoculars/metrics.py (mean mode):
https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py
Model logits stay in the configured dtype. Numerator uses causal shifting;
denominator considers ALL input positions and masks input_id == pad_id (so
EOS is excluded too when the tokenizer uses EOS as its padding ID).
"""
from __future__ import annotations

import numpy as np

from llm_detection.detector_revision import ORIGIN_NLL, ORIGIN_DENOMINATOR, official_nll_mean
from RAID.raid_scoring import RAIDBinocularsScorer, _torch

SCHEMA = "raid-binocular-origin-official-v1"


def official_components(observer_logits, performer_logits, encoding, pad_token_id):
    torch = _torch()
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    ids = encoding["input_ids"]
    mask = encoding["attention_mask"][..., 1:].contiguous()
    nll = loss(performer_logits[..., :-1, :].contiguous().transpose(1, 2),
               ids[..., 1:].contiguous())
    a = ((nll * mask).sum(1) / mask.sum(1)).cpu().float().numpy()
    probabilities = torch.softmax(observer_logits, dim=-1).reshape(-1, observer_logits.shape[-1])
    ce = loss(performer_logits.reshape(-1, performer_logits.shape[-1]),
              probabilities).reshape(ids.shape)
    denominator_mask = (ids != pad_token_id).to(torch.uint8)
    if bool((mask.sum(1) == 0).any()) or bool((denominator_mask.sum(1) == 0).any()):
        raise ValueError("official Binoculars window has no usable positions")
    b = ((ce * denominator_mask).sum(1) / denominator_mask.sum(1)).cpu().float().numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError("invalid official Binoculars components")
    return a, b, nll


class RAIDBinocularsOriginScorer(RAIDBinocularsScorer):
    """Batch-one official protocol; optional independently loaded parity reference."""

    def __init__(self, config, *, reference_metrics=None, **kwargs):
        super().__init__(config, **kwargs)
        if self.max_tokens != 512:
            raise ValueError("binocular-origin RAID limit is fixed at 512")
        if self.requested_dtype not in {"bf16", "bfloat16"}:
            raise ValueError("official RAID origin protocol requires BF16")
        self.reference_metrics = reference_metrics
        self.parity_checks = 0
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def feature_schema(self):
        return SCHEMA

    def token_count(self, row):
        # Use exactly the official tokenizer window, not the parent's optional
        # one-token BOS repair. The full-input preflight rejects such records.
        encoded = self.tokenizer([row["text"]], padding=False, truncation=True,
                                 max_length=512, return_token_type_ids=False)
        ids = encoded["input_ids"][0]
        if len(ids) < 2 or not any(i != self.tokenizer.pad_token_id for i in ids):
            raise ValueError("unscorable official window; no silent BOS insertion")
        return len(ids)

    def validate_existing_row(self, row):
        for key, expected in {
            "scoring_feature_schema": SCHEMA,
            "scoring_max_tokens": 512,
            "scoring_context_policy": "output_only_official_binocular_origin",
            "binoculars_observer_revision": self.observer_resolved_revision,
            "binoculars_performer_revision": self.performer_resolved_revision,
            "binoculars_tokenizer_revision": self.resolved_tokenizer_revision,
            "binoculars_performer_tokenizer_revision": self.performer_resolved_tokenizer_revision,
            "binoculars_observer_dtype": self.observer_dtype,
            "binoculars_performer_dtype": self.performer_dtype,
        }.items():
            if row.get(key) != expected:
                raise ValueError(f"binocular-origin provenance mismatch: {key}")
        values = np.asarray(row["token_features"][ORIGIN_NLL])
        if (values.ndim != 1 or not len(values) or len(values) != row["num_scored_tokens"]
                or not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError("invalid binocular-origin token features")
        for key in ("binocular_origin", ORIGIN_DENOMINATOR):
            value=row["doc_scores"][key]
            if not np.isfinite(value) or value < 0 or (key==ORIGIN_DENOMINATOR and value==0):
                raise ValueError("invalid binocular-origin document score")
        numerator = official_nll_mean(values)
        ds = row["doc_scores"]
        if (numerator != ds["binocular_origin_numerator"] or
                float(np.float32(numerator)/np.float32(ds[ORIGIN_DENOMINATOR])) != ds["binocular_origin"]):
            raise ValueError("saved official numerator/ratio disagrees with BF16 replay")

    def score_batch(self, rows):
        torch = _torch()
        outputs = []
        # Official batch-one protocol avoids batch-dependent padding differences.
        for row in rows:
            encoded = self.tokenizer([row["text"]], return_tensors="pt", padding=False,
                truncation=True, max_length=512, return_token_type_ids=False)
            if encoded["input_ids"].shape[1] < 2:
                raise ValueError("unscorable official window; no silent BOS insertion")
            with torch.inference_mode():
                observed = encoded.to(self.observer_device)
                p = self.observer(**observed).logits.to(self.performer_device)
                performed = encoded.to(self.performer_device)
                q = self.performer(**performed).logits
                a, b, nll = official_components(p, q, performed, self.tokenizer.pad_token_id)
                score = float((a / b)[0])
                if self.reference_metrics is not None:
                    ref_a = self.reference_metrics.perplexity(performed, q)
                    ref_b = self.reference_metrics.entropy(p, q, performed, self.tokenizer.pad_token_id)
                    if not np.array_equal(a, ref_a) or not np.array_equal(b, ref_b):
                        raise ValueError("upstream Binoculars component parity failed")
                    saved_nll = nll[0].cpu().float().numpy()
                    # Check the analysis-time BF16 replay against actual Torch
                    # capped reductions, including a nonrepresentable bound.
                    for upper in (float(saved_nll.min()), float(np.quantile(saved_nll,.9)),
                                  float(saved_nll.max()) - 0.0001):
                        upper = max(0., upper)
                        capped = torch.minimum(nll, torch.tensor(upper,dtype=nll.dtype,device=nll.device))
                        actual = float((capped.sum(1)/capped.shape[1]).cpu().float()[0])
                        if official_nll_mean(saved_nll,upper) != actual:
                            raise ValueError("capped BF16 numerator replay failed")
                    self.parity_checks += 1
                output = {k: v for k, v in row.items() if k not in {"text", "prompt"}}
                output.update({
                    "scoring_feature_schema": SCHEMA,
                    "scoring_context_policy": "output_only_official_binocular_origin",
                    "scoring_max_tokens": 512,
                    "num_scored_tokens": int(nll.shape[1]),
                    "binoculars_observer_revision": self.observer_resolved_revision,
                    "binoculars_performer_revision": self.performer_resolved_revision,
                    "binoculars_tokenizer_revision": self.resolved_tokenizer_revision,
                    "binoculars_performer_tokenizer_revision": self.performer_resolved_tokenizer_revision,
                    "binoculars_observer_model": self.observer_id,
                    "binoculars_performer_model": self.performer_id,
                    "binoculars_observer_dtype": self.observer_dtype,
                    "binoculars_performer_dtype": self.performer_dtype,
                    "token_features": {ORIGIN_NLL: nll[0].cpu().float().tolist()},
                    "doc_scores": {"binocular_origin": score,
                        ORIGIN_DENOMINATOR: float(b[0]), "binocular_origin_numerator": float(a[0])},
                })
            self.validate_existing_row(output)
            outputs.append(output)
            del p, q, nll
        return outputs
