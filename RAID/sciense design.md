> Historical brainstorm. The agreed and implemented protocol is
> [`SCIENTIFIC_DESIGN.md`](SCIENTIFIC_DESIGN.md).

I suggest the design to be like this: 

1. clean the dataset such that we keep one human document for each, and randomly selected one of its machine generations with its 12 conditions.
2. we compute the contamination rate for all the conditions using our proposed method, and then, deciding after we look at the data, how many groups should we split the contamination rates. the final decision should be: contamination until c1, contamination c1 until c2,... For each of these groups, we train a clipping threshold. In addition, we train one clipping threshold using all contamination groups.
3. train a clipping threshold within the tuning set for each group, and then use the calibration group for calibrating it to 5% FPR. Since we only need 5%, not 1%, also because we need to split the tunning set, I suggest the final split be to 40/20/40.
4. for all seven detectors: Log likelihood,Rank,Log rank,LRR,Entropy,Entropy gap, Binoculars, compare: Raw and different contamination-rate-adaptive clipped,universal clipped. report  TPR at 5% by contamination-rate for each different contamination-rate-adaptive clipped vs raw detectors, then report the TPR at 5% by each RAID attack for each universal clipped and raw detectors. we can sanity check if the binocular, raw detector being similar as theirs in this table.
5. Bootstrap confidence interval.
