# Experiment documentation

| Guide | What it answers |
|---|---|
| [Final results guide](FINAL_RESULTS_GUIDE.md) | Which accepted files support the report, and how should their tables be read? |
| [Documentation audit](DOCUMENTATION_AUDIT.md) | What was checked against code/results and what remains unverified? |
| [Repository map](REPOSITORY_MAP.md) | Which folder or program should I use? |
| [Configuration guide](CONFIGURATION.md) | What are JSON configurations and which one applies? |
| [Codebase guide](CODEBASE_GUIDE.md) | How do stages, outputs, restarts, and tests work? |
| [Primary scientific workflow](primary/SCIENTIFIC_WORKFLOW.md) | What is the controlled contamination protocol? |
| [Primary requirements](primary/REQUIREMENT_CHECKLIST.md) | What engineering/scientific requirements were recorded? |
| [Detector amendment](DETECTOR_REVISION.md) | What changed for Binoculars-origin, LRR, and clipping safeguards? |
| [Clipping method](methods/clipping_method.md) | How are clipping bounds selected and evaluated? |
| [Delta operations](delta/DELTA.md) | How is the cluster environment used? |
| [Repository layout check](delta/REPOSITORY_SMOKE.md) | How do I verify the reorganized code on Delta without rerunning experiments? |
| [Delta smoke test](delta/DELTA_SMOKE.md) | How is a bounded pipeline check run? |
| [RAID overview](raid/README.md) | What does the external benchmark package do? |
| [RAID scientific design](raid/SCIENTIFIC_DESIGN.md) | How are data, rates, tuning, calibration, and bootstrap defined? |
| [RAID Delta guide](raid/DELTA_GUIDE.md) | How is RAID launched, resumed, and validated? |
| [Archive catalog](ARCHIVE_CATALOG.md) | What historical work remains locally, outside Git synchronization? |

Shell commands in these guides assume the repository root as the working
directory, even though the guides themselves live under `docs/`.
`README.md` and `AGENTS.md` remain at the repository root for discovery.
User-owned `paper/` content and the allocation-writing checklist were not moved.
