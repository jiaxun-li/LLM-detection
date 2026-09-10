# Local archive catalog

`archive/` is ignored by Git at the user's request. Its contents remain on this
computer and are not included by an ordinary commit, push, pull, or fresh clone.
This catalog is tracked so another checkout can understand the history without
requiring the archived programs. Versions committed before the cleanup remain
recoverable from Git history. Keep a separate backup for uncommitted/local-only
archive material if it is needed on another machine.

| Directory under `archive/` | Contents and purpose |
|---|---|
| `studies/beemo/` | Complete Beemo expert-edit benchmark, configuration, launchers and original study notes |
| `studies/splice_audit/` | Paired Granite–XSum human/machine donor and token/sentence replacement diagnostic |
| `raid_pilots/` | Exploratory clipping selectors, one-sided trimming and Binoculars component comparisons |
| `toy_robust_aggregation/` | Early toy aggregation experiment and its two plotters |
| `legacy_entrypoints/` | Three deprecated aliases for primary preparation, scoring and evaluation |
| `exporters/` | Previous primary-plot and primary/Beemo export formats |
| `diagnostics/` | Clean detector ordering and tokenizer round-trip audits |
| `launchers/` | Former monolithic RAID launcher |
| `notes/` | Historical RAID brainstorm |
| `tests/` | Tests specific to the archived programs |

Retained archived programs reference current `experiment_core/`, `RAID/`, and
`tools/` dependencies and are intended to run from the repository root;
individual runtime compatibility still requires checking. Historical result paths and run
IDs are preserved. Active programs and tests do not import `archive/`.

Intended compatibility does not mean bitwise historical reproduction: a retained launcher
using today's shared implementation may differ from the code recorded by an
old run. Match a historical result to its own commit, configuration, manifests
and source hashes before interpreting a rerun. Archive-local compatibility
tests cover those retained tools, not real Delta execution of every old study.
The final manuscript's primary evidence is the corrected nine-cell + RAID
bundle, not exploratory selector leaderboards or whichever archived output is
newest. Pilot sources used during method exploration were excluded from final
RAID selection; keep their exclusion list as provenance.

Active tests: `python -m unittest discover -s tests -v`.
Optional archive tests: `python -m unittest discover -s archive/tests -v`.

Downloaded artifacts are independently organized under `downloads/current/` and
`downloads/archive/`; the whole downloads tree is ignored. The old Beemo export
contains unique secondary-study results, so it is retained even though the final
nine-cell + RAID bundle supersedes the older primary and RAID exports.

This change does not synchronize or delete anything on Delta. Before applying
the Git changes to a different checkout, retain any historical code there that
is needed independently; ignored local archives are not delivered by Git.

Likewise, ignored `runs`, `results`, logs and machine-local backup directories
are not automatically disposable. Verify unique files and active job writers
before any cleanup. No deletion of scientific artifacts is part of this
documentation audit.
