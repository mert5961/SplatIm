# Record completeness audit (2026-09-14, branch refactor/c-plan-rgba-30k)

Every training experiment has: unique job dir, report.json, alignment.json,
scene.ply, brush log. Verified by direct listing this turn.

| Job | report | align | ply | log | dataset | note |
|-----|--------|-------|-----|-----|---------|------|
| B0 clean-3fbc0a | y | y | y | n/a (legacy) | y | baseline |
| B1 b1-30k | y | y | y | y | reuses B0 | duration isolation |
| B2 b2-rgba30k | y | y | y | y | y (120 RGBA+sparse) | alpha supervision |
| B3 b3-filt30k | y | y | y | y | y | filtered sparse |
| B4 b4-despill30k | y | y | y | y | y | despill v1 |
| B5 b5-sel60 | y | y | y | y | y + selection.json | 60 views |
| B6 b6-despillv2 | y | y | y | y | y | winner line, frozen + manifest |
| E3 aw0.10/0.35/0.50 | y | y | y | y | reuses B2 | aw0.50 = WINNER (110,889 G, 0.9%) |
| E4 res1920 | y | y | y | y | reuses B2 | no gain vs 1600 |
| EVAL-split | y | y | y(x2) | y(x2) | train+full+eval dirs, 12 renders | proxy 4.3%, RGB-only limit logged |

E3/E4/EVAL reuse the B2 dataset by design (single-variable ablations:
only alpha-weight / resolution / view-split change). No dataset copy was
omitted; reuse is recorded in each report.json `source_job`.
