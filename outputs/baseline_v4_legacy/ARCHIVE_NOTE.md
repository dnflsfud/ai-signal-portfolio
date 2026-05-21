# baseline_v4_legacy — archival snapshot

**Archived**: 2026-05-19 during `phases/final-v1-promotion` step 2 (production-cutover).

**What this is**: byte-identical copy of `outputs/baseline_v4/` as it existed
just before the cutover from `variants/iter15_65tkr_reb21_vtg.yaml` to
`variants/baseline_v5_deploy.yaml` in `update_and_deploy.py`.

**Why preserved**: audit trail. The legacy deploy strategy was the production
strategy from 2026-04-24 until 2026-05-19. Its metrics (IR=1.304, turnover=113.6%,
P1=+1.537 / P2=+0.171 / P3=+1.911) are the historical reference for the new
deploy strategy's first few months of live operation.

After cutover, `outputs/baseline_v4/` becomes the alias for `baseline_v5_deploy`
output (dashboard hard-codes that path). Do NOT modify this `_legacy/` copy.

## Headline metrics at archival

| Metric | Value |
|---|---:|
| Information Ratio | 1.304 |
| Annual Turnover (two-way) | 113.6% |
| Annual Return | 28.96% |
| Sharpe Ratio | 1.289 |
| P1 IR | +1.537 |
| P2 IR | +0.171 |
| P3 IR | +1.911 |

## How to reproduce (if ever needed)
```bash
python run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml --no-cache
# Output to outputs/iter15_65tkr_reb21_vtg/
```
