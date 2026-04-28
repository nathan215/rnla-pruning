# Step 1 aggregate (200 problems, b0–b7)

- Generated: `2026-03-30T15:57:44.918280+00:00`
- Match mode: **boxed_only**

## Pass@1 / Oracle Pass@16

| Source | Pass@1 | Oracle Pass@16 |
|--------|--------|----------------|
| Weighted from each batch `analysis/pass_at_k.json` | 0.6500 | 0.7650 |
| Recomputed from all `branch_*.txt` (n=200) | 0.6500 | 0.7650 |

Recomputed vs weighted-from-JSON should match when data are consistent; small drift indicates rounding or a stale `pass_at_k.json`.

## Distribution: number of branches correct (vs reference)

```
{
  "0": 47,
  "1": 5,
  "2": 4,
  "3": 3,
  "6": 2,
  "7": 1,
  "8": 3,
  "9": 4,
  "10": 6,
  "11": 6,
  "12": 7,
  "13": 7,
  "14": 5,
  "15": 15,
  "16": 85
}
```

## Distribution: distinct extracted answers per problem (16 branches)

```
{
  "1": 105,
  "2": 56,
  "3": 28,
  "4": 6,
  "5": 5
}
```

## Batch run times (metadata)

```
{
  "step_1_oracle_b0": 5435.054395404,
  "step_1_oracle_b1": 4325.748852277,
  "step_1_oracle_b2": 5454.002893134,
  "step_1_oracle_b3": 5050.462316648,
  "step_1_oracle_b4": 5526.537344885999,
  "step_1_oracle_b5": 5200.925333903,
  "step_1_oracle_b6": 526.3527751510001,
  "step_1_oracle_b7": 5426.375096755
}
```

**Sum batch wall times:** 36945.5 s (~10.26 h) — not de-duplicated if batches overlapped.

Full JSON: `step_1_oracle_200_summary.json`