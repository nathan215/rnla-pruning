# Table 1 snippet (checkpoint=300)

*Hero selection:* manual --hero_checkpoint

| Method | Oracle Pass@8 | Oracle Pass@4 | Oracle Pass@2 | Oracle Pass@1 |
|--------|---------------|---------------|---------------|---------------|
| Omniscient (ceiling; any correct among alive) | 0.7650 | 0.7650 | 0.7650 | 0.7650 |
| Random Retention | 0.7465 | 0.7271 | 0.6972 | 0.6417 |
| Top-$m$ Logprob | 0.7450 | 0.7200 | 0.6950 | 0.6250 |
| Top-$m$ Leverage (Exact) | 0.7500 | 0.7350 | 0.7100 | 0.6550 |

**Leverage vs Logprob gap (pp):**
- Pass@8: 0.50 pp
- Pass@4: 1.50 pp
- Pass@2: 1.50 pp
- Pass@1: 3.00 pp
