# Exp0: Exact leverage vs correctness (by checkpoint)

Problems: **200**, `k=8`, exact SVD leverage on **alive** rows at each $t$.

**Top-1 columns:** among problems with **≥1 correct** alive branch — is the **single** top branch correct? (Compare leverage vs cumulative logprob; **random** = mean$_i$(\#correct alive / \#alive) on the same problems.)

| cp | Δ lev | pct-rank(corr) | top-1 **lev** | top-1 **logprob** | random baseline |
|---|---:|---:|---:|---:|---:|
| 50 | +0.0051 | 50.2 | 86.3\% | 86.9\% | 84.0\% |
| 100 | +0.0048 | 50.2 | 87.6\% | 86.3\% | 84.0\% |
| 150 | +0.0018 | 50.1 | 82.4\% | 83.0\% | 84.0\% |
| 200 | -0.0054 | 49.7 | 83.0\% | 84.3\% | 84.0\% |
| 250 | -0.0056 | 49.8 | 81.7\% | 79.1\% | 84.0\% |
| 300 | -0.0034 | 49.9 | 85.6\% | 81.7\% | 84.0\% |
| 400 | -0.0022 | 49.9 | 83.7\% | 85.0\% | 84.0\% |
| 500 | +0.0012 | 50.1 | 81.7\% | 86.9\% | 84.0\% |
| 750 | -0.0050 | 49.8 | 81.0\% | 86.9\% | 84.0\% |
| 1000 | +0.0110 | 49.8 | 81.0\% | 85.0\% | 84.0\% |
| 1500 | +0.0686 | 50.0 | 84.0\% | 84.0\% | 84.3\% |
| 2000 | +0.0978 | 49.8 | 81.8\% | 81.8\% | 82.5\% |
| 3000 | +0.1014 | 50.4 | 80.6\% | 74.8\% | 75.8\% |
| 4000 | +0.1016 | 49.4 | 63.5\% | 60.8\% | 65.6\% |
| 5000 | +0.1283 | 48.1 | 58.1\% | 53.2\% | 58.2\% |
| 6000 | +0.1390 | 48.3 | 44.4\% | 46.3\% | 49.7\% |
| 7000 | +0.1740 | 50.3 | 37.2\% | 34.9\% | 39.9\% |
| 8000 | +0.2054 | 55.7 | 22.2\% | 27.8\% | 23.1\% |

**Read:** **Pass@2** in the paper is *not* top-1; it asks whether **either** of the top-2 (by logprob) is correct. Here **top-1 logprob** is the fair apples-to-apples comparison to **top-1 leverage**.
