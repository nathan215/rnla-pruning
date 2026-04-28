# Step 2.2 — Cross-checkpoint variation

When multiple `--checkpoints` are used (or `--step1-checkpoints`), pooled Table 2–3 still 
aggregate all (problem, checkpoint) pairs; the JSON fields `per_checkpoint` break out 
means **within each checkpoint** so you can see whether approximation quality drifts with depth.

*No multi-checkpoint sweep in this run (need ≥2 checkpoints with data).*
