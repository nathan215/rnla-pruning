"""
Runtime configuration. Defaults come from `config.defaults`.

After Step 0, set `T_max` and `N` from
`results/step_0_analysis/part_b_configurations.json` (`step_1_recommendation`), or edit here.
"""

from .defaults import *  # noqa: F401,F403

T_max: int | None = None
N: int | None = None
