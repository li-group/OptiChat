"""L-infinity one-sided SOS1 local-dominance.

Run directly for a single LandS instance.
"""

from __future__ import annotations
from typing import Any, Dict

try:
    from .local_dominance_sos1_norm_common import ExperimentConfig, run_configured_experiment, single_main
except Exception:
    from local_dominance_sos1_norm_common import ExperimentConfig, run_configured_experiment, single_main

CONFIG = ExperimentConfig(
    norm_kind='inf',
    formulation_kind='one_sided',
    slug='norminf_one_sided_sos1',
    summary_stem='Norm_inf_one_sided_sos1',
    output_subdir='norminf_one_sided_sos1_exp',
    canonical_filename='norminf_one_sided_sos1_result.json',
    display_name='L-infinity one-sided SOS1 local-dominance',
)

def run_norminf_one_sided_sos1_local_dominance(instance_name: str, **kwargs: Any) -> Dict[str, Any]:
    return run_configured_experiment(CONFIG, instance_name, **kwargs)

if __name__ == "__main__":
    raise SystemExit(single_main(CONFIG))
