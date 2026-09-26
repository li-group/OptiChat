"""Thin single-instance runner for norminf_one_sided_sos1."""

try:
    from .local_dominance_sos1_norm_common import single_main
    from .local_dominance_radius_norminf_one_sided_sos1 import CONFIG
except Exception:
    from local_dominance_sos1_norm_common import single_main
    from local_dominance_radius_norminf_one_sided_sos1 import CONFIG

if __name__ == "__main__":
    raise SystemExit(single_main(CONFIG))
