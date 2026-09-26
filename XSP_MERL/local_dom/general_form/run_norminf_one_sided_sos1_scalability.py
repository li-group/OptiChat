"""Sequential LandS scalability sweep for L-infinity one-sided SOS1 local-dominance."""

try:
    from .sos1_norm_scalability_common import scalability_main
    from .local_dominance_radius_norminf_one_sided_sos1 import CONFIG
except Exception:
    from sos1_norm_scalability_common import scalability_main
    from local_dominance_radius_norminf_one_sided_sos1 import CONFIG

if __name__ == "__main__":
    raise SystemExit(scalability_main(CONFIG))
