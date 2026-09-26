"""Sequential LandS scalability sweep for L2 extended SOS1 local-dominance."""

try:
    from .sos1_norm_scalability_common import scalability_main
    from .local_dominance_radius_norm2_extended_sos1 import CONFIG
except Exception:
    from sos1_norm_scalability_common import scalability_main
    from local_dominance_radius_norm2_extended_sos1 import CONFIG

if __name__ == "__main__":
    raise SystemExit(scalability_main(CONFIG))
