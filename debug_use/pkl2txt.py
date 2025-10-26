#!/usr/bin/env python3
# pkl2txt.py
import argparse, io, json, pickle, pprint, sys
from typing import Any

# Optional deps (used if available)
try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

# Try Pyomo detection (only if installed)
try:
    import pyomo.environ as pyo
    _PYOMO_AVAILABLE = True
except Exception:
    pyo = None
    _PYOMO_AVAILABLE = False


def _json_default(o: Any):
    """Fallback for non-JSON-serializable objects."""
    try:
        return repr(o)
    except Exception:
        return f"<unrepr: {type(o).__name__}>"


def _is_pyomo_model(obj: Any) -> bool:
    if not _PYOMO_AVAILABLE:
        return False
    # ConcreteModel and Block both work with pprint(ostream=...)
    return isinstance(obj, (pyo.ConcreteModel, pyo.AbstractModel)) or (
        hasattr(obj, "pprint") and "pyomo" in type(obj).__module__
    )


def write_readable(obj: Any, fh, ndarray_threshold: int = 1000):
    """Write a readable textual representation of obj to fh."""
    # 1) Pyomo models
    if _is_pyomo_model(obj):
        # Pyomo's pprint can take a stream-like under ostream=
        obj.pprint(ostream=fh)
        return

    # 2) pandas DataFrame / Series
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            fh.write(obj.to_csv(index=False))
            return
        if isinstance(obj, pd.Series):
            fh.write(obj.to_string())
            return

    # 3) numpy arrays
    if np is not None and isinstance(obj, np.ndarray):
        s = np.array2string(obj, threshold=ndarray_threshold, edgeitems=10)
        fh.write(s + "\n")
        return

    # 4) Simple Python containers / scalars → JSON if possible, else pprint
    simple_types = (dict, list, tuple, set, str, int, float, bool, type(None))
    if isinstance(obj, simple_types):
        try:
            fh.write(json.dumps(obj, indent=2, default=_json_default))
            fh.write("\n")
        except TypeError:
            fh.write(pprint.pformat(obj, compact=False, width=100))
            fh.write("\n")
        return

    # 5) Bytes → try utf-8, else hex
    if isinstance(obj, (bytes, bytearray)):
        try:
            fh.write(bytes(obj).decode("utf-8"))
        except Exception:
            fh.write(bytes(obj).hex())
        fh.write("\n")
        return

    # 6) Fallback
    fh.write(repr(obj) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Convert a .pkl file to a readable .txt file.")
    ap.add_argument("input", help="Path to input .pkl")
    ap.add_argument("-o", "--output", help="Path to output .txt (default: input with .txt)")
    ap.add_argument("--ndarray-threshold", type=int, default=1000,
                    help="Max elements to show before summarizing numpy arrays (default: 1000)")
    args = ap.parse_args()

    out_path = args.output or (args.input.rsplit(".", 1)[0] + ".txt")

    # WARNING: only load trusted pickles!
    with open(args.input, "rb") as f:
        obj = pickle.load(f)

    # If the pickle contains multiple objects (e.g., a tuple/list), we handle that gracefully
    with open(out_path, "w", encoding="utf-8") as out:
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            out.write(f"# Pickle contained a {type(obj).__name__} of length {len(obj)}\n\n")
            for i, item in enumerate(obj):
                out.write(f"## Item {i} — {type(item).__name__}\n")
                write_readable(item, out, ndarray_threshold=args.ndarray_threshold)
                out.write("\n")
        else:
            write_readable(obj, out, ndarray_threshold=args.ndarray_threshold)

    print(f"Wrote readable text to: {out_path}")


if __name__ == "__main__":
    main()
