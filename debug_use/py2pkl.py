"""
py2pkl.py
---------
Convert a Pyomo model .py file to a .pkl file using cloudpickle.

Usage:
    python debug_use/py2pkl.py supply_chain_feasible

The script:
  1. Imports the target module and retrieves `model` from it.
  2. Saves it to tmp/model_objects/<name>.pkl via save_model_object().

Run from the repo root so that the optichat package is on the path.
"""

import sys
import importlib
import importlib.util
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optichat.tools.extract_tool import save_model_object


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_use/py2pkl.py <module_name_without_py>")
        print("Example: python debug_use/py2pkl.py supply_chain_feasible")
        sys.exit(1)

    name = sys.argv[1]
    py_path = Path(__file__).parent / f"{name}.py"

    if not py_path.exists():
        print(f"Error: {py_path} not found.")
        sys.exit(1)

    # Load the .py file as a module
    spec = importlib.util.spec_from_file_location(name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "model"):
        print(f"Error: no 'model' variable found in {py_path.name}.")
        sys.exit(1)

    out_path = save_model_object(module.model, name)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
