from importlib import import_module
import sys as _sys

# The repository's top-level directory is on sys.path, so we can import the
# existing modules (patterns, sim, viz, fetch_data, run, etc.) and expose
# them as submodules of *worldline_sim* so that statements like
# `import worldline_sim.patterns` work without installing the package.

for _name in ("patterns", "sim", "viz", "fetch_data", "run"):
    try:
        _mod = import_module(_name)
        _sys.modules[f"{__name__}.{_name}"] = _mod
        # Also expose as attribute for `worldline_sim._name` access.
        setattr(_sys.modules[__name__], _name, _mod)
    except ModuleNotFoundError:
        # Some optional submodules (e.g. run when imported) may not exist.
        pass

del import_module, _sys, _name, _mod 