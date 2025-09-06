import json
import pickle
import sys
import base64
import dataclasses
from pathlib import Path
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any

# Optional helpers for common scientific types (silently skipped if unavailable)
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None

# Ensure classes referenced by pickles are importable
from src.research.scripts.pipeline import ModelArtifact  # noqa: F401
from src.recorder.poe_currency_recorder import FXCache   # noqa: F401


def _qname(o: Any) -> str:
    t = type(o)
    return f"{t.__module__}.{t.__name__}"


def _to_jsonable(obj: Any, _seen: set[int] | None = None) -> Any:
    """
    Convert arbitrary Python objects to JSON-serializable structures.
    - No depth cap (as requested).
    - Cycle-safe: repeated objects are emitted as {"$ref": "<module.Class>"}.
    """
    if _seen is None:
        _seen = set()

    # Fast path for primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    oid = id(obj)
    if oid in _seen:
        return {"$ref": _qname(obj)}
    _seen.add(oid)

    # Dataclasses -> dict first
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)

    # Common scalar-ish types
    if isinstance(obj, (date, datetime, time)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return {"$decimal": str(obj)}
    if isinstance(obj, Enum):
        return {"$enum": f"{obj.__class__.__name__}.{obj.name}", "value": obj.value}
    if isinstance(obj, Path):
        return str(obj)

    # Bytes: short base64 preview; include full payload only if tiny
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        head = base64.b64encode(b[:1024]).decode("ascii")
        out = {"$type": "bytes", "len": len(b), "b64_head": head}
        if len(b) <= 1024:
            out["b64"] = head
        return out

    # NumPy (optional)
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return {
                "$type": "ndarray",
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "data": obj.tolist(),
            }
        if isinstance(obj, _np.generic):
            return obj.item()

    # Pandas (optional)
    if _pd is not None:
        if isinstance(obj, _pd.DataFrame):
            return {"$type": "DataFrame", "orient": "split", "value": obj.to_dict(orient="split")}
        if isinstance(obj, _pd.Series):
            return {"$type": "Series", "index": obj.index.tolist(), "dtype": str(obj.dtype), "values": obj.tolist()}
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()

    # Mappings
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, _seen) for k, v in obj.items()}

    # Sequences / sets
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [_to_jsonable(x, _seen) for x in obj]
        if isinstance(obj, list):
            return items
        tag = "tuple" if isinstance(obj, tuple) else "set" if isinstance(obj, set) else "frozenset"
        return {"$type": tag, "items": items}

    # Objects with attributes
    if hasattr(obj, "__slots__"):
        attrs = {name: getattr(obj, name) for name in obj.__slots__ if hasattr(obj, name)}
        return {"$class": _qname(obj), "attributes": _to_jsonable(attrs, _seen)}
    if hasattr(obj, "__dict__"):
        return {"$class": _qname(obj), "attributes": _to_jsonable(vars(obj), _seen)}

    # Fallback to string
    try:
        return str(obj)
    except Exception:
        return f"<unserializable {_qname(obj)}>"


def write_pickle_json(path: str | Path, outfile: str | Path | None = None) -> dict:
    """
    Load a pickle and write a pretty JSON view to 'pickle.json' next to it (or to `outfile`).
    Returns the JSON-serializable structure. ⚠️ Only unpickle trusted files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("rb") as f:
        obj = pickle.load(f)

    jsonable = _to_jsonable(obj)
    out_path = Path(outfile) if outfile else p.parent / "pickle.json"
    out_path.write_text(json.dumps(jsonable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote JSON to: {out_path}")
    return jsonable


# --- your original inspector (kept as-is) ---

def inspect_pickle(path):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return
    
    with open(p, "rb") as f:
        try:
            obj = pickle.load(f)
        except Exception as e:
            print(f"Failed to unpickle: {e}")
            return
    
    print(f"=== Inspection of {p} ===")
    print(f"Type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"Keys: {list(obj.keys())}")
    elif hasattr(obj, "__dict__"):
        print("Attributes:")
        for k, v in vars(obj).items():
            print(f"  {k}: {type(v)}")
    else:
        print(obj)

    return obj


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pickle.py <path_to_pickle>")
        sys.exit(1)
    # keep existing behavior
    inspect_pickle(sys.argv[1])
    # if you want JSON too, call:
    write_pickle_json(sys.argv[1])
