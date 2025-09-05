import pickle
import sys
from pathlib import Path
from src.research.scripts.pipeline import ModelArtifact

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
    inspect_pickle(sys.argv[1])
