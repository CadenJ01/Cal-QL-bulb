import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python inspect_npz.py <path>")
        return 1

    path = Path(sys.argv[1])
    data = np.load(path, allow_pickle=True)
    print(f"path: {path}")
    print(f"keys: {list(data.files)}")

    for key in data.files:
        value = data[key]
        print(f"--- {key} ---")
        print(f"type: {type(value)}")
        print(f"shape: {getattr(value, 'shape', None)}")
        print(f"dtype: {getattr(value, 'dtype', None)}")
        if getattr(value, "size", 0):
            first = value.flat[0]
            print(f"first_type: {type(first)}")
            if hasattr(first, "shape"):
                print(f"first_shape: {first.shape}")
            else:
                print(f"first_repr: {repr(first)[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
