import sys
import numpy as np


path = sys.argv[1]
d = np.load(path, allow_pickle=True)
print(list(d.files))
for key in d.files:
    value = d[key]
    scalar = getattr(value, "shape", ()) == ()
    print(key, type(value).__name__, getattr(value, "shape", None), getattr(value, "dtype", None), value if scalar else "")
