import os
import sys


def show_import(name):
    try:
        module = __import__(name)
        print(f"{name}: OK -> {getattr(module, '__file__', '<built-in>')}")
    except Exception as exc:
        print(f"{name}: FAIL -> {exc!r}")


print(f"python={sys.executable}")
print(f"version={sys.version}")
print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
print("sys.path[0:8]=")
for entry in sys.path[:8]:
    print(entry)

show_import("isaacgym")
show_import("isaacgymenvs")
