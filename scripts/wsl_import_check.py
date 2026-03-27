import importlib

mods = ["jax", "flax", "d4rl", "dm_control", "mujoco_py"]

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"{mod}: ok")
    except Exception as exc:
        print(f"{mod}: FAIL {type(exc).__name__}: {exc}")
