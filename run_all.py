"""Run the whole pipeline in order, from the repo root:  python run_all.py"""
import runpy, sys, pathlib
sys.path.insert(0, str(pathlib.Path("pipeline").resolve()))
STEPS = ["load_data", "fake_rrb", "real_rrb", "hedge", "rolling", "drivers",
         "proxy_tests", "subcomponent_tests", "hedgeable_share", "evidence", "export_site"]
for s in STEPS:
    print(f"\n===== {s} =====")
    runpy.run_path(f"pipeline/{s}.py", run_name="__main__")
