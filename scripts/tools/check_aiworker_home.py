"""Compare every configured AI Worker home predicate against the public USD joint graph."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

from check_aiworker_urdf_vs_usd import usd_fk  # Includes USD library bootstrap.
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def configured_homes():
    tree = ast.parse((ROOT / "source/isaaclab_assets/isaaclab_assets/robots/aiworker.py").read_text())
    lower = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                 and getattr(n.target, "id", None) == "AIWORKER_LIFT_HOMES")
    homes = ast.literal_eval(lower.value)
    cfg = next(n for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "AIWORKER_CFG" for t in n.targets))
    init = next(k.value for k in cfg.value.keywords if k.arg == "init_state")
    positions = next(k.value for k in init.keywords if k.arg == "joint_pos")
    homes["0.00"] = {key.value: ast.literal_eval(value)
                     for key, value in zip(positions.keys, positions.values)
                     if isinstance(key, ast.Constant) and str(key.value).startswith("arm_")}
    return homes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/mdp/homes.py"
    spec = importlib.util.spec_from_file_location("home_constants", path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    rows = []
    for lift, joints in configured_homes().items():
        joints["lift_joint"] = float(lift)
        measured = [usd_fk(target=name, joints=joints, usd_path=args.usd)[0][:3, 3]
                    for name in ("ee_link1", "ee_link2")]
        expected = np.asarray(constants.AIWORKER_BY_LIFT[lift])
        error = float(np.max(np.linalg.norm(np.asarray(measured) - expected, axis=1)))
        rows.append(dict(lift_m=float(lift), measured_xyz=[p.tolist() for p in measured],
                         expected_xyz=expected.tolist(), max_error_m=error, passed=error <= 1e-5))
    report = dict(schema_version=1, passed=all(r["passed"] for r in rows), homes=rows,
                  usd_sha256=hashlib.sha256(args.usd.read_bytes()).hexdigest(),
                  scope="Kinematic home predicate alignment, not loaded transport success.")
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
