"""Kitchen generation must stay inside its declared outputs and be retryable after a failure."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
KITCHEN_PKG = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_source(path, name):
    src = path.read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn)


def test_rotation_export_does_not_rewrite_the_tracked_kitchen_init():
    """registration.register_kitchens discovers kitchen_*.py; appending gym.register blocks to the
    tracked __init__.py duplicated that, dirtied the checkout, and left dangling task IDs whenever
    the configs lived in SIMVLA_ENV_CFG_OUTPUT_DIR or a failed run was cleaned up."""
    body = _function_source(ROOT / "scripts/simvla/kitchen_scene_generator.py", "rotate_and_register_envs")
    assert "__init__.py" not in body and "init_file" not in body
    assert "gym.register" not in body


def test_scene_usd_export_keeps_textures_beside_the_kitchen_usd():
    """kitchen.export(file_type='usd') without folder= writes ./textures in the caller's cwd."""
    src = (ROOT / "scripts/simvla/kitchen_scene_generator.py").read_text()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "kitchen.export"]
    assert calls
    for call in calls:
        assert "folder" in {k.arg for k in call.keywords}, ast.unparse(call)


def test_output_dir_configs_are_registered_and_importable(tmp_path):
    registration = _load(KITCHEN_PKG / "registration.py", "kitchen_registration_out")
    (tmp_path / "kitchen_99000_00.py").write_text("raise RuntimeError('must not eagerly import')")
    records = {}
    gym = SimpleNamespace(registry=records, register=lambda **kw: records.setdefault(kw["id"], kw))
    package_path = [str(KITCHEN_PKG)]
    registration.register_output_dir(gym, package_path, "pkg.kitchen", {"SIMVLA_ENV_CFG_OUTPUT_DIR": str(tmp_path)})
    assert "Isaac-Kitchen-v99000-00" in records
    assert records["Isaac-Kitchen-v99000-00"]["kwargs"]["env_cfg_entry_point"] == \
        "pkg.kitchen.kitchen_99000_00:AnubisKitchenEnvCfg"
    assert str(tmp_path.resolve()) in package_path  # submodule import resolves from the output dir
    registration.register_output_dir(gym, package_path, "pkg.kitchen", {"SIMVLA_ENV_CFG_OUTPUT_DIR": str(tmp_path)})
    assert package_path.count(str(tmp_path.resolve())) == 1


def test_output_dir_registration_ignores_unset_missing_or_package_dir(tmp_path):
    registration = _load(KITCHEN_PKG / "registration.py", "kitchen_registration_noop")
    records = {}
    gym = SimpleNamespace(registry=records, register=lambda **kw: records.setdefault(kw["id"], kw))
    for environ in ({}, {"SIMVLA_ENV_CFG_OUTPUT_DIR": ""},
                    {"SIMVLA_ENV_CFG_OUTPUT_DIR": str(tmp_path / "missing")},
                    {"SIMVLA_ENV_CFG_OUTPUT_DIR": str(KITCHEN_PKG)}):
        package_path = [str(KITCHEN_PKG)]
        registration.register_output_dir(gym, package_path, "pkg.kitchen", environ)
        assert package_path == [str(KITCHEN_PKG)]
    assert records == {}


def test_failed_generation_removes_only_its_own_partial_kitchen(tmp_path):
    cleanup = _load(ROOT / "scripts/simvla/generated_kitchen_files.py", "generated_kitchen_files")
    kitchen = tmp_path / "Kitchen"
    (kitchen / "bodex").mkdir(parents=True)
    keep = [kitchen / "kitchen_813_00.usd", kitchen / "kitchen_990001.usd",
            kitchen / "bodex/kitchen_data_813.json"]
    for path in keep:
        path.write_text("keep")
    before = cleanup.kitchen_outputs(kitchen, 99000)
    assert before == set()
    partial = [kitchen / "kitchen_99000.usd", kitchen / "kitchen_99000_00.usd",
               kitchen / "kitchen_99000_11.usd", kitchen / "bodex/kitchen_data_99000.json"]
    for path in partial:
        path.write_text("partial")
    removed = cleanup.remove_new_outputs(kitchen, 99000, before)
    assert sorted(removed) == sorted(partial)
    assert not any(p.exists() for p in partial)
    assert all(p.exists() for p in keep)
