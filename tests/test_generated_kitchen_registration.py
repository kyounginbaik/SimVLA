import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_new_scenes_register_for_every_robot_without_importing_configs(tmp_path):
    path = Path(__file__).parents[1] / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/registration.py"
    spec = importlib.util.spec_from_file_location("kitchen_registration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ["kitchen_00_00.py", "kitchen_01r_03.py", "kitchen_02a_11.py", "kitchen_helper.py"]:
        (tmp_path / name).write_text("raise RuntimeError('must not eagerly import configs')")
    records = {}
    def register(**kwargs):
        assert kwargs["id"] not in records
        records[kwargs["id"]] = kwargs
    gym = SimpleNamespace(registry=records, register=register)
    module.register_kitchens(gym, tmp_path, "test.kitchen")
    module.register_kitchens(gym, tmp_path, "test.kitchen")
    assert set(records) == {"Isaac-Kitchen-v00-00", "Isaac-Kitchen-v01r-03", "Isaac-Kitchen-v02a-11"}
    for task, cls in [("Isaac-Kitchen-v00-00", "AnubisKitchenEnvCfg"),
                      ("Isaac-Kitchen-v01r-03", "RBY1KitchenEnvCfg"),
                      ("Isaac-Kitchen-v02a-11", "AIWorkerKitchenEnvCfg")]:
        assert records[task]["kwargs"]["env_cfg_entry_point"].endswith(":" + cls)
