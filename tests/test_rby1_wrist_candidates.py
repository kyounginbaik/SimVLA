import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "rby1_wrist_candidates", Path(__file__).resolve().parents[1] / "scripts/simvla/rby1_wrist_candidates.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ten_paired_fixed_mounts_preserve_defaults():
    def camera(side):
        return SimpleNamespace(prim_path=f"/Robot/{side}/camera",
                               offset=SimpleNamespace(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                               spawn=SimpleNamespace(focal_length=25))
    scene = SimpleNamespace(wrist_left=camera('left'), wrist_right=camera('right'))
    mounts = module.add_rby1_wrist_candidates(scene)
    assert len(mounts) == 20
    assert scene.wrist_right.offset.pos == (0, 0, 0)
    assert scene.wrist_left.offset.rot == (1, 0, 0, 0)
    for i in range(1, 11):
        r, l = mounts[f'candidate_{i:02d}_right'], mounts[f'candidate_{i:02d}_left']
        assert r['pos'][1] > 0 > l['pos'][1]
        assert r['pos'][1] == -l['pos'][1]
        assert r['rot_wxyz'][1] == l['rot_wxyz'][2]
        assert r['rot_wxyz'][0] == l['rot_wxyz'][3]
        assert l['rot_wxyz'][:2] == (0., 0.)
        assert r['rot_wxyz'][2:] == (0., 0.)
