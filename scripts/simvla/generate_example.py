"""Generate a kitchen with one mug, twelve rotations, and a lift-task template."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kitchen-id', type=int, required=True, help='Unused positive kitchen number')
parser.add_argument('--layout', choices=['single_wall', 'l_shaped', 'u_shaped', 'island', 'peninsula'], default='l_shaped')
parser.add_argument('--mesh', type=Path, required=True, help='BODex mug OBJ with matching grasp data')
parser.add_argument('--output', type=Path, required=True, help='New directory for template and generation facts')
parser.add_argument('--seed', type=int, help='Geometry/material seed; defaults to kitchen ID modulo 2**32')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
assets_root = Path(os.environ.get('SIMVLA_ASSETS_DIR', root / 'source/isaaclab_assets/data')).expanduser().resolve()
assets = assets_root / 'Kitchen'
configs = root / 'source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen'
if args.kitchen_id < 1:
    parser.error('--kitchen-id must be positive')
seed = args.kitchen_id % (2**32) if args.seed is None else args.seed
if not 0 <= seed < 2**32:
    parser.error('--seed must be between 0 and 2**32 - 1')
stem = f'kitchen_{args.kitchen_id:02d}'
if any(assets.glob(stem + '*.usd')) or any(configs.glob(stem + '_*.py')):
    parser.error('Kitchen number already exists; choose a new --kitchen-id')
if not args.mesh.is_file():
    parser.error(f'Mesh does not exist: {args.mesh}')
if args.output.exists():
    parser.error('--output already exists; use a new directory')
assets.mkdir(parents=True, exist_ok=True)
os.environ.update(MUGK_NUM=str(args.kitchen_id), MUGK_TYPE=args.layout,
                  MUGK_SEED=str(seed),
                  MUGK_MESH=os.path.abspath(args.mesh),
                  MUGK_TEMPLATE=str(args.output.resolve() / 'template.json'))
exit_code = 0
try:
    runpy.run_path(str(Path(__file__).with_name('build_new_mug_kitchen.py')), run_name='__main__')
except BaseException:
    print(traceback.format_exc(), flush=True)
    exit_code = 1
finally:
    from workflow_status import record_status
    record_status(exit_code)
    ksg = sys.modules.get('kitchen_scene_generator')
    if ksg is not None:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        ksg.app_launcher.app.close()
raise SystemExit(exit_code)
