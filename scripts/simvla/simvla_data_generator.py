from pxr import UsdGeom, Gf, Usd
import omni.usd, omni.physx
import torch, math, random, json, numpy as np, re, os, ipdb
from shapely.geometry import box, Point
from scipy.spatial.transform import Rotation as R
from isaaclab.simvla.utils import (
	select_thumbnails_cached,
	select_thumbnails,
	merge_free_spaces,
	find_free_spaces_grid,
	make_walls_from_bounds,
	load_grasp_file,
	find_first_free_direction,
)


import glob
import shutil
from collections import defaultdict

# Skill declarations live in skills.py, NOT here.
#
# They used to live here, and that was the bug: simvla_gen.py (the executor) resolves runtime skills
# out of the REGISTRY, which is populated as a side effect of importing the module the @skill
# decorators live in. While that module was THIS one — which the executor cannot import, because of
# the six lines above (pxr, omni, tkinter, PIL) — the registry was empty in the executor's process
# and every runtime step died with KeyError. skills.py imports none of that, and both processes
# import skills.py.
#
# What stays here: the six plan() bodies that genuinely need this stack — they read the USD stage or
# the Tk thumbnail chooser. skills.py declares those skills and injects the body from here via
# register_planner(). See skills.py's module docstring.
from typing import Callable, Any, Tuple, Optional, Dict

import skills
from skills import (
	# the fifteen skills, re-exported so `simvla_data_generator.X` still resolves
	ArmBottlePour,
	ArmBottleToPosition,
	ArmBowlPlace,
	ArmFridgeHandleGrasp,
	ArmGrasp,
	ArmHandleGrasp,
	ArmHandlePregrasp,
	ArmMugToPosition,
	ArmPause,
	ArmPlace,
	ArmReset,
	GripperSet,
	NavCloseArticulation,
	NavOpenArticulation,
	NavToPrim,
	# the authoring-side injection point
	register_planner,
	validate_planners,
	# quaternion helpers, used by plan_arm_grasp below
	gripper_y_up_down,
	local_z_plus_180_wxyz,
	quat_normalize_wxyz,
	rotate_vec_by_quat_wxyz,
	# the v1 bridge. The WRITER no longer needs it — it emits v2. The LOADER still does: every
	# reloadable template on disk is a v1 file whose steps name a skill only by its `usage` string.
	EXECUTOR_ACTIONS,
	LEGACY_USAGE_ALIASES,
	_legacy_skill_for,
)

# The goal file. One file, and every step says which skill produced it — so the GUI's dropdowns,
# the file and (from Task 6) the executor all read the same registry, and none of them restates it.
from goal_format import Step, write_v2
from skill_contract import (
	ACTIONS,
	REGISTRY,
	Bool,
	Choice,
	Float,
	PrimPath,
	skills_for_action,
)

import numpy as np
from scipy.spatial.transform import Rotation as R
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat, quat_mul, quat_conjugate
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

# === simvla path resolution (auto-added) ===
import os as _simvla_os
from pathlib import Path as _SimvlaPath
def _simvla_find_repo_root():
    env = _simvla_os.environ.get("SIMVLA_REPO_ROOT")
    if env:
        return env
    for p in _SimvlaPath(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return str(p)
    raise RuntimeError("Cannot find repo root; set SIMVLA_REPO_ROOT env var")
SIMVLA_REPO_ROOT = _simvla_find_repo_root()
# === end simvla path resolution ===

import simvla_paths
KITCHEN_DIR = str(simvla_paths.assets_dir() / "Kitchen")

# The support-clearance check plan_arm_grasp applies to every BODex candidate. Imported after the
# repo-root block above because tool_spheres() is given SIMVLA_REPO_ROOT to find the vendored cuRobo
# configs — the SAME files the planner loads, so the gripper this measures is the gripper cuRobo
# measures.
import grasp_clearance


# =============================================================================
# What the GUI offers, derived from the registry — never written out by hand
# =============================================================================
# SubtaskDialog used to hardcode its action codes and, per action, a list of free-text "usage"
# strings. That was a second copy of what the registry already knew, and it drifted: the dialog
# offered G_b, simvla_gen's TASK_IDS had never heard of it, and a goal file containing one died
# with KeyError at load. The three functions below are the dialog's whole vocabulary now, so the
# list it offers IS the list that exists.


def gui_actions() -> list[str]:
	"""Every action code the GUI may offer. There is no G_b: no skill declares it, because the
	executor has no such channel (simvla_gen.py:1538)."""
	return list(ACTIONS())


def gui_skills(action: str) -> list[tuple[str, str]]:
	"""(label, skill id) for every skill that can drive `action`. The label is what the dropdown
	shows; the id is what gets written into the goal file."""
	return [(spec.label, spec.id) for spec in skills_for_action(action)]


def gui_params(skill_id: str) -> list[dict]:
	"""The parameter widgets to render for a skill — exactly the params it declared.

	One dict per param: {"name", "kind", "default"} and, for a choice, "options". Returning data
	rather than Tk widgets is what makes this testable; SubtaskDialog turns each dict into a widget.
	"""
	widgets = []
	for p in REGISTRY[skill_id].params:
		if isinstance(p, Choice):
			widgets.append({"name": p.name, "kind": "choice", "options": list(p.options),
			                "default": p.default})
		elif isinstance(p, Bool):
			widgets.append({"name": p.name, "kind": "bool", "default": bool(p.default)})
		elif isinstance(p, Float):
			_default = float(p.default)
			widgets.append({"name": p.name, "kind": "float",
			                "default": _default if math.isfinite(_default) else ""})
		elif isinstance(p, PrimPath):
			widgets.append({"name": p.name, "kind": "prim_path", "default": ""})
		else:
			raise TypeError(f"{skill_id}: no widget for param {p!r}")
	return widgets


def params_for_skill(spec, raw: dict) -> dict:
	"""Exactly the params `spec` declares, coerced to their declared types.

	Both the GUI and the legacy loader come through here. A Tk Entry hands back the string "0.4",
	and a v1 reloadable template hands back a `usage` key and a `prim_path` for a skill that has
	no prim — so this drops what the skill did not declare and casts what it did. Without the
	cast, back_off_m reaches the resolver as the string "0.4" and the arithmetic silently changes
	meaning.
	"""
	params = {}
	for p in spec.params:
		if isinstance(p, Float):
			_raw = raw.get(p.name)
			if _raw in (None, "") and not math.isfinite(float(p.default)):
				continue
			value = float(p.default if _raw in (None, "") else _raw)
			if not math.isfinite(value):
				raise ValueError(f"{spec.id}.{p.name} must be finite; omit it to use the declared default")
			params[p.name] = value
		elif isinstance(p, Bool):
			value = raw.get(p.name, p.default)
			if isinstance(value, str):
				value = value.strip().lower() in ("1", "true", "yes", "on")
			params[p.name] = bool(value)
		elif isinstance(p, Choice):
			params[p.name] = str(raw.get(p.name, p.options[0]))
		elif isinstance(p, PrimPath):
			params[p.name] = str(raw.get(p.name, ""))
		else:
			raise TypeError(f"{spec.id}: no coercion for param {p!r}")
	return params


def authored_step(step_data, index: int) -> dict | None:
	"""One step of the GUI's working list — {"lang", "skill", "action", "params"} — from whatever
	a goal file happens to hold. None if the entry is not a step at all.

	Three shapes reach this:
	  * v2 — the step already names its skill. This is what the writer now emits.
	  * a v1 *reloadable* template — {"language", "action", "parameters": {"usage": ...}}. The
	    15,810 files on disk are all this shape, and the batch generator reads one as its template,
	    so the loader has to keep understanding it. `usage` is resolved to a skill id HERE, once, at
	    load — the free-text key does not survive into the goal file.
	  * a bare [action, params] pair, which some very old templates use.
	"""
	if isinstance(step_data, dict) and "skill" in step_data and "action" in step_data:
		spec = REGISTRY.get(step_data["skill"])
		if spec is None:
			raise ValueError(f"step {index + 1}: no skill registered with id "
			                 f"{step_data['skill']!r}")
		action, params = step_data["action"], dict(step_data.get("params") or {})
		lang = step_data.get("language", f"Loaded Step {index + 1}")
	elif isinstance(step_data, dict) and "action" in step_data and "parameters" in step_data:
		action = step_data["action"]
		spec, params = _legacy_skill_for(action, dict(step_data["parameters"]))
		lang = step_data.get("language", f"Loaded Step {index + 1}")
	elif (isinstance(step_data, list) and len(step_data) == 2
	      and isinstance(step_data[1], dict)):
		action = step_data[0]
		spec, params = _legacy_skill_for(action, dict(step_data[1]))
		lang = f"Loaded: {action} to {params.get('prim_path', '...')}"
	else:
		return None

	return {"lang": lang, "skill": spec.id, "action": action,
	        "params": params_for_skill(spec, params)}





def _load_gui():
	"""Load desktop dependencies only when opening an interactive editor."""
	global tk, ttk, messagebox, simpledialog, filedialog
	import tkinter as tk
	from tkinter import ttk, messagebox, simpledialog, filedialog


# Goal Generator GUI
class GoalGeneratorApp:
	def __init__(self, root):
		_load_gui()
		self.root = root; self.root.title("Kitchen Goal Generator")
		self.stage, self.free_squares, self.N_dir, self.goal_steps = None, [], "N", []
		self.subtask_cache = self._load_subtask_cache()
		self.all_prim_paths = [] 
		# -- Main layout frames --
		top_frame = ttk.Frame(root)
		top_frame.pack(fill="x", padx=10, pady=5)
		
		main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
		main_paned_window.pack(fill="both", expand=True, padx=10, pady=5)
		
		left_frame = ttk.Frame(main_paned_window)
		main_paned_window.add(left_frame, weight=1)

		right_frame = ttk.Frame(main_paned_window)
		main_paned_window.add(right_frame, weight=2)
		
		log_frame = ttk.LabelFrame(root, text="Status Log")
		log_frame.pack(fill="x", padx=10, pady=5)

		# -- Top frame: Setup --
		setup_frame = ttk.LabelFrame(top_frame, text="1. Kitchen Setup")
		setup_frame.pack(fill="x")
		ttk.Label(setup_frame, text="Task Name:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
		self.task_name_var = tk.StringVar(value="PUT BOWL TO SINK"); ttk.Entry(setup_frame, textvariable=self.task_name_var).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
		ttk.Label(setup_frame, text="Kitchen Number:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
		self.kitchen_num_var = tk.IntVar(value=1); ttk.Entry(setup_frame, textvariable=self.kitchen_num_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=2)
		ttk.Label(setup_frame, text="Sub-Number:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
		self.sub_num_var = tk.StringVar(value="01"); ttk.Entry(setup_frame, textvariable=self.sub_num_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=2)
		self.process_btn = ttk.Button(setup_frame, text="Load Kitchen and Generate Config", command=self.process_kitchen); self.process_btn.grid(row=3, column=0, columnspan=2, pady=10)
		setup_frame.columnconfigure(1, weight=1)
		
		
				# --- Batch Generation (Predefined reloadable task) ---
		batch_frame = ttk.LabelFrame(top_frame, text="Batch Generation (Predefined Task)")
		batch_frame.pack(fill="x", pady=(5, 0))

		# row 0: reloadable json path
		ttk.Label(batch_frame, text="Reloadable JSON:").grid(
			row=0, column=0, sticky="w", padx=5, pady=2
		)
		self.batch_template_var = tk.StringVar()
		ttk.Entry(batch_frame, textvariable=self.batch_template_var).grid(
			row=0, column=1, sticky="ew", padx=5, pady=2
		)
		ttk.Button(
			batch_frame,
			text="Browse...",
			command=self._browse_batch_template,
		).grid(row=0, column=2, padx=5, pady=2)

		# row 1: task name for all generated goals
		ttk.Label(batch_frame, text="Task name:").grid(
			row=1, column=0, sticky="w", padx=5, pady=2
		)
		self.batch_task_name_var = tk.StringVar(value="PUT BOWL TO SINK")
		ttk.Entry(batch_frame, textvariable=self.batch_task_name_var).grid(
			row=1, column=1, sticky="ew", padx=5, pady=2
		)

		# row 2: total kitchens + sub-number
		ttk.Label(batch_frame, text="Total kitchens:").grid(
			row=2, column=0, sticky="w", padx=5, pady=2
		)
		self.batch_total_var = tk.IntVar(value=100)
		ttk.Spinbox(
			batch_frame,
			from_=1,
			to=10000,
			textvariable=self.batch_total_var,
			width=7,
		).grid(row=2, column=1, sticky="w", padx=5, pady=2)

		ttk.Label(batch_frame, text="Sub-number:").grid(
			row=2, column=2, sticky="e", padx=5, pady=2
		)
		self.batch_subnum_var = tk.StringVar(value="01")
		ttk.Entry(batch_frame, textvariable=self.batch_subnum_var, width=5).grid(
			row=2, column=3, sticky="w", padx=5, pady=2
		)

		# row 3: per-type counts
		type_frame = ttk.Frame(batch_frame)
		type_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=5, pady=2)

		# internal name, pretty label
		self.batch_kitchen_types = [
			("island", "Island"),
			("l_shaped", "L shaped"),
			("peninsula", "Peninsula"),
			("u_shaped", "U shaped"),
			("single_wall", "Single wall"),
		]
		self.batch_type_vars = {}

		for idx, (internal_name, pretty) in enumerate(self.batch_kitchen_types):
			ttk.Label(type_frame, text=pretty).grid(
				row=0, column=idx, padx=3, pady=2
			)
			var = tk.IntVar(value=0)
			self.batch_type_vars[internal_name] = var
			ttk.Spinbox(
				type_frame,
				from_=0,
				to=10000,
				textvariable=var,
				width=5,
			).grid(row=1, column=idx, padx=3, pady=2)

		# row 4: buttons
		self.batch_divide_btn = ttk.Button(
			batch_frame, text="Divide equally", command=self._batch_divide_equally
		)
		self.batch_divide_btn.grid(row=4, column=0, padx=5, pady=4, sticky="w")

		self.batch_run_btn = ttk.Button(
			batch_frame, text="Generate batch", command=self._run_batch_predefined
		)
		self.batch_run_btn.grid(row=4, column=1, padx=5, pady=4, sticky="w")

		self.batch_autogen_var = tk.BooleanVar(value=True)
		ttk.Checkbutton(
			batch_frame,
			text="Auto-generate missing houses (clone existing kitchens)",
			variable=self.batch_autogen_var,
		).grid(row=4, column=2, columnspan=2, padx=5, pady=4, sticky="w")

		batch_frame.columnconfigure(1, weight=1)



		# -- Left frame: Scene Inspector --
		inspector_frame = ttk.LabelFrame(left_frame, text="Scene Inspector (Double-click to copy)")
		inspector_frame.pack(fill="both", expand=True, padx=5, pady=5)
		
		# ADD: Search bar for Scene Inspector
		self.search_var = tk.StringVar()
		self.search_var.trace_add("write", self._filter_prim_list) # This calls the filter function as you type
		search_entry = ttk.Entry(inspector_frame, textvariable=self.search_var)
		search_entry.pack(fill="x", padx=5, pady=(5, 0))

		prim_list_frame = ttk.Frame(inspector_frame); prim_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
		self.prim_listbox = tk.Listbox(prim_list_frame); self.prim_scrollbar = ttk.Scrollbar(prim_list_frame, orient=tk.VERTICAL, command=self.prim_listbox.yview)
		self.prim_listbox.config(yscrollcommand=self.prim_scrollbar.set)
		self.prim_scrollbar.pack(side="right", fill="y"); self.prim_listbox.pack(side="left", fill="both", expand=True)
		self.prim_listbox.bind("<Double-1>", self._copy_selected_prim_path)
		# -- Right frame: Goal Definition --
		goal_frame = ttk.LabelFrame(right_frame, text="2. Goal Definition")
		goal_frame.pack(fill="both", expand=True)
		self.goal_listbox = tk.Listbox(goal_frame, height=10); self.goal_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
		goal_btn_frame = ttk.Frame(goal_frame); goal_btn_frame.pack(side="left", fill="y", padx=5)
		self.add_goal_btn = ttk.Button(goal_btn_frame, text="Add Subtask", command=self.add_subtask, state="disabled"); self.add_goal_btn.pack(pady=2, fill="x")
		self.load_goals_btn = ttk.Button(goal_btn_frame, text="Load Goals...", command=self.load_goals_from_file, state="disabled"); self.load_goals_btn.pack(pady=2, fill="x")
		self.remove_goal_btn = ttk.Button(goal_btn_frame, text="Remove Selected", command=self.remove_subtask, state="disabled"); self.remove_goal_btn.pack(pady=2, fill="x")
		self.save_goals_btn = ttk.Button(goal_btn_frame, text="Save Goal File", command=self.save_goal_file, state="disabled"); self.save_goals_btn.pack(pady=20, fill="x")

		# -- Bottom frame: Log --
		self.log_text = tk.Text(log_frame, height=6, wrap="word", relief="sunken", borderwidth=1); self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
		
	def log(self, message):
		self.log_text.insert(tk.END, str(message) + "\n"); self.log_text.see(tk.END); self.root.update_idletasks()
		
		# ========= Batch helpers =========

	def _browse_batch_template(self):
		"""File chooser for reloadable JSON."""
		goal_dir = str(simvla_paths.goals_dir())
		filepath = filedialog.askopenfilename(
			initialdir=goal_dir,
			title="Select reloadable goal file",
			filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
		)
		if filepath:
			self.batch_template_var.set(filepath)
			self.log(f"[BATCH] Using template: {filepath}")

	def _batch_divide_equally(self):
		"""Fill per-type counts so they divide total approximately equally."""
		total = max(int(self.batch_total_var.get()), 1)
		n_types = len(self.batch_kitchen_types)
		base = total // n_types
		remainder = total % n_types

		for idx, (internal_name, _) in enumerate(self.batch_kitchen_types):
			count = base + (1 if idx < remainder else 0)
			self.batch_type_vars[internal_name].set(count)

		self.log(
			f"[BATCH] Divided {total} kitchens equally across {n_types} types."
		)

	def _scan_kitchens_by_type(self):
		"""
		Look at $SIMVLA_ASSETS_DIR/Kitchen/bodex/kitchen_data_XX.json
		and return {kitchen_type: [list of kitchen_numbers]}.
		"""
		bodex_dir = os.path.join(KITCHEN_DIR, "bodex")
		pattern = os.path.join(bodex_dir, "kitchen_data_*.json")
		mapping = defaultdict(list)

		for path in glob.glob(pattern):
			try:
				basename = os.path.basename(path)
				num_str = os.path.splitext(basename)[0].split("_")[-1]
				kitchen_num = int(num_str)
				with open(path, "r") as f:
					data = json.load(f)
				ktype = data.get("kitchen_type")
				if not ktype:
					continue
				mapping[ktype].append(kitchen_num)
			except Exception as e:
				self.log(f"[BATCH] Warning: could not read {path}: {e}")

		for k in mapping:
			mapping[k] = sorted(mapping[k])

		self.log(
			"[BATCH] Found "
			+ ", ".join(f"{k}: {len(v)}" for k, v in mapping.items())
		)
		return mapping


	def _next_kitchen_id(self) -> int:
		"""Return next available kitchen id by scanning BODex kitchen_data_*.json."""
		bodex_dir = os.path.join(KITCHEN_DIR, "bodex")
		pattern = os.path.join(bodex_dir, "kitchen_data_*.json")
		max_id = 0
		for path in glob.glob(pattern):
			base = os.path.basename(path)
			try:
				num_str = os.path.splitext(base)[0].split("_")[-1]
				max_id = max(max_id, int(num_str))
			except Exception:
				continue
		return max_id + 1

	def _clone_kitchen_assets(self, base_kitchen_num: int, new_kitchen_num: int, sub_num_str: str) -> None:
		"""Clone (USD + bodex json) from base_kitchen_num to new_kitchen_num for the given sub_num_str."""
		kitchen_dir = KITCHEN_DIR
		bodex_dir = os.path.join(KITCHEN_DIR, "bodex")

		base_num_str = self.format_num(base_kitchen_num)
		new_num_str = self.format_num(new_kitchen_num)

		# --- Clone BODex metadata ---
		base_json = os.path.join(bodex_dir, f"kitchen_data_{base_num_str}.json")
		new_json = os.path.join(bodex_dir, f"kitchen_data_{new_num_str}.json")
		if not os.path.exists(base_json):
			raise FileNotFoundError(f"Missing base kitchen metadata: {base_json}")
		with open(base_json, "r") as f:
			data = json.load(f)
		# annotate provenance (doesn't affect downstream)
		data["kitchen_num"] = int(new_kitchen_num)
		data["source_kitchen_num"] = int(base_kitchen_num)
		with open(new_json, "w") as f:
			json.dump(data, f, indent=4)

		# --- Clone USD ---
		candidates = [sub_num_str, "00", "01"]
		base_usd = None
		for s in candidates:
			p = os.path.join(kitchen_dir, f"kitchen_{base_num_str}_{s}.usd")
			if os.path.exists(p):
				base_usd = p
				break
		if base_usd is None:
			raise FileNotFoundError(
				f"Could not find any USD for base kitchen {base_num_str} with sub in {candidates}"
			)
		new_usd = os.path.join(kitchen_dir, f"kitchen_{new_num_str}_{sub_num_str}.usd")
		shutil.copyfile(base_usd, new_usd)

	def _ensure_kitchens_exist(self, kitchen_type: str, target_count: int, sub_num_str: str) -> None:
		"""
		Ensure at least target_count kitchens exist for kitchen_type.
		If not enough exist, create new ones by cloning existing kitchens of that type.
		"""
		kitchens_by_type = self._scan_kitchens_by_type()
		available = kitchens_by_type.get(kitchen_type, [])
		if len(available) >= target_count:
			return
		if not available:
			raise RuntimeError(f"No existing kitchens found for type '{kitchen_type}' to clone from.")

		need = target_count - len(available)
		self.log(f"[BATCH] Auto-generating {need} new '{kitchen_type}' kitchens by cloning.")
		next_id = self._next_kitchen_id()

		# choose bases with replacement
		bases = random.choices(available, k=need)
		for base_num in bases:
			new_num = next_id
			next_id += 1
			try:
				self._clone_kitchen_assets(base_num, new_num, sub_num_str=sub_num_str)
				self.log(f"[BATCH] Created kitchen {self.format_num(new_num)} (cloned from {self.format_num(base_num)})")
			except Exception as e:
				self.log(f"[BATCH] Failed to clone kitchen from {base_num} -> {new_num}: {e}")
				raise

	def _run_batch_predefined(self):
		"""
		Main entry for 'Generate batch' button.

		Assumes kitchens (and rotated USDs) already exist.
		For each selected kitchen number it will:
		  1) create env cfg
		  2) load reloadable template
		  3) save simulation + reloadable goal json.
		"""
		template_path = self.batch_template_var.get().strip()
		if not template_path or not os.path.exists(template_path):
			messagebox.showerror(
				"Batch error", "Please choose a valid reloadable JSON file first."
			)
			return

		try:
			with open(template_path, "r") as f:
				template_data = json.load(f)
			if "goals" not in template_data or not template_data["goals"]:
				messagebox.showerror(
					"Batch error",
					"Selected file does not contain 'goals' field.",
				)
				return
		except Exception as e:
			messagebox.showerror("Batch error", f"Could not read template:\n{e}")
			return

		total_requested = int(self.batch_total_var.get())
		sub_num_str = self.format_num(self.batch_subnum_var.get())
		task_name = self.batch_task_name_var.get().strip() or "BATCH_TASK"

		# collect counts per type
		per_type_counts = {}
		sum_counts = 0
		for internal_name, _ in self.batch_kitchen_types:
			c = int(self.batch_type_vars[internal_name].get())
			per_type_counts[internal_name] = c
			sum_counts += c

		if sum_counts == 0:
			messagebox.showerror(
				"Batch error",
				"All per-type counts are zero. Set at least one type > 0.",
			)
			return

		if sum_counts != total_requested:
			# simple sanity; we just enforce equality so it's predictable
			messagebox.showerror(
				"Batch error",
				f"Total per-type counts ({sum_counts}) != Total kitchens ({total_requested}).\n"
				f"Either fix numbers or click 'Divide equally'.",
			)
			return

		kitchens_by_type = self._scan_kitchens_by_type()

		# construct final list of kitchen numbers
		selected_kitchens = []
		for internal_name, _ in self.batch_kitchen_types:
			need = per_type_counts[internal_name]
			if need <= 0:
				continue

			available = kitchens_by_type.get(internal_name, [])
			if need > len(available):
				messagebox.showerror(
					"Batch error",
					f"Requested {need} kitchens of type '{internal_name}' "
					f"but only {len(available)} exist.",
				)
				return

			chosen = random.sample(available, need)
			selected_kitchens.extend(chosen)

		if not selected_kitchens:
			messagebox.showerror(
				"Batch error",
				"No kitchens selected after scanning BODex metadata.",
			)
			return

		random.shuffle(selected_kitchens)
		self.log(
			f"[BATCH] Will generate goals for {len(selected_kitchens)} kitchens "
			f"with sub-number {sub_num_str} using template {os.path.basename(template_path)}"
		)

		# loop over kitchens
		successes, failures = 0, 0
		for idx, kitchen_num in enumerate(selected_kitchens, start=1):
			self.log(
				f"[BATCH] ({idx}/{len(selected_kitchens)}) Kitchen {kitchen_num:02d}_{sub_num_str}"
			)
			ok = self._batch_generate_single(
				kitchen_num=kitchen_num,
				sub_num=int(sub_num_str),
				template_data=template_data,
				task_name=task_name,
			)
			if ok:
				successes += 1
			else:
				failures += 1

		messagebox.showinfo(
			"Batch finished",
			f"Batch generation complete.\n"
			f"Success: {successes}\n"
			f"Failed: {failures}",
		)

	def _batch_generate_single(
		self, kitchen_num: int, sub_num: int, template_data: dict, task_name: str
	) -> bool:
		"""
		Core logic for single (kitchen_num, sub_num).

		This reuses the same internal helpers as the GUI:
		  - _generate_env_config
		  - _process_goal_step
		but does not show per-step modal popups.
		"""
		kitchen_num_str = self.format_num(kitchen_num)
		sub_num_str = self.format_num(sub_num)

		usd_file_path = os.path.join(KITCHEN_DIR, f"kitchen_{kitchen_num_str}_{sub_num_str}.usd")
		if not os.path.exists(usd_file_path):
			self.log(f"[BATCH] USD missing: {usd_file_path}")
			return False

		# 1) open USD + generate env config
		if not omni.usd.get_context().open_stage(usd_file_path):
			self.log(f"[BATCH] Failed to open USD: {usd_file_path}")
			return False

		self.stage = omni.usd.get_context().get_stage()
		self._update_prim_list()

		self.kitchen_data = {
			"task_name": task_name,
			"kitchen_num": kitchen_num,
			"kitchen_type": "",
			"island_bound": [],
			"kitchen_sub_num": sub_num,
			"initial_pos_ranges": [],
			"initial_rot_yaw_range": [
				["yaw", math.radians(-10.0), math.radians(10.0)]
			],
			"goals": [],
		}

		try:
			self._generate_env_config(kitchen_num_str, sub_num_str)
		except Exception as e:
			self.log(f"[BATCH] Env config failed for {kitchen_num_str}_{sub_num_str}: {e}")
			return False

		# 2) build self.goal_steps from template_data (same logic as load_goals_from_file)
		try:
			goal_sequence = template_data["goals"][0]
		except Exception as e:
			self.log(f"[BATCH] Invalid template structure: {e}")
			return False

		self.goal_steps.clear()
		self.goal_listbox.delete(0, tk.END)

		for i, step_data in enumerate(goal_sequence):
			try:
				step = authored_step(step_data, i)
			except Exception as e:
				self.log(f"[BATCH] Skipping step {i+1}: {e}")
				continue
			if step is None:
				self.log(
					f"[BATCH] Skipping step {i+1}: unsupported format in template."
				)
				continue

			self.goal_steps.append(step)
			# keep listbox showing the latest generated kitchen (optional)
			self.goal_listbox.insert(
				tk.END, f"{len(self.goal_steps)}. [{step['action']}] {step['lang']}"
			)

		if not self.goal_steps:
			self.log("[BATCH] No valid goal steps built from template.")
			return False

		# 3) process each step into a v2 goal step (same idea as save_goal_file)
		try:
			steps = [self._process_goal_step(s) for s in self.goal_steps]
		except Exception as e:
			self.log(f"[BATCH] Error processing goal steps: {e}")
			return False

		# 4) save the goal file. One file: a v2 step carries the numbers AND the authoring intent,
		# which is the only reason the .reloadable twin ever existed.
		json_file, _reloadable = simvla_paths.goal_files(
			f"Isaac-Kitchen-v{kitchen_num_str}-{sub_num_str}"
		)

		try:
			write_v2(json_file, steps, meta=self.kitchen_data)
			self.log(f"[BATCH] Saved goal file: {json_file}")
			return True
		except Exception as e:
			self.log(f"[BATCH] Error saving goal file: {e}")
			return False


	def _load_subtask_cache(self):
		self.cache_file = "subtask_cache.json"
		if os.path.exists(self.cache_file):
			try:
				with open(self.cache_file, "r") as f: return json.load(f)
			except json.JSONDecodeError: return {}
		return {}

	def _save_subtask_cache(self):
		with open(self.cache_file, "w") as f: json.dump(self.subtask_cache, f, indent=2)
			
	def format_num(self, num):
		try: return f"{int(num):02d}"
		except (ValueError, TypeError): return str(num)
	def _update_prim_list(self):
		self.prim_listbox.delete(0, tk.END)
		self.all_prim_paths = []
		if not self.stage: return
		try:
			paths = [str(prim.GetPath()) for prim in self.stage.Traverse()]
			self.all_prim_paths = sorted(paths)
			self.search_var.set("") # Clear search bar
			self._filter_prim_list() # Populate listbox with full, unfiltered list
			self.log(f"Scene Inspector updated with {len(paths)} prims.")
		except Exception as e:
			self.log(f"Error updating prim list: {e}")
	def _copy_selected_prim_path(self, event=None):
		selected_indices = self.prim_listbox.curselection()
		if not selected_indices: return
		selected_path = self.prim_listbox.get(selected_indices[0])
		self.root.clipboard_clear()
		self.root.clipboard_append(selected_path)
		self.log(f"Copied to clipboard: {selected_path}")
	def _filter_prim_list(self, *args):
		query = self.search_var.get().lower()
		self.prim_listbox.delete(0, tk.END)
		if not query:
			for path in self.all_prim_paths:
				self.prim_listbox.insert(tk.END, path)
		else:
			filtered_paths = [path for path in self.all_prim_paths if query in path.lower()]
			for path in filtered_paths:
				self.prim_listbox.insert(tk.END, path)
	def process_kitchen(self):
		task_name, kitchen_num, sub_num = self.task_name_var.get(), self.kitchen_num_var.get(), self.sub_num_var.get()
		if not task_name: messagebox.showerror("Error", "Task Name cannot be empty."); return
		kitchen_num_str, sub_num_str = self.format_num(kitchen_num), self.format_num(sub_num)
		self.log(f"Processing Kitchen {kitchen_num_str}_{sub_num_str} for task: '{task_name}'")
		self.goal_steps.clear(); self.goal_listbox.delete(0, tk.END)
		self.kitchen_data = {"task_name": task_name, "kitchen_num": kitchen_num, "kitchen_type": "", "island_bound": [], "kitchen_sub_num": sub_num, "initial_pos_ranges": [], "initial_rot_yaw_range": [["yaw", math.radians(-10.0), math.radians(10.0)]], "goals": []}
		usd_file_path = os.path.join(KITCHEN_DIR, f"kitchen_{kitchen_num_str}_{sub_num_str}.usd")
		if not os.path.exists(usd_file_path): messagebox.showerror("File Not Found", f"Could not find USD file:\n{usd_file_path}"); return
		if omni.usd.get_context().open_stage(usd_file_path):
			self.stage = omni.usd.get_context().get_stage()
			self.log(f"Successfully opened {usd_file_path}")
			self._update_prim_list() # Update the scene inspector
			self._generate_env_config(kitchen_num_str, sub_num_str)
			for btn in [self.add_goal_btn, self.remove_goal_btn, self.save_goals_btn, self.load_goals_btn]: btn.config(state="normal")
		else:
			self.log(f"ERROR: Failed to open {usd_file_path}"); self.stage = None; self._update_prim_list(); messagebox.showerror("Error", f"Failed to open USD file:\n{usd_file_path}")
			
	def _generate_env_config(
		self, kitchen_num_str, sub_num_str, primary_obj=None, success_spec=None, retry_spec=None
	):
		try:
			first_level_set = {"/world/" + p.split("/")[-1] for p in [str(prim.GetPath()) for prim in self.stage.Traverse()] if p.startswith("/world/") and len(p.split("/")) == 3}
			first_level_list = sorted(list(first_level_set))
			json_path = os.path.join(KITCHEN_DIR, "bodex", f"kitchen_data_{kitchen_num_str}.json")
			with open(json_path, "r") as f: k_data = json.load(f)
			self.kitchen_data["kitchen_type"] = k_data["kitchen_type"]
			self.kitchen_data["room_shell"] = k_data.get("room_shell", False)
			self.free_squares.clear(); self.kitchen_data["initial_pos_ranges"].clear()
			self._nav_prev_park = None      # a new kitchen: the first nav starts from a spawn band
			world_bounds = None
			if k_data["kitchen_type"] == "island":
				usd_context = omni.usd.get_context()
				success = usd_context.open_stage(os.path.join(KITCHEN_DIR, f"kitchen_{kitchen_num_str}_{sub_num_str}.usd"))
				if success:
					self.stage = omni.usd.get_context().get_stage()
					island_prim = self.stage.GetPrimAtPath("/world/kitchen_island")
					xform = UsdGeom.Xformable(island_prim)
					time = Usd.TimeCode.Default()
					world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
					position: Gf.Vec3d = world_transform.ExtractTranslation()
					orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
					bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
					bbox3d = bbox_cache.ComputeWorldBound(island_prim)	# returns GfBBox3d
					bbox_range = bbox3d.ComputeAlignedRange()
					furniture_min_bound = bbox_range.GetMin()
					furniture_max_bound = bbox_range.GetMax()
					self.kitchen_data["island_bound"] = (furniture_min_bound[0], furniture_min_bound[1], furniture_max_bound[0], furniture_max_bound[1])
			if k_data["kitchen_type"] == "single_wall":
				self.log("Applying special case for 'single_wall' kitchen type.")
				self.kitchen_data["initial_pos_ranges"].append([["x", -0.3, 3.7], ["y", -1.5, -1]])
				world_bounds = (-0.5, -2, 4, 0.5)
				self.free_squares = [((-0.5, -1.5, 3.7, -1), (1.6, -1.25))]
			else:
				self.log("Calculating free space automatically.")
				# Lazy import, and the same one the block emitter below uses: env_cfg_emit pulls in
				# kitchen_build (scene_synthesizer/trimesh), which this GUI has no other reason to
				# load, and a lazy import is how every other consumer here reaches that module
				# (kitchen_wizard, scene_spec). It owns the ONE room-shell name set, so the obstacle
				# list and the emitted configs cannot disagree about what a wall is.
				import env_cfg_emit
				def bbox_probe(path):
					prim = self.stage.GetPrimAtPath(path)
					bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False).ComputeWorldBound(prim).ComputeAlignedRange()
					if bbox.IsEmpty():
						return None
					return (bbox.GetMin()[0], bbox.GetMin()[1], bbox.GetMax()[0], bbox.GetMax()[1])
				# The shell walls are excluded in there. They are the edge of the room, not furniture
				# to walk around: counted as obstacles they also drag world_bounds out to the walls,
				# and the corridor between the counter run and the wall is then wide enough to be
				# promoted into initial_pos_ranges -- a robot spawned facing the back of the counters.
				furniture = [box(*b) for b in env_cfg_emit.furniture_bounds(first_level_list, bbox_probe, self.log)]
				if not furniture: self.log("Warning: No furniture obstacles found!"); min_x, min_y, max_x, max_y = -5.0, -5.0, 5.0, 5.0
				else: min_x, min_y, max_x, max_y = min(o.bounds[0] for o in furniture), min(o.bounds[1] for o in furniture), max(o.bounds[2] for o in furniture), max(o.bounds[3] for o in furniture)
				world_bounds = (min_x, min_y, max_x, max_y)
				merged_spaces = merge_free_spaces(find_free_spaces_grid(world_bounds, furniture))
				robot_radius, safe = 0.23, 0.101
				for bounds, center in merged_spaces:
					if (bounds[2] - bounds[0] - safe) > robot_radius * 2 and (bounds[3] - bounds[1] - safe) > robot_radius * 2:
						self.kitchen_data["initial_pos_ranges"].append([["x", bounds[0] + robot_radius + 0.05, bounds[2] - robot_radius - 0.05], ["y", bounds[1] + robot_radius + 0.05, bounds[3] - robot_radius - 0.05]])
						self.free_squares.append((bounds, center))
			self.log(f"Found {len(self.free_squares)} free spaces for robot placement.")
			self._write_env_config_file(kitchen_num_str, sub_num_str, first_level_list, world_bounds,
			                            primary_obj=primary_obj, success_spec=success_spec,
			                            retry_spec=retry_spec)
		except Exception as e: self.log(f"ERROR during config generation: {e}"); messagebox.showerror("Processing Error", f"An error occurred: {e}")

	def _write_env_config_file(
		self, kitchen_num_str, sub_num_str, first_level_list, world_bounds,
		primary_obj=None, success_spec=None, retry_spec=None
	):
		source_file = f"{SIMVLA_REPO_ROOT}/scripts/simvla/kitchen_env_cfg_source.py"
		out_dir = os.environ.get(
			"SIMVLA_ENV_CFG_OUTPUT_DIR",
			f"{SIMVLA_REPO_ROOT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen",
		)
		output_file = os.path.join(out_dir, f"kitchen_{kitchen_num_str}_{sub_num_str}.py")
		os.makedirs(out_dir, exist_ok=True)
		kitchen_usd = f'''\nkitchen = AssetBaseCfg(\n\tprim_path="{{ENV_REGEX_NS}}/Kitchen",\n\tspawn=sim_utils.UsdFileCfg(usd_path=f"file:{{ISAACLAB_ASSETS_DATA_DIR}}/Kitchen/kitchen_{kitchen_num_str}_{sub_num_str}.usd"),\n)'''
		generated_blocks = []
		generated_blocks.append(kitchen_usd)
		# The stage read is all that has to stay here; which prims get a block, and what the
		# block says, moved to env_cfg_emit so it can be tested without Omniverse. It is what
		# skips the room shell's walls: they carry no RigidBodyAPI, and a RigidObjectCfg
		# pointed at one makes every task file from that kitchen raise "Failed to find a rigid
		# body when resolving ..." at env load. See that module's docstring.
		import env_cfg_emit
		def object_probe(path):
			prim = self.stage.GetPrimAtPath(path)
			xform = UsdGeom.Xformable(prim); world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
			pos, quat_obj = tuple(round(v, 4) for v in world_transform.ExtractTranslation()), world_transform.ExtractRotationQuat()
			quat = (round(quat_obj.GetReal(), 4), *[round(v, 4) for v in quat_obj.GetImaginary()])
			joint_attr = prim.GetAttribute("joint_names"); joint_names = joint_attr.Get() if joint_attr and joint_attr.HasAuthoredValue() else []
			return pos, quat, joint_names
		generated_blocks.extend(env_cfg_emit.object_blocks(first_level_list, object_probe, self.log))
		# A kitchen generated with ROOM_SHELL already carries its walls in the USD. Adding these
		# slabs too would box the room inside a second room -- and because world_bounds comes from
		# the furniture bounding box, which now includes those USD walls, the outer box would creep
		# further out on every regeneration.
		if world_bounds and not self.kitchen_data.get("room_shell"):
			generated_blocks.extend(make_walls_from_bounds(world_bounds))
		indented_blocks = ["\n".join("\t" + line for line in block.splitlines()) for block in generated_blocks]
		generated_code = "\n".join(indented_blocks)
		with open(source_file, "r") as f: config_text = f.read()
		# The template samples its lighting with random.uniform at MODULE level, and this file is
		# copied as text and never evaluated -- so those calls would land verbatim in the task file
		# and run at import in the training/eval process, off the unseeded global `random`: one
		# draw shared by every parallel env, a different draw on every process launch, and a replay
		# lit unlike the episode it replays. Baked into literals here instead, seeded from the
		# kitchen number (see env_cfg_emit.lighting_values), which is also what makes kitchen 42's
		# lighting reproducible. Same marker-block idiom as the Change block below.
		config_text = env_cfg_emit.substitute_lighting(config_text, kitchen_num_str)
		# Same defect, same fix, for the template's floor_material/wall_material random.choice()
		# calls: baked into literals seeded from the kitchen number (see
		# env_cfg_emit.materials_values), off its OWN random.Random instance so this draw can
		# never shift the lighting values substitute_lighting just baked in above, regardless of
		# which of the two calls runs first.
		config_text = env_cfg_emit.substitute_materials(config_text, kitchen_num_str)
		# scene_cam's pos/rot, same marker idiom again but a different defect. The template's
		# literals are kitchen 1400's camera -- a look-at aimed at the floor in front of ITS
		# fridge -- and every kitchen in the corpus was emitted with them, because nothing
		# between the template and the task file ever looked at the room being built. A kitchen
		# whose task is not on the counter run films the wrong half of it (kitchen 1218's
		# push-chair table and both its chairs are outside that frustum entirely, and the eye is
		# outside 1218's walls). substitute_scene_cam re-aims exactly the kitchens listed in
		# env_cfg_emit.SCENE_CAM_AIMS and re-emits the shipped literals for every other, so the
		# mug/fridge framing the corpus was judged against is unchanged to the digit.
		config_text = env_cfg_emit.substitute_scene_cam(config_text, kitchen_num_str)
		# lambda, not the string itself: re.sub reinterprets its replacement as a TEMPLATE, so any
		# backslash in generated source is re-read as a group reference or an escape — '\u' (which
		# json.dumps writes for every non-ASCII character, ensure_ascii being on) raises
		# `re.error: bad escape \u`, and a literal backslash silently collapses. A function
		# replacement is inserted verbatim.
		change_block = "# -------Change-------\n" + generated_code + "\n\t\t# -------Stop-------"
		new_text = re.sub(r"# -------Change-------.*?# -------Stop-------", lambda _m: change_block, config_text, flags=re.DOTALL)
		new_text = self._bind_env_config_objects(new_text, first_level_list, primary_obj)
		# Only when a template supplied them. The tkinter GUI path (no template) calls this with
		# None and must keep emitting the block verbatim, so existing configs regenerate unchanged.
		if success_spec is not None and retry_spec is not None:
			import task_emit
			term_block = (
				"# ----Terminations----\n"
				+ "\n".join("\t" + l for l in
				            task_emit.terminations_block(success_spec, retry_spec).splitlines())
				+ "\n\t\t# ----StopTerminations----")
			new_text = re.sub(
				r"# ----Terminations----.*?# ----StopTerminations----",
				lambda _m: term_block,
				new_text, flags=re.DOTALL)
		with open(output_file, "w") as f: f.write(new_text)
		self.log(f"✅ Created environment config: {output_file}")

	def _bind_env_config_objects(self, config_text, first_level_list, primary_obj):
		"""Rebind the two object roles the env-config TEMPLATE hardcodes — the manipulated object
		(SceneEntityCfg("bowl0"): its reset, physics/mass/color/com randomizers, and the success +
		OOB terminations) and a distractor (SceneEntityCfg("mug0"): domain randomization only) — to
		the prims THIS scene actually contains. The template names them bowl0/mug0 because that was
		the original corpus's scene; a kitchen whose manipulated object is, say, a cup would otherwise
		emit an env config whose success term and randomizers point at a prim the scene lacks —
		KeyError at env load, or a success that can never fire. When the scene already has bowl0/mug0
		the substitution is a no-op, so the original corpus regenerates byte-for-byte."""
		try:
			import scene_spec
			manip_types = set(scene_spec.OBJECT_TYPES)
		except Exception:
			return config_text
		objs = [p.rsplit("/", 1)[-1] for p in first_level_list
		        if p.rsplit("/", 1)[-1].rstrip("0123456789") in manip_types]
		if not objs:
			return config_text
		# primary (bowl0 role): the bound target if it is in the scene; else keep bowl0 if present;
		# else the first manipulable object — never a name the scene does not contain.
		if primary_obj in objs:      primary = primary_obj
		elif "bowl0" in objs:        primary = "bowl0"
		else:                        primary = sorted(objs)[0]
		# secondary (mug0 role): a DIFFERENT manipulable object if the scene has one (preferring mug0);
		# else the primary itself, so the mug0 randomizers re-randomize the primary rather than
		# reference a missing prim.
		others = [o for o in objs if o != primary]
		if "mug0" in others:         secondary = "mug0"
		elif others:                 secondary = sorted(others)[0]
		else:                        secondary = primary
		config_text = config_text.replace('SceneEntityCfg("bowl0")', f'SceneEntityCfg("{primary}")')
		config_text = config_text.replace('SceneEntityCfg("mug0")', f'SceneEntityCfg("{secondary}")')
		return config_text
	
	def add_subtask(self):
		dialog = SubtaskDialog(self.root, "Add Subtask", self.subtask_cache)
		if dialog.result:
			step = dialog.result          # {"lang", "skill", "action", "params"} — the skill is named
			self.goal_steps.append(step)
			self.goal_listbox.insert(tk.END, f"{len(self.goal_steps)}. [{step['action']}] {step['lang']}")
			if step["lang"] not in self.subtask_cache:
				self.subtask_cache[step["lang"]] = step["action"]; self._save_subtask_cache()

	def remove_subtask(self):
		selected_indices = self.goal_listbox.curselection()
		if not selected_indices: return
		for index in sorted(selected_indices, reverse=True): self.goal_listbox.delete(index); del self.goal_steps[index]
			
	def load_goals_from_file(self):
		goal_dir = str(simvla_paths.goals_dir())
		filepath = filedialog.askopenfilename(initialdir=goal_dir, title="Select Goal File", filetypes=(("JSON files", "*.json*"), ("all files", "*.*")))
		if not filepath: return
		try:
			with open(filepath, 'r') as f: loaded_data = json.load(f)
			if "goals" not in loaded_data or not loaded_data["goals"]: messagebox.showwarning("Load Warning", "No 'goals' found in file."); return
			self.goal_steps.clear(); self.goal_listbox.delete(0, tk.END)
			goal_sequence = loaded_data["goals"][0]
			for i, step_data in enumerate(goal_sequence):
				step = authored_step(step_data, i)
				if step is None:
					messagebox.showerror("Load Error", "Cannot load file. It uses an old, processed-only format."); self.goal_steps.clear(); self.goal_listbox.delete(0, tk.END); return
				self.goal_steps.append(step)
				self.goal_listbox.insert(tk.END, f"{len(self.goal_steps)}. [{step['action']}] {step['lang']}")
			self.log(f"Successfully loaded {len(self.goal_steps)} goal steps from {os.path.basename(filepath)}")
		except Exception as e: self.log(f"Failed to load file: {e}"); messagebox.showerror("Load Error", f"An error occurred:\n{e}")

	def save_goal_file(self):
		if not self.goal_steps: messagebox.showwarning("Warning", "No goal steps defined."); return
		steps = []
		for step in self.goal_steps:
			try:
				processed = self._process_goal_step(step)
				# A nav that had to route around furniture returns via-points; they are written as
				# their own nav steps ahead of the park, exactly as task_emit writes them.
				from goal_format import expand_route
				steps.extend(expand_route(processed))
				processed = steps[-1]
				# An island kitchen's nav steps are emitted twice — the robot circles the island, so
				# it needs two goals to get there. Unchanged; only the payload shape moved.
				if processed.action in ["N", "N_s"] and self.kitchen_data.get("kitchen_type") == "island":
					steps.append(processed)
			except Exception as e: self.log(f"Error processing step '{step['lang']}': {e}"); messagebox.showerror("Processing Error", f"Could not process step '{step['lang']}':\n{e}"); return
		kitchen_num_str, sub_num_str = self.format_num(self.kitchen_num_var.get()), self.format_num(self.sub_num_var.get())
		json_file, _reloadable = simvla_paths.goal_files(f"Isaac-Kitchen-v{kitchen_num_str}-{sub_num_str}")
		try:
			# One file. <task>.reloadable.json existed because the numbers and the authoring intent
			# were stored apart; a v2 step carries both, so there is nothing left for a twin to hold.
			write_v2(json_file, steps, meta=self.kitchen_data)
			self.log(f"SUCCESS: Saved goal file to {json_file}")
			messagebox.showinfo("Success", f"Saved goal file to:\n{json_file}")
		except Exception as e: self.log(f"ERROR saving file: {e}"); messagebox.showerror("Save Error", f"Could not save file:\n{e}")


	def _process_goal_step(self, step) -> Step:
		"""One authored step -> one v2 goal step. Dispatches on the skill id, not on (action, usage).

		A skill defines plan() or resolve(), never both, so there are exactly two outcomes here:
		either the goal is computable now, or it is not and the file says `null`. There is no third
		branch in which a skill hands back a raw float and hopes the executor recognises it — which
		is what the refrigerator skill did.
		"""
		spec = REGISTRY.get(step["skill"])
		if spec is None:
			raise ValueError(f"No skill registered with id {step['skill']!r}")

		action = step["action"]
		if action not in spec.actions:
			raise ValueError(f"{spec.id!r} does not drive action {action!r}; it declares "
			                 f"{list(spec.actions)}.")

		params = params_for_skill(spec, step.get("params", {}))

		# `goal: null` means resolve-at-runtime — stated, not encoded as a magic float in a
		# quaternion slot. NOTE arm.place is NOT this: it plans a real, bbox-derived xyz which IS
		# the IK target, and the executor refines only its quaternion.
		goal = None if spec.is_runtime else spec.cls().plan(self, action, params)

		return Step(skill=spec.id, action=action, params=params, goal=goal,
		            language=step.get("lang", ""))


# ===============================================================
# CUSTOM DIALOG FOR ADDING SUBTASKS
# ===============================================================
class _SubtaskDialogMixin:
	"""The dialog is a view of the registry. It hardcodes no action code and no skill name.

	It used to hold its own copy of both: `actions = ["N", ..., "G_b"]` and, per action, a list of
	free-text `usage` strings. Two lists of the same thing drift, and these did — G_b was offered
	here and unknown to the executor. Everything the dialog offers now comes from gui_actions() /
	gui_skills() / gui_params(), so a skill it can author is by construction a skill that exists.
	"""

	def __init__(self, parent, title, cache):
		self.cache = cache; self.result = None; super().__init__(parent, title)

	def body(self, master):
		self.action_var, self.lang_var, self.skill_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
		self.label_to_id = {}
		ttk.Label(master, text="Subtask Language:").pack(anchor="w"); self.lang_combo = ttk.Combobox(master, textvariable=self.lang_var, values=list(self.cache.keys())); self.lang_combo.pack(fill="x", expand=True, padx=5, pady=2); self.lang_combo.bind("<<ComboboxSelected>>", self.on_lang_select)
		ttk.Label(master, text="Action Primitive:").pack(anchor="w", pady=(10,0)); self.action_menu = ttk.OptionMenu(master, self.action_var, "Select Action", *gui_actions(), command=self.on_action_select); self.action_menu.pack(fill="x", padx=5, pady=2)
		self.skill_frame = ttk.Frame(master); self.skill_frame.pack(fill="x", expand=True, padx=5)
		self.param_frame = ttk.Frame(master); self.param_frame.pack(fill="x", expand=True, pady=10, padx=5); self.params = {}
		return self.lang_combo

	def on_lang_select(self, event=None):
		lang = self.lang_var.get()
		if lang in self.cache: self.action_var.set(self.cache[lang]); self.on_action_select(self.cache[lang])

	def on_action_select(self, selected_action):
		"""Offer the skills this action can drive — the registry's answer, not a written-out list."""
		for widget in self.skill_frame.winfo_children(): widget.destroy()
		for widget in self.param_frame.winfo_children(): widget.destroy()
		self.params = {}

		offered = gui_skills(selected_action)
		self.label_to_id = dict(offered)
		if not offered:
			ttk.Label(self.skill_frame, text=f"No skill drives {selected_action!r}.").pack(anchor="w")
			return

		labels = [label for label, _ in offered]
		ttk.Label(self.skill_frame, text="Skill:").pack(anchor="w")
		self.skill_var.set(labels[0])
		ttk.OptionMenu(self.skill_frame, self.skill_var, labels[0], *labels,
		               command=self.on_skill_select).pack(fill="x")
		self.on_skill_select(labels[0])

	def on_skill_select(self, label):
		"""Render exactly the params the chosen skill declared — nothing hardcoded."""
		for widget in self.param_frame.winfo_children(): widget.destroy()
		self.params = {}

		for widget in gui_params(self.label_to_id[label]):
			ttk.Label(self.param_frame, text=f"{widget['name']}:").pack(anchor="w")
			if widget["kind"] == "choice":
				var = tk.StringVar(value=widget["default"])
				ttk.OptionMenu(self.param_frame, var, widget["default"], *widget["options"]).pack(fill="x")
			elif widget["kind"] == "bool":
				var = tk.BooleanVar(value=widget["default"])
				ttk.Checkbutton(self.param_frame, variable=var).pack(anchor="w")
			else:                                    # float / prim_path: a typed text entry
				var = tk.StringVar(value=str(widget["default"]))
				ttk.Entry(self.param_frame, textvariable=var).pack(fill="x")
			self.params[widget["name"]] = var

	def apply(self):
		lang, action, label = self.lang_var.get(), self.action_var.get(), self.skill_var.get()
		if not lang or not action or action == "Select Action" or label not in self.label_to_id:
			messagebox.showwarning("Input Error", "Please provide a subtask language, an action and a skill.", parent=self); return
		skill_id = self.label_to_id[label]
		# params_for_skill casts what the Tk vars hand back: an Entry returns the string "0.4", and
		# a back-off of "0.4" is not a back-off of 0.4 by the time the resolver multiplies by it.
		self.result = {
			"lang": lang,
			"skill": skill_id,
			"action": action,
			"params": params_for_skill(REGISTRY[skill_id], {n: v.get() for n, v in self.params.items()}),
		}

def SubtaskDialog(parent, title, cache):
	"""Create a real Tk dialog without importing Tk during headless goal authoring."""
	_load_gui()
	class TkSubtaskDialog(_SubtaskDialogMixin, simpledialog.Dialog):
		pass
	return TkSubtaskDialog(parent, title, cache)


# =============================================================================
# The authoring-side plan() bodies
# =============================================================================
# Six skills' geometry cannot be computed without the USD stage (bbox caches, xform caches, a physx
# raycast) or, for arm.grasp, without the Tk thumbnail chooser and scipy. skills.py declares them —
# so the executor sees the same fifteen-skill REGISTRY, and SKILL_ID()'s sorted ints mean the same
# thing on both sides — and each class's plan() is `_authored(id)`, which looks the body up here.
#
# The bodies below are the ones that used to sit inside those classes, moved unchanged.

#: The screen under-reports: door_blocked_deg tracks the door's edge POINT and misses the
#: handle's protrusion, so the true minimum is at or above what min_past_for returns.
#: Measured gap on kitchens 1400 and 1410: 0.03 m. 0.05 covers it with room to spare and
#: still leaves the handle in reach (0.32 m offset -> 0.49 m to the handle, against 0.555).
PARK_MARGIN_M = 0.05


def fridge_park_past_m(site, target_sweep_deg: float = 60.0) -> float:
	"""How far past the handle, away from the hinge, the base must park.

	DERIVED, NOT CONFIGURED. This used to be SIMVLA_FRIDGE_PARK_PAST_M defaulting to "0.0", read
	as `if _past:` -- so the shipped default skipped the offset entirely and parked the base at the
	handle's own lateral coordinate, INSIDE the door's swept arc. On kitchen 1400, at the true
	parked pose (base y = handle_bbox_min - 0.35 = -0.7435, matching the goal file the GPU
	consumes), the door's free edge reaches the base at 13.0 degrees (OBB/SAT) -- consistent with
	the honest measured GPU peaks over 343 episodes, 13.1 and 20.8. Two sessions of
	arc/diagonal/step-size tuning were measured against a door that was mechanically jammed.

	DERIVED VALUE = fridge_siting.min_past_for(..., margin_m=PARK_MARGIN_M). min_past_for is built
	on door_blocked_deg, a screen that tracks the door's edge as a single POINT and so misses the
	handle protruding toward the robot -- it is a LOWER BOUND, not the true minimum (see that
	function's own docstring). Measured on kitchens 1400 and 1410, the screen under-reports by
	~0.03 m against the OBB/SAT authority (0.27 vs 0.30 on kitchen 1400). PARK_MARGIN_M=0.05
	covers that gap with room to spare, without pushing the handle out of Anubis's 0.555 m reach.
	Returning the bare screen value parks the robot 0.03 m inside the jam -- silently reproducing
	the exact bug this function exists to remove.

	THE MARGIN IS PASSED INTO min_past_for, NOT ADDED HERE AFTERWARD. min_past_for's own reach
	check used to run on its raw lower bound, before any margin; a caller that added a margin to
	the returned value afterward -- as this function used to -- could silently return an offset
	whose reach exceeds site.reach_m, since nothing re-checked it. min_past_for now takes
	margin_m, adds it before its ONE reach check, and raises SitingError on the value it is about
	to return if THAT is out of reach. There is exactly one place this invariant is enforced, and
	it is enforced on the number that actually goes out the door -- see min_past_for's docstring
	for the reproduction that found the gap (standoff_m=0.465: raw lower bound 0.260 m, in reach
	at 0.533 m; +0.05 m margin -> 0.310 m, reach 0.559 m -- 4 mm past the 0.555 m limit, and the
	old code never raised).

	The env var survives as an OVERRIDE, and BYPASSES THE MARGIN AND THE RAISE ENTIRELY: an
	explicit value means exactly that value, not that value plus a hidden pad, and not subject to
	a reach check this function never asked for. An explicit "0" is honoured as zero -- the A/B
	control group needs the old behaviour deliberately -- which is exactly what `if _past:` could
	not express.

	CALLED AT AUTHOR TIME ONLY. plan_nav_to_prim runs inside task_emit.py when goal files are
	written, and the value this returns feeds a pose that is baked into the goal JSON as a literal
	[x, y, yaw]. Nothing here runs again when a GPU job replays an existing goal file, and nothing
	about SIMVLA_FRIDGE_PARK_PAST_M set at run time (in the eval/rollout process) can retroactively
	change a pose that a separate, earlier authoring process already baked in. An earlier experiment
	this session varied the env var at run time and was, unknowingly, comparing a configuration
	against itself.
	"""
	import fridge_siting

	raw = _simvla_os.environ.get("SIMVLA_FRIDGE_PARK_PAST_M")
	if raw is not None and raw.strip() != "":
		return float(raw)
	return fridge_siting.min_past_for(site, target_sweep_deg, margin_m=PARK_MARGIN_M)


@register_planner("nav.to_prim")
def plan_nav_to_prim(app: "GoalGeneratorApp", action: str, params: dict):
	"""
	Navigation skill for N / N_s:
	  - Computes a base pose in front of a target prim
	  - Sets app.N_dir (N/E/S/W) based on furniture orientation or free-space search
	"""
	prim_path = params.get("prim_path")
	which_arm = params.get("which_arm")

	if not prim_path:
		raise ValueError("Prim path required for navigation")

	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")

	xform = UsdGeom.Xformable(prim)
	world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
	position = world_transform.ExtractTranslation()
	orientation_quat = world_transform.ExtractRotationQuat()

	bbox_range = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
	).ComputeWorldBound(prim).ComputeAlignedRange()

	# bias for which arm will be used after navigation
	#
	# PER ROBOT, because the arm's own offset from the base centreline is. This was a hardcoded
	# 0.05 m for every robot. On the AI Worker the right EEF rests about 0.144 m right of centre,
	# so 0.05 leaves the target 0.094 m ACROSS the body and the arm falls short of it -- every
	# failed-reach delta in that campaign is negative in Y, by 0.08 to 0.45 m, with the goal near
	# base Y=0 and the hand near Y=-0.14. Anubis and RB-Y1 keep 0.05 exactly, so nothing they do
	# changes.
	# SIMVLA_ROBOT first: goals are AUTHORED with app.robot unset (the rby1 goal files are
	# copies made after the fact), so the env var the emit pass already keys the grasp
	# planner on is the only robot identity available here.
	_robot = os.environ.get("SIMVLA_ROBOT") or getattr(app, "robot", "anubis")
	try:
		import nav_tuning
		_prof = nav_tuning.profile_for(_robot)
		_bias = _prof.arm_lateral_bias
		_furn_extra = _prof.furniture_extra_standoff_m
		_obj_extra = _prof.object_extra_standoff_m
	except Exception:                                       # noqa: BLE001
		_bias = 0.05
		_furn_extra = 0.0
		_obj_extra = 0.0
	arm_bias = _bias if which_arm == "Left" else -_bias if which_arm == "Right" else 0.0
	N_pos = torch.zeros(3)
	safety = float(params.get("safety", 0.12))	# default 0.12
	wheel_r = 0.23
	# DOES THE BASE FIT WHERE THIS PARKS IT? Nothing below asks. On kitchen 1570 the free-space walk
	# chose the one side of a corner cabinet that is a refrigerator, and the pose it authored sat
	# 0.38 m inside the fridge -- three identical timed-out runs. nav_clearance checks the pose
	# against the measured base box and every floor-level box in the room, re-sites a colliding
	# one to the nearest clear side (the search the fridge/cabinet tasks already do), and checks
	# the straight drive from the spawn band. A clear pose comes back as the same floats.
	# SIMVLA_NAV_CLEARANCE=0 turns the whole thing off.
	_clearance_on = os.environ.get("SIMVLA_NAV_CLEARANCE", "1").strip().lower() not in ("0", "false", "off")

	# -------------------------------------------------------
	# CASE 1: Furniture (z ~ 0) → use orientation snap logic
	# -------------------------------------------------------
	if position[2] < 0.01:
		min_b, max_b = bbox_range.GetMin(), bbox_range.GetMax()
		quat_wxyz = [orientation_quat.GetReal(), *orientation_quat.GetImaginary()]

		# A FRIDGE HANDLE IS NOT AT THE FRIDGE'S CENTRE, so parking in front of the centre puts
		# the handle out of reach. This used to be spelled `refrigerator_trans = 0.7` -- the door's
		# width -- added or subtracted per direction below.
		#
		# THAT SIGN IS AN ASSUMPTION ABOUT WHICH SIDE THE HINGE IS ON, and it is not true of every
		# fridge. Measured: on kitchen 1300 the hinge is at higher y than the handle and the fixed
		# sign lands the base 1.3 cm from the handle -- fine. On kitchen 1400 the hinge is on the
		# OTHER side, the same sign moves the base the wrong way, and it parks at x = -1.155:
		# 1.541 m from the handle (Anubis reaches 0.555 m) and 0.35 m THROUGH the room's west wall.
		#
		# `prim` here IS the handle, so its own bbox centre is the lateral coordinate to park at.
		# No door width, no sign, no assumption -- and on kitchen 1300 it reproduces the old
		# behaviour to within the 1.3 cm that one was already off by.
		handle_c = None
		if 'refrigerator' in prim_path:
			handle_c = ((min_b[0] + max_b[0]) * 0.5, (min_b[1] + max_b[1]) * 0.5)
			# PARK PAST THE HANDLE, NOT IN FRONT OF IT.
			#
			# A fridge door swings TOWARD the robot. Parked at the handle's own lateral
			# coordinate the base sits inside the swept arc, and the door stops against it --
			# measured (GPU, 343 episodes): honest peak 13.1 deg. This offset used to be
			# SIMVLA_FRIDGE_PARK_PAST_M, defaulting to "0.0" and read as `if _past:` -- so the
			# shipped default skipped this whole block and parked the base inside the arc. It is
			# now DERIVED (fridge_park_past_m, above); the env var survives only as an explicit
			# override, and an explicit "0" is honoured as zero.
			_door = prim.GetParent()
			_dr = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
									useExtentsHint=False).ComputeWorldBound(_door).ComputeAlignedRange()
			_dlo, _dhi = _dr.GetMin(), _dr.GetMax()
			# the door runs along its long horizontal axis; the hinge is the end FARTHEST
			# from the handle, so "away from the hinge" is the direction handle - hinge.
			if (_dhi[0] - _dlo[0]) >= (_dhi[1] - _dlo[1]):
				_ends = ((_dlo[0], handle_c[1]), (_dhi[0], handle_c[1]))
			else:
				_ends = ((handle_c[0], _dlo[1]), (handle_c[0], _dhi[1]))
			_hinge = max(_ends, key=lambda e: (e[0]-handle_c[0])**2 + (e[1]-handle_c[1])**2)
			_vx, _vy = handle_c[0]-_hinge[0], handle_c[1]-_hinge[1]
			_n = math.hypot(_vx, _vy) or 1.0

			# face_normal: read off the SAME canonical quats the N/S/W/E branch below matches, not
			# re-derived from bbox geometry, which does not carry which way the door opens.
			if np.allclose(quat_wxyz, [-0.7071, 0, 0, 0.7071], atol=1e-3):
				_face_normal = (-1.0, 0.0)
			elif np.allclose(quat_wxyz, [0.7071, 0, 0, 0.7071], atol=1e-3):
				_face_normal = (1.0, 0.0)
			elif np.allclose(quat_wxyz, [1.0, 0, 0, 0], atol=1e-3):
				_face_normal = (0.0, -1.0)
			elif np.allclose(quat_wxyz, [0.0, 0, 0, 1.0], atol=1e-3):
				_face_normal = (0.0, 1.0)
			else:
				_face_normal = None		# unrecognised snap; the branch below already warns

			if _face_normal is not None:
				import fridge_siting
				# Site.standoff_m is measured from the handle's CENTRE along the face normal, but
				# the N/S/W/E branch below parks at the handle's bbox MIN/MAX edge (min_b/max_b)
				# minus safety minus wheel_r -- NOT the centre. The two references differ by the
				# handle's own half-depth along the face-normal axis: on kitchen 1400, handle bbox
				# y = [-0.3935, -0.3435], so standoff_m = safety + wheel_r + half_depth =
				# 0.12 + 0.23 + 0.025 = 0.375, not safety + wheel_r = 0.35. Getting this wrong is a
				# 25 mm error against a park window only ~120 mm wide.
				_axis = 0 if _face_normal[0] != 0.0 else 1
				_handle_half_depth = (max_b[_axis] - min_b[_axis]) * 0.5
				# base_radius_m is how big the robot IS (what the door collides with), NOT
				# wheel_r, which is the nav standoff convention -- how far from the handle bbox
				# nav.to_prim chooses to park. Conflating them authors an offset that does not
				# clear the door: on kitchen 1400 at standoff 0.375, wheel_r (0.23) gives
				# min_past_for(60) = 0.14, still inside the jam (OBB/SAT: free only from >= 0.30),
				# while the true footprint radius (0.340) gives 0.27.
				_site = fridge_siting.Site(
					hinge_xy=(_hinge[0], _hinge[1]), handle_xy=(handle_c[0], handle_c[1]),
					radius_m=_n, face_normal=_face_normal,
					standoff_m=safety + wheel_r + _handle_half_depth,
					base_radius_m=fridge_siting.ANUBIS_BASE_RADIUS_M, reach_m=0.555)
				_past = fridge_park_past_m(_site)
				print(f"[nav] fridge park offset {_past:.3f} m past the handle "
					  f"(door free to >{fridge_siting.chord_diag_deg(120.0):.0f} deg from there)")
				handle_c = (handle_c[0] + _past*_vx/_n, handle_c[1] + _past*_vy/_n)

		# PER-ROBOT extra gap for PLAIN furniture parks (tables, counters) only. Handle parks
		# keep the fridge_siting-derived distance -- their park window is ~120 mm wide and
		# reach-critical -- and the object-on-furniture branch below keeps the grasp standoff
		# that test_object_to_plate pins against RB-Y1's arm-reach regression.
		if handle_c is None:
			wheel_r = wheel_r + _furn_extra

		if np.allclose(quat_wxyz, [-0.7071, 0, 0, 0.7071], atol=1e-3):
			app.N_dir = "N"
			N_pos[:] = torch.tensor(
				[min_b[0] - safety - wheel_r,
				 (handle_c[1] if handle_c else position[1]) - arm_bias, math.radians(0.0)]
			)
		elif np.allclose(quat_wxyz, [0.7071, 0, 0, 0.7071], atol=1e-3):
			app.N_dir = "S"
			N_pos[:] = torch.tensor(
				[max_b[0] + safety + wheel_r,
				 (handle_c[1] if handle_c else position[1]) + arm_bias, math.radians(180.0)]
			)
		elif np.allclose(quat_wxyz, [1.0, 0, 0, 0], atol=1e-3):
			app.N_dir = "W"
			N_pos[:] = torch.tensor(
				[(handle_c[0] if handle_c else position[0]) + arm_bias,
				 min_b[1] - safety - wheel_r, math.radians(90.0)]
			)
		elif np.allclose(quat_wxyz, [0.0, 0, 0, 1.0], atol=1e-3):
			app.N_dir = "E"
			N_pos[:] = torch.tensor(
				[(handle_c[0] if handle_c else position[0]) - arm_bias,   # facing -y, LEFT is +x
				 max_b[1] + safety + wheel_r, math.radians(-90.0)]
			)
		else:
			# fallback – you can tweak this if needed
			app.log("Warning: furniture orientation did not match canonical snaps.")
		if _clearance_on:
			# A handle park cannot change side -- the base must face the door -- so this only
			# REPORTS. fridge_siting / cabinet_kitchen own the handle parks' clearance.
			_nav_clearance_report(app, N_pos, _robot)
	# -------------------------------------------------------
	# CASE 2: Object on furniture → raycast + free space grid
	# -------------------------------------------------------
	else:
		physx_interface = omni.physx.get_physx_interface()
		physx_interface.start_simulation()
		scene_query_interface = omni.physx.get_physx_scene_query_interface()

		object_pos = position
		ray_origin = object_pos + Gf.Vec3d(0, 0, 0.4)
		ray_distance = 1.0
		directions_to_check = ["N", "E", "S", "W"]

		furniture_prim = None
		_target_is_fixture = False
		for dir_str in directions_to_check:
			if dir_str == "E":
				ray_direction = Gf.Vec3d(0, 1, -1)
			elif dir_str == "S":
				ray_direction = Gf.Vec3d(1, 0, -1)
			elif dir_str == "W":
				ray_direction = Gf.Vec3d(0, -1, -1)
			elif dir_str == "N":
				ray_direction = Gf.Vec3d(-1, 0, -1)

			hit = scene_query_interface.raycast_closest(
				ray_origin, ray_direction, ray_distance
			)
			if hit["hit"] is False:
				continue
			if prim_path.split("/")[-1] in hit["rigidBody"]:
				continue

			if hit["hit"]:
				hit_path = hit["rigidBody"]
				if hit_path and hit_path.startswith("/world/"):
					top_level_path = "/world/" + hit_path.split("/")[2]
					furniture_prim = app.stage.GetPrimAtPath(top_level_path)
					app.log(f"Raycast hit and identified furniture: {top_level_path}")
					break
		if furniture_prim is None:
			print(prim)
			app.log("Raycast failed to find supporting furniture.")

			# THE TARGET MAY ITSELF BE THE FURNITURE. This branch is "object on furniture", and the
			# raycast looks DOWN from the object for whatever supports it -- skipping any hit whose
			# rigid body carries the target's own name, so a target that IS a top-level fixture
			# skips itself and finds nothing. nav.to_prim on /world/table therefore fell through to
			# the hardcoded island below, which does not exist in an l_shaped kitchen, and every
			# sub-kitchen was SKIPPED with "Could not find furniture prim at path:
			# /world/kitchen_island" -- an error naming a prim the template never mentioned.
			#
			# When the target is a top-level /world/<name> prim, the furniture to stand off from is
			# that prim. Checked BEFORE the island fallback so kitchens that do have an island are
			# unaffected.
			_self_top = "/world/" + prim_path.strip("/").split("/")[1] if prim_path.count("/") >= 2 \
				else prim_path
			_self_prim = app.stage.GetPrimAtPath(_self_top)
			if _self_prim and _self_top == prim_path.rstrip("/"):
				app.log(f"Target is itself a top-level fixture; standing off from {_self_top}")
				furniture_prim = _self_prim
				# PLAIN furniture approach (drive to the table itself, not to an object ON it):
				# the per-robot extra gap applies HERE. The raycast path above is the grasp
				# approach, whose distance test_object_to_plate pins -- it stays untouched.
				_target_is_fixture = True
			else:
				user_prim_path = "/world/kitchen_island"
				furniture_prim = app.stage.GetPrimAtPath(user_prim_path)

				if not furniture_prim:
					raise ValueError(
						f"Could not find furniture prim at path: {user_prim_path} "
						f"(raycast from {prim_path} found no support, and {prim_path} is not "
						f"itself a top-level fixture)"
					)

#		 if furniture_prim is None:
#			 print(prim)
#			 app.log("Raycast failed to find supporting furniture.")
#			 user_prim_path = simpledialog.askstring(
#				 "Input Required",
#				 "Raycast failed. Please provide the prim path of the furniture the object is on:",
#				 parent=app.root,
#			 )
#			 if not user_prim_path:
#				 raise ValueError("Supporting furniture prim path is required.")
#			 furniture_prim = app.stage.GetPrimAtPath(user_prim_path)
#			 if not furniture_prim:
#				 raise ValueError(
#					 f"Could not find furniture prim at path: {user_prim_path}"
#				 )

		f_xform = UsdGeom.Xformable(furniture_prim)
		f_world_transform = f_xform.ComputeLocalToWorldTransform(
			Usd.TimeCode.Default()
		)
		f_position = f_world_transform.ExtractTranslation()
		f_bbox_range = UsdGeom.BBoxCache(
			Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
		).ComputeWorldBound(furniture_prim).ComputeAlignedRange()
		f_min_bound, f_max_bound = f_bbox_range.GetMin(), f_bbox_range.GetMax()
		object_pos_2d = (object_pos[0], object_pos[1])
		dir_point = find_first_free_direction(
			object_pos_2d, app.free_squares, step=0.03, max_steps=100
		)
		if not dir_point and not _clearance_on:
			raise RuntimeError(
				f"Could not find any free space near object {prim_path}"
			)

		# PER-ROBOT extra gap. Fixture parks (drive to the table itself) and object-on-furniture
		# parks (the grasp approach) are DIFFERENT reach budgets, so each has its own field --
		# see nav_tuning.NavProfile.object_extra_standoff_m for why the grasp park needed one.
		if _target_is_fixture:
			wheel_r = wheel_r + _furn_extra
		else:
			wheel_r = wheel_r + _obj_extra

		if dir_point:
			app.N_dir, _ = dir_point
		else:
			app.log(f"Could not find any free space near object {prim_path}; nav_clearance will "
					f"search every side of {furniture_prim.GetPath()}")
			app.N_dir = None
		if app.N_dir == "W":
			N_pos[0] = object_pos_2d[0] + arm_bias
			N_pos[1] = f_min_bound[1] - safety - wheel_r
			N_pos[2] = math.radians(90.0)
		elif app.N_dir == "E":
			# facing -y, the base's LEFT is +x: parking "arm_bias to the left" is x - arm_bias.
			# (This and the N branch had `+ arm_bias`, which put a right arm's target on the
			# robot's LEFT; nav_clearance.lateral_for is the one statement of the convention.)
			N_pos[0] = object_pos_2d[0] - arm_bias
			N_pos[1] = f_max_bound[1] + safety + wheel_r
			N_pos[2] = math.radians(-90.0)
		elif app.N_dir == "S":
			N_pos[1] = object_pos_2d[1] + arm_bias
			N_pos[0] = f_max_bound[0] + safety + wheel_r
			N_pos[2] = math.radians(180.0)
		elif app.N_dir == "N":
			N_pos[1] = object_pos_2d[1] - arm_bias        # facing +x, LEFT is +y (see the E branch)
			N_pos[0] = f_min_bound[0] - safety - wheel_r
			N_pos[2] = math.radians(0.0)

		if _clearance_on:
			import nav_clearance as nc
			_furn_box = nc.Box(furniture_prim.GetName(),
							   float(f_min_bound[0]), float(f_min_bound[1]),
							   float(f_max_bound[0]), float(f_max_bound[1]),
							   float(f_min_bound[2]), float(f_max_bound[2]))
			_usual = None if app.N_dir is None else (float(N_pos[0]), float(N_pos[1]), float(N_pos[2]))
			# The object is above the base (floor_obstacles drops it by z anyway); a FIXTURE target
			# is the furniture being parked at and stays an obstacle.
			_target_name = None if _target_is_fixture else prim_path.strip("/").split("/")[1]
			_res = nc.resolve_park(
				_usual, app.N_dir, (float(object_pos_2d[0]), float(object_pos_2d[1])), _furn_box,
				standoff_m=safety + wheel_r, arm_bias=arm_bias, footprint=nc.footprint_for(_robot),
				boxes=_world_boxes(app.stage), bands=app.kitchen_data.get("initial_pos_ranges", []),
				target_name=_target_name, prev_park=getattr(app, "_nav_prev_park", None))
			for _note in _res.notes:
				app.log(f"[nav_clearance] {_note}")
			app.N_dir = _res.side
			N_pos[0], N_pos[1], N_pos[2] = _res.x, _res.y, _res.yaw
			if _res.bands is not None:
				# IN PLACE: task_emit's kitchen_meta is a shallow copy of kitchen_data taken before
				# planning, so it shares this list -- rebinding it would author the old bands.
				app.kitchen_data["initial_pos_ranges"][:] = _res.bands
			app._nav_prev_park = (float(N_pos[0]), float(N_pos[1]), float(N_pos[2]))
			# A detour comes back as via-points; task_emit writes each as its own nav step ahead
			# of the park. Everything else sees a plain [x, y, yaw].
			return nc.NavRoute(N_pos.tolist(), via=_res.via)

	return N_pos.tolist()


def _world_boxes(stage):
	"""Every /world child's world-space AABB as a nav_clearance.Box: the same BBoxCache read the
	free-space grid and dump_kitchen_geom make, so the clearance gate sees the furniture the
	spawn search saw."""
	import nav_clearance as nc
	cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
	out = []
	for prim in stage.GetPrimAtPath("/world").GetChildren():
		r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
		if r.IsEmpty():
			continue
		lo, hi = r.GetMin(), r.GetMax()
		out.append(nc.Box(prim.GetName(), float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]),
						  float(lo[2]), float(hi[2])))
	return out


def _nav_clearance_report(app, N_pos, robot: str):
	"""Log a furniture (handle) park's clearance; remember it as the chain's current base pose."""
	import nav_clearance as nc
	_, obstacles = nc.scene_obstacles(_world_boxes(app.stage))
	x, y, yaw = float(N_pos[0]), float(N_pos[1]), float(N_pos[2])
	gap, who = nc.clearance(x, y, yaw, nc.footprint_for(robot), obstacles)
	if gap < 0.0:
		app.log(f"[nav_clearance] WARNING: handle park ({x:.3f}, {y:.3f}, {math.degrees(yaw):.0f} deg) "
				f"is INSIDE {who} by {-gap:.3f} m (not re-sited: a handle park cannot change side)")
	else:
		app.log(f"[nav_clearance] handle park ({x:.3f}, {y:.3f}, {math.degrees(yaw):.0f} deg) clears "
				f"{who} by {gap:.3f} m")
	app._nav_prev_park = (x, y, yaw)


@register_planner("nav.to_door_handle")
def plan_nav_to_door_handle(app: "GoalGeneratorApp", action: str, params: dict):
	"""Park in front of a door handle, stepped `past_m` along the door AWAY FROM ITS HINGE.

	WHY NOT nav.to_prim WITH A BIGGER safety. That skill parks at the handle's own lateral
	coordinate and only varies the NORMAL standoff. For a door the binding constraint is LATERAL:
	the handle sits inside the arc the door sweeps, so the door opens until it meets the robot and
	stops there, and backing further off does not move the base out of the arc. Measured by OBB/SAT
	against Anubis's real base box on freshly built kitchens: base_cabinet/door_0_0 jams at 24.0
	degrees and sink_cabinet/door_0_1 at 19.5-26.5, on doors that are otherwise free to a full 90.

	THE HINGE COMES FROM THE JOINT. door_geometry.measure_door reads the revolute joint's localPos0
	through body0, which is the authority; the fridge's SIMVLA_FRIDGE_PARK_PAST_M block below
	guesses it from the door slab's bbox instead, and a guess and a measurement are two facts that
	can disagree. measure_door also REFUSES a horizontal hinge, so pointing this skill at a
	dishwasher or an oven -- both drop-down doors in this corpus, both authoring physics:axis "Z"
	like every swing door -- fails loudly here rather than parking beside a door that opens
	downward.

	Returns [x, y, yaw] in the same frame and shape as nav.to_prim, and sets app.N_dir for the
	arm.handle_pregrasp / arm.handle_grasp steps that follow.
	"""
	import door_geometry as _dg

	prim_path = params.get("prim_path")
	if not prim_path:
		raise ValueError("Prim path required for nav.to_door_handle")
	handle_prim = app.stage.GetPrimAtPath(prim_path)
	if not handle_prim:
		raise ValueError(f"Prim not found: {prim_path}")

	door_path = prim_path.rstrip("/").rsplit("/", 1)[0]
	geom = _dg.measure_door(app.stage, door_path, prim_path.rstrip("/").rsplit("/", 1)[-1])

	past = float(params.get("past_m", 0.21))
	stand = float(params.get("standoff_m", 0.51))

	# Along the door, away from the hinge. Normalised on the XY projection: the hinge is vertical
	# (measure_door guarantees it), so the door's travel is entirely in plan.
	vx = geom.handle_world[0] - geom.hinge_world[0]
	vy = geom.handle_world[1] - geom.hinge_world[1]
	n = math.hypot(vx, vy) or 1.0
	stand_at = (geom.handle_world[0] + past * vx / n, geom.handle_world[1] + past * vy / n)

	# The furniture this door belongs to: /world/<furniture>. Its orientation picks the compass
	# direction, exactly as plan_nav_to_prim's furniture branch does, so N_dir means the same thing
	# to the grasp steps that follow.
	parts = door_path.strip("/").split("/")
	furniture_path = "/world/" + parts[1]
	furniture = app.stage.GetPrimAtPath(furniture_path)
	if not furniture:
		raise ValueError(f"door {door_path!r} has no /world/<furniture> ancestor to stand off from")

	world_transform = UsdGeom.Xformable(furniture).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
	q = world_transform.ExtractRotationQuat()
	quat_wxyz = [q.GetReal(), *q.GetImaginary()]

	# THE STANDOFF IS MEASURED FROM THE HANDLE'S CENTRE, not from the furniture's bbox face, and
	# that difference is the whole reason this is not plan_nav_to_prim with different numbers.
	#
	# plan_nav_to_prim parks at `min_b - safety - wheel_r`, i.e. off the bounding face of the prim
	# it is given. cabinet_kitchen.pick_park_pose -- the OBB/SAT search that produced standoff_m and
	# past_m, and the only evidence either number is right -- places the base at
	#
	#     handle_centre + outward_normal * standoff_m + along_the_door * past_m
	#
	# Applying a standoff measured from one origin at another origin is off by however far the
	# handle centre sits from the cabinet's outer face, which on this door is 3.9 cm against a
	# clearance budget of 10.8 cm. So this reproduces the gate's formula exactly, and the compass
	# snap below is used only for the axis-aligned normal and for app.N_dir.
	N_pos = torch.zeros(3)
	if np.allclose(quat_wxyz, [-0.7071, 0, 0, 0.7071], atol=1e-3):
		app.N_dir, nx, ny, heading = "N", -1.0, 0.0, 0.0
	elif np.allclose(quat_wxyz, [0.7071, 0, 0, 0.7071], atol=1e-3):
		app.N_dir, nx, ny, heading = "S", 1.0, 0.0, 180.0
	elif np.allclose(quat_wxyz, [1.0, 0, 0, 0], atol=1e-3):
		app.N_dir, nx, ny, heading = "W", 0.0, -1.0, 90.0
	elif np.allclose(quat_wxyz, [0.0, 0, 0, 1.0], atol=1e-3):
		app.N_dir, nx, ny, heading = "E", 0.0, 1.0, -90.0
	else:
		nx = ny = heading = None
	if nx is not None:
		N_pos[:] = torch.tensor([stand_at[0] + nx * stand,
								 stand_at[1] + ny * stand,
								 math.radians(heading)])
	if nx is None:
		# NOT a warning-and-carry-on like plan_nav_to_prim's furniture branch, which leaves N_pos at
		# (0, 0, 0) -- the env-frame origin -- and lets the run proceed. This skill exists to place
		# the base to within a couple of centimetres; silently driving to the origin instead would
		# look like a task failure rather than an authoring one.
		raise ValueError(
			f"{furniture_path!r} is rotated to {quat_wxyz}, which is not one of the four canonical "
			f"kitchen snaps, so there is no compass direction to approach it from."
		)

	app.log(f"nav.to_door_handle: {door_path} hinge=({geom.hinge_world[0]:.3f},"
			f"{geom.hinge_world[1]:.3f}) handle=({geom.handle_world[0]:.3f},"
			f"{geom.handle_world[1]:.3f}) past={past:.3f} standoff={stand:.3f} "
			f"-> {app.N_dir} ({float(N_pos[0]):.3f},{float(N_pos[1]):.3f})")
	return N_pos.tolist()


def _oriented_plan_box(app, path):
	"""(world centre xy, local min, local max, BBOX-FRAME yaw, aligned min, aligned max, PRIM yaw).

	TWO YAWS, AND CONFUSING THEM COST A CAMPAIGN. The fourth element is the yaw of the frame the
	BBOX's range is expressed in; the seventh is the yaw of the PRIM'S OWN transform. They are
	different numbers and only the seventh describes where the object is turned.

	THE ORIENTED BOX, NOT THE AABB -- IN PRINCIPLE. UsdGeom.BBoxCache.ComputeWorldBound returns a
	GfBBox3d that CAN keep its range and its matrix apart -- `GetRange()` is the box in that frame and
	`GetMatrix()` is how the frame sits in the world -- and ComputeAlignedRange() is what throws the
	orientation away. Both planners used to call exactly that, and for a yawed prim it inflates the
	extent: kitchen 1218's chair_0 (0.3832 m square) measured 0.3827 where the chair is 0.1916.

	IN PRACTICE, MEASURED ON THE SHIPPED kitchen_1218_00.usd, IT DOES NOT KEEP THEM APART: for
	/world/chair_0 -- a prim carrying xformOp:orient (0.9314151, 0, 0, 0.36395872), i.e. yawed
	+42.687 deg -- ComputeWorldBound hands back an identity matrix and a range that is ALREADY
	world-axis-aligned (GetRange() == ComputeAlignedRange(), both x[-0.0675,0.4740] y[-2.1247,
	-1.5832]). So the bbox-frame yaw reads 0.0 for a plainly rotated chair, and the box form silently
	degrades to the world AABB. That is why the extent moved to mesh points (_prim_mesh_points_xy),
	and it is the whole of the reason -- nothing is baked into any vertex; see that docstring.

	The aligned range is returned BESIDE the oriented one so the planners can log both. That is not
	decoration: the difference between them IS the artefact, and a run whose log shows aligned ==
	oriented is a run in which every prim happened to be axis-aligned and the fix changed nothing.

	Each yaw comes from its matrix's first ROW because USD transforms points as p * M, so row 0 is
	the image of the local +x axis. Reading it off the matrix rather than off a quaternion keeps this
	agreeing with whatever chain of xformOps actually placed the prim.
	"""
	prim = app.stage.GetPrimAtPath(path)
	if not prim:
		raise ValueError(f"Prim not found: {path}")
	bb = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
	).ComputeWorldBound(prim)
	rng = bb.GetRange()
	m = bb.GetMatrix()
	lo, hi = rng.GetMin(), rng.GetMax()
	centroid = bb.ComputeCentroid()
	aligned = bb.ComputeAlignedRange()
	yaw = math.atan2(float(m[0][1]), float(m[0][0]))
	# THE SILENT-REGRESSION CHECK. If a USD path hands back a bbox whose matrix is the identity and
	# whose range is already world-aligned, everything below still runs and `yaw` comes out 0 --
	# which is precisely the old, wrong behaviour, arriving without a word. THIS IS NOT HYPOTHETICAL:
	# it is what kitchen 1218's chairs do (see the docstring). So the prim's OWN xform yaw is read
	# separately, RETURNED (not merely logged), and the two disagreeing is the signature of that case.
	# Reported rather than raised: a prim whose bbox legitimately carries an extra transform is not
	# this function's error to fail a whole build over.
	prim_yaw = yaw
	try:
		xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
		prim_yaw = math.atan2(float(xf[0][1]), float(xf[0][0]))
		if abs(math.degrees(math.atan2(math.sin(prim_yaw - yaw), math.cos(prim_yaw - yaw)))) > 1.0:
			app.log(
				f"WARNING {path}: the bbox frame is yawed {math.degrees(yaw):.1f}deg but the prim's "
				f"own transform is yawed {math.degrees(prim_yaw):.1f}deg. If the bbox reads 0 here the "
				f"oriented extent has silently degraded to the world AABB it replaced."
			)
	except Exception:
		pass
	return ((float(centroid[0]), float(centroid[1])),
		(float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2])),
		float(yaw), aligned.GetMin(), aligned.GetMax(), float(prim_yaw))


def _prim_mesh_points_xy(app, path):
	"""Every mesh vertex under `path`, in WORLD xy. Empty when the subtree carries no readable mesh.

	WHY THE PLANNER READS VERTICES AT ALL. The extent this task stands the base off by has to be the
	prim's reach along the push axis, and every box form of that needs to know which way the prim is
	turned. On kitchen 1218 the box could not say: the emit log read a yaw of 0.0deg off a chair whose
	prim carries xformOp:orient (0.9314151, 0, 0, 0.36395872), i.e. +42.687 deg. The world AABB then
	reports 0.383 for a 0.192 chair and the base parks 0.191 m too far back -- the defect chain this
	task chased. Points cannot be defeated that way: wherever the rotation lives, the vertices are
	already where they are.

	AND THE ROTATION DOES **NOT** LIVE IN THE VERTICES. This docstring used to say scene_synthesizer's
	USD export bakes the rotation into them and leaves the prim identity. That was inferred from a
	MISLABELLED LOG LINE (see _push_extent) and it is measurably false: the exporter writes the
	object's full world transform onto the object Xform (scene_synthesizer/exchange/usd.py:523-531 ->
	usd_export._set_transform_prim, translate + orient) and gives child geometry a transform RELATIVE
	to it, and /world/chair_0/Chair_001_0 in the shipped kitchen_1218_00.usd has a local-to-world yaw
	of 42.687 deg inherited from its parent, not baked. What actually reads 0.0 is the frame
	UsdGeom.BBoxCache.ComputeWorldBound returns its range in -- identity, with the range already
	world-aligned. The mesh-point support is still the right answer; it is right for that reason.

	CHEAP ENOUGH TO DO PER STEP. This task's chair carries 112 vertices (chair_manifest records 200
	faces), and this runs twice per chair per emit, not per frame.

	Each mesh's OWN local-to-world is applied, so a subtree that does carry transforms is handled by
	the same code as one that has them baked -- the two cases differ only in where the numbers were.
	"""
	prim = app.stage.GetPrimAtPath(path)
	if not prim:
		raise ValueError(f"Prim not found: {path}")
	out = []
	for node in Usd.PrimRange(prim):
		if not node.IsA(UsdGeom.Mesh):
			continue
		pts = UsdGeom.Mesh(node).GetPointsAttr().Get()
		if not pts:
			continue
		m = UsdGeom.Xformable(node).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
		for q in pts:
			w = m.Transform(q)
			out.append((float(w[0]), float(w[1])))
	return out


def _prim_mesh_triangles(app, path):
	"""Every triangle under `path`, in WORLD coordinates, as ((x,y,z),(x,y,z),(x,y,z)).

	The same walk as _prim_mesh_points_xy -- same Usd.PrimRange, same per-mesh local-to-world, same
	silence on a subtree with no mesh -- carrying the FACES as well as the points.

	WHY THE FACES ARE NEEDED AND THE POINTS ARE NOT ENOUGH. support_along_points takes a MAXIMUM over
	vertices, and the extreme point of any mesh along any direction IS a vertex, so points answer that
	question exactly. "Is there material under the hand at this height" is not that question: a
	folding chair's backrest is a flat panel with four corner vertices and NOTHING at the hand's
	lateral offset. Measured on b43f9098 (540 triangles): a vertex test found the panel at z 0.84 and
	missed it at 0.81, 0.75 and 0.72, all of them solid. It was measuring the tessellation.

	FAN-TRIANGULATED from faceVertexCounts, which is right for the convex polygons a mesh exporter
	emits and is what scene_synthesizer writes here (every count in this library's chairs is 3 or 4).
	A concave polygon would fan into triangles that stray outside it; nothing in this corpus has one,
	and a chair face that did would over-report material rather than miss it.
	"""
	prim = app.stage.GetPrimAtPath(path)
	if not prim:
		raise ValueError(f"Prim not found: {path}")
	out = []
	for node in Usd.PrimRange(prim):
		if not node.IsA(UsdGeom.Mesh):
			continue
		mesh = UsdGeom.Mesh(node)
		pts = mesh.GetPointsAttr().Get()
		idx = mesh.GetFaceVertexIndicesAttr().Get()
		cnt = mesh.GetFaceVertexCountsAttr().Get()
		if not pts or not idx or not cnt:
			continue
		m = UsdGeom.Xformable(node).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
		world = []
		for q in pts:
			w = m.Transform(q)
			world.append((float(w[0]), float(w[1]), float(w[2])))
		at = 0
		for c in cnt:
			c = int(c)
			if c >= 3 and at + c <= len(idx):
				ring = [world[int(idx[at + k])] for k in range(c)]
				for k in range(1, c - 1):
					out.append((ring[0], ring[k], ring[k + 1]))
			at += c
	return out


def _push_extent(app, skill_id, path, centre, ux, uy):
	"""How far `path` reaches along (ux, uy) from `centre`, and a log line saying how that was got.

	Returns (half, note). `half` is the MESH support when the vertices are readable and the oriented
	box's otherwise.

	AND IT REFUSES THE FALLBACK ON A DIAGONAL PUSH, which is the check that would have failed emit
	job 2109085 outright instead of letting it produce a whole run. The box form is only as good as
	the yaw it is handed, and when the yaw is wrong its answer is the WORLD AABB's -- which for a
	push that is not axis-aligned is provably inflated, by (|cos|+|sin|)^2 on a square. So: no mesh
	plus a non-axis-aligned heading is an error, not a warning. An axis-aligned push is let through
	because there the AABB is tight and the fallback is exactly right.
	"""
	from push_prim_geometry import axis_offset_deg, half_extent_along, support_along_points

	_c, lo, hi, yaw, alo, ahi, prim_yaw = _oriented_plan_box(app, path)
	half_box = half_extent_along(lo, hi, yaw, ux, uy)
	half_aabb = half_extent_along((alo[0], alo[1]), (ahi[0], ahi[1]), 0.0, ux, uy)
	points = _prim_mesh_points_xy(app, path)
	if points:
		half = support_along_points(points, centre, ux, uy)
		src = f"mesh ({len(points)} pts)"
	else:
		off = axis_offset_deg(ux, uy)
		if off > 1.0:
			raise ValueError(
				f"{skill_id}: {path} exposes no mesh points, so its extent falls back to a box measured "
				f"at bbox yaw {math.degrees(yaw):.1f}deg (the prim's own transform is yawed "
				f"{math.degrees(prim_yaw):.1f}deg) -- and this push is {off:.1f}deg off a world "
				f"axis, so that box IS the world AABB and is inflated ({half_box:.4f} against an aligned "
				f"{half_aabb:.4f}). The base would stand off by the inflation. Emit job 2109085 shipped "
				f"exactly this and cost a GPU round."
			)
		half = half_box
		src = "oriented box (no mesh points; push is axis-aligned so the box is tight)"
	# BOTH YAWS ARE NAMED, AND NAMED CORRECTLY. This line used to print the bbox-frame yaw under the
	# label "prim yaw", and emit 2109085's `box@yaw 0.0deg` for a chair whose prim carries a +42.7 deg
	# orient was then read as "the rotation is baked into the vertices and the prim is identity". It
	# is not: /world/chair_0 in the shipped kitchen_1218_00.usd carries xformOp:orient
	# (0.9314151, 0, 0, 0.36395872). That misreading is what sent the campaign looking for a
	# quaternion-convention bug in env_cfg_emit that was never there. A log label that lies about
	# WHICH FRAME a number came from costs more than a missing number.
	note = (f"half={half:.4f} from {src} "
			f"[box@bbox-yaw {math.degrees(yaw):.1f}deg would be {half_box:.4f}, "
			f"world aabb {half_aabb:.4f}, prim yaw {math.degrees(prim_yaw):.1f}deg]")
	return half, note


@register_planner("nav.push_prim")
def plan_nav_push_prim(app: "GoalGeneratorApp", action: str, params: dict):
	"""Base pose that puts `prim_path` between the robot and `toward`, facing `toward`.

	Everything stage-shaped happens here -- two bbox lookups -- and the arithmetic is in
	push_prim_geometry, which is unit-tested on CPU (test_push_prim_geometry.py). See NavPushPrim in
	skills.py for why nav.to_prim cannot serve this case.
	"""
	from push_prim_geometry import push_prim_base_pose

	prim_path = params.get("prim_path")
	toward_path = params.get("toward")
	if not prim_path or not toward_path:
		raise ValueError("nav.push_prim needs both prim_path and toward")

	# Only the CENTRES are needed here: the extent comes from _push_extent, which reads the mesh.
	prim_c = _oriented_plan_box(app, prim_path)[0]
	toward_c = _oriented_plan_box(app, toward_path)[0]

	dx, dy = prim_c[0] - toward_c[0], prim_c[1] - toward_c[1]
	n = math.hypot(dx, dy)
	if n < 1e-6:
		raise ValueError(
			f"nav.push_prim: {prim_path} is centred on {toward_path}; no push direction exists"
		)
	half, extent_note = _push_extent(app, "nav.push_prim", prim_path, prim_c, dx / n, dy / n)
	offset = float(params.get("offset_m", 0.0))

	x, y, yaw = push_prim_base_pose(
		prim_c, toward_c, half,
		wheel_r=0.23,                       # plan_nav_to_prim's own value
		safety=float(params.get("safety", 0.12)),
		offset_m=offset,
	)
	# EVERY NAV SKILL REPORTS THE DIRECTION IT LEFT THE BASE FACING. plan_nav_to_prim's docstring
	# states that as part of the job, and it is not decoration: plan_arm_grasp reads app.N_dir to
	# pick its approach axis and plan_arm_squeeze reads it to pick the squeeze axis. A push has
	# such a direction as surely as a nav does -- the base drives along the chair->table axis, so
	# that axis IS the heading -- so it is reported here. Left unset it would keep the "N" that
	# GoalGeneratorApp.__init__ seeds it with, and the `if not app.N_dir` guard before the grasp
	# never fires on a non-empty string: a later push-then-grasp template would silently grasp
	# along +x whichever way the chair actually went. Inert for this template, which has no arm
	# steps, and cheap insurance for the one that does.
	#
	# The mapping is plan_nav_to_prim's own, read off the yaws it pairs with each letter:
	# N=+x (yaw 0), W=+y (90), S=-x (180), E=-y (-90). An exact diagonal is a tie, and either
	# neighbour is equally defensible there.
	app.N_dir = ("N", "W", "S", "E")[round(math.degrees(yaw) / 90.0) % 4]
	app.log(
		f"nav.push_prim {prim_path} -> {toward_path}: centre=({prim_c[0]:.3f},{prim_c[1]:.3f}) "
		f"{extent_note} offset={offset:.3f} base=({x:.3f},{y:.3f}) "
		f"yaw={math.degrees(yaw):.1f}deg dir={app.N_dir}"
	)
	return [float(x), float(y), float(yaw)]


@register_planner("arm.push_pose")
def plan_arm_push_pose(app: "GoalGeneratorApp", action: str, params: dict):
	"""Both hands, held just clear of `prim_path`'s far face, square to the push at `toward`.

	The A_b payload the executor reads straight through: 14 floats, [:7] LEFT and [7:14] RIGHT.

	SAME TWO BBOX LOOKUPS AS plan_nav_push_prim, deliberately -- the hands and the base that carries
	them into the chair must be on ONE axis, and the way to guarantee that is for both planners to
	derive it from the same two world bounds by the same arithmetic. The arithmetic itself is in
	push_prim_geometry (unit-tested on CPU in test_push_prim_geometry.py); what is here is the stage
	work, which is all this module can contribute.
	"""
	from push_prim_geometry import arm_reach_floor_m, choose_push_height, push_pose_hands

	prim_path = params.get("prim_path")
	toward_path = params.get("toward")
	if not prim_path or not toward_path:
		raise ValueError("arm.push_pose needs both prim_path and toward")

	# The aligned z range is kept for the height check below; the plan-view extent comes from
	# _push_extent, which reads the mesh rather than any box.
	prim_c, _lo, _hi, _yaw, prim_alo, prim_ahi, _prim_yaw = _oriented_plan_box(app, prim_path)
	toward_c = _oriented_plan_box(app, toward_path)[0]

	dx, dy = prim_c[0] - toward_c[0], prim_c[1] - toward_c[1]
	n = math.hypot(dx, dy)
	if n < 1e-6:
		raise ValueError(
			f"arm.push_pose: {prim_path} is centred on {toward_path}; no push direction exists"
		)
	# THE SAME EXTENT nav.push_prim uses, by the same call. The hands sit at half + clearance and the
	# base at half + wheel_r + safety, so a disagreement here would put them on different axes.
	half, extent_note = _push_extent(app, "arm.push_pose", prim_path, prim_c, dx / n, dy / n)

	height_m = float(params.get("height_m", 0.0))
	span_m = float(params.get("span_m", 0.16))
	clearance_m = float(params.get("clearance_m", 0.10))
	safety_m = float(params.get("safety_m", 0.12))
	top_margin_m = float(params.get("top_margin_m", 0.05))

	# HOW FAR IN FRONT OF base_link THE HANDS END UP, which is what decides whether the arm can hold
	# the push orientation at any given height. push_prim_base_pose parks the base at
	# half + wheel_r + safety and these hands sit at half + clearance, so THE HALF-EXTENT CANCELS
	# and one number serves every prim in the library. wheel_r is plan_nav_push_prim's own 0.23.
	#
	# safety_m IS A PARAM NOW, and it has to be. It used to be a 0.12 literal in this planner's log
	# line, under a comment saying the planner cannot see the nav step's value so the number is
	# reported rather than asserted. That was fine while the height was a constant; the height is
	# now DERIVED from this distance, so an assumed standoff would silently derive a height for a
	# base that is standing somewhere else.
	hands_forward = 0.23 + safety_m - clearance_m

	# THE HEIGHT, MEASURED OFF THE PRIM -- unless the step authored one. See ArmPushPose in skills.py
	# for why 0.0 is the default and what a positive value means.
	if height_m > 0.0:
		height_note = f"height={height_m:.3f} AUTHORED (override; not measured)"
	else:
		tris = _prim_mesh_triangles(app, prim_path)
		if not tris:
			raise ValueError(
				f"arm.push_pose: {prim_path} exposes no triangles, so there is nothing to measure a "
				f"press height against. Author height_m explicitly for this prim -- but check FIRST "
				f"that it is reachable: {hands_forward:.3f} m in front of the base the arm cannot "
				f"hold the push orientation below "
				f"{arm_reach_floor_m(hands_forward):.3f} m, and "
				f"cuRobo will not refuse a goal under that line."
			)
		height_m, height_note = choose_push_height(
			tris, prim_c, toward_c, span_m, hands_forward, top_margin_m=top_margin_m)

	# REFUSED, NOT WARNED. A height above the prim's own top is hands that sail over the backrest and
	# a push leg that drives the base into a chair it never touched with anything but its chassis --
	# an episode that looks bimanual on video and is not. It is also the single easiest thing to get
	# wrong here: the SceneSmith reference this pose descends from authored 0.85, and the chair
	# furnished_kitchen actually seats is 0.8143 m tall, so copying the reference lands ABOVE it.
	# The bbox is the only place that fact is available, and this planner is holding it.
	# The Z EXTENT IS THE ALIGNED ONE, deliberately, and it is the one place the aligned range is
	# still right: a yaw about the vertical does not change a prim's height, and the height check is
	# about where the hands sit above the FLOOR, which is a world quantity.
	z_lo, z_hi = float(prim_alo[2]), float(prim_ahi[2])
	if not (z_lo < height_m < z_hi):
		raise ValueError(
			f"arm.push_pose: height_m={height_m:.3f} is outside {prim_path}'s own z extent "
			f"[{z_lo:.3f}, {z_hi:.3f}]. The hands would close on air and the base would arrive alone."
		)

	payload = push_pose_hands(prim_c, toward_c, half,
							  clearance_m=clearance_m, height_m=height_m, span_m=span_m)

	# The base's own approach pose, for the one number this step cannot check on its own: how far in
	# front of the base the hands end up, and therefore whether the arm can reach them. It is
	# `wheel_r + safety - clearance_m` and the half-extent cancels, so it is the same for every chair
	# in the library -- but only while nav.push_prim's safety is the template's, which is why it is
	# reported rather than asserted here (this planner does not see that step's params).
	standoff = hands_forward
	app.log(
		f"arm.push_pose {prim_path} -> {toward_path}: centre=({prim_c[0]:.3f},{prim_c[1]:.3f}) "
		f"{extent_note} "
		f"z_extent=[{z_lo:.3f},{z_hi:.3f}] height={height_m:.3f} span={span_m:.3f} "
		f"clearance={clearance_m:.3f} {height_note} "
		f"left=({payload[0]:.3f},{payload[1]:.3f},{payload[2]:.3f}) "
		f"right=({payload[7]:.3f},{payload[8]:.3f},{payload[9]:.3f}) "
		f"quat=({payload[3]:.4f},{payload[4]:.4f},{payload[5]:.4f},{payload[6]:.4f}) "
		f"~{standoff:.3f}m in front of a base parked at safety={safety_m:.3f}"
	)
	return payload


@register_planner("arm.grasp")
def plan_arm_grasp(app: "GoalGeneratorApp", action: str, params: dict):
	prim_path = params.get("prim_path")

	if not prim_path:
		raise ValueError("Prim path required for grasp")

	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")

	bodex_attr = prim.GetAttribute("BODex_path")

	# -----------------------------------------------------------
	# CASE 1: Full graspdata logic (BODex_path is available)
	# -----------------------------------------------------------
	if bodex_attr and bodex_attr.HasAuthoredValue():
		app.log("Using full graspdata logic for 'Move arm to grasp'")

		grasp_pos = torch.zeros(7)
		bbox_range = UsdGeom.BBoxCache(
			Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
		).ComputeWorldBound(prim).ComputeAlignedRange()
		object_min_bound, object_max_bound = bbox_range.GetMin(), bbox_range.GetMax()

		# SQUEEZE mode (gated). A wide smooth object a parallel-jaw FINGER grasp cannot hold — and at
		# ~10 cm it is also too narrow for two grippers to sit on opposite faces without colliding, so
		# neither single-arm nor bimanual finger grasping works. Instead press the two OPPOSITE faces
		# with the gripper palms and hold by friction. Bypass BODex and place each arm geometrically on
		# the face TRANSVERSE to the approach direction, reusing plan_arm_handle_grasp's proven
		# per-direction orientations. Env-gated so no ordinary grasp changes. SIMVLA_SQUEEZE_OFF is the
		# ee-frame reach PAST the object half-extent — tune it up to press the palms inward.
		if os.environ.get("SIMVLA_SQUEEZE") and action in ("A_r", "A_l"):
			# Every number below is READ FROM THE LIVE OBJECT (world centre + AABB), never hardcoded,
			# so the same heuristic adapts to whatever size the scene placed. hfrac picks the grasp
			# HEIGHT as a fraction of the object's own height — 0.35 = the wide lower body, near the
			# centre of mass, so the squeezed object does not topple. margin is the palm reach past the
			# half-width along the squeeze axis.
			xform = UsdGeom.Xformable(prim)
			wt = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
			cx, cy, _cz = (float(v) for v in wt.ExtractTranslation())
			zmin, zmax = float(object_min_bound[2]), float(object_max_bound[2])
			hfrac = float(os.environ.get("SIMVLA_SQUEEZE_HFRAC", "0.35"))
			cz = zmin + hfrac * (zmax - zmin)              # lower-body height, from live AABB
			ext_x = float(object_max_bound[0] - object_min_bound[0])
			ext_y = float(object_max_bound[1] - object_min_bound[1])
			margin = float(os.environ.get("SIMVLA_SQUEEZE_OFF", "0.08"))
			# Which physical hand reaches which face depends on the robot's yaw, which this planner
			# does not have cleanly; SIMVLA_SQUEEZE_SWAP flips the assignment so the reachable one can
			# be found empirically without re-deriving the kinematics.
			_neg_is_r = (action == "A_r") ^ bool(os.environ.get("SIMVLA_SQUEEZE_SWAP"))
			if app.N_dir in ("N", "S"):          # approach along x -> squeeze along y
				off = ext_y / 2.0 + margin
				if _neg_is_r:                    # hand on the -y face, palm facing +y
					return [[cx, cy - off, cz, 0.5, -0.5, 0.5, 0.5]]
				return [[cx, cy + off, cz, 0.5, 0.5, 0.5, -0.5]]        # +y face, facing -y
			else:                                # approach along y -> squeeze along x
				off = ext_x / 2.0 + margin
				if _neg_is_r:                    # -x face, facing +x
					return [[cx - off, cy, cz, 0.70711, 0.0, 0.70711, 0.0]]
				return [[cx + off, cy, cz, 0.0, -0.70711, 0.0, 0.70711]]  # +x face, facing -x

		robot_name = "sim_parallel"
		base_grasp_path = load_grasp_file(bodex_attr.Get(), robot_name)
		if not base_grasp_path:
			raise FileNotFoundError(f"No grasp base path found for {bodex_attr.Get()}")
		else:
			print(f"Found grasp base path: {base_grasp_path}")

		# 2. Right / Left grasp files
		grasp_file_right = f"{base_grasp_path}_right.npy"
		grasp_file_left = f"{base_grasp_path}_left.npy"

		if not os.path.exists(grasp_file_right):
			raise FileNotFoundError(f"Missing grasp file: {grasp_file_right}")
		if not os.path.exists(grasp_file_left):
			raise FileNotFoundError(f"Missing grasp file: {grasp_file_left}")

		print(f"Loading RIGHT grasps: {os.path.basename(grasp_file_right)}")
		grasp_data_right = np.load(grasp_file_right, allow_pickle=True).item()
		eef_data_right = grasp_data_right["robot_pose"][0, :, 0, :7]

		print(f"Loading LEFT grasps: {os.path.basename(grasp_file_left)}")
		grasp_data_left = np.load(grasp_file_left, allow_pickle=True).item()
		eef_data_left = grasp_data_left["robot_pose"][0, :, 0, :7]

		# 4. Thumbnails + preferences
		thumbnail_path = f"{base_grasp_path}_segments_thumbnails"
		if not os.path.exists(thumbnail_path):
			raise FileNotFoundError(f"Thumbnail folder not found: {thumbnail_path}")

		prefer_list = select_thumbnails_cached(thumbnail_path)
		if not prefer_list:
			raise ValueError("No grasp preference selected from thumbnails.")

		print(f"Found {len(prefer_list)} preferred grasps to load.")

		# 6. Build final data list based on preferences
		valid_eef_data_list = []
		for segment_index, hand in prefer_list:
			if hand == "right":
				valid_eef_data_list.append(eef_data_right[segment_index])
			elif hand == "left":
				valid_eef_data_list.append(eef_data_left[segment_index])

		valid_eef_data = np.stack(valid_eef_data_list, axis=0)
		print(f"Successfully loaded and combined {valid_eef_data.shape[0]} poses.")
		prefer_eef_data = valid_eef_data

		# Positions + orientation
		xyz = prefer_eef_data[:, 0:3]
		quat_wxyz = prefer_eef_data[:, 3:7]

		# Out of the frame BODex synthesised these in, and into the one the object is PLACED in.
		# Those are not the same frame whenever mesh_orientation stands a mesh on a different axis
		# than BODex did -- a plate hand-labelled (0,0,-1) was synthesised at (0,0,1), so its grasps
		# arrive 180 degrees out and no amount of yaw below fixes it. Exactly the identity for the
		# 340 of 357 meshes whose frames already agree, so this changes nothing for them; see
		# grasp_frame and test_grasp_frame, which checks the result against BODex's own recorded
		# contact points. Read off the RIGHT file: both files record the same synthesis pose for
		# every mesh in the dataset, and prefer_eef_data has already merged the two hands.
		import grasp_frame
		from mesh_orientation import resolved_up

		obj_type = grasp_frame.object_type_from_prim_name(prim.GetName())
		placed_up = resolved_up(obj_type, bodex_attr.Get())
		into_placed_frame = grasp_frame.correction(placed_up, grasp_data_right)
		if not grasp_frame.is_identity(into_placed_frame):
			app.log(
				f"Grasp frame: {prim.GetName()} is placed on {placed_up} but its grasps were "
				f"synthesised on another axis; rotating them into the placed frame."
			)
			xyz, quat_wxyz = grasp_frame.apply(into_placed_frame, xyz, quat_wxyz)

		# Rotate based on sub_num
		angle_deg = int(app.kitchen_data["kitchen_sub_num"]) * 30
		rot_z = R.from_euler("z", angle_deg, degrees=True)

		# Rotate position
		rotated_xyz = rot_z.apply(xyz)

		# Rotate orientation (wxyz → xyzw → apply → back to wxyz)
		quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
		norm = np.linalg.norm(quat_xyzw, axis=-1)
		bad = (norm == 0) | (~np.isfinite(norm))
		if bad.any():
			raise ValueError(f"Invalid BoDex grasp quaternions at selected rows {np.flatnonzero(bad).tolist()}")
		new_rotations = R.from_quat(quat_xyzw)
		new_rotations = rot_z * new_rotations
		rotated_quats_xyzw = new_rotations.as_quat()
		rotated_quats_wxyz = np.hstack(
			[rotated_quats_xyzw[:, 3:4], rotated_quats_xyzw[:, 0:3]]
		)

		rotated_data = np.hstack((rotated_xyz, rotated_quats_wxyz))

		if not app.N_dir:
			raise ValueError(
				"Navigation direction 'N_dir' must be set by an 'N' action before grasping."
			)

		# The A_r/A_l side split is a half-plane on the axis TRANSVERSE to the approach: for an N
		# approach the two hands divide along y, for E/W along x. Single-arm keeps the original
		# ±0.01 thresholds, which OVERLAP by design (a lone arm may take a near-central grasp). Under
		# the bimanual escape hatch that overlap is the bug: both hands draw from the same central
		# band and land on nearly the same point, so one arm cannot reach across and "A_r not reached"
		# fires. Widen to a gap (_hi/_lo) so each hand is pushed to its OWN side with clear air
		# between — only when SIMVLA_BIMANUAL_FIXED is set, leaving every single-arm run untouched.
		if os.environ.get("SIMVLA_BIMANUAL_FIXED"):
			_hi, _lo = -0.02, 0.02   # exclusive opposite bands, ~4 cm gap
		else:
			_hi, _lo = 0.01, -0.01   # original overlapping thresholds
		if action == "A_r":
			if app.N_dir == "N":
				avail_data = rotated_data[
					(rotated_data[:, 0] < 0) & (rotated_data[:, 1] < _hi)
				]
			elif app.N_dir == "S":
				avail_data = rotated_data[
					(rotated_data[:, 0] > 0) & (rotated_data[:, 1] > _lo)
				]
			elif app.N_dir == "E":
				avail_data = rotated_data[
					(rotated_data[:, 1] > 0) & (rotated_data[:, 0] < _hi)
				]
			elif app.N_dir == "W":
				avail_data = rotated_data[
					(rotated_data[:, 1] < 0) & (rotated_data[:, 0] > _lo)
				]
			else:
				raise ValueError(f"Invalid navigation direction: {app.N_dir}")
		elif action == "A_l":
			if app.N_dir == "N":
				avail_data = rotated_data[
					(rotated_data[:, 0] < 0) & (rotated_data[:, 1] > _lo)
				]
			elif app.N_dir == "S":
				avail_data = rotated_data[
					(rotated_data[:, 0] > 0) & (rotated_data[:, 1] < _hi)
				]
			elif app.N_dir == "E":
				avail_data = rotated_data[
					(rotated_data[:, 1] > 0) & (rotated_data[:, 0] > _lo)
				]
			elif app.N_dir == "W":
				avail_data = rotated_data[
					(rotated_data[:, 1] < 0) & (rotated_data[:, 0] < _hi)
				]
			else:
				raise ValueError(f"Invalid navigation direction: {app.N_dir}")
		elif action == "A_b":
			if app.N_dir == "N":
				avail_data = rotated_data[(rotated_data[:, 0] < 0)]
			elif app.N_dir == "S":
				avail_data = rotated_data[(rotated_data[:, 0] > 0)]
			elif app.N_dir == "E":
				avail_data = rotated_data[(rotated_data[:, 1] > 0)]
			elif app.N_dir == "W":
				avail_data = rotated_data[(rotated_data[:, 1] < 0)]
			else:
				raise ValueError(f"Invalid navigation direction: {app.N_dir}")
		else:
			raise ValueError(f"Unsupported action for grasp skill: {action}")

		if len(avail_data) == 0:
			raise ValueError(
				f"No valid grasps found for approach direction '{app.N_dir}'"
			)

		# Build final list of grasp poses
		all_grasp_poses = []
		_jardbg_skip = {}
		for grasp_pose_data in avail_data:
			pose_idx_mask = np.all(
				np.isclose(rotated_data, grasp_pose_data, atol=1e-6), axis=1
			)
			if np.sum(pose_idx_mask) != 1:
				app.log(
					"Warning: Could not uniquely identify a grasp pose. Skipping."
				)
				continue

			original_pose_idx = np.where(pose_idx_mask)[0][0]
			pose_idx = prefer_list[original_pose_idx]
			print(pose_idx)

			if pose_idx[1] == "left":
				grasp_data = grasp_data_left
			elif pose_idx[1] == "right":
				grasp_data = grasp_data_right
			else:
				continue

			mean_contact_point = torch.tensor(grasp_data["contact_point"][0])[
				pose_idx[0]
			].squeeze(0).mean(dim=0)

			# Contacts and wrist poses must receive the same object-frame correction.
			rotated_point = rot_z.apply(into_placed_frame @ mean_contact_point.numpy())

			# Scale the grasp contact offset to the cup's ACTUAL placed size, PER AXIS. BODex contacts
			# are at the grasp mesh's natural extents (scene_spec.CLUTTER dims); if the scene places a
			# clutter object smaller or LESS ELONGATED (a shorter cup), each axis of the offset must
			# scale by placed_extent/natural_extent or the gripper reaches past/above the object.
			# No-op when the placed size equals the natural size; skipped for non-clutter objects.
			_obj_t = prim.GetName().rstrip("0123456789")
			try:
				from scene_spec import CLUTTER
				if _obj_t in CLUTTER:
					_nat = CLUTTER[_obj_t]["dims"]
					_ext = [float(object_max_bound[i] - object_min_bound[i]) for i in range(3)]
					_s = np.array([(_ext[i] / _nat[i]) if _nat[i] > 1e-6 else 1.0 for i in range(3)])
					rotated_point = rotated_point * _s
			except Exception:
				pass

			# Adjust based on approach direction
			if app.N_dir == "N":
				rotated_point = rotated_point[[1, 0, 2]]
			elif app.N_dir == "S":
				rotated_point = rotated_point[[1, 0, 2]] * [-1, 1, 1]
			elif app.N_dir == "E":
				rotated_point = rotated_point[[0, 1, 2]] * [-1, -1, 1]
			elif app.N_dir == "W":
				rotated_point = rotated_point
			else:
				raise ValueError(f"Invalid navigation direction: {app.N_dir}")

			current_grasp_pos = torch.zeros(7)
			if "handle" in prim.GetName():
				reference_frame = (
					torch.tensor(object_min_bound) + torch.tensor(object_max_bound)
				) / 2
			else:
				xform = UsdGeom.Xformable(prim)
				world_transform = xform.ComputeLocalToWorldTransform(
					Usd.TimeCode.Default()
				)
				pos = tuple(round(v, 4) for v in world_transform.ExtractTranslation())
				reference_frame = torch.tensor(list(pos))
				reference_frame[-1] = (
					(torch.tensor(object_min_bound) + torch.tensor(object_max_bound)) / 2
				)[-1]

			current_grasp_pos[0:3] = reference_frame + torch.tensor(
				rotated_point, dtype=torch.float32
			)
			quat = torch.tensor(grasp_pose_data[3:7], dtype=torch.float32)

			def quat_mul(a, b):
				w1, x1, y1, z1 = a
				w2, x2, y2, z2 = b
				return torch.tensor(
					[
						w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
						w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
						w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
						w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
					],
					dtype=torch.float32,
				)

			def euler_to_quat(roll, pitch, yaw):
				r = math.radians(roll) / 2
				p = math.radians(pitch) / 2
				y_half = math.radians(yaw) / 2

				cr, sr = math.cos(r), math.sin(r)
				cp, sp = math.cos(p), math.sin(p)
				cy, sy = math.cos(y_half), math.sin(y_half)

				w = cr * cp * cy + sr * sp * sy
				x = sr * cp * cy - cr * sp * sy
				yv = cr * sp * cy + sr * cp * sy
				z = cr * cp * sy - sr * sp * cy
				return torch.tensor([w, x, yv, z], dtype=torch.float32)

			# BODex hand pose -> a target for THIS ROBOT's ee_link1. Was the literal (90, 90, 0)
			# plus a literal "+Y must point down", both of which are facts about Anubis's ee_link1
			# and neither of which named a robot -- so a goal file carried no record of the frame it
			# was authored in. skills.GRASP_TOOL_FRAME derives the pair from each URDF's ee_link1
			# placement; see its comment. SIMVLA_ROBOT selects, defaulting to anubis so the 7,906
			# existing goal files re-emit byte-identically.
			_tf = skills.grasp_tool_frame(os.environ.get("SIMVLA_ROBOT", "anubis"))
			q_rot = torch.tensor(_tf["q_offset"], dtype=torch.float32)
			quat_new = quat_mul(quat, q_rot)
			quat_new = quat_new / torch.norm(quat_new)
			_keep = skills.preferred_grasp_roll(
				quat_new.tolist(), os.environ.get("SIMVLA_ROBOT", "anubis"))
			if _keep:
				current_grasp_pos[3:] = quat_new
			else:
				current_grasp_pos[3:] = local_z_plus_180_wxyz(quat_new)

			all_grasp_poses.append(current_grasp_pos.tolist())

		if not all_grasp_poses:
			raise ValueError(
				"Failed to process any of the available grasps after filtering."
			)

		# Make the authored grasps reachable in a world cuRobo can actually see.
		#
		# Everything above places the gripper on BODex's contact points and never asks what the
		# object is STANDING ON. That was free while cuRobo planned in an empty world; with the
		# world switched on it is the whole failure -- on kitchen 1201 ten of these fifteen
		# candidates put the gripper's own collision spheres inside the countertop (median -15 mm,
		# worst -49 mm), so two thirds of the executor's uniform draws cannot be planned at all.
		# grasp_clearance raises each one by the least it needs and drops what cannot be cleared;
		# see its module docstring for the measurement and for the two mechanisms it rejects.
		_clr = grasp_clearance.required_clearance()
		if _clr > 0.0:
			_tgt = prim.GetPath().pathString
			_bc = UsdGeom.BBoxCache(
				Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
			)
			# The same geom types get_obstacles_from_stage_simvla collects for cuRobo. Filtering to
			# Mesh alone would miss the cabinet panels, which are UsdGeom.Cube -- and silently turn
			# this check into a check against the countertop only.
			_types = (UsdGeom.Mesh, UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Capsule,
					  UsdGeom.Sphere)
			_lo, _hi = [], []
			for _p in app.stage.Traverse():
				_pp = _p.GetPath().pathString
				# The grasp target itself is never an obstacle -- the goal is to close on it -- and
				# the executor excludes it from cuRobo's world for the same reason. Neither is the
				# robot: its own body is cuRobo's job, through collision_spheres.
				if _pp == _tgt or _pp.startswith(_tgt + "/") or "/Robot" in _pp:
					continue
				if not any(_p.IsA(_t) for _t in _types):
					continue
				_r = _bc.ComputeWorldBound(_p).ComputeAlignedRange()
				if _r.IsEmpty():
					continue
				_mn, _mx = _r.GetMin(), _r.GetMax()
				_lo.append([_mn[0], _mn[1], _mn[2]])
				_hi.append([_mx[0], _mx[1], _mx[2]])
			# A_b authors ONE 7-float pose per row, like A_r, so it takes the right arm's tool here.
			# Both robots' two grippers are the same geometry in mirrored links (gripper1* / gripper2*,
			# ee_finger_r* / ee_finger_l*), so the measurement is the same either way.
			_spheres = grasp_clearance.tool_spheres(
				os.environ.get("SIMVLA_ROBOT", "anubis"),
				"left" if action == "A_l" else "right",
				SIMVLA_REPO_ROOT,
			)
			_kept, _notes = grasp_clearance.clear_grasps(
				all_grasp_poses, _spheres, _lo, _hi,
				list(object_min_bound), list(object_max_bound), clearance=_clr,
			)
			app.log(f"{prim.GetName()} {action}: {grasp_clearance.summarize(_notes)}")
			if not _kept:
				raise ValueError(
					f"{prim.GetName()} {action}: every one of the {len(all_grasp_poses)} authored "
					f"grasps puts the gripper inside the scene at the goal pose, and none can be "
					f"raised clear without lifting the jaws off the object. cuRobo will refuse all "
					f"of them. Move the object off the support edge, pick different grasp segments, "
					f"or set SIMVLA_GRASP_CLEARANCE=0 to author them anyway."
				)
			all_grasp_poses = _kept

		processed_value = all_grasp_poses

	# -----------------------------------------------------------
	# CASE 2: No BODex_path → simple bbox-center grasp
	# -----------------------------------------------------------
	else:
		app.log("WARNING: No BODex_path found. Using simplified grasp logic.")
		bbox_range = UsdGeom.BBoxCache(
			Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
		).ComputeWorldBound(prim).ComputeAlignedRange()
		center = (
			torch.tensor(bbox_range.GetMin()) + torch.tensor(bbox_range.GetMax())
		) / 2

		# original simple grasp: center + small z offset, identity quat
		processed_value = (center + torch.tensor([0, 0, 0.1])).tolist() + [
			0.0,
			0.0,
			0.0,
			1.0,
		]

	return processed_value


@register_planner("arm.squeeze")
def plan_arm_squeeze(app: "GoalGeneratorApp", action: str, params: dict):
	"""Bimanual heuristic squeeze — the first mode of the size-reading grasp API.

	Reads the object's LIVE world centre + axis-aligned bounding box from the stage and derives every
	number from them, so the SAME rule fits whatever size the scene placed (no BODex, no hardcoded
	dimensions). Each hand goes on the opposite face TRANSVERSE to the approach direction, at the wide
	lower body (SIMVLA_SQUEEZE_HFRAC of the object's height, default 0.35 — near the centre of mass so
	the object does not topple). The palms press inward (SIMVLA_SQUEEZE_OFF past the half-extent) and
	hold by friction: the answer for wide, smooth objects that one gripper cannot span and two cannot
	pinch. Orientations are the proven per-direction values from plan_arm_handle_grasp.
	"""
	prim_path = params.get("prim_path")
	if not prim_path:
		raise ValueError("Prim path required for squeeze")
	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")
	if not app.N_dir:
		raise ValueError("Navigation direction 'N_dir' must be set by an 'N' action before squeezing.")

	bbox = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
	).ComputeWorldBound(prim).ComputeAlignedRange()
	bmin, bmax = bbox.GetMin(), bbox.GetMax()
	wt = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
	cx, cy, _cz = (float(v) for v in wt.ExtractTranslation())

	# NECK grasp (single arm). A vase's wide body defeats a 9 cm gripper, but its NARROW NECK does not
	# — grasp it there with ONE hand, the way a person lifts a vase by the neck. The gripper comes in
	# horizontally at neck height (high on the object, where it is thin) with the neck between its
	# fingers, reusing the proven per-direction handle-grasp orientations. Env-gated, single-arm.
	if os.environ.get("SIMVLA_NECK") and action in ("A_r", "A_l"):
		neck_frac = float(os.environ.get("SIMVLA_NECK_FRAC", "0.82"))     # 0=base, 1=top of the AABB
		nz = float(bmin[2]) + neck_frac * (float(bmax[2]) - float(bmin[2]))
		so = float(os.environ.get("SIMVLA_NECK_STANDOFF", "0.06"))        # gripper standoff, neck between fingers
		if app.N_dir == "N":
			_pos, _q = [cx - so, cy, nz], [0.70711, 0.0, 0.70711, 0.0]
			_axis = (1.0, 0.0, 0.0)   # approach axis (+x)
		elif app.N_dir == "S":
			_pos, _q = [cx + so, cy, nz], [0.0, -0.70711, 0.0, 0.70711]
			_axis = (1.0, 0.0, 0.0)
		elif app.N_dir == "E":
			_pos, _q = [cx, cy + so, nz], [0.5, 0.5, 0.5, -0.5]
			_axis = (0.0, 1.0, 0.0)   # approach axis (+y)
		else:  # W
			_pos, _q = [cx, cy - so, nz], [0.5, -0.5, 0.5, 0.5]
			_axis = (0.0, 1.0, 0.0)
		# Roll the gripper about its APPROACH axis. The handle-grasp orientations are tuned for a
		# HORIZONTAL handle; a vase neck is a VERTICAL cylinder, so the fingers must close in a
		# different plane. SIMVLA_NECK_ROLL (deg) sweeps that plane to find where the neck ends up
		# BETWEEN the fingers (finger separation stops at the neck's width, not 0).
		_roll = float(os.environ.get("SIMVLA_NECK_ROLL", "0.0"))
		if abs(_roll) > 1e-6:
			import math as _m
			_h = _m.radians(_roll) / 2.0
			_qr = (_m.cos(_h), _m.sin(_h) * _axis[0], _m.sin(_h) * _axis[1], _m.sin(_h) * _axis[2])
			w1, x1, y1, z1 = _qr
			w2, x2, y2, z2 = _q
			_q = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
			      w1*x2 + x1*w2 + y1*z2 - z1*y2,
			      w1*y2 - x1*z2 + y1*w2 + z1*x2,
			      w1*z2 + x1*y2 - y1*x2 + z1*w2]
		return [_pos + _q]

	# Object-CENTRE height + dz — the exact reference the reachability probe validated. hfrac stays as
	# a fallback for callers that don't set SIMVLA_SQUEEZE_DZ.
	_dz_env = os.environ.get("SIMVLA_SQUEEZE_DZ")
	if _dz_env is not None:
		cz = (float(bmin[2]) + float(bmax[2])) / 2.0 + float(_dz_env)
	else:
		hfrac = float(os.environ.get("SIMVLA_SQUEEZE_HFRAC", "0.35"))
		cz = float(bmin[2]) + hfrac * float(bmax[2] - bmin[2])
	margin = float(os.environ.get("SIMVLA_SQUEEZE_OFF", "0.10"))
	ext_x = float(bmax[0] - bmin[0])
	ext_y = float(bmax[1] - bmin[1])
	# Both opposite-face poses, then assign to hands. A squeeze is a COORDINATED motion: on the A_b
	# action both hands go at ONCE (14-float payload, [:7] left [7:14] right), so neither presses the
	# object before the other arrives — the single-arm A_r/A_l forms remain for authoring flexibility.
	if app.N_dir in ("N", "S"):        # approach along x -> squeeze along y
		off = ext_y / 2.0 + margin
		neg_pose = [cx, cy - off, cz, 0.5, -0.5, 0.5, 0.5]        # -y face, palm facing +y
		pos_pose = [cx, cy + off, cz, 0.5, 0.5, 0.5, -0.5]        # +y face, palm facing -y
	else:                              # approach along y -> squeeze along x
		off = ext_x / 2.0 + margin
		neg_pose = [cx - off, cy, cz, 0.70711, 0.0, 0.70711, 0.0]  # -x face, facing +x
		pos_pose = [cx + off, cy, cz, 0.0, -0.70711, 0.0, 0.70711]  # +x face, facing -x
	# Which physical hand reaches which face depends on the robot yaw; SIMVLA_SQUEEZE_SWAP flips it so
	# the reachable assignment can be found without re-deriving the kinematics.
	if bool(os.environ.get("SIMVLA_SQUEEZE_SWAP")):
		right_pose, left_pose = pos_pose, neg_pose
	else:
		right_pose, left_pose = neg_pose, pos_pose
	if action == "A_r":
		return [right_pose]
	if action == "A_l":
		return [left_pose]
	if action == "A_b":
		return [left_pose + right_pose]   # 14 floats: [:7] LEFT, [7:14] RIGHT — both hands at once
	raise ValueError(f"Unsupported action for squeeze skill: {action}")


@register_planner("arm.handle_pregrasp")
def plan_arm_handle_pregrasp(app: "GoalGeneratorApp", action: str, params: dict):
	prim_path = params.get("prim_path") # handle prim

	if not prim_path:
		raise ValueError("Prim path required for grasp")

	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")

	# For pull it is usually horizontal handle or knob 
	# Pos : middle of the handle
	xform = UsdGeom.Xformable(prim)
	world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
	vec3d = world_transform.ExtractTranslation() 
	if app.N_dir == "N":
		return [vec3d[0] - 0.18, vec3d[1], vec3d[2] + 0.05 , 0.70711, 0.0, 0.70711, 0.0]
	elif app.N_dir == "S":
		return [vec3d[0] + 0.18, vec3d[1], vec3d[2] + 0.05 , 0.0, -0.70711, 0.0, 0.70711]
	elif app.N_dir == "E":
		return [vec3d[0], vec3d[1] + 0.18, vec3d[2] + 0.05, 0.5, 0.5, 0.5, -0.5]
	elif app.N_dir == "W":
		return [vec3d[0], vec3d[1] - 0.18, vec3d[2] + 0.05, 0.5, -0.5, 0.5, 0.5]
	else:
		raise ValueError(f"Invalid navigation direction: {app.N_dir}")

	return [vec3d[0], vec3d[1], vec3d[2] , 0.70711, 0.0, .70711, 0.0]


@register_planner("arm.handle_grasp")
def plan_arm_handle_grasp(app: "GoalGeneratorApp", action: str, params: dict):
	prim_path = params.get("prim_path") # handle prim

	if not prim_path:
		raise ValueError("Prim path required for grasp")

	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")

	# For pull it is usually horizontal handle or knob 
	# Pos : middle of the handle
	xform = UsdGeom.Xformable(prim)
	world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
	vec3d = world_transform.ExtractTranslation() 
	if app.N_dir == "N":
		return [vec3d[0] - 0.06, vec3d[1], vec3d[2] , 0.70711, 0.0, 0.70711, 0.0]
	elif app.N_dir == "S":
		return [vec3d[0] + 0.06, vec3d[1], vec3d[2] , 0.0, -0.70711, 0.0, 0.70711]
	elif app.N_dir == "E":
		return [vec3d[0], vec3d[1] + 0.06, vec3d[2], 0.5, 0.5, 0.5, -0.5]
	elif app.N_dir == "W":
		return [vec3d[0], vec3d[1] - 0.06, vec3d[2], 0.5, -0.5, 0.5, 0.5]
	else:
		raise ValueError(f"Invalid navigation direction: {app.N_dir}")

	return [vec3d[0], vec3d[1], vec3d[2] , 0.70711, 0.0, .70711, 0.0]


#: How far the hand stops from the handle's bbox CENTRE, along the approach direction.
#:
#: On Anubis, ee_fixed_joint2 puts `ee_link2` at the gripper and the jaws slide on THAT LINK'S
#: +-X (see skills.py's frame table), so the IK target IS the jaw location, not a wrist behind
#: it -- but "the jaw location" means the pad CENTRE, not the pad TIP: the finger pads extend
#: 47.8 mm past ee_link2 along the approach axis (measured on kitchen 1400: gripper2R collision
#: box spans z = 0.0644-0.1574 off gripper2_base_link, ee_link2 sits at z = 0.10956). Any
#: standoff has to be read against that 47.8 mm offset, not against ee_link2 directly.
#:
#: This started at 0.06, borrowed from plan_arm_handle_grasp on the reasoning that a proven drawer
#: number generalises. It does not: that planner measures from the handle prim's TRANSLATION, which
#: on a drawer already sits off the handle, whereas this measures from the bbox CENTRE, which is
#: the bar itself. Same number, different origin, 6 cm of air. Caught by the user watching the
#: gripper close: "to grasp the handle more move the gripper more front (close to the handle)" --
#: and they were right to say so: measured against kitchen 1400 (bar bbox centre y = -0.368497,
#: door panel outer face y = -0.362874), 0.06 sits above the feasible window below and really did
#: hold the jaws short of the bar.
#:
#: The 0.0 that followed overcorrected. Zero puts ee_link2 AT the bar's bbox centre, which -- once
#: the 47.8 mm pad overhang above is accounted for -- puts the finger pads 41.9 mm INSIDE the door
#: panel: not short of the bar this time, but through it. cuRobo never caught this because
#: `anubis_left_arm.yml`'s `collision_link_names` lists link21..link25 and gripper2_base_link but
#: not `gripper2R`/`gripper2L` -- the pads have collision spheres defined but are not in the
#: checked list, so a plan that rams them 41.9 mm into geometry is invisible to the planner and
#: gets accepted cleanly. Any other gripper-first grasp authored against ee_link2 has the same
#: blind spot; this offset just happened to be the one that walked into it.
#:
#: Diagnosed and measured on kitchen 1400: the geometrically feasible window is [0.0422, 0.0475] -- below
#: 0.0422 the pads penetrate the door panel, above 0.0475 they overshoot the bar's widest line.
#: 0.045 is the midpoint: pad tips 2.8 mm clear of the panel, bar's widest line 2.8 mm inside the
#: pad tip. One coherent story across both corrections: 0.06 too far back, 0.0 too far forward,
#: 0.045 inside the measured window between them.
FRIDGE_GRASP_OFFSET_M = 0.045
FRIDGE_PREGRASP_OFFSET_M = 0.12

#: How far Anubis's finger pads reach past `ee_link2` along the approach axis. Measured on kitchen
#: 1400: the gripper2R collision box spans z = 0.0644-0.1574 off gripper2_base_link while ee_link2
#: sits at z = 0.10956. Named, because the whole standoff argument above is stated against it and
#: it was previously only a number inside a comment.
PAD_OVERHANG_M = 0.0478


#: How much of the feasible window to leave between the pad tips and the panel behind the bar when
#: inserting as deeply as that window allows.
#:
#: IT MUST EXCEED THE GRASP GATE'S TOLERANCE, and at 2 mm it did not -- the gate admits 6 mm, so the
#: pads were driven into the door panel in most episodes. "small enough that a small IK error does
#: not drive the pads into the door" was the intent; 2 mm was a third of the error the gate allows.
#:
#: MEASURED on the dishwasher bar (job 2112357, 294 episodes). Among episodes with a FULL grip
#: (jaw >= 0.048 m on a 0.050 m bar), hand-to-door at the moment of closing separates almost
#: perfectly -- 0.7508 m is a hand exactly on the bar:
#:
#:      door opened        n=7    h2d 0.7479 - 0.7527   (median 0.7523)
#:      door never moved   n=45   h2d 0.7515 - 0.7663   (median 0.7632)
#:
#: The failures are shoved ~12 mm off the bar AFTER the reach gate passed -- i.e. during the close,
#: which is the pad tips hitting the panel 2 mm behind them and pushing the hand back out.
#:
#: 10 mm clears the 6 mm gate with margin and still puts 30 mm of the bar's 40 mm thickness between
#: the pads. The engagement lost is worth far more than it costs: 45 of 52 full grips were being
#: thrown away by the collision.
DEEP_INSERT_MARGIN_M = 0.010


def _handle_grasp_offset(app, handle_prim, normal, fallback=FRIDGE_GRASP_OFFSET_M, bias="mid"):
    """The standoff from THIS handle's bbox centre, measured rather than assumed.

    The feasible window is bounded at both ends by the same 47.8 mm pad overhang:

        pads must not enter the door panel  ->  offset > PAD_OVERHANG_M - gap
        pads must reach past the bar's widest line -> offset < PAD_OVERHANG_M

    where `gap` is how far the bar's bbox centre stands proud of the panel's outer face, along the
    approach normal. The midpoint of that window is what this returns.

    IT REPRODUCES THE FRIDGE'S HAND-TUNED CONSTANT EXACTLY, which is why it can replace it. Kitchen
    1400: bar centre y = -0.368497, panel outer face y = -0.362874, so gap = 0.005623 and the
    midpoint is 0.0478 - 0.002812 = 0.04499 -- the shipped 0.045, and the window is [0.0422,
    0.0478] against the report's measured [0.0422, 0.0475].

    IT IS NOT THE SAME NUMBER FOR EVERY HANDLE, which is the reason to measure. A cabinet door's
    pull stands 0.038 m proud where the fridge's bar stands 0.070; a constant tuned on one is
    outside the other's window.

    Falls back to `fallback` (and says so) when the panel cannot be identified -- a handle that is
    an only child, or a door whose siblings bound to nothing.
    """
    door = handle_prim.GetParent()
    if not door:
        return fallback
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    hb = cache.ComputeWorldBound(handle_prim).ComputeAlignedRange()
    if hb.IsEmpty():
        return fallback
    hc = hb.GetMidpoint()
    centre_along = hc[0] * normal[0] + hc[1] * normal[1]

    # The panel's outermost point TOWARD THE ROBOT, over every sibling of the handle. Max over
    # siblings rather than the door's own bbox, because that bbox includes the handle itself and
    # its outer face IS the bar, which would make the gap zero on every door.
    face = None
    for child in door.GetChildren():
        if child.GetPath() == handle_prim.GetPath():
            continue
        rng = cache.ComputeWorldBound(child).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                v = x * normal[0] + y * normal[1]
                face = v if face is None else max(face, v)
    if face is None:
        return fallback

    gap = centre_along - face
    if gap <= 0.0:
        # The bar does not stand proud of the panel along this axis: nothing to centre between.
        app.log(f"[handle_grasp] {handle_prim.GetPath()} bar centre is {gap:.4f} m proud of its "
                f"panel; keeping the {fallback:.4f} m default")
        return fallback
    # WHICH END OF THE WINDOW, and it is not always the middle.
    #
    # The window is [PAD_OVERHANG_M - gap, PAD_OVERHANG_M]: below it the pads enter the panel behind
    # the handle, above it they stop short of the bar's widest line. Its WIDTH is the gap -- how far
    # the bar stands proud of its panel -- which is 5.6 mm on a refrigerator, where the midpoint is
    # as good as anything, and 20 mm on a dishwasher's bar, where the midpoint leaves the pads
    # 10 mm short of the bar's back face so they hold only its leading edge.
    #
    # Measured on kitchen 1221 (job 2111308): at the midpoint the jaws closed to 0.0539 m on a
    # 0.050 m bar -- just touching -- then to 0.0069 the instant the pull began. The bar came
    # straight out of the fingers and the door never moved.
    #
    # bias="deep" inserts to the far end of the window instead. Right for a THICK bar, wrong for a
    # thin one, so it is the caller's choice rather than a constant here.
    offset = (PAD_OVERHANG_M - gap + DEEP_INSERT_MARGIN_M if bias == "deep"
              else PAD_OVERHANG_M - 0.5 * gap)
    app.log(f"[handle_grasp] {handle_prim.GetPath()} gap={gap:.4f} m -> window "
            f"[{PAD_OVERHANG_M - gap:.4f}, {PAD_OVERHANG_M:.4f}], offset={offset:.4f} m")
    return offset
#: Anubis's shoulder height, in the WORLD frame -- the base sits on the floor, so a base-frame z is
#: also a world one.
#:
#: MEASURED DIRECTLY OFF THE ROBOT ASSET, not eyeballed: `link11` and `link21` -- arm1's and arm2's
#: mount points -- both sit at base-frame z = 0.8233557939529419 in
#: source/isaaclab_assets/data/Robots/anubis_simvla.usd (Usd.Stage.Open + UsdGeom.XformCache, world
#: transform of each link against base_link's world transform, both at the asset's rest pose). It
#: agrees, to 5 decimal places, with two unrelated prior measurements of the same physical mount:
#: the 0.823356 the training pipeline already subtracts out of ee_6d_pos
#: (kitchen/mdp/observations.py) to put the policy's state in a shoulder-relative frame, and the
#: 0.82336 push_prim_geometry.py measured off anubis_final.urdf's arm1_base_link_joint by IK.
SHOULDER_Z = 0.823356

#: How far inside a handle bar's own ends the grasp height is kept clamped, so the fingers are never
#: asked to grip past the end of the bar. The finger pads themselves span roughly 0.035 m along the
#: bar's axis (anubis_final.urdf's finger collision box, see skills.py's ArmPushPose clearance_m
#: comment); 0.04 m is a little over double that, leaving room for ordinary IK settle error on top
#: of the pad's own extent.
FRIDGE_HANDLE_END_MARGIN_M = 0.04


def _fridge_handle_grasp_z(bar_lo_z: float, bar_hi_z: float, shoulder_z: float = SHOULDER_Z,
							end_margin: float = FRIDGE_HANDLE_END_MARGIN_M) -> float:
	"""Where along a VERTICAL handle bar to grip, given the bar's own z extent.

	The bar's bbox CENTRE is not special -- any height along it is a valid grip -- so the hypothesis
	tested here was to grip the point nearest Anubis's own shoulder (`shoulder_z`) instead, clamped
	so the fingers stay `end_margin` clear of the bar's own ends:

		target = clamp(shoulder_z, bar_lo_z + end_margin, bar_hi_z - end_margin)

	STATUS: TESTED AND REJECTED AS A DEFAULT. NOT SHIPPED -- see `_fridge_handle_target_z`, the
	caller-side switch that decides whether this function or the plain bbox centre is used; it
	defaults OFF. This function itself is unchanged and still correct as a pure height DECISION; it
	is kept, and every test in test_fridge_handle_grasp_height.py still passes, because a future
	session may want to pick this back up under different conditions. What follows is why it must
	not run by default today.

	THE ORIGINAL HYPOTHESIS (commit 82d046b6). Aiming at the bbox centre unconditionally, across a
	10-kitchen fridge campaign, the handle's centre height ranged 0.796-1.097 m against a 0.823 m
	shoulder, and straight-line shoulder-to-handle distance APPEARED to separate the outcomes
	cleanly: door-opens at 0.499-0.540 m, door-never-opens at 0.542-0.571 m. The two kitchens whose
	centre already landed near shoulder height were the two that worked, so gripping wherever ON THE
	BAR lands closest to shoulder height -- instead of always gripping the middle -- looked like the
	fix for the rest.

	THE CONTROLLED EXPERIMENT THAT REFUTED IT. That 10-kitchen table was an uncontrolled correlation:
	nav safety margin and base park offset also varied across those runs. Holding both FIXED at the
	values already known to work (nav safety 0.12, park offset 0.35) and changing ONLY the grasp
	height (this function vs. the plain bbox centre) on kitchens 1410 and 1413, 4 episodes each:

		                        kitchen 1410   kitchen 1413   total envs >= 60 deg
		before (bbox centre)       4/4            4/4                8
		after  (shoulder-seek)     0/4            4/4                4

	It cost four working environments and gained zero. Kitchen 1410's handle bbox centre measures
	z = 0.792, BELOW the 0.823356 m shoulder -- so this function's clamp pulled the grasp target UP
	by 31 mm (0.792 -> 0.823356), and that 31 mm alone turned a reliable 4/4 into 0/4.

	THE ACTUAL FINDING: the grasp is knife-edge sensitive to height, to roughly +-30 mm -- a single
	small, well-motivated correction (aim closer to the shoulder, using measured link positions, on
	a real per-kitchen bar) was enough to break a previously-100%-reliable grasp on its own. That
	sensitivity, not shoulder-seeking, is the transferable result of this experiment.

	AND: shoulder-to-handle distance, which looked like a clean separator across the original ten
	kitchens (door-opens at 0.499-0.540 m, door-never-opens at 0.542-0.571 m), IS NOT THE GOVERNING
	VARIABLE. That correlation did not survive the controlled test above -- kitchen 1410 sits well
	inside the "working" band under either height choice, yet one choice opens the door 4/4 and the
	other 0/4. Whatever actually governs the door-opens/door-never-opens split, it is not summarized
	by this one scalar distance; do not re-derive a fix from it without a controlled A/B.

	DEGENERATE CASE (unchanged): a bar shorter than `2 * end_margin` has no z that keeps `end_margin`
	clear at BOTH ends at once. Falling back to the bar's own centre -- rather than picking one end,
	or dropping the margin and risking an overhang -- keeps the same point every fridge handle used
	before this function existed, and is the least-bad compromise when there is no room to do
	better: it is never further from either end than a boundary pick would be from the other one.
	Nothing in this corpus is anywhere close to this short (every measured bar is 0.300 m tall
	against an 0.08 m threshold); the guard is defensive, not sized for a case anyone has observed.
	"""
	span = bar_hi_z - bar_lo_z
	if span <= 2.0 * end_margin:
		return 0.5 * (bar_lo_z + bar_hi_z)
	lo = bar_lo_z + end_margin
	hi = bar_hi_z - end_margin
	return min(max(shoulder_z, lo), hi)


#: Opt-in switch for the shoulder-seeking height above. Unset/empty = OFF, matching this module's
#: other boolean toggles (SIMVLA_SQUEEZE, SIMVLA_NECK, SIMVLA_BIMANUAL_FIXED): read truthy via
#: `_simvla_os.environ.get(...)`, same convention as SIMVLA_FRIDGE_PARK_PAST_M just above. OFF is the
#: shipped default -- see _fridge_handle_grasp_z's docstring for the controlled GPU A/B that measured
#: shoulder-seeking costing 4 of 8 previously-working environments and gaining none. Kept as an
#: opt-in, not deleted, for whoever wants to pick the experiment back up under different conditions.
SIMVLA_FRIDGE_SHOULDER_GRASP_ENV = "SIMVLA_FRIDGE_SHOULDER_GRASP"


def _fridge_handle_target_z(bar_lo_z: float, bar_hi_z: float) -> float:
	"""The grasp height _fridge_handle_target actually uses: bbox centre unless opted in.

	This is the seam that decides between the two candidates in _fridge_handle_grasp_z's docstring.
	Default (SIMVLA_FRIDGE_SHOULDER_GRASP unset) reproduces exactly what this module aimed at before
	commit 82d046b6 introduced the shoulder-seeking alternative: the handle bar's own bbox centre,
	unconditionally. Setting the env var switches to _fridge_handle_grasp_z's rejected hypothesis,
	for experimentation only -- it is not known to help and measured to hurt in the one controlled
	test run so far.
	"""
	if _simvla_os.environ.get(SIMVLA_FRIDGE_SHOULDER_GRASP_ENV):
		return _fridge_handle_grasp_z(bar_lo_z, bar_hi_z)
	return 0.5 * (bar_lo_z + bar_hi_z)


#: N_dir -> (approach normal in world xy, jaw quaternion). The normal points from the handle
#: TOWARD the robot, i.e. the side the hand comes in from; it matches plan_nav_to_prim's own
#: per-direction parking (N parks at lower x facing +x, E parks at higher y facing -y, ...).
#:
#: THE QUATERNIONS ARE CARRIED OVER VERBATIM from the pre-2026-08 planner and are the only part of
#: it kept. They encode Anubis's ee_link1 closing across a VERTICAL bar, and they differ from
#: plan_arm_handle_grasp's on purpose -- that one is for a HORIZONTAL cabinet pull. Checked for
#: N: (0.5,-0.5,0.5,-0.5) maps local +Z (approach) to world +x and local +-X (jaw travel) to world
#: +-y, so the jaws close across the handle's 0.07 m y-span against an 0.080 m opening. Never
#: exercised on hardware, though -- the hero video's fridge beat never ran -- so if the hand
#: arrives at the right PLACE in the wrong ORIENTATION, this table is the first thing to suspect.
_FRIDGE_DIR = {
	"N": ((-1.0,  0.0), (0.5, -0.5, 0.5, -0.5)),
	"S": (( 1.0,  0.0), (-0.5, 0.5, 0.5, -0.5)),
	"E": (( 0.0,  1.0), (0.0, 0.0, 0.70711, -0.70711)),
	"W": (( 0.0, -1.0), (0.70711, -0.70711, 0.0, 0.0)),
}


def _fridge_handle_target(app, params, offset_m):
	"""Shared body of the two fridge-handle planners.

	READS THE HANDLE'S BOUNDING BOX, NOT ITS PIVOT. The previous implementation called
	ExtractTranslation() on the handle prim and got (1.0884, -3.0351, 0.0) on kitchen 1300 -- the
	DOOR XFORM'S PIVOT, i.e. the hinge at floor level, 0.71 m from the handle and 1.03 m below it.
	Every constant that followed (`+ 0.7` for the door width, `- 0.06`, an absolute 0.92-0.98 m
	height) was a correction for that one mistake on that one asset, and it STILL landed 3.8 cm
	past the end of a bar spanning y in [-2.3568, -2.2868]. BBoxCache is the same call
	plan_arm_place already makes, and door_geometry.py measures the same centre offline.
	"""
	prim_path = params.get("prim_path")
	if not prim_path:
		raise ValueError("Prim path required for grasp")

	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")

	bbox = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
	).ComputeWorldBound(prim).ComputeAlignedRange()
	lo, hi = bbox.GetMin(), bbox.GetMax()
	cx, cy = (lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5

	try:
		normal, quat = _FRIDGE_DIR[app.N_dir]
	except KeyError:
		raise ValueError(f"Invalid navigation direction: {app.N_dir}")

	# The bar's bbox centre, by default -- see _fridge_handle_target_z. A shoulder-seeking
	# alternative exists (_fridge_handle_grasp_z, behind SIMVLA_FRIDGE_SHOULDER_GRASP) but a
	# controlled GPU A/B measured it costing working environments and gaining none; read
	# _fridge_handle_grasp_z's docstring before touching this. Either way the height is
	# deterministic (no jitter) and computed from the SAME bbox on both the pregrasp and the grasp
	# call, so the two steps approach and close at the same height on the bar.
	z = _fridge_handle_target_z(lo[2], hi[2])

	if offset_m == FRIDGE_GRASP_OFFSET_M:
		# The grasp, not the pregrasp: FRIDGE_PREGRASP_OFFSET_M is a deliberate 0.12 m approach
		# stand-off with no feasible window to centre in, so it passes through untouched.
		offset_m = _handle_grasp_offset(app, prim, normal)
	return [cx + normal[0] * offset_m, cy + normal[1] * offset_m, z, *quat]


#: N_dir -> (approach normal in world xy, jaw quaternion) for a HORIZONTAL bar.
#:
#: DERIVED, NOT COPIED, and verified rather than asserted. Anubis's ee frame has local +Z along the
#: approach and local +X along the jaw travel (see skills.GRASP_TOOL_FRAME). A bar whose axis is
#: horizontal must be pinched VERTICALLY -- the other perpendicular is straight into the door -- so
#: the frame wanted is
#:
#:     local +Z -> -normal   (the hand moves toward the handle)
#:     local +X -> world +Z  (jaws close top-to-bottom)
#:     local +Y -> Z x X,    right-handed
#:
#: Building the matrix as [X, Y, Z] columns with Y = approach x jaw gives determinant +1; the
#: obvious ordering (Y = jaw x approach) gives -1, a REFLECTION, and the quaternion extracted from
#: it points the hand somewhere else entirely. That is the shape of error that ends in a video of a
#: gripper closing on air, so each entry below was checked by rotating the unit axes back.
#:
#: This differs from _FRIDGE_DIR because that table is for an UPRIGHT bar (fridge, cabinet door),
#: where the jaws close horizontally instead.
_BAR_DIR = {
	"N": ((-1.0,  0.0), (0.0, 0.70711, 0.0, 0.70711)),
	"S": (( 1.0,  0.0), (0.70711, 0.0, -0.70711, 0.0)),
	"E": (( 0.0,  1.0), (0.5, 0.5, -0.5, 0.5)),
	"W": (( 0.0, -1.0), (0.5, -0.5, -0.5, -0.5)),
}


def _bar_handle_target(app, params, offset_m, measured_offset: bool):
	"""Shared body of the two horizontal-bar handle planners.

	READS THE BOUNDING BOX, NOT THE PRIM'S TRANSLATION. On a dishwasher door the handle prim's
	ExtractTranslation() returns the DOOR'S PIVOT -- measured at z = 0.0556 on kitchen 1221, against
	a handle bar sitting at z = 0.7325. A planner that trusts it aims the grasp at the floor. This
	is the same defect door_geometry.py exists to document for the refrigerator.
	"""
	prim_path = params.get("prim_path")
	if not prim_path:
		raise ValueError("Prim path required for grasp")
	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")

	bbox = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
	).ComputeWorldBound(prim).ComputeAlignedRange()
	lo, hi = bbox.GetMin(), bbox.GetMax()
	cx, cy, cz = ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5)

	try:
		normal, quat = _BAR_DIR[app.N_dir]
	except KeyError:
		raise ValueError(f"Invalid navigation direction: {app.N_dir}")

	if measured_offset:
		# deep: a bar handle is thick, and the window's midpoint grips only its leading edge.
		offset_m = _handle_grasp_offset(app, prim, normal, fallback=offset_m, bias="deep")
	return [cx + normal[0] * offset_m, cy + normal[1] * offset_m, cz, *quat]


@register_planner("arm.bar_handle_pregrasp")
def plan_arm_bar_handle_pregrasp(app: "GoalGeneratorApp", action: str, params: dict):
	return _bar_handle_target(app, params, FRIDGE_PREGRASP_OFFSET_M, measured_offset=False)


@register_planner("arm.bar_handle_grasp")
def plan_arm_bar_handle_grasp(app: "GoalGeneratorApp", action: str, params: dict):
	return _bar_handle_target(app, params, FRIDGE_GRASP_OFFSET_M, measured_offset=True)


@register_planner("arm.door_arc_pull")
def plan_arm_door_arc_pull(app: "GoalGeneratorApp", action: str, params: dict):
	"""The grasp pose, carried round the door's OWN hinge to an absolute door angle.

	Everything here is derived from the stage: the closed grasp pose comes from the same
	_bar_handle_target the grasp step authored (so the hand starts exactly where it is already
	holding), and the hinge point and axis come from door_geometry.measure_door, which reads the
	joint rather than guessing -- see door-hinge-axis-joint-frame for why physics:axis alone is not
	the axis.

	BOTH the position and the ORIENTATION are rotated. Rotating only the position would carry the
	hand round the arc while the jaws stayed level, and the bar tilts as the door falls -- by 21 deg
	the jaws would be 21 deg off the bar they are holding, which on a friction grip is a slip.

	SENSE COMES FROM THE JOINT LIMITS, not from an assumption about which way doors open. Half the
	doors in this corpus open under negative rotation; door_sweep_blocker swept every one of them
	the wrong way until that was fixed. Here the legal direction is simply the limit that is not
	zero.
	"""
	import door_geometry as _dg

	prim_path = params.get("prim_path")
	if not prim_path:
		raise ValueError("Prim path required for arm.door_arc_pull")
	to_deg = float(params.get("to_deg", 0.0))

	handle = app.stage.GetPrimAtPath(prim_path)
	if not handle:
		raise ValueError(f"Prim not found: {prim_path}")
	door_path = handle.GetParent().GetPath().pathString
	geom = _dg.measure_door(app.stage, door_path, require_vertical=False)

	# A vertical hinge belongs to nav.open_door_arc: that one arcs the BASE, which is what a
	# cabinet door needs and what this skill deliberately does not do.
	if abs(geom.axis_world[2]) > 0.9:
		raise ValueError(
			f"{door_path} has a VERTICAL hinge (axis {geom.axis_world}); arm.door_arc_pull is for "
			f"drop-down doors. Use nav.open_door_arc, which arcs the base about the hinge.")

	closed = _bar_handle_target(app, params, FRIDGE_GRASP_OFFSET_M, measured_offset=True)
	p_closed = Gf.Vec3d(closed[0], closed[1], closed[2])
	q_closed = Gf.Quatd(closed[3], Gf.Vec3d(closed[4], closed[5], closed[6]))

	sense = 1.0 if abs(geom.limit_upper_deg) >= abs(geom.limit_lower_deg) else -1.0
	rot = Gf.Rotation(Gf.Vec3d(*geom.axis_world), sense * to_deg)
	hinge = Gf.Vec3d(*geom.hinge_world)
	p_new = hinge + rot.TransformDir(p_closed - hinge)

	# ROTATING THE WRIST IS OPTIONAL, and on this arm it has to be.
	#
	# Carrying the orientation round the hinge keeps the jaws square to the tilting bar, which is
	# what a big sweep needs -- at 52 deg a level hand would be 52 deg off the bar it holds. But
	# MEASURED (job 2112070) the arm cannot do it WHILE GRIPPING: the jog converged in position to
	# 0.0014-0.0048 m and left 18.4-20.6 deg of rotation, burning all 300 steps against a 3 deg
	# gate. It only showed up once the gentler retreat let episodes still HOLD the bar at this step;
	# before that the bar was already gone and the free arm rotated easily.
	#
	# At a small waypoint the rotation buys nothing anyway: a 0.050 m bar tilted 15 deg still sits
	# well inside the 0.080 m the jaws span. Note this is NOT the same as tolerating 15 deg of
	# rotation ERROR -- the pads sit 0.09 m out, so 15 deg of error would swing them 0.024 m off a
	# 0.015 m budget. Authoring the level orientation asks for nothing the arm cannot give.
	if params.get("rotate", True):
		q_new = Gf.Quatd(rot.GetQuat()) * q_closed
	else:
		q_new = q_closed
	im = q_new.GetImaginary()
	return [p_new[0], p_new[1], p_new[2], q_new.GetReal(), im[0], im[1], im[2]]


@register_planner("arm.fridge_handle_pregrasp")
def plan_arm_fridge_handle_pregrasp(app: "GoalGeneratorApp", action: str, params: dict):
	return _fridge_handle_target(app, params, FRIDGE_PREGRASP_OFFSET_M)


@register_planner("arm.fridge_handle_grasp")
def plan_arm_fridge_handle_grasp(app: "GoalGeneratorApp", action: str, params: dict):
	return _fridge_handle_target(app, params, FRIDGE_GRASP_OFFSET_M)


@register_planner("arm.place")
def plan_arm_place(app: "GoalGeneratorApp", action: str, params: dict):
	"""
	'Move arm to place': a point above the target's bbox, offset by app.N_dir.

	Returns [x, y, z] — the IK target, and the whole goal. The quaternion is NOT authored: the
	executor fills it from the live end-effector pose. v1 had to say that by writing
	SkillFlag.PLACE (999.0) into quaternion slot 3 and hoping `goal[:, 3] >= THRESHOLD` fired; in
	v2 the step names arm.place, and a skill id cannot be mistaken for a coordinate. That is the
	whole trick, and the reason the 999 is gone from this return.
	"""
	prim_path = params.get("prim_path")

	if not prim_path:
		raise ValueError("Prim path required for place")

	prim = app.stage.GetPrimAtPath(prim_path)
	if not prim:
		raise ValueError(f"Prim not found: {prim_path}")

	bbox_range = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
	).ComputeWorldBound(prim).ComputeAlignedRange()

	object_min_bound, object_max_bound = bbox_range.GetMin(), bbox_range.GetMax()
	position = (
		torch.tensor(object_min_bound) + torch.tensor(object_max_bound)
	) / 2  # center

	if not app.N_dir:
		raise ValueError("Navigation direction 'N_dir' must be set before placing.")

	grasp_pos_place = torch.zeros(3)
	# SIMVLA_PLACE_ABOVE overrides how far above the target's bbox top the goal is authored.
	# WHY IT EXISTS: arm.grasp gets a clearance pass (grasp_clearance raises every candidate until
	# the gripper's OWN cuRobo spheres clear the kitchen, and drops the ones that cannot be
	# cleared) -- that pass is why grasping works at all. arm.place has no equivalent: it authors
	# a point 0.15 m above the target and never asks whether the gripper fits there. On the sink
	# pilot the robot reached the sink (nav trace: dist_to_goal 0.002 m) and then the PLACE plan
	# failed -- 10 of the resets that follow an arrival are plan_fail_r, against 7 timeouts and 2
	# dropped mugs. This knob is the cheap test of that hypothesis; the real fix is to give
	# arm.place the same sphere-clearance treatment arm.grasp has.
	# Unset = 0.15, byte-identical to before.
	# See the arm.place declaration in skills.py: 0.1 is the historical constant, sized for a
	# table top, and a plate needs 0.0. Defaulted here as well as there so a goal file authored
	# before the param existed replays identically. `is None` rather than `or`, because 0.0 is a
	# MEANINGFUL value here and `params.get("inset_m") or 0.1` would silently turn it back into 0.1.
	_inset = params.get("inset_m")
	offset = 0.1 if _inset is None else float(_inset)
	# GENTLE PLACE. When the step declares clearance/eef offsets, the authored height is decided by
	# the SUPPORT and the OBJECT -- bbox_top + gripper-above-object-base + clearance -- instead of
	# one constant that knew about neither. Both zero (the default) keeps SIMVLA_PLACE_ABOVE, so an
	# unparameterised step is byte-identical to every goal file authored before this existed.
	# SIMVLA_PLACE_CLEARANCE overrides the clearance per run, the same way SIMVLA_PLACE_MIN_Z does
	# for arm.bowl_place, so a release height can be swept without a re-emit.
	_clear = float(os.environ.get("SIMVLA_PLACE_CLEARANCE") or params.get("clearance_m") or 0.0)
	_eef_above = float(params.get("eef_above_base_m") or 0.0)
	if _clear or _eef_above:
		put_above = _eef_above + _clear
	else:
		put_above = float(os.environ.get("SIMVLA_PLACE_ABOVE", "0.15") or 0.15)

	if app.N_dir == "N":
		grasp_pos_place[0:3] = torch.tensor(
			[object_min_bound[0] + offset, position[1], object_max_bound[2] + put_above]
		)
	elif app.N_dir == "E":
		grasp_pos_place[0:3] = torch.tensor(
			[position[0], object_max_bound[1] - offset, object_max_bound[2] + put_above]
		)
	elif app.N_dir == "S":
		grasp_pos_place[0:3] = torch.tensor(
			[object_max_bound[0] - offset, position[1], object_max_bound[2] + put_above]
		)
	elif app.N_dir == "W":
		grasp_pos_place[0:3] = torch.tensor(
			[position[0], object_min_bound[1] + offset, object_max_bound[2] + put_above]
		)
	else:
		raise ValueError(f"Invalid navigation direction: {app.N_dir}")

	return grasp_pos_place.tolist()


# Fails at import — in the GUI, before a scene is loaded — if a skill declares
# plan = _authored(...) and no body above supplies it.
validate_planners()


if __name__ == "__main__":
	_load_gui()
	root = tk.Tk()
	root.option_add("*Font", ("DejaVu Sans", 10))
	app = GoalGeneratorApp(root)
	root.mainloop()
