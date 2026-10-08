import torch, math, random, json, numpy as np, re, os, ipdb
from shapely.geometry import box, Point
from scipy.spatial.transform import Rotation as R

def parse_thumbnail_name(filename):
	"""
	Helper function to parse 'segment_05_right.usda.png' into (4, 'right').
	The segment index is 0-based (segment_01 -> index 0).
	"""
	# Match 'segment_XX_hand.usda.png'
	match = re.match(r"segment_(\d+)_(\w+)\.usda\.png", filename)
	if match:
		# segment_index is 0-based, filenames are 1-based
		segment_index = int(match.group(1)) - 1
		hand_str = match.group(2) # 'right' or 'left'
		
		if hand_str not in ('right', 'left'):
			print(f"Warning: Unknown hand '{hand_str}' in {filename}")
			return None
			
		return (segment_index, hand_str)
	
	print(f"Warning: Could not parse filename: {filename}")
	return None


def select_thumbnails(thumbnail_path, thumbs_per_row=5):
	"""
	MODIFIED: This function now displays the GUI and returns a list
	of selected tuples, e.g., [(0, 'right'), (5, 'left'), ...].
	"""
	# GUI libraries are optional for headless collection and GPU workers.
	import tkinter as tk
	from PIL import Image, ImageTk

	png_files = sorted(
		[f for f in os.listdir(thumbnail_path) if f.lower().endswith(".png")],
		# Sort by hand, then segment number
		key=lambda f: (f.split('_')[-1].split('.')[0], int(f.split('_')[1]))
	)
	
	if not png_files:
		print("No PNG files found in thumbnail path.")
		return []

	selected_preferences = [] # Will store the (index, 'hand') tuples
	
	root = tk.Toplevel()
	root.title(f'Select Grasp Preferences ({len(png_files)} options)')

	# --- Create Scrollable Area ---
	canvas = tk.Canvas(root, borderwidth=0)
	scrollable_frame = tk.Frame(canvas)
	scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
	canvas.configure(yscrollcommand=scrollbar.set)
	
	# Bind mouse wheel scrolling
	def _on_mousewheel(event):
		if os.name == 'nt' or os.name == 'posix':
			 canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
		else:
			 canvas.yview_scroll(int(-1 * event.delta), "units")

	canvas.bind_all("<MouseWheel>", _on_mousewheel)

	scrollable_frame.bind(
		"<Configure>",
		lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
	)

	canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
	
	canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
	
	root.geometry("800x600") 

	# --- Populate GUI ---
	photo_images = []  # To prevent garbage collection
	var_list = []
	thumb_size = 128
	
	for idx, file in enumerate(png_files):
		parsed_info = parse_thumbnail_name(file)
		if not parsed_info:
			continue 
		
		segment_id = parsed_info 

		row, col = divmod(idx, thumbs_per_row)
		
		frame = tk.Frame(scrollable_frame, relief=tk.RAISED, borderwidth=1)
		frame.grid(row=row, column=col, padx=5, pady=5)
		
		img_path = os.path.join(thumbnail_path, file)
		try:
			img = Image.open(img_path)
			img.thumbnail((thumb_size, thumb_size))
			photo = ImageTk.PhotoImage(img)
			photo_images.append(photo) 
			
			tk.Label(frame, image=photo).pack()
			
			var = tk.BooleanVar()
			var_list.append((var, segment_id)) 
			
			label_text = f"Seg {segment_id[0]+1} ({segment_id[1]})"
			tk.Checkbutton(frame, text=label_text, variable=var).pack()
			
		except Exception as e:
			print(f"Error loading image {img_path}: {e}")
			tk.Label(frame, text=f"Error\n{file}").pack()


	def confirm_selection():
		for var, segment_id in var_list:
			if var.get():
				selected_preferences.append(segment_id)
		root.destroy() 

	confirm_btn = tk.Button(root, text=f"Confirm Selection(s)", command=confirm_selection)
	confirm_btn.pack(pady=10, side=tk.BOTTOM)
	
	root.wait_window() 
	
	print(f"Selected {len(selected_preferences)} grasps: {selected_preferences}")
	return selected_preferences

# ---
# FUNCTION 3: MODIFIED select_thumbnails_cached (Name unchanged)
# ---
def select_thumbnails_cached(thumbnail_path, cache_file=None):
	"""
	MODIFIED: This function now correctly caches and loads a list of tuples 
	(e.g., [(0, 'right'), ...]) to and from the JSON cache file.
	"""
	if cache_file is None:
		cache_file = os.path.join(thumbnail_path, "selected_indices.json")
	
	# Try to load from cache
	if os.path.exists(cache_file):
		try:
			with open(cache_file, "r") as f:
				cached_data = json.load(f)
				# JSON saves tuples as lists, so we convert them back
				selected_indices = [tuple(item) for item in cached_data.get("selected_indices", [])]
			if selected_indices:
				print(f"Loaded {len(selected_indices)} preferences from cache: {cache_file}")
				return selected_indices
		except json.JSONDecodeError:
			print(f"Cache file {cache_file} is corrupt. Re-selecting.")

	# If no cache, open the GUI
	print("No valid cache found. Opening selection GUI...")
	selected_indices = select_thumbnails(thumbnail_path)
	
	# Save selection to cache
	with open(cache_file, "w") as f:
		# The list of tuples will be saved as a list of lists
		json.dump({"selected_indices": selected_indices}, f, indent=2)
		print(f"Saved {len(selected_indices)} preferences to cache: {cache_file}")
		
	return selected_indices


#def select_thumbnails_cached(thumbnail_path, cache_file=None):
#	if cache_file is None: cache_file = os.path.join(thumbnail_path, "selected_indices.json")
#	if os.path.exists(cache_file):
#		with open(cache_file, "r") as f: return json.load(f).get("selected_indices", [])
#	selected_indices = select_thumbnails(thumbnail_path)
#	with open(cache_file, "w") as f: json.dump({"selected_indices": selected_indices}, f, indent=2)
#	return selected_indices
#
#def select_thumbnails(thumbnail_path, thumbs_per_row=5):
#	png_files = sorted([f for f in os.listdir(thumbnail_path) if f.lower().endswith(".png")])
#	if not png_files: return []
#	selected_indices = []
#	root = tk.Toplevel()
#	root.title("Select Grasp Preferences")
#	canvas = tk.Canvas(root); scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview); scrollable_frame = tk.Frame(canvas)
#	scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
#	canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
#	canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
#	photo_images, var_list, thumb_size = [], [], 128
#	for idx, file in enumerate(png_files):
#		row, col = divmod(idx, thumbs_per_row)
#		frame = tk.Frame(scrollable_frame, relief=tk.RAISED, borderwidth=1); frame.grid(row=row, column=col, padx=5, pady=5)
#		img = Image.open(os.path.join(thumbnail_path, file)); img.thumbnail((thumb_size, thumb_size)); photo = ImageTk.PhotoImage(img)
#		photo_images.append(photo)
#		tk.Label(frame, image=photo).pack()
#		var = tk.BooleanVar(); tk.Checkbutton(frame, text=f"{idx}", variable=var).pack(); var_list.append(var)
#	def confirm_selection():
#		for i, var in enumerate(var_list):
#			if var.get(): selected_indices.append(i)
#		root.destroy()
#	confirm_btn = tk.Button(root, text="Confirm Selection", command=confirm_selection); confirm_btn.pack(pady=10, side=tk.BOTTOM)
#	root.wait_window()
#	return selected_indices

def merge_free_spaces(spaces, tolerance=1e-8):
	if not spaces: return []
	rects = [list(s[0]) for s in spaces]
	while True:
		merged_in_pass, i = False, 0
		while i < len(rects):
			j = i + 1
			while j < len(rects):
				r1, r2 = rects[i], rects[j]
				if (abs(r1[2] - r2[0]) < tolerance and abs(r1[1] - r2[1]) < tolerance and abs(r1[3] - r2[3]) < tolerance):
					r1[2] = r2[2]; rects.pop(j); merged_in_pass = True; continue
				if (abs(r1[3] - r2[1]) < tolerance and abs(r1[0] - r2[0]) < tolerance and abs(r1[2] - r2[2]) < tolerance):
					r1[3] = r2[3]; rects.pop(j); merged_in_pass = True; continue
				j += 1
			i += 1
		if not merged_in_pass: break
	return [(tuple(r), ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)) for r in rects]

def find_free_spaces_grid(bounds, obstacles):
	x0, y0, x1, y1 = bounds
	all_x, all_y = {x0, x1}, {y0, y1}
	for obs in obstacles:
		ox_min, oy_min, ox_max, oy_max = obs.bounds
		all_x.update([ox_min, ox_max]); all_y.update([oy_min, oy_max])
	sorted_x, sorted_y = sorted(list(all_x)), sorted(list(all_y))
	free_spaces = []
	for i in range(len(sorted_x) - 1):
		for j in range(len(sorted_y) - 1):
			cell_x0, cell_x1, cell_y0, cell_y1 = sorted_x[i], sorted_x[i+1], sorted_y[j], sorted_y[j+1]
			center_x, center_y = (cell_x0 + cell_x1) / 2, (cell_y0 + cell_y1) / 2
			center_point = Point(center_x, center_y)
			if not any(obs.contains(center_point) for obs in obstacles):
				free_spaces.append(((cell_x0, cell_y0, cell_x1, cell_y1), (center_x, center_y)))
	return free_spaces


def make_walls_from_bounds(world_bounds, wall_material="wall_material", height=3.0, thickness=0.1):
	"""
	Generates configuration strings for four boundary walls based on world bounds.
	This version is corrected to place walls at the correct Z-height.
	"""
	x_min, y_min, x_max, y_max = world_bounds
	x_min -= 0.2
	y_min -= 0.2
	x_max += 0.2
	y_max += 0.2
	# 1. Pre-calculate all values, including the correct Z-center.
	args = {
		"width": x_max - x_min,
		"depth": y_max - y_min,
		"height": height,
		"thickness": thickness,
		"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
		"x_mid": (x_min + x_max) / 2,
		"y_mid": (y_min + y_max) / 2,
		"z_mid": height / 2,  # <-- ADDED THIS LINE FOR CORRECT Z-COORDINATE
		"wall_material": wall_material
	}

	# 2. Update the templates to use the new 'z_mid' variable.
	templates = [
		("wall_01", "{x_mid:.3f}, {y_min:.3f}, {z_mid:.3f}", "(0.70711, 0.70711, 0.0, 0.0)", "{width:.3f}, {height}, {thickness}"),
		("wall_02", "{x_max:.3f}, {y_mid:.3f}, {z_mid:.3f}", "(0.5, 0.5, 0.5, 0.5)", "{depth:.3f}, {height}, {thickness}"),
		("wall_03", "{x_min:.3f}, {y_mid:.3f}, {z_mid:.3f}", "(0.5, 0.5, 0.5, 0.5)", "{depth:.3f}, {height}, {thickness}"),
		("wall_04", "{x_mid:.3f}, {y_max:.3f}, {z_mid:.3f}", "(0.70711, 0.70711, 0.0, 0.0)", "{width:.3f}, {height}, {thickness}")
	]

	# 3. The generation loop remains the same.
	return tuple(f'''{n} = AssetBaseCfg(
	prim_path="{{ENV_REGEX_NS}}/{n}",
	init_state=AssetBaseCfg.InitialStateCfg(pos=({p.format(**args)}), rot={r}),
	spawn=sim_utils.UsdFileCfg(
		usd_path=f"file:{{ISAACLAB_ASSETS_DATA_DIR}}/floor.usd",
		scale=({s.format(**args)}),
		visual_material=MdlFileCfg(mdl_path={wall_material}),
	),
	collision_group=-1,
)''' for n, p, r, s in templates)

def load_grasp_file(input_path, robot_name):
	if "use_data" not in input_path:
		print("Error: 'use_data' not in input_path.")
		return None

	try:
		# Get the object name, e.g., 'core_bottle_...'
		obj_name = input_path.split("/")[-3]
	except Exception as e:
		print(f"Error splitting input_path '{input_path}': {e}")
		return None

	# Prefer the user's configured dataset root. USDs often preserve the maintainer's absolute
	# `BODex_path`; following it makes a public asset download silently depend on that workstation.
	# Retain the path-derived fallback for older users who have not set BODEX_OBJ_DIR.
	configured_root = os.environ.get("BODEX_OBJ_DIR")
	if configured_root:
		base_dir = os.path.expanduser(configured_root)
	else:
		base_dir = input_path.split("use_data")[0]

	# This is the '.../sem_.../floating' directory in graspdata_final.
	grasp_dir = os.path.join(base_dir, "graspdata_final", robot_name, obj_name, "floating")

	if not os.path.exists(grasp_dir):
		print(f"Error: Grasp directory not found: {grasp_dir}")
		return None

	# Find the base name, e.g., 'scale010_grasp'
	base_names = set()
	for f in os.listdir(grasp_dir):
		if f.endswith("_right.npy"):
			base_names.add(f.replace("_right.npy", ""))
		elif f.endswith("_left.npy"):
			base_names.add(f.replace("_left.npy", ""))

	if not base_names:
		print(f"No _right.npy or _left.npy files found in {grasp_dir}")
		return None

	# Assuming only one grasp type (e.g., 'scale010_grasp') per folder
	base_name = sorted(base_names)[0]

	# Return the full path *without* the extension
	return os.path.join(grasp_dir, base_name)
#	if "/use_data/" not in input_path: return None
#	base_dir, obj_part = input_path.split("/use_data/")
#	obj_name = obj_part.split("/")[0]
#	grasp_dir = os.path.join(base_dir, "graspdata", robot_name, obj_name, "floating")
#	if not os.path.exists(grasp_dir): return None
#	files = [f for f in os.listdir(grasp_dir) if f.endswith("010_grasp.npy")]
#	if not files: files = [f for f in os.listdir(grasp_dir) if f.endswith(".npy")]
#	return os.path.join(grasp_dir, random.choice(files)) if files else None

def find_first_free_direction(position, free_squares, step=0.03, max_steps=50):
	x, y = position[0], position[1]
	directions = {"E": (0, 1), "W": (0, -1), "S": (1, 0), "N": (-1, 0)}
	bounds_list = [b for b, _ in free_squares]
	for i in range(1, max_steps + 1):
		dist = step * i
		for dname, (dx, dy) in directions.items():
			px, py = x + dx * dist, y + dy * dist
			for (x0, y0, x1, y1) in bounds_list:
				if x0 <= px <= x1 and y0 <= py <= y1:
					return dname, (px, py)
	return None
