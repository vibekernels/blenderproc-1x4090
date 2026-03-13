import blenderproc as bproc  # must be first import for BlenderProc
import bpy
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUP_MODEL_PATH = os.path.join(SCRIPT_DIR, "CC0_-_Red_Cup.usdz")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

TARGET_CUP_DIAMETER = 0.09  # ~9 cm, real solo cup top diameter
CUP_SPACING = 0.12        # meters between cup centres
TABLE_LENGTH = 2.44        # ~8 ft regulation table
TABLE_WIDTH = 0.61         # ~2 ft
TABLE_HEIGHT = 0.76        # ~30 in

RENDER_SAMPLES = 32
RESOLUTION = (1920, 1080)


# ---------------------------------------------------------------------------
# USDZ loader (uses Blender's native USD importer)
# ---------------------------------------------------------------------------
def _build_fixed_usdz(source_path: str, output_path: str) -> None:
    """
    Create a fixed USDZ by extracting source_path, applying MaterialBindingAPI
    to any prim that has bindings but lacks the schema declaration, and
    repacking as a new USDZ (ZIP_STORED, as required by the USDZ spec).
    """
    import zipfile
    import tempfile
    from pxr import Usd, UsdShade

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(source_path, 'r') as zf:
            zf.extractall(tmpdir)

        usdc_path = os.path.join(tmpdir, 'scene.usdc')
        stage = Usd.Stage.Open(usdc_path)
        for prim in stage.Traverse():
            if 'MaterialBindingAPI' in prim.GetAppliedSchemas():
                continue
            if UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath():
                UsdShade.MaterialBindingAPI.Apply(prim)
        stage.Save()

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zf:
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    full = os.path.join(root, file)
                    zf.write(full, os.path.relpath(full, tmpdir))


def load_usdz(filepath: str) -> list:
    """Import a .usdz file via bpy and return BlenderProc MeshObjects."""
    if not os.path.isfile(filepath):
        print(f"ERROR: Cup model not found at {filepath}", file=sys.stderr)
        sys.exit(1)

    fixed_path = filepath.rsplit('.', 1)[0] + '_fixed.usdz'
    if not os.path.isfile(fixed_path):
        print(f"Building fixed USDZ: {os.path.basename(fixed_path)}")
        _build_fixed_usdz(filepath, fixed_path)

    # Track which objects exist before import
    existing = set(bpy.data.objects.keys())

    bpy.ops.wm.usd_import(filepath=fixed_path)

    # Find newly imported objects
    new_names = set(bpy.data.objects.keys()) - existing
    mesh_objects = []
    for name in new_names:
        obj = bpy.data.objects[name]
        if obj.type == "MESH":
            mesh_objects.append(bproc.python.types.MeshObjectUtility.MeshObject(obj))

    if not mesh_objects:
        print("ERROR: No mesh objects found in the imported USDZ.", file=sys.stderr)
        sys.exit(1)

    # Unparent meshes while preserving world transforms
    bpy.ops.object.select_all(action='DESELECT')
    for mobj in mesh_objects:
        mobj.blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0].blender_obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.select_all(action='DESELECT')

    # Delete non-mesh imported objects (empty hierarchy)
    for name in list(new_names):
        if name in bpy.data.objects:
            obj = bpy.data.objects[name]
            if obj.type != "MESH":
                bpy.data.objects.remove(obj, do_unlink=True)

    print(f"Loaded {len(mesh_objects)} mesh(es) from {os.path.basename(filepath)}")
    return mesh_objects


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
def triangle_positions(rows: int, spacing: float, origin: np.ndarray) -> list:
    """
    Return cup (x, y) positions for a beer-pong triangle.
    Row 0 has `rows` cups, row 1 has `rows-1`, etc.
    Origin is the centre of the back row (row 0).
    """
    positions = []
    row_dy = spacing * np.sqrt(3) / 2  # equilateral triangle row spacing
    for r in range(rows):
        n = rows - r
        x_start = -(n - 1) * spacing / 2.0
        for c in range(n):
            x = origin[0] + x_start + c * spacing
            y = origin[1] + r * row_dy
            positions.append((x, y))
    return positions


def create_cup(template, location):
    """Linked-duplicate the cup template and place it at (x, y) on the table."""
    cup = template.duplicate(linked=True)
    cup.set_location([location[0], location[1], TABLE_HEIGHT])
    return cup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
bproc.init()

# Force GPU-only rendering
_cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
_cycles_prefs.compute_device_type = 'OPTIX'
_cycles_prefs.get_devices()
for _device in _cycles_prefs.devices:
    _device.use = _device.type == 'OPTIX'
bpy.context.scene.cycles.device = 'GPU'

# --- Load cup model --------------------------------------------------------
cup_objs = load_usdz(CUP_MODEL_PATH)
# If the model has multiple parts, join them into one object
if len(cup_objs) > 1:
    cup_template = cup_objs[0]
    for other in cup_objs[1:]:
        other_bpy = other.blender_obj
        cup_template.blender_obj.select_set(True)
        other_bpy.select_set(True)
        bpy.context.view_layer.objects.active = cup_template.blender_obj
    bpy.ops.object.join()
    bpy.ops.object.select_all(action='DESELECT')
else:
    cup_template = cup_objs[0]

# First bake existing transforms (including parent scale) into mesh
cup_template.persist_transformation_into_mesh(location=True, rotation=True, scale=True)

# Now bbox is in true world units; compute scale to target cup diameter
bbox_raw = cup_template.get_bound_box()
raw_diameter = max(bbox_raw[:, 0]) - min(bbox_raw[:, 0])
cup_scale = TARGET_CUP_DIAMETER / raw_diameter if raw_diameter > 0 else 0.01
cup_template.set_scale([cup_scale, cup_scale, cup_scale])
cup_template.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
cup_template.move_origin_to_bottom_mean_point()

bbox = cup_template.get_bound_box()
cup_diameter = max(bbox[:, 0]) - min(bbox[:, 0])
CUP_SPACING = cup_diameter * 1.05

# Hide template off-screen (we use duplicates)
cup_template.set_location([0, 0, -10])

# --- Table -----------------------------------------------------------------
table_top = bproc.object.create_primitive(
    "CUBE",
    scale=[TABLE_WIDTH / 2, TABLE_LENGTH / 2, 0.02],
    location=[0, 0, TABLE_HEIGHT - 0.02],
)
table_mat = bproc.material.create("table_wood")
table_mat.set_principled_shader_value("Base Color", [0.35, 0.22, 0.10, 1.0])
table_mat.set_principled_shader_value("Roughness", 0.6)
table_top.replace_materials(table_mat)

# Legs
for lx, ly in [
    (-TABLE_WIDTH / 2 + 0.04, -TABLE_LENGTH / 2 + 0.06),
    (TABLE_WIDTH / 2 - 0.04, -TABLE_LENGTH / 2 + 0.06),
    (-TABLE_WIDTH / 2 + 0.04, TABLE_LENGTH / 2 - 0.06),
    (TABLE_WIDTH / 2 - 0.04, TABLE_LENGTH / 2 - 0.06),
]:
    leg = bproc.object.create_primitive(
        "CUBE",
        scale=[0.03, 0.03, TABLE_HEIGHT / 2 - 0.02],
        location=[lx, ly, TABLE_HEIGHT / 2 - 0.02],
    )
    leg.replace_materials(table_mat)

# Floor
floor = bproc.object.create_primitive(
    "PLANE", scale=[5, 5, 1], location=[0, 0, 0]
)
floor_mat = bproc.material.create("floor")
floor_mat.set_principled_shader_value("Base Color", [0.15, 0.15, 0.15, 1.0])
floor_mat.set_principled_shader_value("Roughness", 0.9)
floor.replace_materials(floor_mat)

# --- Place cups in two triangles (one per side) ----------------------------
cups = []

# Side A: triangle pointing toward +Y, centred at y = -TABLE_LENGTH/2 + offset
side_a_origin = np.array([0, -TABLE_LENGTH / 2 + CUP_SPACING * 2.5])
for pos in triangle_positions(4, CUP_SPACING, side_a_origin):
    cups.append(create_cup(cup_template, pos))

# Side B: triangle pointing toward -Y, centred at y = +TABLE_LENGTH/2 - offset
side_b_origin = np.array([0, TABLE_LENGTH / 2 - CUP_SPACING * 2.5])
for pos in triangle_positions(4, CUP_SPACING, side_b_origin):
    cup = create_cup(cup_template, pos)
    # Flip so triangle points the other way
    cup.set_location([
        pos[0],
        side_b_origin[1] - (cup.get_location()[1] - side_b_origin[1]),
        TABLE_HEIGHT,
    ])
    cups.append(cup)

# --- Ping pong balls (optional fun detail) ---------------------------------
for i in range(2):
    ball = bproc.object.create_primitive(
        "SPHERE",
        scale=[0.02, 0.02, 0.02],
        location=[
            0.05 * (i * 2 - 1),
            0,
            TABLE_HEIGHT + 0.02,
        ],
    )
    ball_mat = bproc.material.create(f"ball_{i}")
    ball_mat.set_principled_shader_value("Base Color", [1.0, 1.0, 1.0, 1.0])
    ball_mat.set_principled_shader_value("Roughness", 0.3)
    ball.replace_materials(ball_mat)

# --- Lighting --------------------------------------------------------------
# Overhead warm light
key = bproc.types.Light()
key.set_type("AREA")
key.set_location([0, 0, TABLE_HEIGHT + 2.0])
key.set_rotation_euler([0, 0, 0])
key.set_energy(300)
key.set_color([1.0, 0.92, 0.82])

# Fill from the side
fill = bproc.types.Light()
fill.set_type("POINT")
fill.set_location([2, -1, TABLE_HEIGHT + 1.5])
fill.set_energy(150)
fill.set_color([0.85, 0.9, 1.0])

# Subtle backlight
back = bproc.types.Light()
back.set_type("POINT")
back.set_location([-1.5, 2, TABLE_HEIGHT + 1.0])
back.set_energy(80)
back.set_color([1.0, 0.95, 0.85])

# Dark ambient
bproc.renderer.set_world_background([0.01, 0.01, 0.02], strength=1)

# --- Camera (ZED 2i, 2.1mm lens, single camera at 1080p) ------------------
# Sensor: 1/3" 4MP CMOS, 2688x1520 native, 2µm pixel pitch
# At 1080p (1920x1080): scale = 1920/2688 ≈ 0.714
# fx = fy = 2.1mm / 0.002mm * (1920/2688) ≈ 750 px
ZED_FX = 750.0
ZED_FY = 750.0
ZED_CX = 960.0   # 1920 / 2
ZED_CY = 540.0   # 1080 / 2

bproc.camera.set_resolution(*RESOLUTION)
K = np.array([
    [ZED_FX, 0,      ZED_CX],
    [0,      ZED_FY, ZED_CY],
    [0,      0,      1     ],
])
bproc.camera.set_intrinsics_from_K_matrix(K, RESOLUTION[0], RESOLUTION[1],
                                           clip_start=0.01, clip_end=50)

# Camera height: Unitree H1 head ≈ 1.80 m from floor
CAM_Z = 1.80
# Tilt angle to look down at table (table is at 0.76m, so ~1.04m below camera)
# atan2(1.04, distance) gives a reasonable downward pitch

# Pose 1: Standing at one end of the table, looking down the length
cam1 = bproc.math.build_transformation_mat(
    [0, -TABLE_LENGTH / 2 - 0.3, CAM_Z],
    [np.radians(50), 0, 0],
)
bproc.camera.add_camera_pose(cam1)

# Pose 2: Standing at the side of the table, looking across
cam2 = bproc.math.build_transformation_mat(
    [TABLE_WIDTH / 2 + 0.4, 0, CAM_Z],
    [np.radians(50), 0, np.radians(90)],
)
bproc.camera.add_camera_pose(cam2)

# Pose 3: Standing at the corner, looking diagonally
cam3 = bproc.math.build_transformation_mat(
    [TABLE_WIDTH / 2 + 0.3, -TABLE_LENGTH / 2 - 0.2, CAM_Z],
    [np.radians(50), 0, np.radians(30)],
)
bproc.camera.add_camera_pose(cam3)

# Pose 4: Standing at the other end, looking back
cam4 = bproc.math.build_transformation_mat(
    [0.1, TABLE_LENGTH / 2 + 0.3, CAM_Z],
    [np.radians(50), 0, np.radians(180)],
)
bproc.camera.add_camera_pose(cam4)

# --- Render ----------------------------------------------------------------
bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)
bproc.renderer.set_noise_threshold(0.1)
bproc.renderer.set_light_bounces(
    diffuse_bounces=2, glossy_bounces=2, max_bounces=4,
    transparent_max_bounces=4,
)
bproc.renderer.enable_depth_output(activate_antialiasing=False)

import time as _time
_t0 = _time.time()
print("Rendering...")
data = bproc.renderer.render()
print(f"render+load took {_time.time()-_t0:.2f}s")

os.makedirs(OUTPUT_DIR, exist_ok=True)
bproc.writer.write_hdf5(OUTPUT_DIR, data)
print(f"Done! Output written to {OUTPUT_DIR}/")
