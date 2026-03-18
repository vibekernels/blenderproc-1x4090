import blenderproc as bproc  # must be first import
import bpy
import numpy as np
import os
import sys
import json
import glob
import time
import argparse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUP_MODEL_PATH = os.path.join(SCRIPT_DIR, "CC0_-_Red_Cup.usdz")
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
HDRI_DIR = os.path.join(ASSETS_DIR, "hdris")
CCTEXTURES_DIR = os.path.join(ASSETS_DIR, "cctextures")

TARGET_CUP_DIAMETER = 0.09
TABLE_LENGTH = 2.44
TABLE_WIDTH = 0.61
TABLE_HEIGHT = 0.76

RENDER_SAMPLES = 32
RESOLUTION = (1920, 1080)

# Camera intrinsics (ZED 2i at 1080p)
ZED_FX = 750.0
ZED_FY = 750.0
ZED_CX = 960.0
ZED_CY = 540.0

# Max cups we'll ever need (10 per side = triangle(4) = 10)
MAX_CUPS_PER_SIDE = 10

# Distractor object limits
MAX_DISTRACTORS = 6


# ---------------------------------------------------------------------------
# USDZ loader
# ---------------------------------------------------------------------------
def _build_fixed_usdz(source_path: str, output_path: str) -> None:
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
    if not os.path.isfile(filepath):
        print(f"ERROR: Cup model not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    fixed_path = filepath.rsplit('.', 1)[0] + '_fixed.usdz'
    if not os.path.isfile(fixed_path):
        _build_fixed_usdz(filepath, fixed_path)
    existing = set(bpy.data.objects.keys())
    bpy.ops.wm.usd_import(filepath=fixed_path)
    new_names = set(bpy.data.objects.keys()) - existing
    mesh_objects = []
    for name in new_names:
        obj = bpy.data.objects[name]
        if obj.type == "MESH":
            mesh_objects.append(bproc.python.types.MeshObjectUtility.MeshObject(obj))
    if not mesh_objects:
        print("ERROR: No mesh objects found in USDZ.", file=sys.stderr)
        sys.exit(1)
    bpy.ops.object.select_all(action='DESELECT')
    for mobj in mesh_objects:
        mobj.blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0].blender_obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.select_all(action='DESELECT')
    for name in list(new_names):
        if name in bpy.data.objects:
            obj = bpy.data.objects[name]
            if obj.type != "MESH":
                bpy.data.objects.remove(obj, do_unlink=True)
    return mesh_objects


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
def triangle_positions(rows: int, spacing: float, origin: np.ndarray) -> list:
    positions = []
    row_dy = spacing * np.sqrt(3) / 2
    for r in range(rows):
        n = rows - r
        x_start = -(n - 1) * spacing / 2.0
        for c in range(n):
            x = origin[0] + x_start + c * spacing
            y = origin[1] + r * row_dy
            positions.append((x, y))
    return positions


def rows_for_count(n):
    """Return number of rows for a triangle with <= n cups."""
    # triangle(k) = k*(k+1)/2
    # 1->1, 3->2, 6->3, 10->4
    for k in range(1, 10):
        if k * (k + 1) // 2 >= n:
            return k
    return 4


def get_cup_3d_bbox(cup_obj):
    """Get 8 corners of the 3D bounding box in world coordinates."""
    bbox = cup_obj.get_bound_box()  # 8x3 array
    return bbox.tolist()


def get_camera_K():
    return [[ZED_FX, 0, ZED_CX], [0, ZED_FY, ZED_CY], [0, 0, 1]]


# ---------------------------------------------------------------------------
# Randomization functions
# ---------------------------------------------------------------------------
def random_table_color(rng):
    """Return a random plausible table color."""
    table_colors = [
        [0.35, 0.22, 0.10, 1.0],  # wood brown
        [0.10, 0.10, 0.10, 1.0],  # black
        [0.60, 0.60, 0.60, 1.0],  # grey/silver
        [0.15, 0.25, 0.15, 1.0],  # dark green (beer pong table)
        [0.05, 0.10, 0.35, 1.0],  # dark blue
        [0.45, 0.30, 0.15, 1.0],  # light wood
        [0.80, 0.80, 0.80, 1.0],  # white/light
        [0.30, 0.05, 0.05, 1.0],  # dark red
    ]
    return table_colors[rng.integers(len(table_colors))]


def random_floor_color(rng):
    colors = [
        [0.15, 0.15, 0.15, 1.0],  # dark grey
        [0.30, 0.25, 0.20, 1.0],  # wood floor
        [0.10, 0.10, 0.10, 1.0],  # near black
        [0.25, 0.25, 0.25, 1.0],  # medium grey
        [0.40, 0.35, 0.25, 1.0],  # light wood
        [0.05, 0.08, 0.05, 1.0],  # dark green (grass/astroturf)
        [0.20, 0.15, 0.12, 1.0],  # brown
    ]
    return colors[rng.integers(len(colors))]


def random_background_color(rng):
    """World background color - dark/dim environments typical for beer pong."""
    r = rng.uniform(0.005, 0.06)
    g = rng.uniform(0.005, 0.06)
    b = rng.uniform(0.005, 0.08)
    return [r, g, b]


def random_camera_pose(rng):
    """
    Generate a random camera pose around the beer pong table.
    Camera is at roughly humanoid height (1.5-1.9m) looking at the table.
    """
    # Random angle around the table
    angle = rng.uniform(0, 2 * np.pi)

    # Distance from table center (standing near table edge to a few steps back)
    # Table extends to ~1.22m in Y, ~0.305m in X
    distance = rng.uniform(1.0, 2.5)

    cam_x = distance * np.sin(angle)
    cam_y = distance * np.cos(angle)
    cam_z = rng.uniform(1.50, 1.90)  # humanoid head height range

    # Look at a point on the table (with some randomness)
    look_x = rng.uniform(-0.15, 0.15)
    look_y = rng.uniform(-TABLE_LENGTH / 4, TABLE_LENGTH / 4)
    look_z = TABLE_HEIGHT + rng.uniform(-0.05, 0.10)

    # Build rotation matrix to look at target
    cam_pos = np.array([cam_x, cam_y, cam_z])
    target = np.array([look_x, look_y, look_z])

    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    # BlenderProc uses -Z as forward, +Y as up in camera space
    # We need to build a rotation that maps camera -Z to forward direction
    right = np.cross(np.array([0, 0, 1]), forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm

    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # Camera matrix: columns are right, up, -forward (Blender convention)
    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = up
    rot[:3, 2] = -forward
    rot[:3, 3] = cam_pos

    return rot


def random_lighting(rng, lights):
    """Randomize the 3 lights (key, fill, back)."""
    key, fill, back = lights

    # Key light: overhead, varies in position and warmth
    key.set_location([
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        TABLE_HEIGHT + rng.uniform(1.5, 3.0),
    ])
    key_energy = rng.uniform(150, 500)
    key.set_energy(key_energy)
    warmth = rng.uniform(0.8, 1.0)
    key.set_color([1.0, warmth, warmth * 0.85])

    # Fill light: from side
    fill_angle = rng.uniform(0, 2 * np.pi)
    fill.set_location([
        2.0 * np.cos(fill_angle),
        2.0 * np.sin(fill_angle),
        TABLE_HEIGHT + rng.uniform(0.5, 2.0),
    ])
    fill.set_energy(rng.uniform(50, 200))
    fill.set_color([rng.uniform(0.8, 1.0), rng.uniform(0.85, 1.0), rng.uniform(0.9, 1.0)])

    # Back light: opposite side
    back.set_location([
        -2.0 * np.cos(fill_angle) + rng.uniform(-0.5, 0.5),
        -2.0 * np.sin(fill_angle) + rng.uniform(-0.5, 0.5),
        TABLE_HEIGHT + rng.uniform(0.5, 1.5),
    ])
    back.set_energy(rng.uniform(30, 120))
    back.set_color([1.0, rng.uniform(0.9, 1.0), rng.uniform(0.8, 1.0)])


def randomize_cups(rng, cup_template, cup_spacing):
    """
    Create a random beer pong cup arrangement. Returns list of cup objects
    and their metadata (position, rotation, upright status).
    """
    cups = []
    cup_meta = []

    for side in ['A', 'B']:
        # Random number of cups on this side (0-10)
        n_cups = rng.integers(0, 11)  # 0 to 10
        if n_cups == 0:
            continue

        # Determine triangle size and remove random cups
        rows = rows_for_count(n_cups)
        max_in_triangle = rows * (rows + 1) // 2

        if side == 'A':
            origin = np.array([0, -TABLE_LENGTH / 2 + cup_spacing * 2.5])
            y_sign = 1.0
        else:
            origin = np.array([0, TABLE_LENGTH / 2 - cup_spacing * 2.5])
            y_sign = -1.0

        positions = triangle_positions(rows, cup_spacing, origin)

        # For side B, flip y positions
        if side == 'B':
            positions = [
                (p[0], origin[1] - (p[1] - origin[1]))
                for p in positions
            ]

        # Randomly remove cups to get down to n_cups
        if len(positions) > n_cups:
            indices = rng.choice(len(positions), size=n_cups, replace=False)
            positions = [positions[i] for i in sorted(indices)]

        for pos in positions:
            # Decide if cup is tipped over (5% chance)
            tipped = rng.random() < 0.05

            cup = cup_template.duplicate(linked=True)

            if tipped:
                # Tip the cup on its side with random rotation
                tip_angle = rng.uniform(60, 100)  # degrees from vertical
                tip_dir = rng.uniform(0, 360)  # direction of tip
                rot = [np.radians(tip_angle), 0, np.radians(tip_dir)]
                cup.set_rotation_euler(rot)
                # Place slightly on the table surface
                cup.set_location([pos[0], pos[1], TABLE_HEIGHT + 0.02])
            else:
                # Small random positional jitter (cups aren't perfectly placed)
                jitter_x = rng.uniform(-0.005, 0.005)
                jitter_y = rng.uniform(-0.005, 0.005)
                cup.set_location([pos[0] + jitter_x, pos[1] + jitter_y, TABLE_HEIGHT])
                # Tiny rotation jitter
                rot_jitter = rng.uniform(-2, 2, size=3)
                cup.set_rotation_euler(np.radians(rot_jitter).tolist())

            cups.append(cup)
            loc = cup.get_location().tolist()
            rot = cup.get_rotation_euler().tolist()
            bbox_3d = get_cup_3d_bbox(cup)
            cup_meta.append({
                'side': side,
                'position_3d': loc,
                'rotation_euler_rad': rot,
                'bbox_3d_corners': bbox_3d,
                'tipped_over': tipped,
            })

    return cups, cup_meta


def randomize_balls(rng, balls):
    """Randomize ball positions - some on table, some off, some mid-air."""
    for ball in balls:
        choice = rng.integers(4)
        if choice == 0:
            # On the table surface
            ball.set_location([
                rng.uniform(-TABLE_WIDTH / 2 + 0.03, TABLE_WIDTH / 2 - 0.03),
                rng.uniform(-TABLE_LENGTH / 2 + 0.05, TABLE_LENGTH / 2 - 0.05),
                TABLE_HEIGHT + 0.02,
            ])
        elif choice == 1:
            # Rolling on floor
            ball.set_location([
                rng.uniform(-1.5, 1.5),
                rng.uniform(-2.0, 2.0),
                0.02,
            ])
        elif choice == 2:
            # Hidden (off-scene)
            ball.set_location([0, 0, -10])
        else:
            # Mid-air (being thrown)
            ball.set_location([
                rng.uniform(-0.5, 0.5),
                rng.uniform(-1.0, 1.0),
                rng.uniform(TABLE_HEIGHT + 0.1, TABLE_HEIGHT + 0.8),
            ])


# ---------------------------------------------------------------------------
# [IMPROVEMENT 1] HDRI environment maps
# ---------------------------------------------------------------------------
def discover_hdri_files(hdri_dir):
    """Find all .hdr files under the HDRI directory."""
    if not os.path.isdir(hdri_dir):
        return []
    files = sorted(glob.glob(os.path.join(hdri_dir, "**", "*.hdr"), recursive=True))
    return files


def set_hdri_background(path_to_hdr, strength=1.0, rotation_euler=None):
    """Apply an HDRI to the world background, clearing any previous HDRI nodes."""
    if rotation_euler is None:
        rotation_euler = [0.0, 0.0, 0.0]

    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Remove old environment/mapping/texcoord nodes from previous HDRI calls
    for node in list(nodes):
        if node.type in ("TEX_ENVIRONMENT", "MAPPING", "TEX_COORD"):
            nodes.remove(node)

    # Find the existing Background node
    bg_node = None
    for node in nodes:
        if node.type == "BACKGROUND":
            bg_node = node
            break
    if bg_node is None:
        bg_node = nodes.new("ShaderNodeBackground")

    # Create environment texture
    tex_node = nodes.new("ShaderNodeTexEnvironment")
    tex_node.image = bpy.data.images.load(path_to_hdr, check_existing=True)

    # Create mapping + texcoord for rotation control
    mapping_node = nodes.new("ShaderNodeMapping")
    texcoord_node = nodes.new("ShaderNodeTexCoord")

    links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], tex_node.inputs["Vector"])
    links.new(tex_node.outputs["Color"], bg_node.inputs["Color"])

    mapping_node.inputs["Rotation"].default_value = rotation_euler
    bg_node.inputs["Strength"].default_value = strength


def clear_hdri_background():
    """Remove HDRI nodes and revert to flat color background."""
    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in list(nodes):
        if node.type in ("TEX_ENVIRONMENT", "MAPPING", "TEX_COORD"):
            nodes.remove(node)


def random_hdri(rng, hdri_files):
    """Pick a random HDRI and apply it with randomized rotation and strength."""
    if not hdri_files:
        return None
    idx = rng.integers(len(hdri_files))
    hdr_path = hdri_files[idx]
    strength = rng.uniform(0.3, 1.5)
    rotation = [0.0, 0.0, rng.uniform(0, 2 * np.pi)]
    set_hdri_background(hdr_path, strength=strength, rotation_euler=rotation)
    return os.path.basename(os.path.dirname(hdr_path))


# ---------------------------------------------------------------------------
# [IMPROVEMENT 2] CC texture materials for table and floor
# ---------------------------------------------------------------------------
def load_cc_materials_if_available(cctextures_dir):
    """Load CC textures if the asset directory exists. Returns list of Materials."""
    if not os.path.isdir(cctextures_dir):
        return []
    try:
        materials = bproc.loader.load_ccmaterials(
            folder_path=cctextures_dir,
            used_assets=None,  # use the default curated list
            use_all_materials=True,
        )
        print(f"Loaded {len(materials)} CC texture materials")
        return materials
    except Exception as e:
        print(f"Warning: Could not load CC textures: {e}")
        return []


def random_cc_texture_for_surface(rng, cc_materials, obj, fallback_mat, fallback_color):
    """
    With 70% probability, apply a random CC texture to the object.
    Otherwise fall back to the flat-color material.
    Returns description string for metadata.
    """
    if cc_materials and rng.random() < 0.7:
        idx = rng.integers(len(cc_materials))
        mat = cc_materials[idx]
        obj.replace_materials(mat)
        return f"cc:{mat.get_name()}"
    else:
        obj.replace_materials(fallback_mat)
        fallback_mat.set_principled_shader_value("Base Color", fallback_color)
        return f"flat:{fallback_color[:3]}"


# ---------------------------------------------------------------------------
# [IMPROVEMENT 3] Distractor objects
# ---------------------------------------------------------------------------
def create_distractor_objects(rng, n_max=MAX_DISTRACTORS):
    """
    Create random distractor objects (bottles, cans, boxes, solo cups) on or
    near the table. Returns a list of (MeshObject, material) tuples.
    """
    n = rng.integers(0, n_max + 1)
    distractors = []

    # Common distractor colors
    distractor_colors = [
        [0.02, 0.15, 0.02, 1.0],   # dark green (bottle)
        [0.60, 0.55, 0.10, 1.0],   # amber (beer bottle)
        [0.75, 0.75, 0.78, 1.0],   # silver (can)
        [0.08, 0.08, 0.40, 1.0],   # blue (can)
        [0.70, 0.05, 0.05, 1.0],   # red (can)
        [0.90, 0.90, 0.90, 1.0],   # white (paper plate)
        [0.30, 0.18, 0.08, 1.0],   # brown (cardboard box)
        [0.85, 0.80, 0.55, 1.0],   # beige (chip bag)
    ]

    for i in range(n):
        shape_type = rng.integers(4)

        if shape_type == 0:
            # Tall cylinder (bottle)
            radius = rng.uniform(0.025, 0.04)
            height = rng.uniform(0.15, 0.30)
            obj = bproc.object.create_primitive(
                "CYLINDER",
                scale=[radius, radius, height / 2],
                location=[0, 0, -10],
            )
        elif shape_type == 1:
            # Short cylinder (can)
            radius = rng.uniform(0.028, 0.035)
            height = rng.uniform(0.10, 0.13)
            obj = bproc.object.create_primitive(
                "CYLINDER",
                scale=[radius, radius, height / 2],
                location=[0, 0, -10],
            )
        elif shape_type == 2:
            # Box (snack box, pizza box, etc.)
            sx = rng.uniform(0.05, 0.15)
            sy = rng.uniform(0.05, 0.15)
            sz = rng.uniform(0.02, 0.08)
            obj = bproc.object.create_primitive(
                "CUBE",
                scale=[sx / 2, sy / 2, sz / 2],
                location=[0, 0, -10],
            )
        else:
            # Sphere (fruit, ball, crumpled napkin)
            radius = rng.uniform(0.02, 0.05)
            obj = bproc.object.create_primitive(
                "SPHERE",
                scale=[radius, radius, radius],
                location=[0, 0, -10],
            )

        # Material
        mat = bproc.material.create(f"distractor_{i}")
        color = distractor_colors[rng.integers(len(distractor_colors))]
        mat.set_principled_shader_value("Base Color", color)
        mat.set_principled_shader_value("Roughness", float(rng.uniform(0.2, 0.9)))
        mat.set_principled_shader_value("Metallic", float(rng.uniform(0.0, 0.5)))
        obj.replace_materials(mat)

        # Place: 80% on table, 20% on floor nearby
        if rng.random() < 0.8:
            x = rng.uniform(-TABLE_WIDTH / 2 + 0.05, TABLE_WIDTH / 2 - 0.05)
            y = rng.uniform(-TABLE_LENGTH / 2 + 0.10, TABLE_LENGTH / 2 - 0.10)
            z = TABLE_HEIGHT + 0.01
        else:
            x = rng.uniform(-1.5, 1.5)
            y = rng.uniform(-2.0, 2.0)
            z = 0.01

        obj.set_location([x, y, z])

        # Random rotation for objects on their side
        if rng.random() < 0.15:
            obj.set_rotation_euler([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
            ])

        distractors.append((obj, mat))

    return distractors


def remove_distractor_objects(distractors):
    """Remove all distractor objects and their materials from the scene."""
    for obj, mat in distractors:
        bpy_obj = obj.blender_obj
        if bpy_obj and bpy_obj.name in bpy.data.objects:
            bpy.data.objects.remove(bpy_obj, do_unlink=True)
        # Clean up material
        bpy_mat = mat.blender_obj if hasattr(mat, 'blender_obj') else None
        if bpy_mat and bpy_mat.name in bpy.data.materials:
            bpy.data.materials.remove(bpy_mat)


# ---------------------------------------------------------------------------
# [IMPROVEMENT 4] Camera post-processing (noise, exposure, JPEG artifacts)
# ---------------------------------------------------------------------------
def apply_post_processing(img_array, rng):
    """
    Apply realistic camera imperfections to a rendered image.
    Returns modified uint8 numpy array.
    """
    img = img_array.astype(np.float32)

    # --- Exposure shift (brightness/gamma variation) ---
    exposure = rng.uniform(0.85, 1.20)
    img = img * exposure

    # --- White balance shift (subtle color temperature) ---
    wb_r = rng.uniform(0.95, 1.05)
    wb_b = rng.uniform(0.95, 1.05)
    img[:, :, 0] = img[:, :, 0] * wb_r
    img[:, :, 2] = img[:, :, 2] * wb_b

    # --- Sensor noise (Gaussian + shot noise) ---
    noise_sigma = rng.uniform(1.0, 8.0)
    noise = rng.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = img + noise

    # --- Vignetting (subtle corner darkening, 30% chance) ---
    if rng.random() < 0.3:
        h, w = img.shape[:2]
        cy, cx = h / 2, w / 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        vignette_strength = rng.uniform(0.1, 0.35)
        vignette = 1.0 - vignette_strength * (r / r_max) ** 2
        img = img * vignette[:, :, np.newaxis]

    # Clamp and convert back to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def apply_jpeg_compression(pil_img, rng):
    """
    Simulate JPEG compression artifacts by re-encoding at random quality.
    Applied with 40% probability.
    """
    if rng.random() < 0.4:
        import io
        quality = int(rng.integers(60, 95))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        from PIL import Image
        pil_img = Image.open(buffer).convert("RGB")
    return pil_img


def setup_dof(rng, cam_pos, look_target):
    """Configure depth-of-field with randomized aperture. Applied 30% of the time."""
    cam = bpy.context.scene.camera
    if cam is None:
        return False

    use_dof = rng.random() < 0.3
    cam.data.dof.use_dof = use_dof

    if use_dof:
        # Focus on the table area
        focus_dist = float(np.linalg.norm(cam_pos - look_target))
        cam.data.dof.focus_distance = focus_dist
        # f-stop: lower = more blur. Beer pong photos are usually phone cameras (f/1.8-2.8)
        cam.data.dof.aperture_fstop = float(rng.uniform(1.8, 5.6))

    return use_dof


# ---------------------------------------------------------------------------
# [IMPROVEMENT 5] Cup material variation
# ---------------------------------------------------------------------------
# Realistic cup color/material presets
CUP_MATERIAL_PRESETS = [
    # (name, base_color, roughness, metallic, alpha, transmission)
    # Classic red Solo cup
    {"name": "red_solo",      "color": [0.70, 0.04, 0.04, 1.0], "roughness": 0.35, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    {"name": "red_dark",      "color": [0.50, 0.02, 0.02, 1.0], "roughness": 0.30, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    {"name": "red_bright",    "color": [0.85, 0.08, 0.05, 1.0], "roughness": 0.40, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # Blue cups
    {"name": "blue_solo",     "color": [0.04, 0.08, 0.65, 1.0], "roughness": 0.35, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    {"name": "blue_dark",     "color": [0.02, 0.04, 0.45, 1.0], "roughness": 0.30, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # Green cups
    {"name": "green_solo",    "color": [0.04, 0.50, 0.08, 1.0], "roughness": 0.35, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # Yellow cups
    {"name": "yellow_solo",   "color": [0.85, 0.75, 0.05, 1.0], "roughness": 0.35, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # Orange cups
    {"name": "orange_solo",   "color": [0.90, 0.40, 0.05, 1.0], "roughness": 0.35, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # Pink cups
    {"name": "pink_solo",     "color": [0.85, 0.20, 0.50, 1.0], "roughness": 0.35, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # Black cups
    {"name": "black_solo",    "color": [0.03, 0.03, 0.03, 1.0], "roughness": 0.30, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # White/frost cups
    {"name": "white_opaque",  "color": [0.90, 0.90, 0.90, 1.0], "roughness": 0.40, "metallic": 0.0, "alpha": 1.0, "transmission": 0.0},
    # Clear/translucent cups
    {"name": "clear_cup",     "color": [0.95, 0.95, 0.95, 1.0], "roughness": 0.10, "metallic": 0.0, "alpha": 0.3, "transmission": 0.8},
    {"name": "frosted_clear", "color": [0.90, 0.90, 0.92, 1.0], "roughness": 0.30, "metallic": 0.0, "alpha": 0.5, "transmission": 0.5},
    # Metallic/party cups
    {"name": "gold_metallic", "color": [0.80, 0.65, 0.15, 1.0], "roughness": 0.20, "metallic": 0.7, "alpha": 1.0, "transmission": 0.0},
    {"name": "silver_metal",  "color": [0.75, 0.75, 0.78, 1.0], "roughness": 0.20, "metallic": 0.7, "alpha": 1.0, "transmission": 0.0},
]


def randomize_cup_material(rng, cup_template):
    """
    Apply a random cup material preset to the cup template, with additional
    per-scene jitter. Returns the chosen preset name for metadata.
    """
    preset = CUP_MATERIAL_PRESETS[rng.integers(len(CUP_MATERIAL_PRESETS))]

    # Get or create the cup material
    bpy_obj = cup_template.blender_obj
    if not bpy_obj.data.materials:
        mat = bproc.material.create("cup_material")
        bpy_obj.data.materials.append(mat.blender_obj)
    else:
        mat = bproc.python.types.MaterialUtility.Material(bpy_obj.data.materials[0])

    # Apply base properties with slight jitter
    color = list(preset["color"])
    for i in range(3):
        color[i] = np.clip(color[i] + rng.uniform(-0.05, 0.05), 0, 1)
    mat.set_principled_shader_value("Base Color", color)

    roughness = np.clip(preset["roughness"] + rng.uniform(-0.05, 0.05), 0.05, 1.0)
    mat.set_principled_shader_value("Roughness", float(roughness))

    metallic = np.clip(preset["metallic"] + rng.uniform(-0.05, 0.05), 0.0, 1.0)
    mat.set_principled_shader_value("Metallic", float(metallic))

    alpha = float(preset["alpha"])
    mat.set_principled_shader_value("Alpha", alpha)

    transmission = float(preset["transmission"])
    mat.set_principled_shader_value("Transmission Weight", transmission)

    # For transparent cups, set blend mode
    bpy_mat = mat.blender_obj
    if alpha < 1.0 or transmission > 0.0:
        bpy_mat.blend_method = 'HASHED'
        bpy_mat.shadow_method = 'HASHED'

    return preset["name"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate beer pong dataset')
    parser.add_argument('--num-scenes', type=int, default=100,
                        help='Number of scene variations to generate')
    parser.add_argument('--views-per-scene', type=int, default=4,
                        help='Camera views per scene variation')
    parser.add_argument('--output-dir', type=str,
                        default='/workspace/beer_pong_dataset',
                        help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--start-scene', type=int, default=0,
                        help='Starting scene index (for resuming)')
    parser.add_argument('--assets-dir', type=str, default=ASSETS_DIR,
                        help='Path to downloaded assets (HDRIs + cctextures)')
    args = parser.parse_args()

    hdri_dir = os.path.join(args.assets_dir, "hdris")
    cctextures_dir = os.path.join(args.assets_dir, "cctextures")

    # Each scene gets its own deterministic RNG seeded from (base_seed, scene_index).
    # This ensures identical results regardless of parallelization strategy —
    # scene 42 produces the same output whether run in a single process or
    # split across N workers.
    base_seed = args.seed

    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # --- One-time initialization ---
    bproc.init()

    # Render settings (set once)
    bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)
    bproc.renderer.set_noise_threshold(0.1)
    bproc.renderer.set_light_bounces(
        diffuse_bounces=3, glossy_bounces=3, max_bounces=6,
        transparent_max_bounces=8,
    )

    # Camera intrinsics (set once)
    bproc.camera.set_resolution(*RESOLUTION)
    K = np.array([
        [ZED_FX, 0, ZED_CX],
        [0, ZED_FY, ZED_CY],
        [0, 0, 1],
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K, RESOLUTION[0], RESOLUTION[1],
                                               clip_start=0.01, clip_end=50)

    # --- [IMPROVEMENT 1] Discover HDRI files ---
    hdri_files = discover_hdri_files(hdri_dir)
    if hdri_files:
        print(f"Found {len(hdri_files)} HDRI environment maps")
    else:
        print("No HDRIs found - using flat color backgrounds. "
              "Run download_assets.py to enable HDRI backgrounds.")

    # --- [IMPROVEMENT 2] Load CC texture materials ---
    cc_materials = load_cc_materials_if_available(cctextures_dir)
    if not cc_materials:
        print("No CC textures found - using flat color surfaces. "
              "Run download_assets.py to enable PBR textures.")

    # --- Load cup template (once) ---
    cup_objs = load_usdz(CUP_MODEL_PATH)
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

    cup_template.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
    bbox_raw = cup_template.get_bound_box()
    raw_diameter = max(bbox_raw[:, 0]) - min(bbox_raw[:, 0])
    cup_scale = TARGET_CUP_DIAMETER / raw_diameter if raw_diameter > 0 else 0.01
    cup_template.set_scale([cup_scale, cup_scale, cup_scale])
    cup_template.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
    cup_template.move_origin_to_bottom_mean_point()

    bbox = cup_template.get_bound_box()
    cup_diameter = max(bbox[:, 0]) - min(bbox[:, 0])
    cup_spacing = cup_diameter * 1.05

    # Hide template off-screen
    cup_template.set_location([0, 0, -10])

    # --- Create persistent scene objects ---
    # Table
    table_top = bproc.object.create_primitive(
        "CUBE",
        scale=[TABLE_WIDTH / 2, TABLE_LENGTH / 2, 0.02],
        location=[0, 0, TABLE_HEIGHT - 0.02],
    )
    table_mat = bproc.material.create("table_surface")
    table_mat.set_principled_shader_value("Roughness", 0.6)
    table_top.replace_materials(table_mat)

    # Legs
    legs = []
    leg_mat = bproc.material.create("table_legs")
    leg_mat.set_principled_shader_value("Roughness", 0.5)
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
        leg.replace_materials(leg_mat)
        legs.append(leg)

    # Floor
    floor = bproc.object.create_primitive(
        "PLANE", scale=[5, 5, 1], location=[0, 0, 0]
    )
    floor_mat = bproc.material.create("floor")
    floor_mat.set_principled_shader_value("Roughness", 0.9)
    floor.replace_materials(floor_mat)

    # Ping pong balls (reused across scenes)
    balls = []
    for i in range(2):
        ball = bproc.object.create_primitive(
            "SPHERE", scale=[0.02, 0.02, 0.02], location=[0, 0, -10],
        )
        ball_mat = bproc.material.create(f"ball_{i}")
        ball_mat.set_principled_shader_value("Base Color", [1.0, 1.0, 1.0, 1.0])
        ball_mat.set_principled_shader_value("Roughness", 0.3)
        ball.replace_materials(ball_mat)
        balls.append(ball)

    # Lights (reused, randomized each scene)
    key = bproc.types.Light()
    key.set_type("AREA")
    fill = bproc.types.Light()
    fill.set_type("POINT")
    back = bproc.types.Light()
    back.set_type("POINT")
    lights = [key, fill, back]

    print(f"Scene initialized. Generating {args.num_scenes} scenes "
          f"x {args.views_per_scene} views = "
          f"{args.num_scenes * args.views_per_scene} images")
    print(f"Domain randomization: HDRIs={len(hdri_files)}, "
          f"CC_textures={len(cc_materials)}, "
          f"cup_presets={len(CUP_MATERIAL_PRESETS)}, "
          f"max_distractors={MAX_DISTRACTORS}, "
          f"post_processing=enabled")

    total_start = time.time()
    prev_distractors = []

    for scene_i in range(args.start_scene, args.start_scene + args.num_scenes):
        scene_start = time.time()

        # Per-scene deterministic RNG: same scene_i always produces same output
        rng = np.random.default_rng([base_seed, scene_i])

        # --- [IMPROVEMENT 1] HDRI background ---
        # 75% HDRI, 25% flat color (for diversity)
        if hdri_files and rng.random() < 0.75:
            hdri_name = random_hdri(rng, hdri_files)
        else:
            clear_hdri_background()
            bg = random_background_color(rng)
            bproc.renderer.set_world_background(bg, strength=1)
            hdri_name = None

        # --- [IMPROVEMENT 2] Randomize table/floor textures ---
        tc = random_table_color(rng)
        table_tex_desc = random_cc_texture_for_surface(
            rng, cc_materials, table_top, table_mat, tc)
        # Legs always match the flat table color (CC textures don't tile well on thin legs)
        leg_mat.set_principled_shader_value("Base Color", tc)

        fc = random_floor_color(rng)
        floor_tex_desc = random_cc_texture_for_surface(
            rng, cc_materials, floor, floor_mat, fc)

        # Lighting
        random_lighting(rng, lights)

        # Balls
        randomize_balls(rng, balls)

        # --- [IMPROVEMENT 5] Cup material variation ---
        cup_preset_name = randomize_cup_material(rng, cup_template)

        # Cups - remove old ones first
        for obj_name in list(bpy.data.objects.keys()):
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                continue
            if (obj.type == 'MESH'
                and obj != cup_template.blender_obj
                and obj.data == cup_template.blender_obj.data
                and obj.location.z > -5):
                bpy.data.objects.remove(obj, do_unlink=True)

        cups, cup_meta = randomize_cups(rng, cup_template, cup_spacing)

        # --- [IMPROVEMENT 3] Distractor objects ---
        remove_distractor_objects(prev_distractors)
        prev_distractors = create_distractor_objects(rng)

        # --- Camera poses ---
        cam_obj = bpy.context.scene.camera
        if cam_obj and cam_obj.animation_data:
            cam_obj.animation_data_clear()

        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = args.views_per_scene - 1

        cam_poses = []
        for v in range(args.views_per_scene):
            pose = random_camera_pose(rng)
            bproc.camera.add_camera_pose(pose, frame=v)
            cam_poses.append(pose)

        # --- [IMPROVEMENT 4] Depth of field (per-scene, applied to all views) ---
        # Use the first view's look target as focus reference
        look_target = np.array([0, 0, TABLE_HEIGHT])
        cam_pos = cam_poses[0][:3, 3]
        dof_enabled = setup_dof(rng, cam_pos, look_target)

        # --- Render ---
        data = bproc.renderer.render()

        # Disable DOF after render to avoid carrying state
        if cam_obj:
            cam_obj.data.dof.use_dof = False

        # --- Save outputs ---
        from PIL import Image

        for v in range(args.views_per_scene):
            img_name = f"scene_{scene_i:06d}_view_{v:02d}"

            img_array = data['colors'][v]

            # [IMPROVEMENT 4] Apply post-processing
            img_array = apply_post_processing(img_array, rng)

            img = Image.fromarray(img_array)

            # [IMPROVEMENT 4] JPEG compression artifacts (sometimes)
            img = apply_jpeg_compression(img, rng)

            img.save(os.path.join(images_dir, f"{img_name}.png"))

            cam_pose = cam_poses[v].tolist()

            meta = {
                'scene_index': scene_i,
                'view_index': v,
                'image_file': f"{img_name}.png",
                'image_width': RESOLUTION[0],
                'image_height': RESOLUTION[1],
                'camera_intrinsics': get_camera_K(),
                'camera_extrinsics_4x4': cam_pose,
                'cups': cup_meta,
                'cup_material': cup_preset_name,
                'table': {
                    'length': TABLE_LENGTH,
                    'width': TABLE_WIDTH,
                    'height': TABLE_HEIGHT,
                    'surface': table_tex_desc,
                },
                'floor_surface': floor_tex_desc,
                'hdri': hdri_name,
                'dof_enabled': dof_enabled,
                'n_distractors': len(prev_distractors),
            }

            with open(os.path.join(labels_dir, f"{img_name}.json"), 'w') as f:
                json.dump(meta, f, indent=2)

        elapsed = time.time() - scene_start
        total_elapsed = time.time() - total_start
        scenes_done = scene_i - args.start_scene + 1
        avg_per_scene = total_elapsed / scenes_done
        remaining = (args.num_scenes - scenes_done) * avg_per_scene

        print(f"Scene {scene_i:4d} | {scenes_done}/{args.num_scenes} | "
              f"{elapsed:.1f}s | avg {avg_per_scene:.1f}s/scene | "
              f"ETA {remaining / 60:.1f}min | "
              f"cup={cup_preset_name} hdri={'yes' if hdri_name else 'no'} "
              f"dist={len(prev_distractors)}")

    # Clean up last scene's distractors
    remove_distractor_objects(prev_distractors)

    total_time = time.time() - total_start
    total_images = args.num_scenes * args.views_per_scene
    print(f"\nDone! Generated {total_images} images in {total_time:.1f}s "
          f"({total_time / total_images:.2f}s/image)")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
