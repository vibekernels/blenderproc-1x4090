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
    Includes close-up shots, low angles, and high/overhead perspectives
    in addition to standard human-height views.
    """
    # Random angle around the table
    angle = rng.uniform(0, 2 * np.pi)

    # Distance from table center — includes close-ups (0.4m) to far back (2.5m)
    # Table extends to ~1.22m in Y, ~0.305m in X
    distance = rng.uniform(0.4, 2.5)

    cam_x = distance * np.sin(angle)
    cam_y = distance * np.cos(angle)
    cam_z = rng.uniform(0.6, 2.5)  # low angle (table-level) to overhead

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
# [IMPROVEMENT 1] HDRI environment maps — with preloading
# ---------------------------------------------------------------------------
def discover_hdri_files(hdri_dir):
    """Find all .hdr files under the HDRI directory."""
    if not os.path.isdir(hdri_dir):
        return []
    files = sorted(glob.glob(os.path.join(hdri_dir, "**", "*.hdr"), recursive=True))
    return files


class HDRIManager:
    """
    Pre-loads all HDRI images and creates persistent world shader nodes once.
    Per-scene switching only changes the image pointer and uniform values —
    no node creation/destruction, no image re-loading from disk.
    """

    def __init__(self, hdri_files):
        self.hdri_files = hdri_files
        self.images = {}       # path -> bpy.data.images entry
        self.tex_node = None
        self.mapping_node = None
        self.bg_node = None
        self._nodes_created = False

        # Pre-load all HDRI images into Blender's image cache
        if hdri_files:
            print(f"Pre-loading {len(hdri_files)} HDRI images...")
            for path in hdri_files:
                self.images[path] = bpy.data.images.load(path, check_existing=True)
            print(f"HDRI images cached in memory")
            self._setup_world_nodes()

    def _setup_world_nodes(self):
        """Create the world shader node graph once."""
        world = bpy.context.scene.world
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clean any existing env nodes
        for node in list(nodes):
            if node.type in ("TEX_ENVIRONMENT", "MAPPING", "TEX_COORD"):
                nodes.remove(node)

        self.bg_node = None
        for node in nodes:
            if node.type == "BACKGROUND":
                self.bg_node = node
                break
        if self.bg_node is None:
            self.bg_node = nodes.new("ShaderNodeBackground")

        self.tex_node = nodes.new("ShaderNodeTexEnvironment")
        self.mapping_node = nodes.new("ShaderNodeMapping")
        texcoord_node = nodes.new("ShaderNodeTexCoord")

        links.new(texcoord_node.outputs["Generated"], self.mapping_node.inputs["Vector"])
        links.new(self.mapping_node.outputs["Vector"], self.tex_node.inputs["Vector"])
        links.new(self.tex_node.outputs["Color"], self.bg_node.inputs["Color"])

        self._nodes_created = True

    def set_hdri(self, rng):
        """Switch to a random HDRI. Only updates image pointer + uniforms."""
        if not self.hdri_files:
            return None

        idx = rng.integers(len(self.hdri_files))
        path = self.hdri_files[idx]

        if not self._nodes_created:
            self._setup_world_nodes()

        # Just swap the image — no node creation, no disk I/O
        self.tex_node.image = self.images[path]
        self.mapping_node.inputs["Rotation"].default_value = [
            0.0, 0.0, rng.uniform(0, 2 * np.pi)]
        self.bg_node.inputs["Strength"].default_value = rng.uniform(0.3, 1.5)

        return os.path.basename(os.path.dirname(path))

    def disable_hdri(self):
        """Disconnect HDRI so flat color background takes effect."""
        if self._nodes_created:
            world = bpy.context.scene.world
            links = world.node_tree.links
            # Remove the link from tex_node to bg_node so the flat color input wins
            for link in list(links):
                if link.to_node == self.bg_node and link.from_node == self.tex_node:
                    links.remove(link)
                    break
            self._nodes_created = False

    def enable_hdri(self):
        """Reconnect HDRI nodes if previously disabled."""
        if not self._nodes_created and self.tex_node and self.bg_node:
            world = bpy.context.scene.world
            links = world.node_tree.links
            links.new(self.tex_node.outputs["Color"], self.bg_node.inputs["Color"])
            self._nodes_created = True


# ---------------------------------------------------------------------------
# [IMPROVEMENT 2] CC texture materials for table and floor
#
# Performance-critical: we avoid replace_materials() entirely because swapping
# materials on a mesh triggers Cycles shader recompilation (~150s overhead).
# Instead we pre-scan texture image paths and hot-swap only the Image datablock
# pointers inside a persistent material's texture nodes.
# ---------------------------------------------------------------------------
def discover_cc_texture_images(cctextures_dir):
    """
    Scan the CC texture directory and return a list of dicts, each containing
    the image file paths for one texture asset (color, roughness, normal, etc.).
    Does NOT use load_ccmaterials() — we manage images directly for speed.
    """
    if not os.path.isdir(cctextures_dir):
        return []

    textures = []
    for asset in sorted(os.listdir(cctextures_dir)):
        asset_dir = os.path.join(cctextures_dir, asset)
        if not os.path.isdir(asset_dir):
            continue

        # Look for color image (required)
        color_path = os.path.join(asset_dir, f"{asset}_2K_Color.jpg")
        if not os.path.exists(color_path):
            color_path = os.path.join(asset_dir, f"{asset}_2K-JPG_Color.jpg")
        if not os.path.exists(color_path):
            continue

        entry = {"name": asset, "color": color_path}

        # Optional maps — same naming convention
        for map_name in ["Roughness", "NormalGL", "Normal", "Displacement",
                         "AmbientOcclusion", "Metalness"]:
            p = color_path.replace("Color", map_name)
            if os.path.exists(p):
                entry[map_name.lower()] = p

        textures.append(entry)

    return textures


def preload_cc_images(texture_entries):
    """Pre-load all CC texture images into Blender's image cache."""
    image_cache = {}  # path -> bpy.data.images
    for entry in texture_entries:
        for key, path in entry.items():
            if key == "name":
                continue
            if path not in image_cache:
                img = bpy.data.images.load(path, check_existing=True)
                # Mark non-color for non-diffuse maps
                if key != "color":
                    img.colorspace_settings.name = "Non-Color"
                image_cache[path] = img
    return image_cache


def create_pbr_surface_material(name):
    """
    Create a persistent PBR material with texture nodes for color, roughness,
    normal, and displacement. Returns (Material, node_refs_dict).
    The same material stays assigned to the mesh forever — we only swap images.
    """
    mat = bproc.material.create(name)
    bpy_mat = mat.blender_obj
    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links

    principled = None
    output_node = None
    for n in nodes:
        if n.type == "BSDF_PRINCIPLED":
            principled = n
        elif n.type == "OUTPUT_MATERIAL":
            output_node = n

    # Texture coordinate + mapping (shared)
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])

    # Color texture
    color_tex = nodes.new("ShaderNodeTexImage")
    color_tex.name = "cc_color"
    links.new(mapping.outputs["Vector"], color_tex.inputs["Vector"])
    links.new(color_tex.outputs["Color"], principled.inputs["Base Color"])

    # Roughness texture
    rough_tex = nodes.new("ShaderNodeTexImage")
    rough_tex.name = "cc_roughness"
    links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
    links.new(rough_tex.outputs["Color"], principled.inputs["Roughness"])

    # Normal map
    normal_tex = nodes.new("ShaderNodeTexImage")
    normal_tex.name = "cc_normal"
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])
    links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])

    node_refs = {
        "principled": principled,
        "color_tex": color_tex,
        "rough_tex": rough_tex,
        "normal_tex": normal_tex,
        "mapping": mapping,
    }
    return mat, node_refs


def apply_cc_texture_to_surface(rng, texture_entries, image_cache, node_refs,
                                 fallback_color):
    """
    Hot-swap texture images in a persistent material. Only changes Image pointers
    and a few uniform values — no material replacement, no shader recompilation.
    Returns description string for metadata.
    """
    if texture_entries and rng.random() < 0.7:
        idx = rng.integers(len(texture_entries))
        entry = texture_entries[idx]

        node_refs["color_tex"].image = image_cache[entry["color"]]

        if "roughness" in entry:
            node_refs["rough_tex"].image = image_cache[entry["roughness"]]
            node_refs["rough_tex"].mute = False
        else:
            node_refs["rough_tex"].mute = True

        normal_key = "normalgl" if "normalgl" in entry else "normal" if "normal" in entry else None
        if normal_key:
            node_refs["normal_tex"].image = image_cache[entry[normal_key]]
            node_refs["normal_tex"].mute = False
        else:
            node_refs["normal_tex"].mute = True

        # Mute the color_tex node=False to ensure it's active
        node_refs["color_tex"].mute = False

        return f"cc:{entry['name']}"
    else:
        # Use flat color: mute all texture nodes so principled defaults take over
        node_refs["color_tex"].mute = True
        node_refs["rough_tex"].mute = True
        node_refs["normal_tex"].mute = True
        node_refs["principled"].inputs["Base Color"].default_value = fallback_color
        node_refs["principled"].inputs["Roughness"].default_value = rng.uniform(0.4, 0.9)
        return f"flat:{fallback_color[:3]}"


# ---------------------------------------------------------------------------
# [IMPROVEMENT 3] Distractor objects — pre-created pool (no per-scene alloc)
# ---------------------------------------------------------------------------
DISTRACTOR_COLORS = [
    [0.02, 0.15, 0.02, 1.0],   # dark green (bottle)
    [0.60, 0.55, 0.10, 1.0],   # amber (beer bottle)
    [0.75, 0.75, 0.78, 1.0],   # silver (can)
    [0.08, 0.08, 0.40, 1.0],   # blue (can)
    [0.70, 0.05, 0.05, 1.0],   # red (can)
    [0.90, 0.90, 0.90, 1.0],   # white (paper plate)
    [0.30, 0.18, 0.08, 1.0],   # brown (cardboard box)
    [0.85, 0.80, 0.55, 1.0],   # beige (chip bag)
]


def create_distractor_pool(n_max=MAX_DISTRACTORS):
    """
    Pre-create a pool of distractor objects at startup. Each object gets its
    own material. All start hidden off-screen. Returns list of (obj, mat) tuples.
    """
    pool = []
    # Create a mix of shapes: cylinders, cubes, spheres
    shapes = ["CYLINDER", "CYLINDER", "CUBE", "CUBE", "SPHERE", "SPHERE"]
    for i in range(n_max):
        shape = shapes[i % len(shapes)]
        obj = bproc.object.create_primitive(
            shape,
            scale=[0.03, 0.03, 0.06],  # default size, will be randomized
            location=[0, 0, -10],       # hidden
        )
        mat = bproc.material.create(f"distractor_{i}")
        mat.set_principled_shader_value("Base Color", [0.5, 0.5, 0.5, 1.0])
        mat.set_principled_shader_value("Roughness", 0.5)
        obj.replace_materials(mat)
        pool.append((obj, mat))
    return pool


def randomize_distractor_pool(rng, pool):
    """
    Each scene: pick how many to show (0..len(pool)), randomize their
    scale/position/color/rotation, hide the rest. No object creation/deletion.
    Returns count of visible distractors.
    """
    n_show = rng.integers(0, len(pool) + 1)

    for i, (obj, mat) in enumerate(pool):
        if i < n_show:
            # Randomize scale to simulate different object types
            sx = rng.uniform(0.02, 0.08)
            sy = rng.uniform(0.02, 0.08)
            sz = rng.uniform(0.03, 0.15)
            obj.set_scale([sx, sy, sz])

            # Randomize color
            color = list(DISTRACTOR_COLORS[rng.integers(len(DISTRACTOR_COLORS))])
            mat.set_principled_shader_value("Base Color", color)
            mat.set_principled_shader_value("Roughness", float(rng.uniform(0.2, 0.9)))
            mat.set_principled_shader_value("Metallic", float(rng.uniform(0.0, 0.5)))

            # Place: 80% on table, 20% on floor
            if rng.random() < 0.8:
                x = rng.uniform(-TABLE_WIDTH / 2 + 0.05, TABLE_WIDTH / 2 - 0.05)
                y = rng.uniform(-TABLE_LENGTH / 2 + 0.10, TABLE_LENGTH / 2 - 0.10)
                z = TABLE_HEIGHT + 0.01
            else:
                x = rng.uniform(-1.5, 1.5)
                y = rng.uniform(-2.0, 2.0)
                z = 0.01
            obj.set_location([x, y, z])

            # Random rotation (15% tipped)
            if rng.random() < 0.15:
                obj.set_rotation_euler([
                    rng.uniform(0, 2 * np.pi),
                    rng.uniform(0, 2 * np.pi),
                    rng.uniform(0, 2 * np.pi),
                ])
            else:
                obj.set_rotation_euler([0, 0, rng.uniform(0, 2 * np.pi)])
        else:
            # Hide off-screen
            obj.set_location([0, 0, -10])

    return n_show


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
# Red cup shade presets — HSV shifts applied to the original texture so the
# white interior is preserved (only saturated red regions are affected).
# hue: 0.5 = no shift; saturation: 1.0 = original; value: 1.0 = original brightness
CUP_MATERIAL_PRESETS = [
    {"name": "red_solo",    "hue": 0.500, "saturation": 1.00, "value": 1.00},  # original model red
    {"name": "red_dark",    "hue": 0.500, "saturation": 1.10, "value": 0.75},  # darker
    {"name": "red_bright",  "hue": 0.500, "saturation": 0.90, "value": 1.20},  # brighter/lighter
    {"name": "red_deep",    "hue": 0.500, "saturation": 1.20, "value": 0.80},  # deep/saturated
    {"name": "red_faded",   "hue": 0.500, "saturation": 0.70, "value": 1.10},  # washed-out/faded
    {"name": "red_warm",    "hue": 0.520, "saturation": 1.00, "value": 1.00},  # slightly orange shift
    {"name": "red_cool",    "hue": 0.480, "saturation": 1.00, "value": 1.00},  # slightly pink/cool shift
    {"name": "red_cherry",  "hue": 0.490, "saturation": 1.15, "value": 0.90},  # cherry
]

# Reference to the HSV node once inserted (set up on first call)
_cup_hsv_node = None


def randomize_cup_material(rng, cup_template):
    """
    Randomize the cup's red shade by inserting/updating a Hue Saturation Value
    node between the base color texture and the Principled BSDF.  The white
    interior is unaffected because HSV shifts only move chromatic pixels.
    Returns the chosen preset name for metadata.
    """
    global _cup_hsv_node

    preset = CUP_MATERIAL_PRESETS[rng.integers(len(CUP_MATERIAL_PRESETS))]

    bpy_obj = cup_template.blender_obj
    bpy_mat = bpy_obj.data.materials[0]
    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links

    # --- One-time setup: insert HSV node between texture and BSDF ---
    if _cup_hsv_node is None:
        principled = None
        tex_node = None
        tex_link = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
        # Find the link feeding Base Color
        bc_input = principled.inputs['Base Color']
        for link in links:
            if link.to_socket == bc_input:
                tex_link = link
                tex_node = link.from_node
                tex_output = link.from_socket
                break

        hsv = nodes.new('ShaderNodeHueSaturation')
        hsv.name = 'CupHSVShift'
        hsv.location = (principled.location.x - 200, principled.location.y)

        # Rewire: texture -> HSV -> principled
        if tex_link:
            links.remove(tex_link)
            links.new(tex_output, hsv.inputs['Color'])
        links.new(hsv.outputs['Color'], bc_input)

        _cup_hsv_node = hsv

    # --- Per-scene: update HSV values with jitter ---
    hue = np.clip(preset["hue"] + rng.uniform(-0.02, 0.02), 0.0, 1.0)
    saturation = np.clip(preset["saturation"] + rng.uniform(-0.10, 0.10), 0.5, 1.5)
    value = np.clip(preset["value"] + rng.uniform(-0.10, 0.10), 0.6, 1.4)

    _cup_hsv_node.inputs['Hue'].default_value = float(hue)
    _cup_hsv_node.inputs['Saturation'].default_value = float(saturation)
    _cup_hsv_node.inputs['Value'].default_value = float(value)
    _cup_hsv_node.inputs['Fac'].default_value = 1.0

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
        diffuse_bounces=2, glossy_bounces=2, max_bounces=4,
        transparent_max_bounces=4,
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

    # --- [IMPROVEMENT 1] Pre-load HDRI environment maps ---
    hdri_files = discover_hdri_files(hdri_dir)
    hdri_mgr = HDRIManager(hdri_files)
    if not hdri_files:
        print("No HDRIs found - using flat color backgrounds. "
              "Run download_assets.py to enable HDRI backgrounds.")

    # --- [IMPROVEMENT 2] Discover and preload CC texture images ---
    cc_texture_entries = discover_cc_texture_images(cctextures_dir)
    if cc_texture_entries:
        print(f"Found {len(cc_texture_entries)} CC texture assets, preloading images...")
        cc_image_cache = preload_cc_images(cc_texture_entries)
        print(f"Preloaded {len(cc_image_cache)} texture images into memory")
    else:
        cc_image_cache = {}
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
    # Table — uses persistent PBR material (image-swap only, no shader recompile)
    table_top = bproc.object.create_primitive(
        "CUBE",
        scale=[TABLE_WIDTH / 2, TABLE_LENGTH / 2, 0.02],
        location=[0, 0, TABLE_HEIGHT - 0.02],
    )
    table_mat, table_nodes = create_pbr_surface_material("table_surface")
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

    # Floor — uses persistent PBR material (image-swap only, no shader recompile)
    floor = bproc.object.create_primitive(
        "PLANE", scale=[5, 5, 1], location=[0, 0, 0]
    )
    floor_mat, floor_nodes = create_pbr_surface_material("floor")
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

    # --- [IMPROVEMENT 3] Pre-create distractor object pool ---
    distractor_pool = create_distractor_pool()

    print(f"Scene initialized. Generating {args.num_scenes} scenes "
          f"x {args.views_per_scene} views = "
          f"{args.num_scenes * args.views_per_scene} images")
    print(f"Domain randomization: HDRIs={len(hdri_files)}, "
          f"CC_textures={len(cc_texture_entries)}, "
          f"cup_presets={len(CUP_MATERIAL_PRESETS)}, "
          f"max_distractors={MAX_DISTRACTORS}, "
          f"post_processing=enabled")

    total_start = time.time()

    for scene_i in range(args.start_scene, args.start_scene + args.num_scenes):
        scene_start = time.time()

        # Per-scene deterministic RNG: same scene_i always produces same output
        rng = np.random.default_rng([base_seed, scene_i])

        # --- [IMPROVEMENT 1] HDRI background ---
        # 75% HDRI, 25% flat color (for diversity)
        if hdri_files and rng.random() < 0.75:
            hdri_mgr.enable_hdri()
            hdri_name = hdri_mgr.set_hdri(rng)
        else:
            hdri_mgr.disable_hdri()
            bg = random_background_color(rng)
            bproc.renderer.set_world_background(bg, strength=1)
            hdri_name = None

        # --- [IMPROVEMENT 2] Randomize table/floor textures (image-swap, no recompile) ---
        tc = random_table_color(rng)
        table_tex_desc = apply_cc_texture_to_surface(
            rng, cc_texture_entries, cc_image_cache, table_nodes, tc)
        leg_mat.set_principled_shader_value("Base Color", tc)

        fc = random_floor_color(rng)
        floor_tex_desc = apply_cc_texture_to_surface(
            rng, cc_texture_entries, cc_image_cache, floor_nodes, fc)

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

        # --- [IMPROVEMENT 3] Distractor objects (pooled) ---
        n_distractors = randomize_distractor_pool(rng, distractor_pool)

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
                'n_distractors': int(n_distractors),
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
              f"dist={n_distractors}")

    total_time = time.time() - total_start
    total_images = args.num_scenes * args.views_per_scene
    print(f"\nDone! Generated {total_images} images in {total_time:.1f}s "
          f"({total_time / total_images:.2f}s/image)")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
