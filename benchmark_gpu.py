import blenderproc as bproc  # must be first import
# Benchmark OptiX vs CUDA rendering for beer pong scenes.
# Renders 5 scenes with each backend and compares per-scene times.
import bpy
import numpy as np
import os
import sys
import time
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import everything from generate_dataset
from generate_dataset import (
    load_usdz, CUP_MODEL_PATH, TARGET_CUP_DIAMETER,
    TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT,
    RENDER_SAMPLES, RESOLUTION, ZED_FX, ZED_FY, ZED_CX, ZED_CY,
    ASSETS_DIR, MAX_DISTRACTORS,
    discover_hdri_files, HDRIManager,
    discover_cc_texture_images, preload_cc_images,
    create_pbr_surface_material, apply_cc_texture_to_surface,
    create_distractor_pool, randomize_distractor_pool,
    random_table_color, random_floor_color, random_background_color,
    random_camera_pose, random_lighting, randomize_cups,
    randomize_balls, randomize_cup_material,
    CUP_MATERIAL_PRESETS,
)

N_SCENES = 5


def setup_scene():
    """One-time scene setup. Returns all reusable objects."""
    bproc.init()
    bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)
    bproc.renderer.set_noise_threshold(0.1)
    bproc.renderer.set_light_bounces(
        diffuse_bounces=2, glossy_bounces=2, max_bounces=4,
        transparent_max_bounces=4,
    )
    bproc.camera.set_resolution(*RESOLUTION)
    K = np.array([[ZED_FX, 0, ZED_CX], [0, ZED_FY, ZED_CY], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, RESOLUTION[0], RESOLUTION[1],
                                               clip_start=0.01, clip_end=50)

    hdri_dir = os.path.join(ASSETS_DIR, "hdris")
    cctextures_dir = os.path.join(ASSETS_DIR, "cctextures")

    hdri_files = discover_hdri_files(hdri_dir)
    hdri_mgr = HDRIManager(hdri_files)

    cc_entries = discover_cc_texture_images(cctextures_dir)
    cc_cache = preload_cc_images(cc_entries) if cc_entries else {}

    cup_objs = load_usdz(CUP_MODEL_PATH)
    if len(cup_objs) > 1:
        cup_template = cup_objs[0]
        for other in cup_objs[1:]:
            cup_template.blender_obj.select_set(True)
            other.blender_obj.select_set(True)
            bpy.context.view_layer.objects.active = cup_template.blender_obj
        bpy.ops.object.join()
        bpy.ops.object.select_all(action='DESELECT')
    else:
        cup_template = cup_objs[0]

    cup_template.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
    bbox_raw = cup_template.get_bound_box()
    raw_d = max(bbox_raw[:, 0]) - min(bbox_raw[:, 0])
    s = TARGET_CUP_DIAMETER / raw_d if raw_d > 0 else 0.01
    cup_template.set_scale([s, s, s])
    cup_template.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
    cup_template.move_origin_to_bottom_mean_point()
    bbox = cup_template.get_bound_box()
    cup_spacing = (max(bbox[:, 0]) - min(bbox[:, 0])) * 1.05
    cup_template.set_location([0, 0, -10])

    table_top = bproc.object.create_primitive("CUBE",
        scale=[TABLE_WIDTH/2, TABLE_LENGTH/2, 0.02],
        location=[0, 0, TABLE_HEIGHT - 0.02])
    table_mat, table_nodes = create_pbr_surface_material("table_surface")
    table_top.replace_materials(table_mat)

    leg_mat = bproc.material.create("table_legs")
    leg_mat.set_principled_shader_value("Roughness", 0.5)
    for lx, ly in [(-TABLE_WIDTH/2+0.04, -TABLE_LENGTH/2+0.06),
                    (TABLE_WIDTH/2-0.04, -TABLE_LENGTH/2+0.06),
                    (-TABLE_WIDTH/2+0.04, TABLE_LENGTH/2-0.06),
                    (TABLE_WIDTH/2-0.04, TABLE_LENGTH/2-0.06)]:
        leg = bproc.object.create_primitive("CUBE",
            scale=[0.03, 0.03, TABLE_HEIGHT/2-0.02],
            location=[lx, ly, TABLE_HEIGHT/2-0.02])
        leg.replace_materials(leg_mat)

    floor = bproc.object.create_primitive("PLANE", scale=[5, 5, 1], location=[0, 0, 0])
    floor_mat, floor_nodes = create_pbr_surface_material("floor")
    floor.replace_materials(floor_mat)

    balls = []
    for i in range(2):
        ball = bproc.object.create_primitive("SPHERE", scale=[0.02, 0.02, 0.02], location=[0, 0, -10])
        bm = bproc.material.create(f"ball_{i}")
        bm.set_principled_shader_value("Base Color", [1, 1, 1, 1])
        bm.set_principled_shader_value("Roughness", 0.3)
        ball.replace_materials(bm)
        balls.append(ball)

    key = bproc.types.Light(); key.set_type("AREA")
    fill = bproc.types.Light(); fill.set_type("POINT")
    back = bproc.types.Light(); back.set_type("POINT")
    lights = [key, fill, back]

    distractor_pool = create_distractor_pool()

    return {
        'hdri_mgr': hdri_mgr, 'hdri_files': hdri_files,
        'cc_entries': cc_entries, 'cc_cache': cc_cache,
        'cup_template': cup_template, 'cup_spacing': cup_spacing,
        'table_top': table_top, 'table_nodes': table_nodes,
        'floor': floor, 'floor_nodes': floor_nodes,
        'leg_mat': leg_mat, 'balls': balls, 'lights': lights,
        'distractor_pool': distractor_pool,
    }


def render_n_scenes(ctx, n, label):
    """Render n scenes with full randomization. Returns list of per-scene times."""
    times = []
    for i in range(n):
        t0 = time.time()
        rng = np.random.default_rng([99, i])

        # HDRI
        if ctx['hdri_files'] and rng.random() < 0.75:
            ctx['hdri_mgr'].enable_hdri()
            ctx['hdri_mgr'].set_hdri(rng)
        else:
            ctx['hdri_mgr'].disable_hdri()
            bproc.renderer.set_world_background(random_background_color(rng), strength=1)

        # Textures
        tc = random_table_color(rng)
        apply_cc_texture_to_surface(rng, ctx['cc_entries'], ctx['cc_cache'], ctx['table_nodes'], tc)
        ctx['leg_mat'].set_principled_shader_value("Base Color", tc)
        fc = random_floor_color(rng)
        apply_cc_texture_to_surface(rng, ctx['cc_entries'], ctx['cc_cache'], ctx['floor_nodes'], fc)

        random_lighting(rng, ctx['lights'])
        randomize_balls(rng, ctx['balls'])
        randomize_cup_material(rng, ctx['cup_template'])

        # Clear old cups
        for name in list(bpy.data.objects.keys()):
            obj = bpy.data.objects.get(name)
            if obj and obj.type == 'MESH' and obj != ctx['cup_template'].blender_obj \
               and obj.data == ctx['cup_template'].blender_obj.data and obj.location.z > -5:
                bpy.data.objects.remove(obj, do_unlink=True)

        randomize_cups(rng, ctx['cup_template'], ctx['cup_spacing'])
        randomize_distractor_pool(rng, ctx['distractor_pool'])

        # Camera
        cam_obj = bpy.context.scene.camera
        if cam_obj and cam_obj.animation_data:
            cam_obj.animation_data_clear()
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 3
        for v in range(4):
            pose = random_camera_pose(rng)
            bproc.camera.add_camera_pose(pose, frame=v)

        data = bproc.renderer.render()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  [{label}] Scene {i}: {elapsed:.1f}s")

    return times


# --- Main ---
ctx = setup_scene()

# Benchmark OptiX (default)
print("\n=== Benchmarking OptiX ===")
bproc.renderer.set_render_devices(desired_gpu_device_type=["OPTIX"])
optix_times = render_n_scenes(ctx, N_SCENES, "OptiX")

# Benchmark CUDA
print("\n=== Benchmarking CUDA ===")
bproc.renderer.set_render_devices(desired_gpu_device_type=["CUDA"])
cuda_times = render_n_scenes(ctx, N_SCENES, "CUDA")

print("\n" + "=" * 50)
print(f"OptiX: avg={np.mean(optix_times):.1f}s  median={np.median(optix_times):.1f}s  times={[f'{t:.1f}' for t in optix_times]}")
print(f"CUDA:  avg={np.mean(cuda_times):.1f}s  median={np.median(cuda_times):.1f}s  times={[f'{t:.1f}' for t in cuda_times]}")
print(f"Winner: {'CUDA' if np.median(cuda_times) < np.median(optix_times) else 'OptiX'} "
      f"(by {abs(np.median(cuda_times) - np.median(optix_times)):.1f}s/scene)")
print("=" * 50)
