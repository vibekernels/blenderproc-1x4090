# Render Performance Optimization Notes

## Summary

Total wall-clock time for 4 frames at 1920×1080 is ~50 seconds. The GPU renders
all 4 frames in approximately **1 second**. The remaining ~49 seconds is fixed
Blender cold-start overhead that cannot be meaningfully reduced within the
current single-run architecture.

---

## What was investigated

### GPU utilization
Monitored with `nvidia-smi dmon` during render. The RTX 4090 peaked at 98% SM
utilization for exactly one second, then returned to 0%. This confirmed the GPU
is being used correctly and is not the bottleneck.

### Sample count (256 → 32)
No measurable effect on render time. Adaptive sampling with `noise_threshold`
converges well below 32 samples per pixel for this scene, so the max-sample cap
was never the limiting factor.

### Noise threshold (0.01 → 0.1)
No measurable effect. Same reason as above — adaptive sampling was already
stopping far short of the threshold.

### Depth output pass
No measurable effect (~0s difference with/without).

### OptiX denoising
Added `bpy.context.scene.cycles.use_denoising = True` with `denoiser = 'OPTIX'`.
No meaningful speedup; removed because it requires auxiliary passes (albedo,
normals) that add complexity without benefit for this use case.

### Kernel cache (`CYCLES_KERNEL_CACHE_PATH`)
Blender 4.2 ships with pre-compiled PTX kernels (`kernel_optix.ptx.zst`).
The NVIDIA driver caches the PTX→GPU binary compilation in
`/var/tmp/OptixCache_ubuntu/optix7cache.db` (1.1 MB, persisted across runs).
The cache is populated and being reused — kernel compilation is not the
bottleneck.

### Linked cup instances
Changed `cup.duplicate()` → `cup.duplicate(linked=True)` so all 20 cups share
one mesh datablock and one BVH. Marginal improvement; the dominant overhead is
not per-object BVH but OptiX pipeline and scene-export work inside
`bpy.ops.render.render()`.

### Manual GPU device override
Blenderproc already selects OptiX by default on Linux
(`desired_gpu_device_type = ["OPTIX", "CUDA", "HIP"]`). Manual overrides of
`bpy.context.preferences.addons['cycles'].preferences` and
`bpy.context.scene.cycles.device` were redundant and have been removed.

### Light bounces
Reduced from `diffuse=3, glossy=3, max=6, transparent=8` to
`diffuse=2, glossy=2, max=4, transparent=4`. No scene elements require deep
transparency or glossy chains, so this is safe and slightly reduces per-sample
cost.

---

## Root cause of the ~49s overhead

The time is spent inside `bpy.ops.render.render()` on work that happens before
and after the GPU fires:

- OptiX pipeline initialization and scene export to Cycles/OptiX format
- BVH construction
- Texture upload to GPU VRAM
- Post-processing and image writing

A minimal single-cube scene costs ~10s just to call `bpy.ops.render.render()`
once. The beer pong scene (20 textured cup instances, 3 lights) adds ~37s on
top. This is characteristic of Blender as a cold-start process.

---

## How to actually go faster

The only lever that would make a material difference is **rendering multiple
scene variations in a single Blender session**. Currently each `render.sh`
invocation pays the full ~49s overhead for 4 frames. If the script were
restructured to loop over N scene variations (different cup arrangements, camera
angles, lighting, etc.) within one Blender session, the overhead would be paid
once and the marginal cost per frame would be ~0.25s.

For example, 100 variations × 4 views = 400 frames:
- **Current approach**: ~50s × 100 runs = ~83 minutes
- **Single-session approach**: ~50s startup + 400 × ~0.25s = ~2 minutes
