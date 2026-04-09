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

### Batching more views per scene

Tested rendering 16 views per scene (vs default 4) in a single
`bproc.renderer.render()` call.  Result: **no speedup**.  The ~13s/frame
overhead is per-frame, not per-render-call — Blender rebuilds the OptiX
pipeline for each frame even when camera poses are batched as keyframes.

- 4 views × 1 scene:  ~50s total → 12.5s/image
- 16 views × 1 scene: ~213s total → 13.3s/image

The original hypothesis that the overhead was a one-time cost per render call
was incorrect.

## How to actually go faster

The ~13s/frame overhead is dominated by Blender's per-frame OptiX pipeline
work (scene export, BVH rebuild, texture upload).  This cannot be reduced
within a single process.  The effective options are:

1. **Parallel workers** — `generate_parallel.sh` runs N independent Blender
   processes, each handling a non-overlapping range of scene indices.  Workers
   share the GPU.  Default: 4 workers.  With 4 workers on a single RTX 4090,
   throughput scales roughly linearly (4× → ~3.3s/image effective).

2. **Lower resolution** — reducing from 1920×1080 would cut image write and
   post-processing time, though the OptiX overhead dominates.

3. **Simpler scenes** — fewer distractors and cups reduce BVH build time per
   frame.
