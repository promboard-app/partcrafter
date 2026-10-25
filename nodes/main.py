"""promboard.partcrafter — PartCrafter 이미지 → 부품별로 나뉜 3D 메시(GLB) 노드.

모델 코드는 wgsxm/PartCrafter 의 src 를 nodes/partcrafter 로 옮긴 것이다(NOTICE — 트랜스포머 일부는
Tencent Hunyuan Community License). 공식 예제(scripts/inference_partcrafter.py)와 바꾼 점:
  - 배경 지우기: RMBG-1.4(비상업) 대신 BiRefNet 공식 ONNX(MIT)
  - 파이프라인을 model_index.json(모듈 이름 `src.…`) 대신 부품을 하나씩 불러 조립한다
  - 부품 색: 무작위 대신 공식 팔레트(data_utils.RGB) 순서, 면 줄이기는 trimesh quadric(fast-simplification, MIT)
"""
from __future__ import annotations

import os
import time

REPO = "wgsxm/PartCrafter"
BIREFNET = "birefnet/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
OOM_HELP = "그래픽카드 메모리가 모자랍니다. 부품 수나 부품마다 점 수를 줄이거나, 다른 프로그램이 그래픽카드를 쓰고 있으면 닫아 주세요."
# PartCrafter src/utils/data_utils.py 의 RGB
PALETTE = [
    (82, 170, 220), (215, 91, 78), (45, 136, 117), (247, 172, 83), (124, 121, 121),
    (127, 171, 209), (243, 152, 101), (145, 204, 192), (150, 59, 121), (181, 206, 78),
    (189, 119, 149), (199, 193, 222), (200, 151, 54), (236, 110, 102), (238, 182, 212),
]


def _load_pipeline(ctx, dtype):
    from transformers import BitImageProcessor, Dinov2Model

    from .partcrafter.models.autoencoders import TripoSGVAEModel
    from .partcrafter.models.transformers import PartCrafterDiTModel
    from .partcrafter.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from .partcrafter.schedulers import RectifiedFlowScheduler

    root = os.path.dirname(ctx.model_path(f"{REPO}/model_index.json"))
    sub = lambda name: os.path.join(root, name)  # noqa: E731
    pipe = PartCrafterPipeline(
        vae=TripoSGVAEModel.from_pretrained(sub("vae")),
        transformer=PartCrafterDiTModel.from_pretrained(sub("transformer")),
        scheduler=RectifiedFlowScheduler.from_pretrained(sub("scheduler")),
        image_encoder_dinov2=Dinov2Model.from_pretrained(sub("image_encoder_dinov2")),
        feature_extractor_dinov2=BitImageProcessor.from_pretrained(sub("feature_extractor_dinov2")),
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(ctx.device, dtype)


def image_to_parts(ctx, inputs, settings):
    import numpy as np
    import torch
    import trimesh

    from .cutout import prepare

    if ctx.device != "cuda":
        # 옮겨 온 메시 뽑기 코드가 CUDA 를 못 박아 쓴다(expand_edge_region_fast).
        raise ValueError("PartCrafter 는 NVIDIA 그래픽카드(CUDA)가 있어야 돕니다.")
    parts = int(settings["parts"])
    tokens = int(settings["tokens"])
    steps = int(settings["steps"])
    guidance = float(settings["guidance"])
    faces = int(settings["faces"])
    seed = int(settings["seed"])
    started = time.monotonic()

    ctx.progress(0.02, "cutout")
    bg = str(settings["background"])
    image, removed = prepare(ctx.in_path(inputs["image"]), bg, ctx.model_path(BIREFNET) if bg != "none" else None)
    image.save(ctx.out_path("input-prepared.png"))
    ctx.log(f"input prepared {image.size} (background removed: {removed}) in {time.monotonic() - started:.1f}s")
    ctx.check_cancel()

    dtype = torch.float16
    pipe = None
    try:
        ctx.progress(0.06, "load-model")
        pipe = _load_pipeline(ctx, dtype)
        ctx.log(f"PartCrafter loaded in {time.monotonic() - started:.1f}s")

        def on_step(_pipe, i, _t, kw):
            ctx.check_cancel()
            ctx.progress(0.1 + 0.55 * (i + 1) / steps, "parts")
            return kw

        gen_started = time.monotonic()
        meshes = pipe(
            image=[image] * parts,
            attention_kwargs={"num_parts": parts},
            num_tokens=tokens,
            generator=torch.Generator(device=ctx.device).manual_seed(seed),
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_num_expanded_coords=int(1e9),
            use_flash_decoder=False,
            callback_on_step_end=on_step,
        ).meshes
        ctx.log(f"sampling + {parts} surfaces in {time.monotonic() - gen_started:.1f}s")
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
            raise RuntimeError(OOM_HELP) from exc
        raise
    finally:
        if pipe is not None:
            del pipe
        torch.cuda.empty_cache()
    ctx.check_cancel()

    ctx.progress(0.9, "mesh")
    ok = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not ok:
        raise ValueError("부품을 만들지 못했습니다. 다른 시드나 부품 수로 다시 해 보거나 물체가 더 크게 나온 사진을 넣어 주세요.")
    if len(ok) < len(meshes):
        ctx.log(f"{len(meshes) - len(ok)} of {len(meshes)} parts were empty and are left out", "warn")
    total = sum(len(m.faces) for m in ok)
    scene = trimesh.Scene()
    for i, mesh in enumerate(ok):
        if faces > 0 and total > faces:
            target = max(500, int(len(mesh.faces) * faces / total))
            if len(mesh.faces) > target:
                mesh = mesh.simplify_quadric_decimation(face_count=target)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.array(PALETTE[i % len(PALETTE)], np.uint8))
        scene.add_geometry(mesh, node_name=f"part_{i:02d}", geom_name=f"part_{i:02d}")
    dst = ctx.out_path("parts.glb")
    scene.export(dst)
    ctx.log(f"{len(ok)} parts, {total} -> {sum(len(g.faces) for g in scene.geometry.values())} faces, done in {time.monotonic() - started:.1f}s")
    ctx.progress(1.0, "done")
    return {"model": dst}


NODES = {"image-to-parts": image_to_parts}
