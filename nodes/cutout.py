"""물체 사진 → 흰 배경 위에 물체만 가운데 둔 정사각 그림(TripoSG · PartCrafter 입력 모양).

배경은 BiRefNet(MIT, 공식 ONNX — ZhengPeng7/BiRefNet GitHub 릴리스 v1)으로 지운다. 원래 TripoSG 예제는 RMBG-1.4(비상업)를
쓰지만 이 플러그인은 그것을 받지 않는다. 투명 PNG 처럼 알파가 이미 쓸 만하면 그 알파를 쓴다.
잘라 내기 · 여백 · 흰 배경 합성은 TripoSG scripts/image_process.py(load_image)와 같은 순서다.
"""
from __future__ import annotations

from typing import Optional, Tuple

SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _valid_alpha(alpha, min_ratio: float = 0.01) -> bool:
    """TripoSG is_valid_alpha 와 같다 — 아주 투명한 칸과 아주 불투명한 칸이 각각 1% 이상이면 쓸 만한 알파."""
    import numpy as np

    hist, _ = np.histogram(alpha, bins=20, range=(0, 256))
    need = alpha.shape[0] * alpha.shape[1] * min_ratio
    return hist[0] >= need and hist[-1] >= need


def birefnet_alpha(rgb, model_path: str):
    """RGB uint8 (H, W, 3) → 알파 uint8 (H, W). onnxruntime CPU 로 1024×1024 에서 돌린다."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    h, w = rgb.shape[:2]
    small = np.asarray(Image.fromarray(rgb).resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = ((small - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)).transpose(2, 0, 1)[None]
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x, dtype=np.float32)})[-1]
    while out.ndim > 2:
        out = out[0]
    mask = 0.5 * (np.tanh(0.5 * out.astype(np.float32)) + 1.0)
    del sess
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
    return mask


def _clean(alpha):
    """오츠 이진화 + 작은 조각 지우기(TripoSG load_image 와 같다)."""
    import numpy as np
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.morphology import remove_small_objects

    t = threshold_otsu(alpha) if alpha.max() > alpha.min() else 127
    binary = alpha > t
    cleaned = remove_small_objects(label(binary), min_size=200) > 0
    return cleaned.astype(np.uint8) * 255


def prepare(path: str, mode: str, birefnet_path: Optional[str], padding_ratio: float = 0.1) -> Tuple[object, bool]:
    """→ (흰 배경 정사각 PIL 그림, 배경을 지웠나). mode: auto · birefnet · none."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if max(im.size) > 2000:
            k = 2000 / max(im.size)
            im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        rgba = np.asarray(im.convert("RGBA"))
    rgb = np.ascontiguousarray(rgba[..., :3])
    alpha = None
    removed = False
    if mode == "none":
        alpha = np.full(rgb.shape[:2], 255, np.uint8)
    elif mode == "auto" and has_alpha and _valid_alpha(rgba[..., 3]):
        alpha = rgba[..., 3]
    else:
        if not birefnet_path:
            raise ValueError("배경 지우기 모델(BiRefNet)이 준비되지 않았습니다.")
        alpha = _clean(birefnet_alpha(rgb, birefnet_path))
        removed = True
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        raise ValueError("그림에서 물체를 찾지 못했습니다. 물체 하나가 크게 나온 사진을 넣거나 '배경 지우기'를 '안 함'으로 바꿔 보세요.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    comp = rgb.astype(np.float32) * a + 255.0 * (1.0 - a)
    crop = comp[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = int(max(w, h) * (1 + 2 * padding_ratio))
    canvas = np.full((side, side, 3), 255.0, np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)), removed
