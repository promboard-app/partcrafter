"""README 그림: 입력 사진 → 결과 그림들을 한 줄로(글자 없음, 불투명 배경).

python compose.py <out.jpg|png> <height> <img1> <img2> ...  (첫 장 뒤에 화살표)
"""
import sys

from PIL import Image, ImageDraw

BG = (236, 238, 241)


def fit(im, h):
    im = im.convert("RGB") if im.mode != "RGBA" else im
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def trim(im, margin=0.08):
    """렌더 그림의 빈 배경을 잘라 물체가 크게 보이게 한다(사방 여백 margin)."""
    from PIL import ImageChops

    rgb = im.convert("RGB")
    box = ImageChops.difference(rgb, Image.new("RGB", rgb.size, BG)).convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    if not box:
        return rgb
    x0, y0, x1, y1 = box
    m = int(max(x1 - x0, y1 - y0) * margin)
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(rgb.width, x1 + m), min(rgb.height, y1 + m)
    return rgb.crop((x0, y0, x1, y1))


def main():
    out, h = sys.argv[1], int(sys.argv[2])
    ims = [fit(Image.open(p) if k == 0 else trim(Image.open(p)), h) for k, p in enumerate(sys.argv[3:])]
    gap, arrow, pad = 16, 56, 16
    w = pad * 2 + sum(i.width for i in ims) + gap * (len(ims) - 2) + arrow
    canvas = Image.new("RGB", (w, h + pad * 2), BG)
    d = ImageDraw.Draw(canvas)
    x = pad
    for k, im in enumerate(ims):
        canvas.paste(im, (x, pad))
        x += im.width
        if k == 0:
            cy = pad + h // 2
            d.line([(x + 12, cy), (x + arrow - 14, cy)], fill=(120, 124, 132), width=4)
            d.polygon([(x + arrow - 10, cy), (x + arrow - 24, cy - 9), (x + arrow - 24, cy + 9)], fill=(120, 124, 132))
            x += arrow
        elif k < len(ims) - 1:
            x += gap
    canvas.save(out, quality=90) if out.lower().endswith(".jpg") else canvas.save(out)
    print("ok", out, canvas.size)


if __name__ == "__main__":
    main()
