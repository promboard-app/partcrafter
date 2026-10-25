"""GLB 를 앱과 같은 three.js(GLTFLoader)로 그려 정지 그림 · 한 바퀴 GIF 를 만든다.

python render3d.py <model.glb> <out prefix> [--size 512] [--frames 36] [--still-az 35]
→ <prefix>.png (정지), <prefix>.gif (한 바퀴)
"""
import argparse
import base64
import functools
import http.server
import io
import os
import socketserver
import threading

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
# three.js: THREE_DIR(로컬 node_modules/three)가 있으면 그것을, 없으면 같은 판을 unpkg 에서 읽는다(PromBoard 앱과 같은 0.185.1).
THREE = os.environ.get("THREE_DIR")
THREE_BASE = "/three" if THREE else "https://unpkg.com/three@0.185.1"


class Handler(http.server.SimpleHTTPRequestHandler):
    glb = None

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/view.html":
            with open(os.path.join(HERE, "view.html"), encoding="utf-8") as fh:
                body = fh.read().replace("THREE_BASE", THREE_BASE).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def translate_path(self, path):
        p = path.split("?", 1)[0]
        if p == "/model.glb":
            return self.glb
        if THREE and p.startswith("/three/"):
            return os.path.join(THREE, *p[len("/three/"):].split("/"))
        return os.path.join(HERE, *p.lstrip("/").split("/"))

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("prefix")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--still-az", type=float, default=35)
    ap.add_argument("--el", type=float, default=18)
    a = ap.parse_args()
    Handler.glb = os.path.abspath(a.glb)
    Handler.extensions_map[".js"] = "text/javascript"
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": a.size, "height": a.size})
        pg.goto(f"http://127.0.0.1:{port}/view.html?size={a.size}")
        info = pg.evaluate("window.ready")
        print("loaded", info)

        def shot(az):
            url = pg.evaluate(f"window.renderAt({az}, {a.el})")
            return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")

        shot(a.still_az).save(a.prefix + ".png")
        if a.frames > 0:
            frames = [shot(a.still_az + 360 * i / a.frames) for i in range(a.frames)]
            pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
            pal[0].save(a.prefix + ".gif", save_all=True, append_images=pal[1:], duration=80, loop=0, optimize=True)
        b.close()
    httpd.shutdown()
    print("ok", a.prefix)


if __name__ == "__main__":
    main()
