from __future__ import annotations
import argparse, json, runpy, sys, webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

Here, There, Meat = (λ := Path(__file__).resolve().parent), λ / "web", λ / "web" / "weights.json"
EXT = {f".{e}" for e in "jpg jpeg png webp gif bmp tif tiff".split()}
KEYS = "k1 b1 k2 b2 w3 b3".split()


def Σ(path=Meat):
    raw = json.loads(Path(path).read_text())
    return {k: np.asarray(raw[k], np.float32) for k in KEYS}


def _pix(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR), np.float32) * (1 / 255)


def _col(x, w, b):
    # im2col: view every kh×kw patch, then one matmul
    h, wd, c = x.shape
    kh, kw, _, f = w.shape
    oh, ow = h - kh + 1, wd - kw + 1
    s0, s1, s2 = x.strides
    patches = np.lib.stride_tricks.as_strided(x, (oh, ow, kh, kw, c), (s0, s1, s0, s1, s2))
    return (patches.reshape(oh * ow, -1) @ w.reshape(-1, f) + b).reshape(oh, ow, f)


def _mx2(x):
    h, w, c = x.shape
    return x.reshape(h >> 1, 2, w >> 1, 2, c).max((1, 3))


def _go(x, p):
    g = _mx2(np.maximum(_col(_mx2(np.maximum(_col(x, p["k1"], p["b1"]), 0)), p["k2"], p["b2"]), 0)).max((0, 1))
    z = g @ p["w3"] + p["b3"]
    e = np.exp(z - z.max())  # softmax
    return e / e.sum(), z


load_weights, load_image, predict = Σ, _pix, _go


def _die(msg, code=1):
    print(msg, file=sys.stderr)
    return code


def cmd_predict(image, weights, as_json):
    if not (image := Path(image)).is_file():
        return _die(f"not found: {image}")
    yhat, logits = _go(_pix(image), Σ(weights))
    tag = "cat" if yhat[1] >= yhat[0] else "not cat"
    print(
        json.dumps(
            {
                "file": str(image),
                "label": tag,
                "p_not_cat": float(yhat[0]),
                "p_cat": float(yhat[1]),
                "logits": [float(logits[0]), float(logits[1])],
            }
        )
        if as_json
        else f"{tag}  not-cat {yhat[0]*100:.1f}%  cat {yhat[1]*100:.1f}%"
    )
    return 0


def cmd_train():
    runpy.run_path(str(Here / "scripts" / "train_cat.py"), run_name="__main__")
    return 0


def cmd_serve(port, open_browser):
    if not There.is_dir():
        return _die(f"missing {There}")
    box = ThreadingHTTPServer(("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(There)))
    url = f"http://127.0.0.1:{port}/"
    print(f"serving {There.name}/ at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        box.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py", description="Train, serve, or classify with the tiny cat CNN.")
    ap.add_argument("command", nargs="?", default="serve", help="serve | train | predict | image path")
    ap.add_argument("image", nargs="?", help="image for predict")
    ap.add_argument("-p", "--port", type=int, default=8765)
    ap.add_argument("--weights", type=Path, default=Meat)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    cmd, img, known = a.command, a.image, {"serve", "train", "predict"}

    # `python run.py photo.jpg` → predict
    if img is None and Path(cmd).suffix.lower() in EXT:
        img, cmd = Path(cmd), "predict"
    elif cmd == "predict" and img is not None:
        img = Path(img)
    elif cmd not in known and Path(cmd).exists():
        img, cmd = Path(cmd), "predict"

    match cmd:
        case "serve":
            return cmd_serve(a.port, not a.no_browser)
        case "train":
            return cmd_train()
        case "predict":
            return cmd_predict(img, a.weights, a.json) if img else ap.error("predict needs an image path")
        case _:
            ap.error(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
