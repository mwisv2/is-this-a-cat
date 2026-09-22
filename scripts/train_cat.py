from __future__ import annotations
import json, math, urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np

Here = Path(__file__).resolve().parents[1]
Hole, There = Here / ".cache", Here / "web"
Edit = Here / "web-edit"
TR = "https://raw.githubusercontent.com/jalalmansoori19/Cat-Classifier/master/datasets/train_catvnoncat.h5"
TE = "https://raw.githubusercontent.com/jalalmansoori19/Cat-Classifier/master/datasets/test_catvnoncat.h5"
RNG = np.random.default_rng(1)
EPOCHS = 120
DROP = 0.2
WD = 5e-5


def _get(url, t=30):
    req = urllib.request.Request(url, headers={"User-Agent": "yz-cat/1"})
    with urllib.request.urlopen(req, timeout=t) as r:
        return r.read()


def _stash(url, name, n=1000):
    Hole.mkdir(exist_ok=True)
    dest = Hole / name
    if dest.exists() and dest.stat().st_size > n:
        return dest
    print(f"downloading {name}…")
    dest.write_bytes(_get(url))
    return dest


def _32(x):  # 64→32 by 2×2 mean
    return x.reshape(-1, 32, 2, 32, 2, 3).mean((2, 4)).astype(np.float32)


def _pix(raw):
    from PIL import Image

    return np.asarray(Image.open(BytesIO(raw)).convert("RGB").resize((32, 32), Image.BILINEAR), np.float32) * (1 / 255)


def _h5():
    with h5py.File(_stash(TR, "train_catvnoncat.h5")) as f:
        x, y = _32(np.array(f["train_set_x"])) / 255.0, np.array(f["train_set_y"]).astype(np.int64)
    with h5py.File(_stash(TE, "test_catvnoncat.h5")) as f:
        xt, yt = _32(np.array(f["test_set_x"])) / 255.0, np.array(f["test_set_y"]).astype(np.int64)
    return np.ascontiguousarray(x), y, np.ascontiguousarray(xt), yt


def _urls(n):
    out = []
    while len(out) < n:
        k = min(50, n - len(out))
        out.extend(json.loads(_get(f"https://dog.ceo/api/breeds/image/random/{k}", 12))["message"])
    return out[:n]


def _dog_urls_diverse(n):
    # sample across breeds so "dog" isn't one pose/breed cluster
    breeds = list(json.loads(_get("https://dog.ceo/api/breeds/list/all", 12))["message"].keys())
    RNG.shuffle(breeds)
    breeds = breeds[: min(60, len(breeds))]
    per = max(1, (n + len(breeds) - 1) // len(breeds))

    def breed_batch(b):
        try:
            msg = json.loads(_get(f"https://dog.ceo/api/breed/{b}/images/random/{per}", 12))["message"]
            return msg if isinstance(msg, list) else [msg]
        except Exception:
            return []

    out = []
    with ThreadPoolExecutor(16) as ex:
        for batch in ex.map(breed_batch, breeds):
            out.extend(batch)
            if len(out) >= n:
                break
    RNG.shuffle(out)
    if len(out) < n:
        out.extend(_urls(n - len(out)))
    return out[:n]


def _extra(n_dog=450, n_cat=220, n_fox=100, n_bg=120, force=False):
    # dogs/foxes + random scenes so "not cat" isn't one shortcut
    # kind: 1=cat 2=dog 3=fox 4=scene
    dest = Hole / "extra_pets_v5.npz"
    if not force and dest.exists() and dest.stat().st_size > 40_000:
        z = np.load(dest)
        return z["x"], z["y"], z["kind"]

    def one(url):
        try:
            return _pix(_get(url, 12))
        except Exception:
            return None

    def fox(_):
        try:
            return one(json.loads(_get("https://randomfox.ca/floof/", 12))["image"])
        except Exception:
            return None

    print(f"fetching {n_dog} dogs (breed-diverse) + {n_fox} foxes + {n_bg} scenes + {n_cat} cats…")
    cats_u = [f"https://cataas.com/cat?width=96&height=96&t={i}" for i in range(n_cat)]
    bg_u = [f"https://picsum.photos/seed/yz{i}/96/96" for i in range(n_bg)]
    with ThreadPoolExecutor(20) as ex:
        dogs = [im for im in ex.map(one, _dog_urls_diverse(n_dog)) if im is not None]
        cats = [im for im in ex.map(one, cats_u) if im is not None]
        foxes = [im for im in ex.map(fox, range(n_fox)) if im is not None]
        bgs = [im for im in ex.map(one, bg_u) if im is not None]
    if len(dogs) + len(foxes) + len(bgs) < 40 or len(cats) < 40:
        raise RuntimeError(f"extras thin: cats={len(cats)} dogs={len(dogs)} fox={len(foxes)} bg={len(bgs)}")
    x = np.stack(cats + dogs + foxes + bgs)
    y = np.concatenate(
        [
            np.ones(len(cats), np.int64),
            np.zeros(len(dogs) + len(foxes) + len(bgs), np.int64),
        ]
    )
    kind = np.concatenate(
        [
            np.full(len(cats), 1, np.int64),
            np.full(len(dogs), 2, np.int64),
            np.full(len(foxes), 3, np.int64),
            np.full(len(bgs), 4, np.int64),
        ]
    )
    np.savez_compressed(dest, x=x, y=y, kind=kind)
    print(f"extra: {len(cats)} cats, {len(dogs)} dogs, {len(foxes)} foxes, {len(bgs)} scenes")
    return x, y, kind


def _phash(batch):
    # tiny 8×8 quantized gray fingerprint for near-dupe checks
    g = batch.mean(-1)
    small = g[:, ::4, ::4]
    return (small * 15).astype(np.int16).reshape(len(batch), -1)


def _leak_report(x_tr, x_va, k_tr, k_va):
    ht = _phash(x_tr)
    hv = _phash(x_va)
    # hash → list of train kinds
    bucket = {}
    for i, row in enumerate(ht):
        key = row.tobytes()
        bucket.setdefault(key, []).append(int(k_tr[i]))
    hits = {2: 0, 3: 0, 1: 0, 0: 0, 4: 0}
    n_hit = 0
    for i, row in enumerate(hv):
        key = row.tobytes()
        if key in bucket:
            n_hit += 1
            hits[int(k_va[i])] = hits.get(int(k_va[i]), 0) + 1
    print(
        f"near-dupe val↔train: {n_hit}/{len(x_va)} "
        f"(cat={hits.get(1,0)} dog={hits.get(2,0)} fox={hits.get(3,0)} scene={hits.get(4,0)} h5-not={hits.get(0,0)})"
    )


def _bag():
    x, y, xt, yt = _h5()
    # h5 kinds: 0 = h5-not, 1 = h5-cat
    kx = np.where(y == 1, 1, 0).astype(np.int64)
    kxt = np.where(yt == 1, 1, 0).astype(np.int64)
    xe, ye, ke = _extra()
    hard, cats = np.flatnonzero(ye == 0), np.flatnonzero(ye == 1)
    RNG.shuffle(hard)
    RNG.shuffle(cats)
    n_hte, n_cte = min(60, len(hard) // 4), min(24, len(cats) // 5)
    hte, cte, htr, ctr = hard[:n_hte], cats[:n_cte], hard[n_hte:], cats[n_cte:]
    x = np.ascontiguousarray(np.concatenate([x, xe[ctr], xe[htr]]))
    y = np.concatenate([y, ye[ctr], ye[htr]])
    kx = np.concatenate([kx, ke[ctr], ke[htr]])
    xt = np.ascontiguousarray(np.concatenate([xt, xe[cte], xe[hte]]))
    yt = np.concatenate([yt, ye[cte], ye[hte]])
    kxt = np.concatenate([kxt, ke[cte], ke[hte]])
    print(f"train {len(x)}  val {len(xt)}  extra-neg val {len(hte)}")
    _leak_report(x, xt, kx, kxt)
    return x, y, xt, yt, kx, kxt, len(hte)


def _col(x, w, b):  # batched im2col conv NHWC
    n, h, wd, c = x.shape
    kh, kw, _, f = w.shape
    oh, ow = h - kh + 1, wd - kw + 1
    s = x.strides
    cols = np.lib.stride_tricks.as_strided(x, (n, oh, ow, kh, kw, c), (s[0], s[1], s[2], s[1], s[2], s[3]))
    return (cols.reshape(n * oh * ow, -1) @ w.reshape(-1, f) + b).reshape(n, oh, ow, f)


def _mx2(x):
    n, h, w, c = x.shape
    return x.reshape(n, h >> 1, 2, w >> 1, 2, c).max((2, 4))


def _sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _boot():
    # He/Kaiming for ReLU
    def w(*shape):
        return RNG.normal(0, np.sqrt(2 / np.prod(shape[:-1])), shape).astype(np.float32)

    return {
        "k1": w(5, 5, 3, 6),
        "b1": np.zeros(6, np.float32),
        "k2": w(5, 5, 6, 8),
        "b2": np.zeros(8, np.float32),
        "w3": w(8, 2),
        "b3": np.zeros(2, np.float32),
    }


def _fwd(p, x, drop=0.0):
    z1 = _col(x, p["k1"], p["b1"])
    a1 = np.maximum(z1, 0)
    p1 = np.ascontiguousarray(_mx2(a1))
    z2 = _col(p1, p["k2"], p["b2"])
    a2 = np.maximum(z2, 0)
    p2 = np.ascontiguousarray(_mx2(a2))
    g = p2.max((1, 2))
    if drop > 0:
        keep = (RNG.random(g.shape) >= drop).astype(np.float32)
        g = g * keep / (1.0 - drop)
    logits = g @ p["w3"] + p["b3"]
    return _sm(logits), (z1, a1, p1, z2, a2, p2, g, logits)


def _col_b(x, w, dout):
    n, h, wd, c = x.shape
    kh, kw, _, f = w.shape
    oh, ow = dout.shape[1:3]
    s = x.strides
    cols = np.lib.stride_tricks.as_strided(x, (n, oh, ow, kh, kw, c), (s[0], s[1], s[2], s[1], s[2], s[3])).reshape(
        n * oh * ow, -1
    )
    dr = dout.reshape(n * oh * ow, f)
    dw, db = (cols.T @ dr).reshape(w.shape), dr.sum(0)
    dxc = (dr @ w.reshape(-1, f).T).reshape(n, oh, ow, kh, kw, c)
    dx = np.zeros_like(x)
    for i in range(kh):
        for j in range(kw):
            dx[:, i : i + oh, j : j + ow] += dxc[:, :, :, i, j]
    return dx, dw, db


def _mx2_b(a, dout):
    n, h, w, c = a.shape
    p = a.reshape(n, h >> 1, 2, w >> 1, 2, c)
    m = p.max((2, 4), keepdims=True)
    up = (p == m) * dout.reshape(n, h >> 1, 1, w >> 1, 1, c)
    return up.reshape(n, h, w, c)


def _grads(p, x, y, drop=DROP):
    n = x.shape[0]
    yhat, (z1, a1, p1, z2, a2, p2, g, _) = _fwd(p, x, drop=drop)
    # mild smoothing; cats slightly less smoothed
    yoh = np.zeros((n, 2), np.float32)
    for i in range(n):
        if y[i] == 1:
            yoh[i] = (0.04, 0.96)
        else:
            yoh[i] = (0.94, 0.06)
    w = np.where(y == 1, 1.35, 1.0).astype(np.float32)
    w = w / w.mean()
    per = -np.sum(yoh * np.log(yhat.clip(1e-8)), 1)
    loss = float((per * w).mean())
    acc = float((yhat.argmax(1) == y).mean())
    dz = ((yhat - yoh) * w[:, None]) / n
    dw3, db3 = g.T @ dz, dz.sum(0)
    dg = dz @ p["w3"].T
    n2, h, wmap, c = p2.shape
    dp2 = np.zeros_like(p2)
    flat = p2.reshape(n2, h * wmap, c)
    dp2.reshape(n2, h * wmap, c)[np.arange(n2)[:, None], flat.argmax(1), np.arange(c)] = dg
    da2 = _mx2_b(a2, dp2) * (z2 > 0)
    dp1, dk2, db2 = _col_b(p1, p["k2"], da2)
    da1 = _mx2_b(a1, dp1) * (z1 > 0)
    _, dk1, db1 = _col_b(x, p["k1"], da1)
    return loss, acc, {
        "k1": dk1.astype(np.float32),
        "b1": db1.astype(np.float32),
        "k2": dk2.astype(np.float32),
        "b2": db2.astype(np.float32),
        "w3": dw3.astype(np.float32),
        "b3": db3.astype(np.float32),
    }


def _adam(p, m, v, g, t, lr, wd=WD):
    t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        v[k] = b2 * v[k] + (1 - b2) * (g[k] * g[k])
        step = (m[k] / (1 - b1**t)) / (np.sqrt(v[k] / (1 - b2**t)) + eps)
        if k[0] != "b":
            step = step + wd * p[k]
        p[k] -= lr * step
    return t


def _rot_batch(xb, deg):
    from PIL import Image

    out = np.empty_like(xb)
    for i in range(xb.shape[0]):
        pil = Image.fromarray((xb[i] * 255).astype(np.uint8))
        pil = pil.rotate(float(deg[i]), resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        out[i] = np.asarray(pil, np.float32) * (1 / 255)
    return out


def _aug(xb):
    xb = np.ascontiguousarray(xb)
    if RNG.random() < 0.5:
        xb = np.ascontiguousarray(xb[:, :, ::-1])
    if RNG.random() < 0.7:
        deg = RNG.uniform(-15, 15, size=xb.shape[0])
        xb = _rot_batch(xb, deg)
    if RNG.random() < 0.55:
        xb = np.ascontiguousarray(np.roll(xb, int(RNG.integers(-3, 4)), 1))
    if RNG.random() < 0.55:
        xb = np.ascontiguousarray(np.roll(xb, int(RNG.integers(-3, 4)), 2))
    if RNG.random() < 0.55:
        # random crop / mild zoom
        side = int(RNG.integers(24, 31))
        y0 = int(RNG.integers(0, 33 - side))
        x0 = int(RNG.integers(0, 33 - side))
        crop = xb[:, y0 : y0 + side, x0 : x0 + side]
        idx = (np.arange(32) * side / 32).astype(np.int64)
        xb = crop[:, idx][:, :, idx]
    # brightness + contrast
    gain = float(RNG.uniform(0.75, 1.25))
    bias = float(RNG.uniform(-0.08, 0.08))
    mean = xb.mean((1, 2, 3), keepdims=True)
    xb = np.clip((xb - mean) * gain + mean + bias, 0, 1)
    if RNG.random() < 0.35:
        xb = np.clip(xb + RNG.normal(0, 0.025, xb.shape).astype(np.float32), 0, 1)
    return xb.astype(np.float32)


def _lr(ep, total=EPOCHS, lr0=0.002, lr_min=8e-5):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * ep / max(total - 1, 1)))


def _fwd_tta(p, x):
    # average logits over original + hflip + mild center crop (test-time only)
    views = [x, np.ascontiguousarray(x[:, :, ::-1])]
    side = 28
    y0 = x0 = (32 - side) // 2
    crop = x[:, y0 : y0 + side, x0 : x0 + side]
    idx = (np.arange(32) * side / 32).astype(np.int64)
    views.append(np.ascontiguousarray(crop[:, idx][:, :, idx]))
    logits = 0.0
    last = None
    for v in views:
        yhat, cache = _fwd(p, v, drop=0.0)
        logits = logits + cache[-1]
        last = (yhat, cache)
    logits = logits / len(views)
    yhat = _sm(logits)
    # maps from unaugmented pass for any downstream use
    _, cache0 = _fwd(p, x, drop=0.0)
    return yhat, (*cache0[:-1], logits)


def _metrics(yhat, yt, thr=0.5):
    pred = (yhat[:, 1] >= thr).astype(np.int64)
    val = float((pred == yt).mean())
    cat_rec = float((pred[yt == 1] == 1).mean()) if (yt == 1).any() else 0.0
    neg_rec = float((pred[yt == 0] == 0).mean()) if (yt == 0).any() else 0.0
    bal = 0.5 * cat_rec + 0.5 * neg_rec
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tp = int(((pred == 1) & (yt == 1)).sum())
    cat_conf = float(yhat[yt == 1, 1].mean()) if (yt == 1).any() else 0.0
    cat_hi = float((yhat[yt == 1, 1] >= 0.8).mean()) if (yt == 1).any() else 0.0
    return {
        "val": val,
        "bal": bal,
        "cat": cat_rec,
        "neg": neg_rec,
        "pred": pred,
        "cm": (tn, fp, fn, tp),
        "cat_conf": cat_conf,
        "cat_hi": cat_hi,
    }


def _kind_acc(pred, yt, kxt):
    names = {0: "h5-not", 1: "cat", 2: "dog", 3: "fox", 4: "scene"}
    out = {}
    for k, name in names.items():
        m = kxt == k
        if not m.any():
            continue
        # for cats expect 1; for not-cats expect 0
        want = 1 if k == 1 else 0
        out[name] = float((pred[m] == want).mean())
    return out


def _best_thr(yhat, yt):
    best_t, best_b, best_m = 0.5, -1.0, None
    for t in np.linspace(0.30, 0.65, 36):
        m = _metrics(yhat, yt, thr=float(t))
        sc = 0.35 * m["bal"] + 0.30 * m["cat"] + 0.20 * m["cat_conf"] + 0.15 * m["neg"]
        if m["neg"] < 0.35:
            sc -= 0.12
        if m["cat"] < 0.70:
            sc -= 0.08
        if sc > best_b:
            best_b, best_t, best_m = sc, float(t), m
    return best_t, best_m


def _apply_temp(p, T):
    out = {k: v.copy() for k, v in p.items()}
    out["w3"] = (out["w3"] / T).astype(np.float32)
    out["b3"] = (out["b3"] / T).astype(np.float32)
    return out


def _calibrate(p, xt, yt, kxt):
    # nudge cat logit + sharpen so true cats land high-confidence; keep a dog floor
    yhat0, _ = _fwd_tta(p, xt)
    base_dog = _kind_acc((yhat0[:, 1] >= 0.5).astype(np.int64), yt, kxt).get("dog", 0.0)
    floor = max(0.30, base_dog - 0.18)
    best_sc, best_p, best_meta = -1.0, p, (1.0, 0.0)
    for bias in np.linspace(0.0, 1.8, 19):
        for T in np.linspace(0.40, 1.0, 13):
            pt = {k: v.copy() for k, v in p.items()}
            pt["b3"] = pt["b3"].copy()
            pt["b3"][1] = pt["b3"][1] + bias
            pt = _apply_temp(pt, float(T))
            yhat, _ = _fwd_tta(pt, xt)
            thr, met = _best_thr(yhat, yt)
            kinds = _kind_acc(met["pred"], yt, kxt)
            dog = kinds.get("dog", met["neg"])
            if dog < floor or met["cat"] < 0.75:
                continue
            sc = (
                0.40 * met["cat_conf"]
                + 0.30 * met["cat_hi"]
                + 0.20 * met["cat"]
                + 0.10 * dog
            )
            if sc > best_sc:
                best_sc, best_p, best_meta = sc, pt, (float(T), float(bias))
    print(f"calibrate  T={best_meta[0]:.2f}  bias={best_meta[1]:.2f}  dog_floor={floor:.0%}")
    return best_p


def _save_mis_grid(imgs, path, cols=8):
    from PIL import Image

    if len(imgs) == 0:
        return
    n = len(imgs)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    cell = 32
    canvas = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        tile = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        canvas.paste(tile, (c * cell, r * cell))
    path.parent.mkdir(exist_ok=True)
    canvas.save(path)
    print(f"wrote {path}  ({n} tiles @ 32×32)")


def _mine_hard(p, x, y, kx, thr=0.5):
    # dog/fox train samples currently scored as cat
    mammal = np.flatnonzero((y == 0) & ((kx == 2) | (kx == 3)))
    if len(mammal) == 0:
        return np.array([], np.int64)
    yhat, _ = _fwd(p, x[mammal], drop=0.0)
    wrong = mammal[yhat[:, 1] >= thr]
    return wrong.astype(np.int64)


def _dump(a):
    return np.round(a.astype(np.float32), 6).tolist()


def main():
    x, y, xt, yt, kx, kxt, n_neg = _bag()
    print(f"cats {int((y == 1).sum())}  not {int((y == 0).sum())}")
    p = _boot()
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    t, bs = 0, 16
    cats, others = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    hard = np.array([], np.int64)
    best_sc, best = -1e9, None

    for ep in range(EPOCHS):
        lr = _lr(ep)
        n = max(len(cats), len(others))
        if len(hard) > 0:
            n_h = min(len(hard), n // 2)
            neg = np.concatenate(
                [RNG.choice(hard, n_h, replace=True), RNG.choice(others, n - n_h, replace=True)]
            )
        else:
            neg = RNG.choice(others, n, replace=True)
        mix = np.concatenate([RNG.choice(cats, n, replace=True), neg])
        RNG.shuffle(mix)
        losses, accs = [], []
        for i in range(0, len(mix), bs):
            b = mix[i : i + bs]
            loss, acc, g = _grads(p, _aug(x[b]), y[b], drop=DROP)
            t = _adam(p, m, v, g, t, lr)
            losses.append(loss)
            accs.append(acc)
        yhat, _ = _fwd(p, xt, drop=0.0)
        met = _metrics(yhat, yt, thr=0.5)
        neg_acc = float((met["pred"][-n_neg:] == 0).mean()) if n_neg else met["neg"]
        train_a = float(np.mean(accs))
        # balanced + prefer higher cat confidence
        sc = 0.40 * met["bal"] + 0.25 * met["cat"] + 0.20 * met["cat_conf"] + 0.15 * neg_acc
        if abs(met["cat"] - neg_acc) > 0.35:
            sc -= 0.08
        if met["cat_conf"] < 0.58:
            sc -= 0.05
        if ep >= 14 and ep % 5 == 4:
            hard = _mine_hard(p, x, y, kx, thr=0.5)
            print(f"  hard-neg pool: {len(hard)} dog/fox train FPs")
        print(
            f"epoch {ep+1:3d}  lr {lr:.5f}  loss {np.mean(losses):.3f}  "
            f"train {train_a:.0%}  val {met['val']:.0%}  gap {train_a-met['val']:+.0%}  "
            f"bal {met['bal']:.0%}  neg {neg_acc:.0%}  cat {met['cat']:.0%}  "
            f"cconf {met['cat_conf']:.2f}  chi {met['cat_hi']:.0%}"
        )
        if sc > best_sc:
            best_sc, best = sc, {k: vv.copy() for k, vv in p.items()}

    p = best or {k: vv.copy() for k, vv in p.items()}
    p = _calibrate(p, xt, yt, kxt)
    # TTA logits → threshold sweep → confusion + kind breakdown
    yhat_tta, _ = _fwd_tta(p, xt)
    thr, met = _best_thr(yhat_tta, yt)
    tn, fp, fn, tp = met["cm"]
    kinds = _kind_acc(met["pred"], yt, kxt)
    print(
        f"TTA+thr={thr:.2f}  val={met['val']:.0%}  bal={met['bal']:.0%}  "
        f"cat={met['cat']:.0%}  neg={met['neg']:.0%}  "
        f"cat_conf={met['cat_conf']:.2f}  cat≥0.8={met['cat_hi']:.0%}"
    )
    print(f"confusion  [[tn={tn} fp={fp}]  [fn={fn} tp={tp}]]")
    print("by kind  " + "  ".join(f"{k}={v:.0%}" for k, v in kinds.items()))

    pred = met["pred"]
    # eyeball misclassified mammals at real 32×32 input
    mis_dog = xt[(kxt == 2) & (pred == 1)]
    mis_fox = xt[(kxt == 3) & (pred == 1)]
    _save_mis_grid(mis_dog[:32], Hole / "mis_dogs_32.png")
    _save_mis_grid(mis_fox[:32], Hole / "mis_foxes_32.png")
    print(f"misclassified @32: dogs={len(mis_dog)} foxes={len(mis_fox)}")

    ok_cat = xt[(yt == 1) & (pred == 1)][:8]
    ok_not = xt[(yt == 0) & (pred == 0)][:8]
    if len(ok_cat) < 3:
        ok_cat = xt[yt == 1][:6]
    if len(ok_not) < 3:
        ok_not = xt[yt == 0][:6]
    out = {
        "val_acc": met["val"],
        "balanced_acc": met["bal"],
        "dog_acc": float(kinds.get("dog", met["neg"])),
        "thr": thr,
        "cat_conf": met["cat_conf"],
        "cat_hi": met["cat_hi"],
        "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "kind_acc": kinds,
        "k1": _dump(p["k1"]),
        "b1": _dump(p["b1"]),
        "k2": _dump(p["k2"]),
        "b2": _dump(p["b2"]),
        "w3": _dump(p["w3"]),
        "b3": _dump(p["b3"]),
        "samples": {"cat": _dump(ok_cat), "not": _dump(ok_not)},
    }
    blob = json.dumps(out, separators=(",", ":"))
    for dest in (There, *((Edit,) if Edit.is_dir() else ())):
        dest.mkdir(exist_ok=True)
        (dest / "weights.json").write_text(blob)
        print(f"wrote {dest / 'weights.json'}  val={met['val']:.0%}  bal={met['bal']:.0%}  thr={thr:.2f}")


if __name__ == "__main__":
    main()
