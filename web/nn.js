(function (global) {
  let K1, K2, B1, B2, B3, W3, _W;

  function packK(k) {
    const kh = k.length, kw = k[0].length, cin = k[0][0].length, f = k[0][0][0].length;
    const data = new Float32Array(kh * kw * cin * f);
    let t = 0;
    for (let u = 0; u < kh; u++)
      for (let v = 0; v < kw; v++)
        for (let c = 0; c < cin; c++)
          for (let fi = 0; fi < f; fi++) data[t++] = k[u][v][c][fi];
    return { data, kh, kw, cin, f };
  }

  function packX(x) {
    const h = x.length, w = x[0].length, c = x[0][0].length;
    const data = new Float32Array(h * w * c);
    let t = 0;
    for (let i = 0; i < h; i++)
      for (let j = 0; j < w; j++) {
        const p = x[i][j];
        for (let k = 0; k < c; k++) data[t++] = p[k];
      }
    return { data, h, w, c };
  }

  function nestMap(flat, h, w, f) {
    const y = new Array(h);
    for (let i = 0; i < h; i++) {
      y[i] = new Array(w);
      const row = i * w * f;
      for (let j = 0; j < w; j++) {
        const o = row + j * f;
        y[i][j] = flat.subarray(o, o + f);
      }
    }
    return y;
  }

  function conv(xin, ker, bias) {
    const { data: xd, h, w, c: cin } = xin;
    const { data: kd, kh, kw, f } = ker;
    const oh = h - kh + 1, ow = w - kw + 1;
    const yd = new Float32Array(oh * ow * f);
    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        const yo = (i * ow + j) * f;
        for (let fi = 0; fi < f; fi++) yd[yo + fi] = bias[fi];
        for (let u = 0; u < kh; u++) {
          for (let v = 0; v < kw; v++) {
            const xv = ((i + u) * w + j + v) * cin;
            const base = ((u * kw + v) * cin) * f;
            for (let c = 0; c < cin; c++) {
              const xval = xd[xv + c];
              const kb = base + c * f;
              for (let fi = 0; fi < f; fi++) yd[yo + fi] += xval * kd[kb + fi];
            }
          }
        }
      }
    }
    return { data: yd, h: oh, w: ow, c: f };
  }

  function reluIn(m) {
    const d = m.data;
    for (let i = 0; i < d.length; i++) if (d[i] < 0) d[i] = 0;
    return m;
  }

  function nestSw(sw, h, w, f) {
    const y = new Array(h);
    for (let i = 0; i < h; i++) {
      y[i] = new Array(w);
      for (let j = 0; j < w; j++) {
        const o = (i * w + j) * f;
        y[i][j] = sw.subarray(o, o + f);
      }
    }
    return y;
  }

  function pool(m) {
    const { data: xd, h, w, c: f } = m;
    const oh = h >> 1, ow = w >> 1;
    const yd = new Float32Array(oh * ow * f);
    const sw = new Int8Array(oh * ow * f);
    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        const yo = (i * ow + j) * f;
        const i0 = i << 1, j0 = j << 1;
        for (let fi = 0; fi < f; fi++) {
          let best = -1e9, who = 0, t = 0;
          for (let u = 0; u < 2; u++)
            for (let v = 0; v < 2; v++, t++) {
              const v0 = xd[((i0 + u) * w + (j0 + v)) * f + fi];
              if (v0 > best) {
                best = v0;
                who = t;
              }
            }
          yd[yo + fi] = best;
          sw[yo + fi] = who;
        }
      }
    }
    return { y: { data: yd, h: oh, w: ow, c: f }, sw: nestSw(sw, oh, ow, f) };
  }

  function gmax(m) {
    const { data: xd, h, w, c: f } = m;
    const g = new Float32Array(f);
    g.fill(-1e9);
    const n = h * w;
    for (let p = 0; p < n; p++) {
      const o = p * f;
      for (let fi = 0; fi < f; fi++) if (xd[o + fi] > g[fi]) g[fi] = xd[o + fi];
    }
    return g;
  }

  function softmax(z) {
    const m = z[0] > z[1] ? z[0] : z[1];
    const e0 = Math.exp(z[0] - m), e1 = Math.exp(z[1] - m);
    const s = e0 + e1;
    return new Float32Array([e0 / s, e1 / s]);
  }

  function load(W) {
    _W = W;
    K1 = packK(W.k1);
    K2 = packK(W.k2);
    B1 = Float32Array.from(W.b1);
    B2 = Float32Array.from(W.b2);
    B3 = Float32Array.from(W.b3);
    const n = B2.length;
    W3 = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      W3[i * 2] = W.w3[i][0];
      W3[i * 2 + 1] = W.w3[i][1];
    }
  }

  function forward(xNest) {
    const x = packX(xNest);
    const z1 = conv(x, K1, B1);
    const a1m = reluIn(z1);
    const p1 = pool(a1m);
    const z2 = conv(p1.y, K2, B2);
    const a2m = reluIn(z2);
    const p2 = pool(a2m);
    const g = gmax(p2.y);
    const logits = [B3[0], B3[1]];
    const n = g.length;
    for (let i = 0; i < n; i++) {
      logits[0] += g[i] * W3[i * 2];
      logits[1] += g[i] * W3[i * 2 + 1];
    }
    return {
      x: xNest,
      a1: nestMap(a1m.data, a1m.h, a1m.w, a1m.c),
      p1: { y: nestMap(p1.y.data, p1.y.h, p1.y.w, p1.y.c), sw: p1.sw },
      a2: nestMap(a2m.data, a2m.h, a2m.w, a2m.c),
      p2: { y: nestMap(p2.y.data, p2.y.h, p2.y.w, p2.y.c), sw: p2.sw },
      g,
      logits,
      yhat: softmax(logits),
    };
  }

  function argmax2(map, ch) {
    let m = -1e9, yi = 0, xi = 0;
    for (let i = 0; i < map.length; i++)
      for (let j = 0; j < map[0].length; j++)
        if (map[i][j][ch] > m) {
          m = map[i][j][ch];
          yi = i;
          xi = j;
        }
    return [yi, xi, m];
  }

  function flipH(xNest) {
    const h = xNest.length, w = xNest[0].length;
    const y = new Array(h);
    for (let i = 0; i < h; i++) {
      y[i] = new Array(w);
      for (let j = 0; j < w; j++) y[i][j] = xNest[i][w - 1 - j];
    }
    return y;
  }

  function centerZoom(xNest, side) {
    const o = (32 - side) >> 1;
    const crop = new Array(side);
    for (let i = 0; i < side; i++) {
      crop[i] = new Array(side);
      for (let j = 0; j < side; j++) crop[i][j] = xNest[o + i][o + j];
    }
    const y = new Array(32);
    for (let i = 0; i < 32; i++) {
      y[i] = new Array(32);
      const si = Math.min(side - 1, (i * side / 32) | 0);
      for (let j = 0; j < 32; j++) {
        const sj = Math.min(side - 1, (j * side / 32) | 0);
        y[i][j] = crop[si][sj];
      }
    }
    return y;
  }

  function thr() {
    const t = _W && _W.thr;
    return typeof t === "number" ? t : 0.5;
  }

  function decide(yhat) {
    return yhat[1] >= thr() ? 1 : 0;
  }

  function forwardTTA(xNest) {
    const base = forward(xNest);
    const views = [xNest, flipH(xNest), centerZoom(xNest, 28)];
    let l0 = 0, l1 = 0;
    for (let i = 0; i < views.length; i++) {
      const f = forward(views[i]);
      l0 += f.logits[0];
      l1 += f.logits[1];
    }
    const n = views.length;
    const logits = [l0 / n, l1 / n];
    return {
      x: base.x,
      a1: base.a1,
      p1: base.p1,
      a2: base.a2,
      p2: base.p2,
      g: base.g,
      logits,
      yhat: softmax(logits),
    };
  }

  function contribs(f) {
    const cls = decide(f.yhat);
    const rows = [];
    const n = f.g.length;
    for (let i = 0; i < n; i++) {
      const w = W3[i * 2 + cls];
      rows.push({ i, v: f.g[i] * w, g: f.g[i], w });
    }
    rows.sort((a, b) => b.v - a.v);
    return { cls, rows };
  }

  function findPath(f) {
    const { cls, rows } = contribs(f);
    const ch = rows[0].i;
    const [y2, x2, peak2] = argmax2(f.p2.y, ch);
    const sw = f.p2.sw[y2][x2][ch];
    const a2y = y2 * 2 + (sw >> 1), a2x = x2 * 2 + (sw & 1);
    const k2 = _W.k2;
    const cinN = f.p1.y[0][0].length;
    const cinScore = [];
    for (let c = 0; c < cinN; c++) {
      let s = 0;
      for (let u = 0; u < 5; u++)
        for (let v = 0; v < 5; v++) s += f.p1.y[a2y + u][a2x + v][c] * k2[u][v][c][ch];
      cinScore.push({ c, s });
    }
    cinScore.sort((a, b) => b.s - a.s);
    const cin = cinScore[0].c;
    let p1y = a2y, p1x = a2x, m1 = -1e9;
    for (let u = 0; u < 5; u++)
      for (let v = 0; v < 5; v++) {
        const v0 = f.p1.y[a2y + u][a2x + v][cin];
        if (v0 > m1) {
          m1 = v0;
          p1y = a2y + u;
          p1x = a2x + v;
        }
      }
    const s1 = f.p1.sw[p1y][p1x][cin];
    const a1y = p1y * 2 + (s1 >> 1), a1x = p1x * 2 + (s1 & 1);
    return {
      cls, ch, cin, y2, x2, peak2, a2y, a2x, p1y, p1x, a1y, a1x,
      inY: a1y, inX: a1x, inS: 5,
      rfY: Math.max(0, 2 * a2y), rfX: Math.max(0, 2 * a2x), rfS: 14,
      rows, cinScore, m1,
    };
  }

  global.NN = {
    packK,
    packX,
    nestMap,
    conv,
    reluIn,
    pool,
    nestSw,
    gmax,
    softmax,
    argmax2,
    contribs,
    load,
    forward,
    forwardTTA,
    decide,
    thr,
    findPath,
    get W3() {
      return W3;
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
