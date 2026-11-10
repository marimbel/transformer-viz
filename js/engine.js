/* ============================================================
   engine.js — a tiny matrix autograd + GPT-style transformer
   Everything here runs in the browser: forward pass, backprop,
   and the Adam optimizer. No libraries.
   ============================================================ */
(function (root) {
  'use strict';

  // ---------- random ----------
  let seed = 1337;
  function rand() { // mulberry32
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function setSeed(s) { seed = s; }

  // ---------- Tensor (always 2-D: rows x cols) ----------
  class Tensor {
    constructor(data, rows, cols, parents, backward) {
      this.data = data; this.rows = rows; this.cols = cols;
      this.grad = null;
      this.parents = parents || [];
      this._backward = backward || null;
    }
    static zeros(r, c) { return new Tensor(new Float32Array(r * c), r, c); }
    static fill(r, c, v) { const t = Tensor.zeros(r, c); t.data.fill(v); return t; }
    static randn(r, c, std) {
      const t = Tensor.zeros(r, c);
      for (let i = 0; i < t.data.length; i++) t.data[i] = randn() * std;
      return t;
    }
    g() { if (!this.grad) this.grad = new Float32Array(this.data.length); return this.grad; }
    item() { return this.data[0]; }
  }

  // ---------- ops ----------
  function matmul(A, B) {
    const n = A.rows, k = A.cols, m = B.cols;
    if (k !== B.rows) throw new Error(`matmul shape ${n}x${k} @ ${B.rows}x${m}`);
    const out = new Float32Array(n * m), a = A.data, b = B.data;
    for (let i = 0; i < n; i++) {
      const io = i * m, ia = i * k;
      for (let p = 0; p < k; p++) {
        const av = a[ia + p]; if (av === 0) continue;
        const pb = p * m;
        for (let j = 0; j < m; j++) out[io + j] += av * b[pb + j];
      }
    }
    const O = new Tensor(out, n, m, [A, B]);
    O._backward = () => {
      const dO = O.grad, dA = A.g(), dB = B.g();
      for (let i = 0; i < n; i++) {
        const io = i * m, ia = i * k;
        for (let p = 0; p < k; p++) {
          const pb = p * m; let s = 0; const av = a[ia + p];
          for (let j = 0; j < m; j++) { const d = dO[io + j]; s += d * b[pb + j]; dB[pb + j] += av * d; }
          dA[ia + p] += s;
        }
      }
    };
    return O;
  }

  // A + B, where B may be 1 x cols (broadcast over rows)
  function add(A, B) {
    const n = A.rows, m = A.cols, bc = B.rows === 1 && n !== 1;
    const out = new Float32Array(n * m), a = A.data, b = B.data;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++)
      out[i * m + j] = a[i * m + j] + (bc ? b[j] : b[i * m + j]);
    const O = new Tensor(out, n, m, [A, B]);
    O._backward = () => {
      const dO = O.grad, dA = A.g(), dB = B.g();
      for (let i = 0; i < dO.length; i++) dA[i] += dO[i];
      if (bc) { for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) dB[j] += dO[i * m + j]; }
      else for (let i = 0; i < dO.length; i++) dB[i] += dO[i];
    };
    return O;
  }

  function relu(A) {
    const out = new Float32Array(A.data.length);
    for (let i = 0; i < out.length; i++) out[i] = A.data[i] > 0 ? A.data[i] : 0;
    const O = new Tensor(out, A.rows, A.cols, [A]);
    O._backward = () => { const dA = A.g(); for (let i = 0; i < out.length; i++) if (A.data[i] > 0) dA[i] += O.grad[i]; };
    return O;
  }

  function scale(A, s) {
    const out = new Float32Array(A.data.length);
    for (let i = 0; i < out.length; i++) out[i] = A.data[i] * s;
    const O = new Tensor(out, A.rows, A.cols, [A]);
    O._backward = () => { const dA = A.g(); for (let i = 0; i < out.length; i++) dA[i] += O.grad[i] * s; };
    return O;
  }

  function transpose(A) {
    const n = A.rows, m = A.cols, out = new Float32Array(n * m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j * n + i] = A.data[i * m + j];
    const O = new Tensor(out, m, n, [A]);
    O._backward = () => { const dA = A.g(); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) dA[i * m + j] += O.grad[j * n + i]; };
    return O;
  }

  function layerNorm(X, gain, bias) {
    const n = X.rows, m = X.cols, eps = 1e-5;
    const out = new Float32Array(n * m), xhat = new Float32Array(n * m), inv = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let mu = 0; for (let j = 0; j < m; j++) mu += X.data[i * m + j]; mu /= m;
      let v = 0; for (let j = 0; j < m; j++) { const d = X.data[i * m + j] - mu; v += d * d; } v /= m;
      inv[i] = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < m; j++) {
        const xh = (X.data[i * m + j] - mu) * inv[i];
        xhat[i * m + j] = xh; out[i * m + j] = xh * gain.data[j] + bias.data[j];
      }
    }
    const O = new Tensor(out, n, m, [X, gain, bias]);
    O._backward = () => {
      const dO = O.grad, dX = X.g(), dg = gain.g(), db = bias.g();
      for (let i = 0; i < n; i++) {
        let s1 = 0, s2 = 0;
        for (let j = 0; j < m; j++) {
          const d = dO[i * m + j], dxh = d * gain.data[j];
          dg[j] += d * xhat[i * m + j]; db[j] += d;
          s1 += dxh; s2 += dxh * xhat[i * m + j];
        }
        for (let j = 0; j < m; j++) {
          const dxh = dO[i * m + j] * gain.data[j];
          dX[i * m + j] += inv[i] / m * (m * dxh - s1 - xhat[i * m + j] * s2);
        }
      }
    };
    return O;
  }

  // softmax over each row, only columns j <= i allowed (causal mask)
  function causalSoftmax(S) {
    const n = S.rows, m = S.cols, out = new Float32Array(n * m);
    for (let i = 0; i < n; i++) {
      let mx = -Infinity; for (let j = 0; j <= i; j++) mx = Math.max(mx, S.data[i * m + j]);
      let sum = 0; for (let j = 0; j <= i; j++) { const e = Math.exp(S.data[i * m + j] - mx); out[i * m + j] = e; sum += e; }
      for (let j = 0; j <= i; j++) out[i * m + j] /= sum;
    }
    const O = new Tensor(out, n, m, [S]);
    O._backward = () => {
      const dO = O.grad, dS = S.g();
      for (let i = 0; i < n; i++) {
        let dot = 0; for (let j = 0; j <= i; j++) dot += out[i * m + j] * dO[i * m + j];
        for (let j = 0; j <= i; j++) dS[i * m + j] += out[i * m + j] * (dO[i * m + j] - dot);
      }
    };
    return O;
  }

  function gather(E, ids) { // rows of E selected by ids
    const m = E.cols, n = ids.length, out = new Float32Array(n * m);
    for (let t = 0; t < n; t++) out.set(E.data.subarray(ids[t] * m, ids[t] * m + m), t * m);
    const O = new Tensor(out, n, m, [E]);
    O._backward = () => { const dE = E.g(); for (let t = 0; t < n; t++) for (let j = 0; j < m; j++) dE[ids[t] * m + j] += O.grad[t * m + j]; };
    return O;
  }

  function rowSlice(A, n) { // first n rows
    const m = A.cols, out = A.data.slice(0, n * m);
    const O = new Tensor(out, n, m, [A]);
    O._backward = () => { const dA = A.g(); for (let i = 0; i < n * m; i++) dA[i] += O.grad[i]; };
    return O;
  }

  function colSlice(A, start, len) {
    const n = A.rows, m = A.cols, out = new Float32Array(n * len);
    for (let i = 0; i < n; i++) for (let j = 0; j < len; j++) out[i * len + j] = A.data[i * m + start + j];
    const O = new Tensor(out, n, len, [A]);
    O._backward = () => { const dA = A.g(); for (let i = 0; i < n; i++) for (let j = 0; j < len; j++) dA[i * m + start + j] += O.grad[i * len + j]; };
    return O;
  }

  function concatCols(list) {
    const n = list[0].rows, m = list.reduce((s, t) => s + t.cols, 0), out = new Float32Array(n * m);
    let off = 0;
    for (const T of list) { for (let i = 0; i < n; i++) for (let j = 0; j < T.cols; j++) out[i * m + off + j] = T.data[i * T.cols + j]; off += T.cols; }
    const O = new Tensor(out, n, m, list);
    O._backward = () => {
      let o = 0;
      for (const T of list) { const dT = T.g(); for (let i = 0; i < n; i++) for (let j = 0; j < T.cols; j++) dT[i * T.cols + j] += O.grad[i * m + o + j]; o += T.cols; }
    };
    return O;
  }

  // mean cross-entropy of logits (n x V) vs integer targets
  function crossEntropy(L, targets) {
    const n = L.rows, V = L.cols, probs = new Float32Array(n * V);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let mx = -Infinity; for (let j = 0; j < V; j++) mx = Math.max(mx, L.data[i * V + j]);
      let sum = 0; for (let j = 0; j < V; j++) { const e = Math.exp(L.data[i * V + j] - mx); probs[i * V + j] = e; sum += e; }
      for (let j = 0; j < V; j++) probs[i * V + j] /= sum;
      loss -= Math.log(probs[i * V + targets[i]] + 1e-12);
    }
    const O = new Tensor(new Float32Array([loss / n]), 1, 1, [L]);
    O.probs = probs;
    O._backward = () => {
      const dL = L.g(), d = O.grad[0] / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < V; j++)
        dL[i * V + j] += d * (probs[i * V + j] - (j === targets[i] ? 1 : 0));
    };
    return O;
  }

  function backward(loss) {
    const order = [], seen = new Set();
    (function visit(t) {
      if (seen.has(t)) return; seen.add(t);
      for (const p of t.parents) visit(p);
      order.push(t);
    })(loss);
    loss.g()[0] = 1;
    for (let i = order.length - 1; i >= 0; i--) if (order[i]._backward) order[i]._backward();
  }

  // ---------- the model ----------
  class GPT {
    constructor(cfg) {
      this.cfg = Object.assign({ vocab: 32, ctx: 16, dModel: 32, heads: 2, layers: 2, dFF: 64, seed: 1337 }, cfg);
      setSeed(this.cfg.seed);
      const { vocab: V, ctx: T, dModel: d, layers: L, dFF: f } = this.cfg;
      const std = 0.08, projStd = 0.08 / Math.sqrt(2 * L);
      this.params = {};     // name -> Tensor
      this.meta = {};       // name -> {group, layer, desc}
      const P = (name, t, group, layer, desc) => { this.params[name] = t; this.meta[name] = { group, layer, desc }; return t; };
      P('tok_emb', Tensor.randn(V, d, std), 'embed', -1, 'Token embedding table: one learned vector per token');
      P('pos_emb', Tensor.randn(T, d, std), 'embed', -1, 'Position embedding table: one learned vector per position');
      this.blocks = [];
      for (let l = 0; l < L; l++) {
        const b = {};
        b.ln1_g = P(`L${l}.ln1.gain`, Tensor.fill(1, d, 1), 'norm', l, 'LayerNorm scale before attention');
        b.ln1_b = P(`L${l}.ln1.bias`, Tensor.zeros(1, d), 'norm', l, 'LayerNorm shift before attention');
        b.Wq = P(`L${l}.attn.Wq`, Tensor.randn(d, d, std), 'attn', l, 'Query projection: what each token is looking for');
        b.Wk = P(`L${l}.attn.Wk`, Tensor.randn(d, d, std), 'attn', l, 'Key projection: what each token offers');
        b.Wv = P(`L${l}.attn.Wv`, Tensor.randn(d, d, std), 'attn', l, 'Value projection: the information each token passes on');
        b.Wo = P(`L${l}.attn.Wo`, Tensor.randn(d, d, projStd), 'attn', l, 'Output projection: mixes the heads back together');
        b.ln2_g = P(`L${l}.ln2.gain`, Tensor.fill(1, d, 1), 'norm', l, 'LayerNorm scale before the MLP');
        b.ln2_b = P(`L${l}.ln2.bias`, Tensor.zeros(1, d), 'norm', l, 'LayerNorm shift before the MLP');
        b.W1 = P(`L${l}.mlp.W1`, Tensor.randn(d, f, std), 'mlp', l, 'MLP expand: d_model → d_ff');
        b.b1 = P(`L${l}.mlp.b1`, Tensor.zeros(1, f), 'mlp', l, 'MLP hidden bias');
        b.W2 = P(`L${l}.mlp.W2`, Tensor.randn(f, d, projStd), 'mlp', l, 'MLP contract: d_ff → d_model');
        b.b2 = P(`L${l}.mlp.b2`, Tensor.zeros(1, d), 'mlp', l, 'MLP output bias');
        this.blocks.push(b);
      }
      this.lnf_g = P('lnf.gain', Tensor.fill(1, d, 1), 'norm', -1, 'Final LayerNorm scale');
      this.lnf_b = P('lnf.bias', Tensor.zeros(1, d), 'norm', -1, 'Final LayerNorm shift');
      this.W_out = P('head.W', Tensor.randn(d, V, std), 'head', -1, 'Unembedding: turns a vector into a score per token');
      this.b_out = P('head.b', Tensor.zeros(1, V), 'head', -1, 'Output bias per token');
      // Adam state
      this.m = {}; this.v = {}; this.t = 0;
      for (const k in this.params) { this.m[k] = new Float32Array(this.params[k].data.length); this.v[k] = new Float32Array(this.params[k].data.length); }
    }

    numParams() { let n = 0; for (const k in this.params) n += this.params[k].data.length; return n; }

    // ids: array of token ids (length <= ctx). Returns logits + a trace for visualization.
    forward(ids, record) {
      const { heads: H, dModel: d } = this.cfg, hd = d / H, n = ids.length;
      const trace = record ? { ids: ids.slice(), layers: [] } : null;
      const tok = gather(this.params.tok_emb, ids);
      const pos = rowSlice(this.params.pos_emb, n);
      let x = add(tok, pos);
      if (trace) { trace.tok = tok.data.slice(); trace.pos = pos.data.slice(); trace.x0 = x.data.slice(); }
      for (const b of this.blocks) {
        const lt = record ? { attn: [] } : null;
        const h = layerNorm(x, b.ln1_g, b.ln1_b);
        const q = matmul(h, b.Wq), k = matmul(h, b.Wk), v = matmul(h, b.Wv);
        const outs = [];
        for (let hh = 0; hh < H; hh++) {
          const qh = colSlice(q, hh * hd, hd), kh = colSlice(k, hh * hd, hd), vh = colSlice(v, hh * hd, hd);
          const att = causalSoftmax(scale(matmul(qh, transpose(kh)), 1 / Math.sqrt(hd)));
          if (lt) lt.attn.push(att.data.slice());
          outs.push(matmul(att, vh));
        }
        x = add(x, matmul(H > 1 ? concatCols(outs) : outs[0], b.Wo));
        const h2 = layerNorm(x, b.ln2_g, b.ln2_b);
        const hid = relu(add(matmul(h2, b.W1), b.b1));
        if (lt) { lt.mlpIn = h2.data.slice(); lt.mlpHidden = hid.data.slice(); }
        const mo = add(matmul(hid, b.W2), b.b2);
        if (lt) lt.mlpOut = mo.data.slice();
        x = add(x, mo);
        if (lt) { lt.resid = x.data.slice(); trace.layers.push(lt); }
      }
      const logits = add(matmul(layerNorm(x, this.lnf_g, this.lnf_b), this.W_out), this.b_out);
      return { logits, trace };
    }

    zeroGrad() { for (const k in this.params) if (this.params[k].grad) this.params[k].grad.fill(0); }

    trainStep(batch, lr, opts) {
      const r = this.computeGrads(batch, opts);
      this.applyUpdate(lr, r.gradNorm, opts);
      return r;
    }

    // forward + loss + backward on a batch of {x, y} sequences (no update yet)
    computeGrads(batch, opts) {
      opts = opts || {};
      this.zeroGrad();
      let total = 0, lastTrace = null, lastProbs = null;
      batch.forEach((ex, i) => {
        const rec = i === batch.length - 1 && opts.record;
        const { logits, trace } = this.forward(ex.x, rec);
        const loss = crossEntropy(logits, ex.y);
        loss.grad = new Float32Array([1 / batch.length]);
        backward(loss);
        total += loss.item();
        if (rec) { lastTrace = trace; lastProbs = loss.probs; }
      });
      // gradient norms per parameter, with global clipping at 1.0
      const gnorm = {}; let gsq = 0;
      for (const k in this.params) {
        const g = this.params[k].grad; let s = 0;
        for (let i = 0; i < g.length; i++) s += g[i] * g[i];
        gnorm[k] = Math.sqrt(s); gsq += s;
      }
      return { loss: total / batch.length, gradNorm: Math.sqrt(gsq), gnorm, trace: lastTrace, probs: lastProbs };
    }

    // Adam update using the gradients currently stored on each parameter
    applyUpdate(lr, gradNorm, opts) {
      opts = opts || {};
      const clip = Math.min(1, 1 / (gradNorm + 1e-8));
      this.t++;
      const b1 = 0.9, b2 = 0.99, eps = 1e-8, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      const wd = opts.weightDecay || 0;
      for (const k in this.params) {
        const p = this.params[k].data, g = this.params[k].grad, m = this.m[k], v = this.v[k];
        const decay = this.meta[k].group === 'norm' || k.endsWith('.b') || /\.b\d$/.test(k) ? 0 : wd;
        for (let i = 0; i < p.length; i++) {
          const gi = g[i] * clip;
          m[i] = b1 * m[i] + (1 - b1) * gi;
          v[i] = b2 * v[i] + (1 - b2) * gi * gi;
          p[i] -= lr * ((m[i] / c1) / (Math.sqrt(v[i] / c2) + eps) + decay * p[i]);
        }
      }
    }

    // next-token probabilities for the last position
    predict(ids, temperature) {
      const ctx = ids.slice(-this.cfg.ctx);
      const { logits, trace } = this.forward(ctx, true);
      const V = this.cfg.vocab, row = logits.data.subarray((ctx.length - 1) * V, ctx.length * V);
      const t = Math.max(temperature || 1, 1e-3);
      let mx = -Infinity; for (let j = 0; j < V; j++) mx = Math.max(mx, row[j] / t);
      const p = new Float32Array(V); let s = 0;
      for (let j = 0; j < V; j++) { p[j] = Math.exp(row[j] / t - mx); s += p[j]; }
      for (let j = 0; j < V; j++) p[j] /= s;
      return { probs: p, trace };
    }

    sample(ids, n, temperature) {
      const out = ids.slice();
      for (let i = 0; i < n; i++) {
        const { probs } = this.predict(out, temperature);
        let r = Math.random(), j = 0;
        for (; j < probs.length - 1; j++) { r -= probs[j]; if (r <= 0) break; }
        out.push(j);
      }
      return out;
    }
  }

  // ---------- character tokenizer ----------
  class CharTokenizer {
    constructor(text) {
      this.chars = Array.from(new Set(text.split(''))).sort();
      this.stoi = {}; this.chars.forEach((c, i) => (this.stoi[c] = i));
    }
    get size() { return this.chars.length; }
    encode(s) { return s.split('').filter(c => c in this.stoi).map(c => this.stoi[c]); }
    decode(ids) { return ids.map(i => this.chars[i]).join(''); }
  }

  root.Engine = { Tensor, GPT, CharTokenizer, ops: { matmul, add, relu, scale, transpose, layerNorm, causalSoftmax, gather, rowSlice, colSlice, concatCols, crossEntropy, backward }, setSeed, rand };
})(typeof window !== 'undefined' ? window : globalThis);
