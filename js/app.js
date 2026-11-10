/* ============================================================
   app.js — interactive visuals wired to the live model
   ============================================================ */
(function () {
  'use strict';
  const E = window.Engine;
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const DPR = Math.min(window.devicePixelRatio || 1, 2);

  const C = {
    text: '#f0efec', text2: '#c3c2b7', muted: '#8d8c84', line: '#2e2e2b', line2: '#3a3a36',
    surface: '#1a1a19', surface2: '#222220', bg: '#111110',
    blue: '#3987e5', orange: '#d95926', aqua: '#199e70', yellow: '#c98500', magenta: '#d55181', mid: '#383835',
  };
  const RGB = { pos: [57, 135, 229], neg: [217, 89, 38], mid: [56, 56, 53] };
  const GROUPS = {
    embed: { label: 'Embeddings', color: C.blue },
    attn: { label: 'Attention', color: C.orange },
    mlp: { label: 'MLP', color: C.aqua },
    norm: { label: 'LayerNorm', color: C.yellow },
    head: { label: 'Output head', color: C.magenta },
  };

  const DEFAULT_CORPUS = [
    'the robot picks up the red block.',
    'the robot puts down the red block.',
    'the robot picks up the blue block.',
    'the robot puts down the blue block.',
    'the arm moves left. the arm moves right.',
    'the gripper opens. the gripper closes.',
    'the robot sees the green ball and rolls it to the wall.',
    'the robot picks up the blue block and puts it on the red block.',
    'a small model learns to guess the next letter.',
    'attention lets each letter look back at the letters before it.',
    'the arm moves up. the arm moves down.',
    'the robot picks up the green ball.',
    'the gripper closes on the red block and the arm moves left.',
    'the robot puts the blue block on the green block.',
  ].join('\n');

  // ---------------- state ----------------
  const S = {
    corpus: DEFAULT_CORPUS,
    cfg: { ctx: 16, dModel: 32, heads: 2, layers: 2, dFF: 64 },
    seed: 1337,
    lr: Math.pow(10, -2.5), batch: 8, speed: 2,
    running: false, step: 0, losses: [], emas: [], tokensSeen: 0, gradNorm: null,
    lastGnorm: null, lastEx: null, lastProbs: null, lastDelta: null,
    samples: [], phase: -1, pending: null,
    archSel: 'attn', archParam: null,
    attnLayer: 0, attnHead: 0, attnQuery: -1,
    nnLayer: 0, nnTok: -1, nnMode: 'w', nnDens: 6, nnHover: null,
    pmSel: 'L0.attn.Wq',
    visible: {}, lossHover: null,
  };
  let tok, data, model;

  // ---------------- helpers ----------------
  const fmt = (n) => n >= 1e12 ? (n / 1e12).toFixed(2) + 'T' : n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e4 ? (n / 1e3).toFixed(1) + 'K' : n.toLocaleString();
  const showChar = (c) => c === ' ' ? '␣' : c === '\n' ? '↵' : c;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const maxAbs = (arr) => { let m = 0; for (let i = 0; i < arr.length; i++) { const a = Math.abs(arr[i]); if (a > m) m = a; } return m || 1; };
  function el(tag, attrs, html) { const e = document.createElement(tag); if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]); if (html != null) e.innerHTML = html; return e; }
  function svgEl(tag, attrs) { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }

  // tooltip
  const tipEl = $('#tip');
  function tip(html, x, y) {
    tipEl.innerHTML = html; tipEl.classList.add('on');
    const r = tipEl.getBoundingClientRect();
    let tx = x + 14, ty = y + 14;
    if (tx + r.width > innerWidth - 8) tx = x - r.width - 14;
    if (ty + r.height > innerHeight - 8) ty = y - r.height - 14;
    tipEl.style.left = tx + 'px'; tipEl.style.top = ty + 'px';
  }
  function untip() { tipEl.classList.remove('on'); }

  function divRGB(v, s) {
    const t = clamp(v / (s || 1), -1, 1), a = Math.pow(Math.abs(t), 0.7), c = t >= 0 ? RGB.pos : RGB.neg;
    return [RGB.mid[0] + (c[0] - RGB.mid[0]) * a, RGB.mid[1] + (c[1] - RGB.mid[1]) * a, RGB.mid[2] + (c[2] - RGB.mid[2]) * a];
  }
  function seqRGB(v) { // 0..1 -> mid..orange
    const a = Math.pow(clamp(v, 0, 1), 0.6), c = RGB.neg;
    return [RGB.mid[0] + (c[0] - RGB.mid[0]) * a, RGB.mid[1] + (c[1] - RGB.mid[1]) * a, RGB.mid[2] + (c[2] - RGB.mid[2]) * a];
  }

  // pixel heatmap: 1 canvas pixel per cell, scaled up by CSS
  function heat(cv, arr, rows, cols, o) {
    o = o || {};
    const s = o.scale || maxAbs(arr);
    cv.width = cols; cv.height = rows;
    const cell = o.cell || clamp(Math.floor((o.maxW || 320) / cols), 2, 18);
    cv.style.width = cols * cell + 'px'; cv.style.height = Math.max(rows * cell, o.minH || 0) + 'px';
    const ctx = cv.getContext('2d'), img = ctx.createImageData(cols, rows);
    for (let i = 0; i < rows * cols; i++) {
      const masked = o.mask && o.mask(Math.floor(i / cols), i % cols);
      const c = masked ? [26, 26, 25] : o.seq ? seqRGB(arr[i]) : divRGB(arr[i], s);
      img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    cv._h = { arr, rows, cols, o, s };
    if (!cv._hooked) {
      cv._hooked = true;
      cv.addEventListener('mousemove', (ev) => {
        const h = cv._h, r = cv.getBoundingClientRect();
        const row = Math.floor((ev.clientY - r.top) / r.height * h.rows), col = Math.floor((ev.clientX - r.left) / r.width * h.cols);
        if (row < 0 || col < 0 || row >= h.rows || col >= h.cols) return untip();
        if (h.o.mask && h.o.mask(row, col)) return tip('<span class="muted">masked (future token)</span>', ev.clientX, ev.clientY);
        const v = h.arr[row * h.cols + col];
        const lab = h.o.tip ? h.o.tip(row, col, v) : `${h.o.name || 'value'}[${row}, ${col}]`;
        tip(`${lab}<br><b>${v.toFixed(4)}</b>`, ev.clientX, ev.clientY);
      });
      cv.addEventListener('mouseleave', untip);
    }
  }

  function fitCanvas(cv, w, h) {
    cv.width = Math.round(w * DPR); cv.height = Math.round(h * DPR);
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    const ctx = cv.getContext('2d'); ctx.setTransform(DPR, 0, 0, DPR, 0, 0); return ctx;
  }

  // ---------------- model lifecycle ----------------
  function build(resetSeed) {
    tok = new E.CharTokenizer(S.corpus);
    data = tok.encode(S.corpus);
    if (resetSeed) S.seed++;
    model = new E.GPT(Object.assign({}, S.cfg, { vocab: tok.size, seed: S.seed }));
    Object.assign(S, { step: 0, losses: [], emas: [], tokensSeen: 0, gradNorm: null, lastGnorm: null, lastProbs: null, lastDelta: null, samples: [], phase: -1, pending: null, lastEx: null });
    S.attnLayer = Math.min(S.attnLayer, S.cfg.layers - 1); S.attnHead = Math.min(S.attnHead, S.cfg.heads - 1);
    S.nnLayer = Math.min(S.nnLayer, S.cfg.layers - 1);
    if (!model.params[S.pmSel]) S.pmSel = 'L0.attn.Wq';
    S.archParam = null;
    newExample();
    renderStatic();
    renderAll();
  }

  function getWindow() {
    const T = S.cfg.ctx, i = Math.floor(Math.random() * (data.length - T - 1));
    return { x: data.slice(i, i + T), y: data.slice(i + 1, i + T + 1) };
  }
  function getBatch() { const b = []; for (let i = 0; i < S.batch; i++) b.push(getWindow()); return b; }
  function newExample() { S.example = getWindow(); }

  // probability the model gives each correct target in a window
  function targetProbs(ex) {
    const { logits, trace } = model.forward(ex.x, true), V = tok.size, n = ex.x.length, out = [];
    for (let t = 0; t < n; t++) {
      const row = logits.data.subarray(t * V, t * V + V); let mx = -Infinity, s = 0;
      for (let j = 0; j < V; j++) mx = Math.max(mx, row[j]);
      for (let j = 0; j < V; j++) s += Math.exp(row[j] - mx);
      out.push(Math.exp(row[ex.y[t]] - mx) / s);
    }
    return { p: out, trace };
  }

  // ---------------- training ----------------
  function recordStep(r) {
    S.step++; S.tokensSeen += S.batch * S.cfg.ctx; S.gradNorm = r.gradNorm; S.lastGnorm = r.gnorm;
    S.losses.push(r.loss);
    const prev = S.emas.length ? S.emas[S.emas.length - 1] : r.loss;
    S.emas.push(S.emas.length ? prev * 0.95 + r.loss * 0.05 : r.loss);
    if (S.step % 150 === 0) addSample();
  }

  function trainOnce(record) {
    const batch = getBatch();
    const r = model.computeGrads(batch, { record });
    let before = null;
    if (record) before = model.params[S.pmSel].data.slice();
    model.applyUpdate(S.lr, r.gradNorm);
    if (record) {
      const now = model.params[S.pmSel].data, d = new Float32Array(now.length);
      for (let i = 0; i < d.length; i++) d[i] = now[i] - before[i];
      S.lastDelta = { name: S.pmSel, d };
      S.lastEx = batch[batch.length - 1]; S.lastProbs = r.probs;
    }
    recordStep(r);
  }

  let lastHeavy = 0, lastArch = 0, frames = 0, spsT = performance.now(), spsSteps = 0;
  function loop(now) {
    if (!S.running) return;
    for (let i = 0; i < S.speed; i++) trainOnce(i === S.speed - 1);
    spsSteps += S.speed; frames++;
    if (now - spsT > 1000) { $('#d-sps').textContent = Math.round(spsSteps * 1000 / (now - spsT)) + ' steps/s'; spsT = now; spsSteps = 0; }
    setLoopPhase(Math.floor(now / 140) % 5);
    renderLight();
    if (now - lastHeavy > 350) { lastHeavy = now; renderHeavy(); }
    if (now - lastArch > 1600) { lastArch = now; animateArch('fwd', () => animateArch('bwd')); }
    requestAnimationFrame(loop);
  }
  function setRunning(on) {
    S.running = on;
    if (on) { S.phase = -1; S.pending = null; spsT = performance.now(); spsSteps = 0; requestAnimationFrame(loop); explain('<b>Training continuously.</b> Each frame runs ' + S.speed + ' full iterations of the loop above. Scroll through the page: every visual updates from the live weights.'); }
    else { setLoopPhase(-1); $('#d-sps').textContent = ''; renderAll(); }
    $('#t-run').textContent = on ? '❚❚ Pause' : '▶ Train';
    $('#d-run').textContent = on ? '❚❚ Pause' : '▶ Train';
    $('#hero-train').textContent = on ? '❚❚ Pause training' : '▶ Start training';
    $('#d-dot').classList.toggle('live', on);
  }

  function explain(html) { $('#explain').innerHTML = html; }
  function setLoopPhase(p) {
    $$('#loop .phase').forEach((e) => e.classList.toggle('on', +e.dataset.p === p));
    $('#explain').style.borderLeftColor = p === 3 ? C.orange : C.blue;
  }

  function stepPhase() {
    if (S.running) setRunning(false);
    S.phase = (S.phase + 1) % 5;
    const B = S.batch, T = S.cfg.ctx, V = tok.size, base = Math.log(V);
    if (S.phase === 0) {
      S.pending = { batch: getBatch() };
      S.lastEx = S.pending.batch[B - 1]; S.lastProbs = null;
      renderBatch();
      explain(`<b>1 · Sample.</b> Picked ${B} random windows of ${T} characters. Each window holds ${T} next-character exercises, so this batch is <b>${B * T} predictions</b>. The last window is shown under "Current batch" below.`);
    } else if (S.phase === 1) {
      const r = model.computeGrads(S.pending.batch, { record: true });
      S.pending.r = r; S.lastProbs = r.probs;
      animateArch('fwd');
      renderBatch();
      explain(`<b>2 · Forward.</b> Each window flows up the stack: embedding lookup → ${S.cfg.layers} transformer block${S.cfg.layers > 1 ? 's' : ''} (attention + MLP) → output layer. The result is a <span class="mono">${T} × ${V}</span> grid of scores (logits): one score per vocabulary character, at every position.`);
    } else if (S.phase === 2) {
      const r = S.pending.r, ex = S.lastEx, p = r.probs[(0) * V + ex.y[0]];
      explain(`<b>3 · Loss.</b> Softmax turns scores into probabilities. Example: after "<span class="mono">${showChar(tok.chars[ex.x[0]])}</span>" the right answer is "<span class="mono">${showChar(tok.chars[ex.y[0]])}</span>"; the model gave it p = ${p.toFixed(3)}, so its loss is −ln(${p.toFixed(3)}) = <b>${(-Math.log(p)).toFixed(3)}</b>. Averaged over the batch: <b>${r.loss.toFixed(3)}</b> (random guessing would be ${base.toFixed(3)}).`);
    } else if (S.phase === 3) {
      const r = S.pending.r;
      S.lastGnorm = r.gnorm; S.gradNorm = r.gradNorm;
      animateArch('bwd');
      renderGrads(); renderParamMatrix();
      const top = Object.entries(r.gnorm).sort((a, b) => b[1] - a[1])[0];
      explain(`<b>4 · Backward.</b> Backpropagation applies the chain rule from the loss down through every operation, producing a gradient for all ${fmt(model.numParams())} parameters: which direction to move each one to reduce the loss. Overall gradient norm: <b>${r.gradNorm.toFixed(3)}</b>. Largest push this step: <span class="mono">${top[0]}</span>.`);
    } else if (S.phase === 4) {
      const r = S.pending.r, before = model.params[S.pmSel].data.slice();
      model.applyUpdate(S.lr, r.gradNorm);
      const now = model.params[S.pmSel].data, d = new Float32Array(now.length);
      for (let i = 0; i < d.length; i++) d[i] = now[i] - before[i];
      S.lastDelta = { name: S.pmSel, d };
      recordStep(r);
      S.pending = null;
      renderAll();
      explain(`<b>5 · Update.</b> Adam moved every weight a small step against its gradient (learning rate ${S.lr.toExponential(1)}), adapting the step size per weight from its gradient history. The largest change to <span class="mono">${S.pmSel}</span> was ${maxAbs(d).toExponential(2)}. That's one iteration; step ${S.step} done. Press again to start the next one.`);
    }
    setLoopPhase(S.phase);
  }

  function addSample() {
    const prompt = $('#gen-prompt').value || 'the ';
    const ids = tok.encode(prompt);
    if (!ids.length) return;
    const out = model.sample(ids, 70, +$('#gen-temp').value);
    S.samples.unshift({ step: S.step, prompt: tok.decode(ids), text: tok.decode(out.slice(ids.length)) });
    if (S.samples.length > 12) S.samples.pop();
    renderSamples();
  }

  // ---------------- render: static-ish ----------------
  function renderStatic() {
    const n = model.numParams();
    $('#h-params').textContent = n.toLocaleString();
    $('#h-layers').textContent = `${S.cfg.layers} × ${S.cfg.heads}`;
    $('#h-vocab').textContent = tok.size;
    $('#h-ctx').textContent = S.cfg.ctx;
    $('#corpus-info').textContent = `${data.length.toLocaleString()} characters`;
    $('#vocab-count').textContent = `· ${tok.size} tokens`;
    $('#ex-len').textContent = S.cfg.ctx;
    $('#k-base').textContent = Math.log(tok.size).toFixed(3);
    const vb = $('#vocab'); vb.innerHTML = '';
    tok.chars.forEach((c, i) => vb.appendChild(el('span', { class: 'tk' }, `${showChar(c)}<small>${i}</small>`)));
    // selects
    const fill = (sel, n, cur, lab) => { sel.innerHTML = ''; for (let i = 0; i < n; i++) { const o = el('option', { value: i }, lab(i)); if (i === cur) o.selected = true; sel.appendChild(o); } };
    fill($('#attn-layer'), S.cfg.layers, S.attnLayer, (i) => 'Layer ' + (i + 1));
    fill($('#attn-head'), S.cfg.heads, S.attnHead, (i) => 'Head ' + (i + 1));
    fill($('#nn-layer'), S.cfg.layers, S.nnLayer, (i) => 'Layer ' + (i + 1));
    const pm = $('#pm-sel'); pm.innerHTML = '';
    Object.keys(model.params).forEach((k) => { const o = el('option', { value: k }, k); if (k === S.pmSel) o.selected = true; pm.appendChild(o); });
    renderParamStack();
    buildArch();
    renderGradLegend();
  }

  function renderParamStack() {
    const tot = {}; let all = 0;
    for (const k in model.params) { const g = model.meta[k].group, n = model.params[k].data.length; tot[g] = (tot[g] || 0) + n; all += n; }
    const st = $('#pstack'), lg = $('#plegend'); st.innerHTML = ''; lg.innerHTML = '';
    Object.keys(GROUPS).forEach((g) => {
      if (!tot[g]) return;
      const s = el('span'); s.style.width = (tot[g] / all * 100) + '%'; s.style.background = GROUPS[g].color;
      s.addEventListener('mousemove', (ev) => tip(`${GROUPS[g].label}<br><b>${tot[g].toLocaleString()}</b> params · ${(tot[g] / all * 100).toFixed(1)}%`, ev.clientX, ev.clientY));
      s.addEventListener('mouseleave', untip);
      st.appendChild(s);
      lg.appendChild(el('span', null, `<i style="background:${GROUPS[g].color}"></i>${GROUPS[g].label} <b class="mono">${tot[g].toLocaleString()}</b> (${(tot[g] / all * 100).toFixed(0)}%)`));
    });
    lg.appendChild(el('span', { class: 'mono' }, `total ${all.toLocaleString()}`));
  }

  // ---------------- 01 data ----------------
  function renderPairs(container, ex, probsFlat, pList) {
    const n = ex.x.length, V = tok.size;
    container.style.gridTemplateColumns = `auto repeat(${n}, minmax(30px, 1fr))`;
    let h = '<div class="lbl">input</div>';
    for (let t = 0; t < n; t++) h += `<div class="tk">${showChar(tok.chars[ex.x[t]])}<small>${ex.x[t]}</small></div>`;
    h += '<div class="lbl"></div>';
    for (let t = 0; t < n; t++) h += '<div class="arrow">↓</div>';
    h += '<div class="lbl">target</div>';
    for (let t = 0; t < n; t++) {
      const p = pList ? pList[t] : probsFlat ? probsFlat[t * V + ex.y[t]] : null;
      const bar = p == null ? '' : `<div class="prob-bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div>`;
      h += `<div class="tk" title="${p == null ? '' : 'p(correct) = ' + p.toFixed(3)}">${showChar(tok.chars[ex.y[t]])}<small>${ex.y[t]}</small>${bar}</div>`;
    }
    container.innerHTML = h;
  }

  function renderData() {
    const live = S.running && S.lastEx;
    const ex = live ? S.lastEx : S.example;
    $('#ex-src').textContent = live ? 'live: last sequence of the current training batch' : 'random window from the corpus';
    const { p, trace } = targetProbs(ex);
    renderPairs($('#ex-grid'), ex, null, p);
    const n = ex.x.length, d = S.cfg.dModel;
    const mw = Math.max(160, Math.min(300, ($('#emb-row').clientWidth - 90) / 3));
    const tipFn = (name) => (r, c) => `${name} · pos ${r} "${showChar(tok.chars[ex.x[r]])}" · dim ${c}`;
    heat($('#emb-tok'), trace.tok, n, d, { maxW: mw, tip: tipFn('token emb') });
    heat($('#emb-pos'), trace.pos, n, d, { maxW: mw, tip: tipFn('position emb') });
    heat($('#emb-sum'), trace.x0, n, d, { maxW: mw, tip: tipFn('input x₀') });
  }

  // ---------------- 02 architecture ----------------
  const ARCH = [
    { id: 'loss', y: 30, label: 'Cross-entropy loss', group: null },
    { id: 'softmax', y: 80, label: 'Softmax', group: null },
    { id: 'head', y: 130, label: 'Linear (unembedding)', group: 'head' },
    { id: 'lnf', y: 180, label: 'Final LayerNorm', group: 'norm' },
    { id: 'add2', y: 250, label: '+', circle: true },
    { id: 'mlp', y: 295, label: 'MLP (feed-forward)', group: 'mlp' },
    { id: 'ln2', y: 345, label: 'LayerNorm', group: 'norm' },
    { id: 'add1', y: 390, label: '+', circle: true },
    { id: 'attn', y: 435, label: 'Multi-head attention', group: 'attn' },
    { id: 'ln1', y: 485, label: 'LayerNorm', group: 'norm' },
    { id: 'embadd', y: 555, label: '+', circle: true },
    { id: 'tok_emb', y: 605, label: 'Token emb.', group: 'embed', x: 190, w: 110 },
    { id: 'pos_emb', y: 605, label: 'Position emb.', group: 'embed', x: 310, w: 110 },
    { id: 'tokens', y: 660, label: 'Input token IDs', group: null },
  ];
  const ARCH_FWD = ['tokens', 'tok_emb', 'pos_emb', 'embadd', 'ln1', 'attn', 'add1', 'ln2', 'mlp', 'add2', 'lnf', 'head', 'softmax', 'loss'];
  const ARCH_PARAMS = {
    tok_emb: /^tok_emb$/, pos_emb: /^pos_emb$/, ln1: /^L\d+\.ln1\./, attn: /^L\d+\.attn\./, ln2: /^L\d+\.ln2\./,
    mlp: /^L\d+\.mlp\./, lnf: /^lnf\./, head: /^head\./,
  };
  function archInfo(id) {
    const { ctx: T, dModel: d, heads: H, dFF: f, layers: L } = S.cfg, V = tok.size, hd = d / H;
    const I = {
      tokens: ['The input: a sequence of up to T token IDs. During training, each sequence is one window from the corpus.', [['output', `[${T}] integers in 0…${V - 1}`]]],
      tok_emb: ['Looks up one learned vector per token ID: row <i>id</i> of a V × d table. Similar characters end up with similar vectors as training goes on.', [['input', `[${T}] IDs`], ['output', `[${T} × ${d}]`]]],
      pos_emb: ['Attention by itself ignores order, so a learned vector for each position (0, 1, 2…) is added. This is how the model knows which character came first.', [['input', `positions 0…${T - 1}`], ['output', `[${T} × ${d}]`]]],
      embadd: ['Token and position vectors are added element-wise. The result is the <b>residual stream</b>: a T × d matrix that every block reads from and adds its output back into.', [['shape', `[${T} × ${d}]`]]],
      ln1: [`LayerNorm rescales each token's vector to mean 0 and variance 1, then applies a learned scale (gain) and shift (bias). It keeps values in a stable range so a deep stack trains reliably. Like GPT-2, this model normalizes <i>before</i> each sublayer (pre-norm). This box repeats in all ${L} blocks.`, [['input', `[${T} × ${d}]`], ['output', `[${T} × ${d}]`]]],
      attn: [`The only place tokens exchange information. Each token is projected into query, key and value vectors (Wq, Wk, Wv), split into ${H} head${H > 1 ? 's' : ''} of size ${hd}. Each head computes softmax(QKᵀ/√${hd}) with the causal mask and takes a weighted mix of values; Wo merges the heads. See section 03.`, [['input', `[${T} × ${d}]`], ['Q, K, V', `[${T} × ${d}] each → ${H} × [${T} × ${hd}]`], ['attention weights', `${H} × [${T} × ${T}]`], ['output', `[${T} × ${d}]`]]],
      add1: ['Residual connection: the attention output is <i>added</i> to its input instead of replacing it. Gradients can flow straight down this shortcut, which is a big reason deep transformers are trainable.', [['shape', `[${T} × ${d}]`]]],
      ln2: ['A second LayerNorm, this time before the MLP.', [['input', `[${T} × ${d}]`], ['output', `[${T} × ${d}]`]]],
      mlp: [`A two-layer neural network applied to each position on its own: expand ${d} → ${f}, ReLU, project ${f} → ${d}. Most of a large model's parameters live here; it's often described as where learned features are stored. See section 04.`, [['input', `[${T} × ${d}]`], ['hidden', `[${T} × ${f}]`], ['output', `[${T} × ${d}]`]]],
      add2: ['Second residual connection: the MLP output is added back into the stream. The stream then enters the next block (or the output layer after the last one).', [['shape', `[${T} × ${d}]`]]],
      lnf: ['One final normalization of the residual stream before reading out predictions.', [['shape', `[${T} × ${d}]`]]],
      head: [`Multiplies each final vector by a ${d} × ${V} matrix, producing one score (logit) per vocabulary token at every position.`, [['input', `[${T} × ${d}]`], ['output (logits)', `[${T} × ${V}]`]]],
      softmax: ['Turns each row of logits into probabilities that sum to 1: pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ). No parameters.', [['shape', `[${T} × ${V}]`]]],
      loss: ['Compares each predicted distribution with the true next token: loss = −log p(target), averaged over positions and the batch. Backpropagation starts here and flows down through every box.', [['input', `[${T} × ${V}] probs + [${T}] targets`], ['output', 'one number']]],
    };
    return I[id];
  }

  function buildArch() {
    const svg = $('#arch-svg'); svg.innerHTML = '';
    svg.setAttribute('viewBox', '0 0 460 690');
    const g = (a) => svgEl('g', a || {});
    // edges
    const ys = ARCH.filter((n) => !n.x).map((n) => n.y);
    for (let i = 0; i < ys.length - 1; i++) svg.appendChild(svgEl('line', { x1: 250, y1: ys[i], x2: 250, y2: ys[i + 1], class: 'aedge' }));
    svg.appendChild(svgEl('path', { d: 'M190 605 L190 580 L250 568 M310 605 L310 580 L250 568', class: 'aedge' }));
    // block frame
    svg.appendChild(svgEl('rect', { x: 70, y: 222, width: 360, height: 296, rx: 10, class: 'block-frame' }));
    const ft = svgEl('text', { x: 418, y: 240, 'text-anchor': 'end', fill: C.muted, 'font-size': 12, 'font-family': 'JetBrains Mono, monospace' });
    ft.textContent = `transformer block × ${S.cfg.layers}`; svg.appendChild(ft);
    // residual skips
    svg.appendChild(svgEl('path', { d: 'M250 512 L105 512 L105 390 L237 390', class: 'aresid' }));
    svg.appendChild(svgEl('path', { d: 'M250 368 L105 368 L105 250 L237 250', class: 'aresid' }));
    const rt = svgEl('text', { x: 98, y: 450, fill: C.muted, 'font-size': 10, 'font-family': 'JetBrains Mono, monospace', transform: 'rotate(-90 98 450)', 'text-anchor': 'middle' });
    rt.textContent = 'residual'; svg.appendChild(rt);

    const counts = {};
    for (const k in model.params) for (const id in ARCH_PARAMS) if (ARCH_PARAMS[id].test(k)) counts[id] = (counts[id] || 0) + model.params[k].data.length;

    ARCH.forEach((n) => {
      const a = g({ class: 'anode' + (S.archSel === n.id ? ' sel' : ''), 'data-id': n.id });
      if (n.circle) {
        a.appendChild(svgEl('circle', { cx: 250, cy: n.y, r: 13, fill: C.surface2, stroke: C.line2 }));
        const t = svgEl('text', { x: 250, y: n.y + 5, 'text-anchor': 'middle' }); t.textContent = '+'; a.appendChild(t);
        // reuse rect styling for highlight
        a.querySelector('circle').setAttribute('class', 'circ');
      } else {
        const w = n.w || 240, cx = n.x || 250, x = cx - w / 2;
        a.appendChild(svgEl('rect', { x, y: n.y - 17, width: w, height: 34, rx: 7, class: 'box' }));
        if (n.group) a.appendChild(svgEl('rect', { x: x + 8, y: n.y - 5, width: 10, height: 10, rx: 3, fill: GROUPS[n.group].color, class: 'swatch' }));
        const t = svgEl('text', { x: n.group ? x + 26 : cx, y: n.y + 4.5, 'text-anchor': n.group ? 'start' : 'middle' });
        t.textContent = n.label; a.appendChild(t);
        if (counts[n.id] && !n.x) {
          const c = svgEl('text', { x: x + w - 10, y: n.y + 4, 'text-anchor': 'end', class: 'dim' });
          c.textContent = fmt(counts[n.id]); a.appendChild(c);
        }
      }
      a.addEventListener('click', () => { S.archSel = n.id; S.archParam = null; $$('.anode').forEach((e) => e.classList.toggle('sel', e.dataset.id === n.id)); renderArchDetail(); });
      svg.appendChild(a);
    });
    // highlight style for circles
    const st = svgEl('style', {}); st.textContent = '.anode .circ{transition:fill .15s}.anode.fwd .circ{fill:rgba(57,135,229,.5);stroke:#3987e5}.anode.bwd .circ{fill:rgba(217,89,38,.5);stroke:#d95926}.anode.sel .circ{stroke:#f0efec;stroke-width:2}';
    svg.appendChild(st);
    renderArchDetail();
  }

  function animateArch(dir, done) {
    const order = dir === 'fwd' ? ARCH_FWD : ARCH_FWD.slice().reverse(), cls = dir;
    order.forEach((id, i) => {
      setTimeout(() => {
        const e = document.querySelector(`.anode[data-id="${id}"]`); if (!e) return;
        e.classList.add(cls); setTimeout(() => e.classList.remove(cls), 260);
        if (i === order.length - 1 && done) setTimeout(done, 120);
      }, i * 42);
    });
  }

  function renderArchDetail() {
    const id = S.archSel, n = ARCH.find((a) => a.id === id), [desc, shapes] = archInfo(id);
    const names = ARCH_PARAMS[id] ? Object.keys(model.params).filter((k) => ARCH_PARAMS[id].test(k)) : [];
    if (names.length && (!S.archParam || !names.includes(S.archParam))) S.archParam = names.find((k) => model.params[k].rows > 1) || names[0];
    let h = `<h3>${n.group ? `<i style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${GROUPS[n.group].color}"></i>` : ''}${n.circle ? 'Residual add' : n.label}</h3><p>${desc}</p>`;
    h += '<dl class="detail-shape">' + shapes.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('') + '</dl>';
    if (names.length) {
      const tot = names.reduce((s, k) => s + model.params[k].data.length, 0);
      h += `<table class="ptab"><tr><th>parameter</th><th>shape</th><th style="text-align:right">count</th></tr>`;
      names.forEach((k) => { const p = model.params[k]; h += `<tr class="click${k === S.archParam ? ' on' : ''}" data-k="${k}"><td class="mono">${k}</td><td class="mono">${p.rows} × ${p.cols}</td><td class="n">${p.data.length.toLocaleString()}</td></tr>`; });
      h += `<tr><td colspan="2" style="color:var(--muted)">total in this component</td><td class="n"><b>${tot.toLocaleString()}</b></td></tr></table>`;
      h += `<div style="margin-top:14px" class="heat-box"><canvas class="heat" id="arch-heat"></canvas><div class="cap"><b class="mono">${S.archParam}</b> · ${model.meta[S.archParam].desc}. Click a row above to switch; it updates live during training.</div></div>`;
    } else h += '<p class="note">This step has no learnable parameters.</p>';
    const box = $('#arch-detail'); box.innerHTML = h;
    box.querySelectorAll('tr.click').forEach((tr) => tr.addEventListener('click', () => { S.archParam = tr.dataset.k; renderArchDetail(); }));
    renderArchHeat();
  }
  function renderArchHeat() {
    const cv = $('#arch-heat'); if (!cv || !S.archParam) return;
    const p = model.params[S.archParam], w = Math.min(420, Math.max(200, $('#arch-detail').clientWidth - 40));
    heat(cv, p.data, p.rows, p.cols, { maxW: w, cell: Math.min(12, Math.max(3, Math.floor(w / p.cols))), minH: p.rows === 1 ? 14 : 0, name: S.archParam, tip: (r, c) => S.archParam === 'tok_emb' ? `token "${showChar(tok.chars[r])}" · dim ${c}` : `${S.archParam}[${r}, ${c}]` });
  }

  // ---------------- 03 attention ----------------
  function attnData() {
    let ids = tok.encode($('#attn-text').value.toLowerCase());
    if (!ids.length) ids = tok.encode(S.corpus.slice(0, 8));
    ids = ids.slice(0, S.cfg.ctx);
    const { logits, trace } = model.forward(ids, true);
    return { ids, logits, trace };
  }
  function renderAttention() {
    const { ids, logits, trace } = attnData(), n = ids.length, V = tok.size;
    if (S.attnQuery < 0 || S.attnQuery >= n) S.attnQuery = n - 1;
    const q = S.attnQuery, A = trace.layers[S.attnLayer].attn[S.attnHead];
    // node-link graph
    const svg = $('#attn-svg'), W = 640, H = 250, top = 34, bot = 214;
    const sp = Math.min(46, (W - 40) / n), x0 = (W - sp * (n - 1)) / 2;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = '';
    const lab = (t, y, txt) => { const e = svgEl('text', { x: 8, y, fill: C.muted, 'font-size': 11 }); e.textContent = txt; e.style.font = '11px Inter, sans-serif'; e.style.fill = C.muted; svg.appendChild(e); };
    for (let j = 0; j <= q; j++) {
      const w = A[q * n + j];
      svg.appendChild(svgEl('line', { x1: x0 + q * sp, y1: top + 16, x2: x0 + j * sp, y2: bot - 16, stroke: C.orange, 'stroke-width': (0.6 + 9 * w).toFixed(2), 'stroke-opacity': (0.12 + 0.88 * w).toFixed(3), 'stroke-linecap': 'round' }));
    }
    const box = (i, y, isQ) => {
      const x = x0 + i * sp, on = isQ && i === q, masked = !isQ && i > q;
      svg.appendChild(svgEl('rect', { x: x - 15, y: y - 15, width: 30, height: 30, rx: 6, class: 'tok-bg' + (on ? ' on' : ''), opacity: masked ? 0.3 : 1 }));
      const t = svgEl('text', { x, y: y + 5, 'text-anchor': 'middle', opacity: masked ? 0.3 : 1 }); t.textContent = showChar(tok.chars[ids[i]]); svg.appendChild(t);
      if (!isQ && !masked) {
        const p = svgEl('text', { x, y: y + 32, 'text-anchor': 'middle' }); p.textContent = Math.round(A[q * n + i] * 100) + '%';
        p.style.font = '10px JetBrains Mono, monospace'; p.style.fill = C.muted; svg.appendChild(p);
      }
      if (isQ) {
        const hit = svgEl('rect', { x: x - sp / 2, y: y - 18, width: sp, height: 36, class: 'tok-hit' });
        hit.addEventListener('click', () => { S.attnQuery = i; renderAttention(); });
        svg.appendChild(hit);
      }
    };
    for (let i = 0; i < n; i++) { box(i, top, true); box(i, bot, false); }
    // heatmap with labels
    renderAttnHeat(A, n, ids, q);
    // predictions after token q
    const row = logits.data.subarray(q * V, q * V + V); let mx = -Infinity, s = 0;
    for (let j = 0; j < V; j++) mx = Math.max(mx, row[j]);
    const probs = Array.from(row, (z) => Math.exp(z - mx)); probs.forEach((v) => (s += v));
    const top8 = probs.map((p, j) => [j, p / s]).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const truth = q + 1 < n ? ids[q + 1] : null;
    $('#attn-pred').innerHTML = top8.map(([j, p]) => `<div class="pred"><span class="tk" style="padding:1px 4px${j === truth ? ';border-color:var(--aqua)' : ''}">${showChar(tok.chars[j])}</span><div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div><span class="p">${(p * 100).toFixed(1)}%</span></div>`).join('') +
      (truth != null ? `<div class="note" style="margin-top:4px">Actual next character in your text: <span class="mono">"${showChar(tok.chars[truth])}"</span> (outlined green if in the top 8)</div>` : '');
    // minis
    const mg = $('#attn-minis'); mg.style.gridTemplateColumns = `repeat(${Math.min(8, S.cfg.heads * S.cfg.layers)}, 92px)`;
    mg.innerHTML = '';
    trace.layers.forEach((L, l) => L.attn.forEach((M, h) => {
      const d = el('div', { class: 'mini' + (l === S.attnLayer && h === S.attnHead ? ' on' : '') });
      const cv = el('canvas'); d.appendChild(cv); d.appendChild(document.createTextNode(`L${l + 1} · H${h + 1}`));
      heat(cv, M, n, n, { seq: true, cell: Math.max(2, Math.floor(80 / n)), mask: (r, c) => c > r, name: 'attn' });
      d.addEventListener('click', () => { S.attnLayer = l; S.attnHead = h; $('#attn-layer').value = l; $('#attn-head').value = h; renderAttention(); });
      mg.appendChild(d);
    }));
  }
  function renderAttnHeat(A, n, ids, q) {
    const cv = $('#attn-heat'), cell = Math.min(26, Math.floor(Math.min(360, $('#attn-heat').parentElement.clientWidth - 60) / n)), pad = 22;
    const W = pad + n * cell + 4, H = pad + n * cell + 4, ctx = fitCanvas(cv, W, H);
    cv.style.imageRendering = 'auto';
    ctx.font = '11px JetBrains Mono, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = C.muted;
      ctx.fillText(showChar(tok.chars[ids[i]]), pad + i * cell + cell / 2, pad / 2);
      ctx.fillText(showChar(tok.chars[ids[i]]), pad / 2, pad + i * cell + cell / 2);
      for (let j = 0; j < n; j++) {
        if (j > i) { ctx.fillStyle = C.surface; } else { const c = seqRGB(A[i * n + j]); ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; }
        ctx.fillRect(pad + j * cell + 1, pad + i * cell + 1, cell - 2, cell - 2);
      }
    }
    ctx.strokeStyle = C.text; ctx.lineWidth = 1.5; ctx.strokeRect(pad + 0.5, pad + q * cell + 0.5, n * cell - 1, cell - 1);
    cv._attn = { A, n, ids, cell, pad };
    if (!cv._hooked) {
      cv._hooked = true;
      cv.addEventListener('mousemove', (ev) => {
        const a = cv._attn, r = cv.getBoundingClientRect();
        const i = Math.floor((ev.clientY - r.top - a.pad) / a.cell), j = Math.floor((ev.clientX - r.left - a.pad) / a.cell);
        if (i < 0 || j < 0 || i >= a.n || j >= a.n) return untip();
        if (j > i) return tip('<span class="muted">masked: a token cannot see the future</span>', ev.clientX, ev.clientY);
        tip(`"${showChar(tok.chars[a.ids[i]])}" (pos ${i}) attends to "${showChar(tok.chars[a.ids[j]])}" (pos ${j})<br><b>${(a.A[i * a.n + j] * 100).toFixed(1)}%</b>`, ev.clientX, ev.clientY);
      });
      cv.addEventListener('mouseleave', untip);
      cv.addEventListener('click', (ev) => { const a = cv._attn, r = cv.getBoundingClientRect(); const i = Math.floor((ev.clientY - r.top - a.pad) / a.cell); if (i >= 0 && i < a.n) { S.attnQuery = i; renderAttention(); } });
    }
  }

  // ---------------- 04 neurons ----------------
  let nnGeom = null;
  function renderNeurons() {
    const { ids, trace } = attnData(), n = ids.length, d = S.cfg.dModel, f = S.cfg.dFF, l = S.nnLayer;
    if (S.nnTok < 0 || S.nnTok >= n) S.nnTok = n - 1;
    const p = S.nnTok, L = trace.layers[l], b = model.blocks[l];
    // token chips
    const tc = $('#nn-tokens'); tc.innerHTML = '';
    ids.forEach((id, i) => { const s = el('span', { class: 'tk click' + (i === p ? ' sel' : '') }, showChar(tok.chars[id])); s.addEventListener('click', () => { S.nnTok = i; renderNeurons(); }); tc.appendChild(s); });
    const a0 = L.mlpIn.subarray(p * d, p * d + d), a1 = L.mlpHidden.subarray(p * f, p * f + f), a2 = L.mlpOut.subarray(p * d, p * d + d);
    const cv = $('#nn-canvas'), W = cv.parentElement.clientWidth - 36, H = Math.max(420, Math.min(640, f * 8 + 90));
    const ctx = fitCanvas(cv, W, H), top = 52, bottom = 16;
    const cols = [{ x: W * 0.14, n: d, a: a0 }, { x: W * 0.5, n: f, a: a1 }, { x: W * 0.86, n: d, a: a2 }];
    cols.forEach((c) => { c.sp = (H - top - bottom) / c.n; c.y = (i) => top + (i + 0.5) * c.sp; c.r = clamp(c.sp * 0.36, 1.6, 6); c.max = maxAbs(c.a); });
    nnGeom = { cols, W, H, a0, a1, a2 };
    const hv = S.nnHover; // {col, i}
    // edges
    const drawEdges = (src, dst, Wt, colIdx) => {
      const rows = src.n, cc = dst.n, list = [];
      for (let i = 0; i < rows; i++) for (let j = 0; j < cc; j++) {
        const w = Wt.data[i * cc + j], v = S.nnMode === 'w' ? w : w * src.a[i];
        const conn = hv ? ((hv.col === colIdx && hv.i === i) || (hv.col === colIdx + 1 && hv.i === j)) : true;
        if (!conn) continue;
        list.push([i, j, v]);
      }
      list.sort((A, B) => Math.abs(B[2]) - Math.abs(A[2]));
      const keep = hv ? list.length : Math.max(1, Math.round(list.length * S.nnDens / 100));
      const mx = Math.abs(list[0] ? list[0][2] : 1) || 1;
      for (let k = keep - 1; k >= 0; k--) {
        const [i, j, v] = list[k], s = Math.abs(v) / mx;
        ctx.strokeStyle = v >= 0 ? `rgba(57,135,229,${0.06 + 0.8 * s})` : `rgba(217,89,38,${0.06 + 0.8 * s})`;
        ctx.lineWidth = hv ? 0.6 + 2 * s : 0.5 + 1.4 * s;
        ctx.beginPath(); ctx.moveTo(src.x, src.y(i)); ctx.lineTo(dst.x, dst.y(j)); ctx.stroke();
      }
    };
    drawEdges(cols[0], cols[1], b.W1, 0);
    drawEdges(cols[1], cols[2], b.W2, 1);
    // nodes
    cols.forEach((c, ci) => {
      for (let i = 0; i < c.n; i++) {
        const v = c.a[i], s = Math.min(1, Math.abs(v) / c.max);
        ctx.beginPath(); ctx.arc(c.x, c.y(i), c.r, 0, Math.PI * 2);
        if (ci === 1) ctx.fillStyle = v > 0 ? `rgba(240,239,236,${0.15 + 0.85 * s})` : C.surface;
        else ctx.fillStyle = v >= 0 ? `rgba(57,135,229,${0.15 + 0.85 * s})` : `rgba(217,89,38,${0.15 + 0.85 * s})`;
        ctx.fill(); ctx.lineWidth = hv && hv.col === ci && hv.i === i ? 2 : 1;
        ctx.strokeStyle = hv && hv.col === ci && hv.i === i ? C.text : C.line2; ctx.stroke();
      }
    });
    // titles
    ctx.fillStyle = C.text2; ctx.font = '600 12px Inter, sans-serif'; ctx.textAlign = 'center';
    const t1 = [['MLP input', `${d} values, after LayerNorm`], ['Hidden neurons', `${f}, after ReLU`], ['MLP output', `${d} values, added to the stream`]];
    cols.forEach((c, i) => { ctx.fillStyle = C.text; ctx.font = '600 12px Inter, sans-serif'; ctx.fillText(t1[i][0], c.x, 18); ctx.fillStyle = C.muted; ctx.font = '11px Inter, sans-serif'; ctx.fillText(t1[i][1], c.x, 34); });
    ctx.fillStyle = C.muted; ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText(`W1 · ${d}×${f} = ${(d * f).toLocaleString()} weights`, (cols[0].x + cols[1].x) / 2, H - 2);
    ctx.fillText(`W2 · ${f}×${d} = ${(d * f).toLocaleString()} weights`, (cols[1].x + cols[2].x) / 2, H - 2);
    const active = Array.from(a1).filter((v) => v > 0).length;
    $('#nn-info').innerHTML = `For token <span class="mono">"${showChar(tok.chars[ids[p]])}"</span> in layer ${l + 1}: <b>${active} of ${f}</b> hidden neurons are active (above 0); the rest were zeroed by ReLU. ${S.nnMode === 's' ? 'Signal mode shows weight × source activation: the paths information actually took for this token.' : 'Hover a neuron to isolate its connections.'}`;
  }
  function nnHit(ev) {
    if (!nnGeom) return null;
    const r = $('#nn-canvas').getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    for (let ci = 0; ci < 3; ci++) {
      const c = nnGeom.cols[ci]; if (Math.abs(x - c.x) > c.r + 6) continue;
      const i = Math.round((y - 52) / c.sp - 0.5); if (i >= 0 && i < c.n) return { col: ci, i };
    }
    return null;
  }

  // ---------------- 05 training ----------------
  function renderBatch() {
    if (!S.lastEx) { $('#batch-grid').innerHTML = '<p class="note">Start training or step through to see a batch.</p>'; return; }
    renderPairs($('#batch-grid'), S.lastEx, S.lastProbs, null);
  }
  function renderSamples() {
    $('#samples').innerHTML = S.samples.length ? S.samples.map((s) => `<div class="s"><span class="st">step ${s.step}</span><span class="pr">${esc(s.prompt)}</span>${esc(s.text)}</div>`).join('') : '<p class="note">No samples yet. One is written every 150 training steps, or press Generate now. An untrained model produces random characters.</p>';
  }
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function renderGradLegend() {
    $('#g-legend').innerHTML = Object.keys(GROUPS).map((g) => `<span><i style="background:${GROUPS[g].color}"></i>${GROUPS[g].label}</span>`).join('');
  }
  function renderGrads() {
    const box = $('#gbars');
    if (!S.lastGnorm) { box.innerHTML = '<p class="note">Gradients appear after the first backward pass.</p>'; return; }
    const ent = Object.entries(S.lastGnorm), mx = Math.max(...ent.map((e) => e[1])) || 1;
    box.innerHTML = ent.map(([k, v]) => `<div class="gbar" title="${model.meta[k].desc}"><span class="nm">${k}</span><span class="tr"><i style="width:${(v / mx * 100).toFixed(1)}%;background:${GROUPS[model.meta[k].group].color}"></i></span><span class="vv">${v.toFixed(3)}</span></div>`).join('');
  }
  function renderParamMatrix() {
    const k = S.pmSel, p = model.params[k], mw = Math.max(150, Math.min(260, ($('#pm-w').closest('.card').clientWidth - 60) / 3));
    const o = (nm) => ({ maxW: mw, minH: p.rows === 1 ? 14 : 0, tip: (r, c) => `${nm} ${k}[${r}, ${c}]` });
    heat($('#pm-w'), p.data, p.rows, p.cols, o('W'));
    $('#pm-w-s').textContent = `±${maxAbs(p.data).toFixed(3)}`;
    const g = p.grad || new Float32Array(p.data.length);
    heat($('#pm-g'), g, p.rows, p.cols, o('∂L/∂W'));
    $('#pm-g-s').textContent = p.grad ? `±${maxAbs(g).toExponential(1)}` : '(none yet)';
    const d = S.lastDelta && S.lastDelta.name === k ? S.lastDelta.d : new Float32Array(p.data.length);
    heat($('#pm-d'), d, p.rows, p.cols, o('ΔW'));
    $('#pm-d-s').textContent = S.lastDelta && S.lastDelta.name === k ? `±${maxAbs(d).toExponential(1)}` : '(after next step)';
    $('#pm-desc').innerHTML = `<span class="mono">${k}</span> · ${p.rows} × ${p.cols} = ${p.data.length.toLocaleString()} parameters. ${model.meta[k].desc}. The update is roughly the opposite sign of the gradient: blue gradient → orange update.`;
  }

  function renderLossChart() {
    const cv = $('#loss-canvas'), W = cv.parentElement.clientWidth - 36, H = 260, ctx = fitCanvas(cv, W, H);
    const pl = 44, pr = 12, pt = 10, pb = 26, iw = W - pl - pr, ih = H - pt - pb;
    const base = Math.log(tok.size), N = S.losses.length;
    const ymax = Math.max(base * 1.08, N ? Math.max(...S.losses.slice(0, 20)) * 1.02 : 0), ymin = 0;
    const xmax = Math.max(N, 50);
    const X = (i) => pl + (i / (xmax - 1 || 1)) * iw, Y = (v) => pt + ih - (v - ymin) / (ymax - ymin) * ih;
    // grid
    ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.fillStyle = C.muted; ctx.font = '11px JetBrains Mono, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const step = ymax > 3 ? 1 : 0.5;
    for (let v = 0; v <= ymax; v += step) { ctx.beginPath(); ctx.moveTo(pl, Y(v) + 0.5); ctx.lineTo(W - pr, Y(v) + 0.5); ctx.stroke(); ctx.fillText(v.toFixed(1), pl - 6, Y(v)); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const xt = niceStep(xmax / 5);
    for (let s = 0; s <= xmax; s += xt) ctx.fillText(s, X(s), H - pb + 6);
    // baseline
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.muted; ctx.beginPath(); ctx.moveTo(pl, Y(base)); ctx.lineTo(W - pr, Y(base)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(`random guessing = ln(${tok.size}) = ${base.toFixed(2)}`, W - pr, Y(base) - 3);
    if (!N) { ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('press Train to start', pl + iw / 2, pt + ih / 2 + 20); return; }
    const stride = Math.max(1, Math.floor(N / iw));
    const line = (arr, color, w) => { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.beginPath(); for (let i = 0; i < N; i += stride) { const y = Y(Math.min(arr[i], ymax)); i ? ctx.lineTo(X(i), y) : ctx.moveTo(X(i), y); } ctx.lineTo(X(N - 1), Y(Math.min(arr[N - 1], ymax))); ctx.stroke(); };
    line(S.losses, 'rgba(57,135,229,0.3)', 1);
    line(S.emas, C.blue, 2);
    // end label
    ctx.fillStyle = C.text; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.font = '600 12px JetBrains Mono, monospace';
    const ey = Y(S.emas[N - 1]);
    ctx.fillText(S.emas[N - 1].toFixed(3), Math.min(X(N - 1), W - pr), ey - 8);
    // hover
    if (S.lossHover != null) {
      const i = clamp(Math.round(S.lossHover), 0, N - 1), x = X(i);
      ctx.strokeStyle = C.text2; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x + 0.5, pt); ctx.lineTo(x + 0.5, pt + ih); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, Y(S.emas[i]), 4.5, 0, Math.PI * 2); ctx.fillStyle = C.blue; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = C.surface; ctx.stroke();
    }
    cv._lc = { X, pl, iw, xmax, N };
  }
  function niceStep(r) { const p = Math.pow(10, Math.floor(Math.log10(Math.max(r, 1)))); for (const m of [1, 2, 5, 10]) if (m * p >= r) return m * p; return 10 * p; }

  function renderSpark() {
    const cv = $('#d-spark'), ctx = fitCanvas(cv, 90, 28), a = S.emas.slice(-300);
    if (a.length < 2) return;
    const mx = Math.log(tok.size) * 1.05;
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.5; ctx.beginPath();
    a.forEach((v, i) => { const x = i / (a.length - 1) * 88 + 1, y = 26 - Math.min(v, mx) / mx * 24; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }

  function renderKPIs() {
    const N = S.emas.length;
    $('#k-step').textContent = S.step.toLocaleString();
    $('#k-loss').textContent = N ? S.emas[N - 1].toFixed(3) : '–';
    $('#k-tok').textContent = fmt(S.tokensSeen);
    $('#k-gn').textContent = S.gradNorm == null ? '–' : S.gradNorm.toFixed(2);
    $('#d-step').textContent = S.step.toLocaleString();
    $('#d-loss').textContent = N ? S.emas[N - 1].toFixed(3) : '–';
  }

  // ---------------- 06 scale ----------------
  const CV = [32, 256, 1000, 8000, 32000, 50257, 100000, 128000];
  const CT = [16, 64, 256, 512, 1024, 2048, 4096, 8192, 32768];
  const CD = [16, 32, 64, 128, 256, 512, 768, 1024, 1600, 2048, 4096, 5120, 8192, 12288, 16384];
  const PRESETS = [
    { name: 'This page (approx.)', v: null },
    { name: 'GPT-2 Small', v: 50257, t: 1024, d: 768, l: 12 },
    { name: 'GPT-2 XL', v: 50257, t: 1024, d: 1600, l: 48 },
    { name: 'GPT-3', v: 50257, t: 2048, d: 12288, l: 96 },
  ];
  const nearest = (arr, v) => arr.reduce((bi, x, i) => Math.abs(x - v) < Math.abs(arr[bi] - v) ? i : bi, 0);
  function setPreset(p) {
    if (!p.v) { $('#c-v').value = nearest(CV, tok.size); $('#c-t').value = nearest(CT, S.cfg.ctx); $('#c-d').value = nearest(CD, S.cfg.dModel); $('#c-l').value = S.cfg.layers; }
    else { $('#c-v').value = nearest(CV, p.v); $('#c-t').value = nearest(CT, p.t); $('#c-d').value = nearest(CD, p.d); $('#c-l').value = p.l; }
    renderScale();
  }
  function renderScale() {
    const V = CV[+$('#c-v').value], T = CT[+$('#c-t').value], d = CD[+$('#c-d').value], L = +$('#c-l').value;
    $('#c-v-v').textContent = V.toLocaleString(); $('#c-t-v').textContent = T.toLocaleString(); $('#c-d-v').textContent = d.toLocaleString(); $('#c-l-v').textContent = L;
    const parts = { embed: V * d + T * d, attn: L * (4 * d * d + 4 * d), mlp: L * (8 * d * d + 5 * d), norm: L * 4 * d + 2 * d };
    const tot = Object.values(parts).reduce((a, b) => a + b, 0);
    $('#c-total').textContent = fmt(tot);
    const st = $('#c-stack'), lg = $('#c-legend'); st.innerHTML = ''; lg.innerHTML = '';
    Object.keys(parts).forEach((g) => {
      const s = el('span'); s.style.width = (parts[g] / tot * 100) + '%'; s.style.background = GROUPS[g].color; st.appendChild(s);
      s.addEventListener('mousemove', (ev) => tip(`${GROUPS[g].label}<br><b>${fmt(parts[g])}</b> · ${(parts[g] / tot * 100).toFixed(1)}%`, ev.clientX, ev.clientY));
      s.addEventListener('mouseleave', untip);
      lg.appendChild(el('span', null, `<i style="background:${GROUPS[g].color}"></i>${GROUPS[g].label} <b class="mono">${fmt(parts[g])}</b> (${(parts[g] / tot * 100).toFixed(parts[g] / tot < 0.01 ? 2 : 0)}%)`));
    });
    const bytes = tot * 2;
    $('#c-mem').textContent = bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + ' GB' : (bytes / 1e6).toFixed(bytes >= 1e7 ? 0 : 2) + ' MB';
    const fl = 6 * tot; $('#c-flop').textContent = fl.toExponential(1) + ' FLOPs';
    $('#c-ratio').textContent = (tot / model.numParams()).toLocaleString(undefined, { maximumFractionDigits: tot / model.numParams() < 10 ? 1 : 0 }) + '×';
  }

  // ---------------- orchestration ----------------
  function renderLight() { renderKPIs(); renderLossChart(); renderSpark(); }
  function renderHeavy() {
    renderBatch(); renderGrads();
    if (S.visible.data) renderData();
    if (S.visible.arch) renderArchHeat();
    if (S.visible.attention) renderAttention();
    if (S.visible.neurons) renderNeurons();
    if (S.visible.training) renderParamMatrix();
  }
  function renderAll() {
    renderLight(); renderData(); renderArchDetail(); renderAttention(); renderNeurons();
    renderBatch(); renderGrads(); renderParamMatrix(); renderSamples(); renderScale();
  }

  // ---------------- events ----------------
  function bind() {
    $('#corpus').value = S.corpus;
    $('#corpus-apply').addEventListener('click', () => {
      const t = $('#corpus').value.toLowerCase();
      if (t.length < S.cfg.ctx * 4) { $('#corpus-info').textContent = 'Corpus too short: add more text.'; return; }
      $('#corpus').value = t; S.corpus = t; setRunning(false); build(false);
    });
    $('#ex-next').addEventListener('click', () => { newExample(); renderData(); });
    $('#cfg-apply').addEventListener('click', () => {
      S.cfg = { dModel: +$('#cfg-d').value, heads: +$('#cfg-h').value, layers: +$('#cfg-l').value, dFF: +$('#cfg-f').value, ctx: +$('#cfg-t').value };
      if (S.cfg.dModel % S.cfg.heads) S.cfg.heads = 1;
      setRunning(false); build(false);
    });
    $('#attn-text').addEventListener('input', () => { S.attnQuery = -1; S.nnTok = -1; renderAttention(); renderNeurons(); });
    $('#attn-layer').addEventListener('change', (e) => { S.attnLayer = +e.target.value; renderAttention(); });
    $('#attn-head').addEventListener('change', (e) => { S.attnHead = +e.target.value; renderAttention(); });
    $('#nn-layer').addEventListener('change', (e) => { S.nnLayer = +e.target.value; renderNeurons(); });
    $$('#nn-mode button').forEach((b) => b.addEventListener('click', () => { S.nnMode = b.dataset.m; $$('#nn-mode button').forEach((x) => x.classList.toggle('on', x === b)); renderNeurons(); }));
    $('#nn-dens').addEventListener('input', (e) => { S.nnDens = +e.target.value; $('#nn-dens-v').textContent = S.nnDens + '%'; renderNeurons(); });
    const nc = $('#nn-canvas');
    nc.addEventListener('mousemove', (ev) => {
      const h = nnHit(ev), same = (h && S.nnHover && h.col === S.nnHover.col && h.i === S.nnHover.i) || (!h && !S.nnHover);
      if (h) { const c = nnGeom.cols[h.col], nm = ['input dim', 'hidden neuron', 'output dim'][h.col]; tip(`${nm} ${h.i}<br>activation <b>${c.a[h.i].toFixed(3)}</b>`, ev.clientX, ev.clientY); } else untip();
      if (!same) { S.nnHover = h; renderNeurons(); }
    });
    nc.addEventListener('mouseleave', () => { untip(); if (S.nnHover) { S.nnHover = null; renderNeurons(); } });

    const toggle = () => setRunning(!S.running);
    $('#t-run').addEventListener('click', toggle);
    $('#d-run').addEventListener('click', toggle);
    $('#hero-train').addEventListener('click', () => { if (!S.running) { setRunning(true); document.getElementById('training').scrollIntoView(); } else setRunning(false); });
    $('#t-step').addEventListener('click', stepPhase);
    $('#t-reset').addEventListener('click', () => { setRunning(false); build(true); explain('Weights re-initialized with new random values. The model has forgotten everything.'); });
    const lr = () => { S.lr = Math.pow(10, +$('#lr').value); $('#lr-v').textContent = S.lr.toExponential(1); };
    const bs = () => { S.batch = +$('#bs').value; $('#bs-v').textContent = S.batch; };
    const sp = () => { S.speed = +$('#sp').value; $('#sp-v').textContent = S.speed; };
    $('#lr').addEventListener('input', lr); $('#bs').addEventListener('input', bs); $('#sp').addEventListener('input', sp); lr(); bs(); sp();
    $('#gen-temp').addEventListener('input', (e) => ($('#gen-temp-v').textContent = (+e.target.value).toFixed(2)));
    $('#gen-btn').addEventListener('click', addSample);
    $('#pm-sel').addEventListener('change', (e) => { S.pmSel = e.target.value; renderParamMatrix(); });

    const lc = $('#loss-canvas');
    lc.addEventListener('mousemove', (ev) => {
      const g = lc._lc; if (!g || !g.N) return;
      const r = lc.getBoundingClientRect(), i = Math.round((ev.clientX - r.left - g.pl) / g.iw * (g.xmax - 1));
      if (i < 0 || i >= g.N) { S.lossHover = null; untip(); renderLossChart(); return; }
      S.lossHover = i; renderLossChart();
      tip(`step ${i + 1}<br>batch loss <b>${S.losses[i].toFixed(3)}</b><br>smoothed <b>${S.emas[i].toFixed(3)}</b>`, ev.clientX, ev.clientY);
    });
    lc.addEventListener('mouseleave', () => { S.lossHover = null; untip(); renderLossChart(); });

    // scale
    const pr = $('#presets');
    PRESETS.forEach((p) => { const b = el('button', null, p.name); b.addEventListener('click', () => setPreset(p)); pr.appendChild(b); });
    ['#c-v', '#c-t', '#c-d', '#c-l'].forEach((s) => $(s).addEventListener('input', renderScale));

    // visibility + nav
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) $$('.nav a.link').forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
    }), { rootMargin: '-30% 0px -50% 0px' });
    const vis = new IntersectionObserver((es) => es.forEach((e) => (S.visible[e.target.id] = e.isIntersecting)), { rootMargin: '100px' });
    $$('section.sec').forEach((s) => { io.observe(s); vis.observe(s); });

    let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(renderAll, 150); });
    document.addEventListener('keydown', (e) => { if (e.code === 'Space' && !/INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement.tagName)) { e.preventDefault(); toggle(); } });
  }

  bind();
  build(false);
  setPreset(PRESETS[1]);
})();
