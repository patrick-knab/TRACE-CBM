/* TRACE-CBM intervention playground: a hand-sized TRACE rollout over the Figure 3 examples */
(function () {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const svgEl = window.TRACE.svgEl;
  const showTip = window.TRACE.showTip, hideTip = window.TRACE.hideTip;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const sig = z => 1 / (1 + Math.exp(-z));
  const logit = p => Math.log(p / (1 - p));
  const softmax = a => { const m = Math.max(...a); const e = a.map(v => Math.exp(v - m)); const s = e.reduce((x, y) => x + y, 0); return e.map(v => v / s); };
  const f2 = v => v.toFixed(2);

  let DATA = null, ex = null, state = null;

  /* ---------- model ---------- */
  function rollout(E, edits) {
    const C = E.concepts.length;
    let r = E.states_observed.map(row => row[2]);
    for (const j in edits.nodes) r[+j] = clamp(edits.nodes[j], 0.01, 0.99);
    const out = [];
    for (let h = 0; h < 4; h++) {
      const lg = r.map(v => logit(clamp(v, 1e-4, 1 - 1e-4)));
      let y = softmax(E.W.map((w, k) => w.reduce((s, wj, j) => s + wj * lg[j], 0) + E.b[k]));
      if (h === 0 && edits.act) {
        const idx = E.activities.indexOf(edits.act.name), v = edits.act.p;
        const rest = 1 - y[idx];
        y = y.map((p, k) => k === idx ? v : p * (1 - v) / rest);
      }
      out.push({ r: r.slice(), y });
      if (h === 3) break;
      const msg = new Array(C).fill(0);
      for (let j = 0; j < C; j++) {
        for (let s = 0; s < C; s++) {
          if (edits.edges[`same:${s}:${j}`] !== false) msg[j] += r[s] * E.A_same[s][j];
          if (edits.edges[`cross:${s}:${j}`] !== false) msg[j] += r[s] * E.A_cross[s][j];
        }
        if (edits.edges[`self:${j}`] !== false) msg[j] += r[j] * E.a_self[j];
      }
      const fb = new Array(C).fill(0);
      for (let k = 0; k < y.length; k++) for (let j = 0; j < C; j++) fb[j] += y[k] * E.A_act[k][j];
      r = r.map((v, j) => sig(clamp(logit(clamp(v, 1e-4, 1 - 1e-4)) + E.lam[j] * msg[j] + fb[j], -6, 6)));
    }
    return out;
  }
  const edgeList = (E) => {
    const L = [];
    const C = E.concepts.length;
    for (let j = 0; j < C; j++) if (Math.abs(E.a_self[j]) > 1e-6) L.push({ key: `self:${j}`, type: 'self', s: j, t: j, w: E.a_self[j], label: `${E.concepts[j]} → itself (t→t+1)` });
    for (let s = 0; s < C; s++) for (let t = 0; t < C; t++) {
      if (Math.abs(E.A_cross[s][t]) > 1e-6) L.push({ key: `cross:${s}:${t}`, type: 'cross', s, t, w: E.A_cross[s][t], label: `${E.concepts[s]} → ${E.concepts[t]} (t→t+1)` });
      if (Math.abs(E.A_same[s][t]) > 1e-6) L.push({ key: `same:${s}:${t}`, type: 'same', s, t, w: E.A_same[s][t], label: `${E.concepts[s]} → ${E.concepts[t]} (same window)` });
    }
    return L;
  };

  /* ---------- state ---------- */
  function resetState() {
    state = { nodes: {}, act: null, edges: {} };
  }
  const isEdited = () => Object.keys(state.nodes).length || state.act || Object.values(state.edges).some(v => v === false);

  /* ---------- rendering ---------- */
  function renderFrames() {
    const host = $('#pgFrames'); host.innerHTML = '';
    const labs = ['t-2', 't-1', 't', 't+1', 't+2', 't+3'];
    for (let i = 0; i < 6; i++) {
      const d = document.createElement('div');
      if (i < 3) {
        d.className = 'f';
        d.innerHTML = `<img src="assets/img/frames/${state.ex}_${i}.jpg?v=2" alt="${ex.title} window ${labs[i]}"><div class="gt">GT: ${ex.gt[i]}</div>`;
      } else {
        d.className = 'f future'; d.innerHTML = `<div>${labs[i]}<br><span class="small">GT: ${ex.gt[i]}</span></div>`;
      }
      host.appendChild(d);
    }
  }

  function renderSvg(base, cur) {
    const host = $('#pgSvg'); host.innerHTML = '';
    const C = ex.concepts.length, K = ex.activities.length;
    const W = 760;
    const left = 128, right = 16;
    const cols = Array.from({ length: 6 }, (_, i) => left + 48 + i * ((W - left - right - 96) / 5));
    const labs = ['t-2', 't-1', 't', 't+1', 't+2', 't+3'];
    // activity panel geometry
    const actTop = 26, actRows = 4, actRowH = 20, actH = actRows * actRowH + 26;
    const gridTop = actTop + actH + 30, rowH = 52;
    const H = gridTop + C * rowH + 24;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'TRACE rollout over three concepts' });
    svg.appendChild(svgEl('text', { x: 0, y: actTop - 10, class: 'hdr' }, 'activity distribution (top 4)'));
    svg.appendChild(svgEl('text', { x: 0, y: gridTop - 14, class: 'hdr' }, 'concept states & relations'));
    svg.appendChild(svgEl('line', { x1: 0, x2: W, y1: gridTop - 30, y2: gridTop - 30, class: 'grid' }));

    // ---- activity bars for t..t+3 (columns 2..5) ----
    const colW = cols[1] - cols[0];
    for (let h = 0; h < 4; h++) {
      const aw = W / 4, cx = aw * h + aw / 2, x0 = aw * h + 6, barMax = aw - 150;
      const yb = cur[h].y, y0 = base[h].y;
      let order = yb.map((p, k) => k).filter(k => ex.activities[k] !== 'Other').sort((a, b) => yb[b] - yb[a]).slice(0, actRows);
      const gtK = ex.activities.indexOf(ex.gt[h + 2]);
      if (gtK >= 0 && !order.includes(gtK)) order[order.length - 1] = gtK;
      const gt = ex.gt[h + 2];
      const predK = order[0];
      const g = svgEl('g');
      g.appendChild(svgEl('text', { x: cx, y: actTop + 2, class: 'col-lab' + (h ? ' fut' : ''), 'text-anchor': 'middle' }, labs[h + 2]));
      order.forEach((k, i) => {
        const yy = actTop + 10 + i * actRowH;
        const name = ex.activities[k];
        g.appendChild(svgEl('text', { x: x0, y: yy + 12, class: 'act-lab' }, name.length > 17 ? name.slice(0, 16) + '…' : name));
        const bx = x0 + 96;
        const w = Math.max(2, yb[k] * barMax), wb = y0[k] * barMax;
        g.appendChild(svgEl('rect', { x: bx, y: yy + 3, width: w, height: 12, rx: 2, class: 'act-bar' + (h === 0 ? ' obs' : '') + (k === predK ? ' pred' : ''), opacity: k === predK ? 1 : .55 }));
        if (isEdited() && Math.abs(y0[k] - yb[k]) > 0.005) g.appendChild(svgEl('rect', { x: bx, y: yy + 3, width: Math.max(2, wb), height: 12, rx: 2, class: 'act-ghost' }));
        g.appendChild(svgEl('text', { x: bx + w + 4, y: yy + 12, class: 'act-val' }, f2(yb[k]) + (name === gt && k !== predK ? ' GT' : '')));
        const hit = svgEl('rect', { x: x0 - 4, y: yy, width: aw - 8, height: actRowH, fill: 'transparent' });
        hit.addEventListener('mousemove', e => showTip(`<b>${name}</b> at ${labs[h + 2]}<br>now ${f2(yb[k])}${isEdited() ? ` · unedited ${f2(y0[k])}` : ''}${name === gt ? '<br>ground truth' : ''}`, e.clientX, e.clientY));
        hit.addEventListener('mouseleave', hideTip);
        g.appendChild(hit);
      });
      const ok = ex.activities[predK] === gt;
      g.appendChild(svgEl('text', { x: cx, y: actTop + 10 + actRows * actRowH + 10, class: 'badge ' + (ok ? 'ok' : 'bad'), 'text-anchor': 'middle' }, ok ? '✓ Pred = GT' : `✗ GT: ${gt.length > 18 ? gt.slice(0, 17) + '…' : gt}`));
      svg.appendChild(g);
    }

    // ---- concept grid ----
    const rowY = j => gridTop + 14 + j * rowH;
    const nodeW = 50, nodeH = 26;
    ex.concepts.forEach((c, j) => svg.appendChild(svgEl('text', { x: left - 10, y: rowY(j) + 4, class: 'row-lab', 'text-anchor': 'end' }, c)));
    cols.forEach((x, i) => svg.appendChild(svgEl('text', { x, y: H - 4, class: 'col-lab' + (i > 2 ? ' fut' : ''), 'text-anchor': 'middle' }, labs[i])));
    const vals = ex.concepts.map((_, j) => [...ex.states_observed[j], ...cur.slice(1).map(o => o.r[j])]);
    const baseVals = ex.concepts.map((_, j) => [...ex.states_observed[j], ...base.slice(1).map(o => o.r[j])]);
    // edges
    const defs = svgEl('defs');
    ['var(--text-2)', 'var(--gold)'].forEach((c, i) => {
      const m = svgEl('marker', { id: 'pgarr' + i, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto' });
      m.appendChild(svgEl('path', { d: 'M0 0 L10 5 L0 10 z', fill: c })); defs.appendChild(m);
    });
    svg.appendChild(defs);
    const eg = svgEl('g');
    const edges = edgeList(ex);
    const onEdge = e => state.edges[e.key] !== false;
    const addHit = (pathD, e) => {
      const hit = svgEl('path', { d: pathD, class: 'edge hit' });
      hit.addEventListener('click', () => { state.edges[e.key] = !onEdge(e); update(); });
      hit.addEventListener('mousemove', ev => showTip(`<b>${e.type}-${e.type === 'same' ? 'window' : 'temporal'} relation</b><br>${e.label}<br>weight ${e.w >= 0 ? '+' : ''}${e.w.toFixed(2)} · ${onEdge(e) ? 'active — click to cut' : 'cut — click to restore'}`, ev.clientX, ev.clientY));
      hit.addEventListener('mouseleave', hideTip);
      eg.appendChild(hit);
    };
    for (let i = 0; i < 5; i++) {
      edges.filter(e => e.type === 'self').forEach(e => {
        const d = `M${cols[i] + nodeW / 2} ${rowY(e.s)} H${cols[i + 1] - nodeW / 2 - 2}`;
        eg.appendChild(svgEl('path', { d, class: 'edge self' + (onEdge(e) ? '' : ' off'), 'marker-end': onEdge(e) ? 'url(#pgarr1)' : null, opacity: onEdge(e) ? clamp(0.35 + Math.abs(e.w), 0.35, 1) : null }));
        addHit(d, e);
      });
      edges.filter(e => e.type === 'cross').forEach(e => {
        const sy = rowY(e.s), ty = rowY(e.t);
        const d = `M${cols[i] + nodeW / 2 - 6} ${sy + (ty > sy ? nodeH / 2 - 4 : -nodeH / 2 + 4)} L${cols[i + 1] - nodeW / 2 - 2} ${ty + (ty > sy ? -nodeH / 2 + 6 : nodeH / 2 - 6)}`;
        eg.appendChild(svgEl('path', { d, class: 'edge cross' + (onEdge(e) ? '' : ' off'), 'marker-end': onEdge(e) ? 'url(#pgarr0)' : null }));
        addHit(d, e);
      });
    }
    for (let i = 2; i < 6; i++) {
      edges.filter(e => e.type === 'same').forEach(e => {
        const sy = rowY(e.s), ty = rowY(e.t);
        const dir = ty > sy ? 1 : -1;
        const side = e.s < e.t ? 1 : -1; // bulge to the right if going down, left if up
        const x = cols[i] + side * (nodeW / 2 - 4);
        const d = `M${x} ${sy + dir * nodeH / 2} Q${x + side * 18} ${(sy + ty) / 2} ${x} ${ty - dir * (nodeH / 2 + 2)}`;
        eg.appendChild(svgEl('path', { d, class: 'edge same' + (onEdge(e) ? '' : ' off'), 'marker-end': onEdge(e) ? 'url(#pgarr0)' : null }));
        addHit(d, e);
      });
    }
    svg.appendChild(eg);
    // nodes
    ex.concepts.forEach((c, j) => cols.forEach((x, i) => {
      const v = vals[j][i], bv = baseVals[j][i];
      const g = svgEl('g', { class: 'node' + (Math.abs(v - bv) > 0.005 ? ' changed' : '') });
      const fut = i > 2;
      const fill = fut ? 'var(--forecast)' : 'var(--accent)';
      g.appendChild(svgEl('rect', { x: x - nodeW / 2, y: rowY(j) - nodeH / 2, width: nodeW, height: nodeH, rx: 7, fill, 'fill-opacity': 0.15 + 0.85 * v, stroke: fill }));
      g.appendChild(svgEl('text', { x, y: rowY(j) + 4, 'text-anchor': 'middle', fill: v > 0.5 ? '#fff' : 'var(--text)' }, f2(v)));
      if (i === 2) { g.style.cursor = 'ns-resize'; g.addEventListener('click', () => { const inp = $(`#pgNode${j}`); if (inp) { inp.focus(); } }); }
      g.addEventListener('mousemove', e => showTip(`<b>${c}</b> at ${labs[i]}<br>${fut ? 'forecast' : 'observed'} state ${f2(v)}${Math.abs(v - bv) > 0.005 ? ` · unedited ${f2(bv)}` : ''}${i === 2 ? '<br>edit with the slider →' : ''}`, e.clientX, e.clientY));
      g.addEventListener('mouseleave', hideTip);
      svg.appendChild(g);
    }));
    host.appendChild(svg);
  }

  function renderControls(base, cur) {
    const host = $('#pgControls');
    if (!host.dataset.built || host.dataset.ex !== state.ex) {
      host.innerHTML = ''; host.dataset.built = '1'; host.dataset.ex = state.ex;
      const mk = (html) => { const d = document.createElement('div'); d.innerHTML = html; return d.firstElementChild; };
      // node sliders
      const nodeBox = mk(`<div><h3>Concept node edits at t</h3><p class="sub">What if a concept were different?</p></div>`);
      ex.concepts.forEach((c, j) => {
        const v0 = ex.states_observed[j][2];
        const ctl = mk(`<div class="ctl"><label for="pgNode${j}"><span>${c}</span><b id="pgNodeV${j}">${f2(v0)}</b></label><input type="range" id="pgNode${j}" min="0.02" max="0.98" step="0.01" value="${v0}" aria-label="${c} at t"></div>`);
        ctl.querySelector('input').addEventListener('input', e => { const v = +e.target.value; if (Math.abs(v - v0) < 0.005) delete state.nodes[j]; else state.nodes[j] = v; update(); });
        nodeBox.appendChild(ctl);
      });
      host.appendChild(nodeBox);
      // activity belief
      const actBox = mk(`<div><h3>Activity-belief edit at t</h3><p class="sub">Set one class probability; the rest is rescaled and fed back via A<sub>act</sub>.</p></div>`);
      const sel = mk(`<div class="ctl fc"><select id="pgAct" aria-label="Activity to edit"><option value="">— no activity edit —</option>${ex.activities.filter(a => a !== 'Other').map(a => `<option value="${a}">${a}</option>`).join('')}</select></div>`);
      const sl = mk(`<div class="ctl fc"><label for="pgActP"><span>probability</span><b id="pgActV">0.80</b></label><input type="range" id="pgActP" min="0.05" max="0.95" step="0.01" value="0.8" aria-label="probability for edited activity"></div>`);
      const syncAct = () => { const name = sel.querySelector('select').value; const p = +sl.querySelector('input').value; $('#pgActV').textContent = f2(p); state.act = name ? { name, p } : null; update(); };
      sel.querySelector('select').addEventListener('change', syncAct); sl.querySelector('input').addEventListener('input', syncAct);
      actBox.appendChild(sel); actBox.appendChild(sl); host.appendChild(actBox);
      // edges
      const edgeBox = mk(`<details><summary>Learned relations · uncheck to cut an edge (or click it in the graph)</summary><div class="edges" id="pgEdges"></div></details>`);
      const list = edgeBox.querySelector('#pgEdges');
      edgeList(ex).sort((a, b) => ['self', 'cross', 'same'].indexOf(a.type) - ['self', 'cross', 'same'].indexOf(b.type)).forEach(e => {
        const l = mk(`<label><input type="checkbox" data-key="${e.key}" checked><span class="k ${e.type}"></span><span>${e.label} <span class="muted">(${e.w >= 0 ? '+' : ''}${e.w.toFixed(2)})</span></span></label>`);
        l.querySelector('input').addEventListener('change', ev => { state.edges[e.key] = ev.target.checked; update(); });
        list.appendChild(l);
      });
      // presets
      const pr = mk(`<div><h3>Try the paper's edits</h3><div class="presets" id="pgPresets"></div></div>`);
      const pl = pr.querySelector('#pgPresets');
      (ex.paper_interventions || []).forEach((iv, i) => {
        const parts = [];
        for (const j in iv.nodes) parts.push(`${ex.concepts[+j]} → ${iv.nodes[j]}`);
        for (const a in iv.acts) parts.push(`belief ${a} → ${iv.acts[a]}`);
        const b = mk(`<button type="button" class="chip">${parts.join(', ')}</button>`);
        b.addEventListener('click', () => { resetState(); state.ex = ex.key; for (const j in iv.nodes) state.nodes[+j] = iv.nodes[j]; const a = Object.keys(iv.acts || {})[0]; if (a) state.act = { name: a, p: iv.acts[a] }; update(true); });
        pl.appendChild(b);
      });
      const rb = mk(`<button type="button" class="chip">Reset</button>`);
      rb.addEventListener('click', () => { resetState(); state.ex = ex.key; update(true); });
      pl.appendChild(rb);
      pr.appendChild(mk(`<div class="pg-status" id="pgStatus" style="margin-top:10px"></div>`));
      host.appendChild(pr);
      host.appendChild(edgeBox);
    }
    // sync widgets to state
    ex.concepts.forEach((c, j) => { const v = state.nodes[j] ?? ex.states_observed[j][2]; $(`#pgNode${j}`).value = v; $(`#pgNodeV${j}`).textContent = f2(v); });
    $('#pgAct').value = state.act ? state.act.name : ''; if (state.act) { $('#pgActP').value = state.act.p; $('#pgActV').textContent = f2(state.act.p); }
    $$('#pgEdges input').forEach(i => i.checked = state.edges[i.dataset.key] !== false);
    // status
    const st = $('#pgStatus');
    const h3b = base[3].y, h3c = cur[3].y;
    const amax = a => a.map((p, k) => k).filter(k => ex.activities[k] !== 'Other').sort((x, y) => a[y] - a[x])[0];
    const kb = amax(h3b), kc = amax(h3c);
    const gt = ex.gt[5], gi = ex.activities.indexOf(gt);
    if (!isEdited()) st.innerHTML = `<b>Unedited rollout.</b> t+3 prediction: <b>${ex.activities[kb]}</b> (${f2(h3b[kb])}), ground truth <b>${gt}</b>${gi >= 0 ? ` (${f2(h3b[gi])})` : ''}.`;
    else {
      const dp = gi >= 0 ? h3c[gi] - h3b[gi] : 0;
      st.innerHTML = `<b>After your edits</b>, t+3: ${ex.activities[kb]} ${f2(h3b[kb])} → <b>${ex.activities[kc]}</b> ${f2(h3c[kc])}. True class <b>${gt}</b>: ${f2(h3b[gi])} → ${f2(h3c[gi])} (${dp >= 0 ? '+' : ''}${(dp * 100).toFixed(1)} pp)${ex.activities[kc] === gt ? ' · <b>corrected ✓</b>' : ex.activities[kb] === gt ? ' · <b>broke a correct forecast</b>' : ''}.`;
    }
  }

  function update(rebuild) {
    const base = rollout(ex, { nodes: {}, act: null, edges: {} });
    const cur = rollout(ex, state);
    if (rebuild) { $('#pgControls').dataset.built = ''; }
    renderSvg(base, cur);
    renderControls(base, cur);
  }

  function setExample(key) {
    ex = DATA[key]; ex.key = key;
    resetState(); state.ex = key;
    $$('#pgExample button').forEach(b => b.setAttribute('aria-pressed', b.dataset.ex === key));
    renderFrames();
    update(true);
  }

  $('#pgExample').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setExample(b.dataset.ex); });
  document.addEventListener('trace-theme', () => { if (ex) update(); });

  fetch('assets/data/playground.json?v=2').then(r => r.json()).then(d => { DATA = d; setExample('barista'); })
    .catch(err => { $('#pgSvg').innerHTML = `<p class="muted">Could not load playground.json (${err}). Serve this folder over HTTP to run the demo.</p>`; });
})();
