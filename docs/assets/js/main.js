/* TRACE-CBM project page: theme, hero animation, architecture explorer, results explorer */
(function () {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs = {}, text) => {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) el.setAttribute(k, attrs[k]);
    if (text !== undefined) el.textContent = text;
    return el;
  };
  const fmt = (v, d = 1) => (v === null || v === undefined) ? 'n/a' : (Math.round(v * 10 ** d) / 10 ** d).toFixed(d);
  window.TRACE = window.TRACE || {};
  window.TRACE.svgEl = svgEl;

  /* ---------------- theme ---------------- */
  const themeBtn = $('#themeBtn');
  const applyTheme = (t) => {
    if (t) document.documentElement.setAttribute('data-theme', t); else document.documentElement.removeAttribute('data-theme');
    try { t ? localStorage.setItem('trace-theme', t) : localStorage.removeItem('trace-theme'); } catch (e) {}
    document.dispatchEvent(new CustomEvent('trace-theme'));
  };
  themeBtn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    applyTheme(isDark ? 'light' : 'dark');
  });

  /* ---------------- tooltip ---------------- */
  const tip = $('#tooltip');
  const showTip = (html, x, y) => { tip.innerHTML = html; tip.style.left = x + 'px'; tip.style.top = y + 'px'; tip.classList.add('show'); };
  const hideTip = () => tip.classList.remove('show');
  window.TRACE.showTip = showTip; window.TRACE.hideTip = hideTip;

  /* ---------------- hero animation ---------------- */
  // One story in four beats: observe -> roll out -> edit a concept -> a second trajectory branches off.
  function buildHero() {
    const host = $('#heroAnim');
    const W = 1000, H = 430, T = 14; // seconds per loop
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
    const timed = []; // {el, show, hide, dim}
    const add = (parent, el, show, hide = Infinity, dim = Infinity) => { el.style.opacity = 0; parent.appendChild(el); timed.push({ el, show, hide, dim }); return el; };
    const concepts = ['portafilter', 'inserting sth.', 'coffee grinder'];
    const obsActs = ['Add coffee', 'Press coffee', 'Insert portafilter'];
    const futActs = ['Grab cup', 'Press button', 'Place cup'];
    const futActsEdited = ['Press button', 'Grab cup', 'Clean portafilter'];
    const cls = ['c0', 'c1', 'c2'];
    const B = { observe: 0.3, roll: [1.6, 2.6, 3.6], edit: 6.0, branch: 7.3, pills: 9.6, end: 13.3 };

    // ---- beat captions (bottom center) ----
    const cap = (txt, show, hide) => add(svg, svgEl('text', { x: W / 2 + 90, y: H - 8, class: 'lab', 'text-anchor': 'middle', 'font-weight': 600 }, txt), show, hide);
    cap('1 · Observe: both models see the same video windows', B.observe, B.roll[0]);
    cap('2 · Forecast: the black box rolls out latents, TRACE rolls out named concepts', B.roll[0], B.edit);
    cap('3 · Intervene: a human lowers one concept at t', B.edit, B.branch);
    cap('4 · A second trajectory branches off: the edited state re-rolls through the same learned relations', B.branch, B.end);

    // ---- left lane: black box ----
    svg.appendChild(svgEl('text', { x: 20, y: 28, class: 'lane-title' }, 'Black-box forecasting'));
    const fx = [20, 95, 170];
    fx.forEach((x, i) => {
      const g = svgEl('g');
      g.appendChild(svgEl('image', { href: `assets/img/frames/hero_${i}.jpg?v=2`, x, y: 48, width: 62, height: 44, preserveAspectRatio: 'xMidYMid slice', 'clip-path': 'inset(0 round 6px)' }));
      g.appendChild(svgEl('text', { x: x + 31, y: 106, class: 'lab-s', 'text-anchor': 'middle' }, obsActs[i]));
      g.appendChild(svgEl('text', { x: x + 31, y: 118, class: 'lab-s', 'text-anchor': 'middle' }, ['t-2', 't-1', 't'][i]));
      add(svg, g, B.observe);
    });
    const bb = svgEl('g');
    bb.appendChild(svgEl('path', { d: 'M126 122 v16', class: 'edge' }));
    bb.appendChild(svgEl('rect', { x: 46, y: 140, width: 160, height: 56, rx: 10, class: 'bb' }));
    bb.appendChild(svgEl('text', { x: 126, y: 160, class: 'bb-text', 'text-anchor': 'middle' }, 'Black-box model'));
    bb.appendChild(svgEl('text', { x: 126, y: 190, class: 'q', 'text-anchor': 'middle' }, '?'));
    add(svg, bb, B.roll[0]);
    const lat = svgEl('g');
    for (let i = 0; i < 18; i++) lat.appendChild(svgEl('rect', { x: 50 + (i % 9) * 17, y: 204 + Math.floor(i / 9) * 10, width: 13, height: 7, rx: 1.5, class: 'latent' }));
    lat.appendChild(svgEl('text', { x: 126, y: 236, class: 'lab-s', 'text-anchor': 'middle' }, 'opaque latents'));
    add(svg, lat, B.roll[1]);
    futActs.forEach((a, i) => {
      const g = svgEl('g');
      g.appendChild(svgEl('text', { x: 46 + i * 80, y: 262, class: 'lab' }, a));
      g.appendChild(svgEl('text', { x: 46 + i * 80, y: 275, class: 'lab-s' }, ['t+1', 't+2', 't+3'][i]));
      add(svg, g, B.roll[i]);
    });
    const noEdit = svgEl('g');
    noEdit.appendChild(svgEl('rect', { x: 36, y: 300, width: 180, height: 40, rx: 10, fill: 'var(--bad)', opacity: .10 }));
    noEdit.appendChild(svgEl('text', { x: 126, y: 316, class: 'lab-s', 'text-anchor': 'middle', fill: 'var(--bad)', 'font-weight': 600 }, '✗ nothing to inspect or edit'));
    noEdit.appendChild(svgEl('text', { x: 126, y: 331, class: 'lab-s', 'text-anchor': 'middle', fill: 'var(--bad)' }, 'no alternative trajectory'));
    add(svg, noEdit, B.edit);
    svg.appendChild(svgEl('line', { x1: 280, y1: 20, x2: 280, y2: H - 24, class: 'divider' }));

    // ---- right lane: TRACE ----
    svg.appendChild(svgEl('text', { x: 304, y: 28, class: 'lane-title' }, 'TRACE-CBM: forecasting in concept space'));
    const cols = [360, 440, 520, 610, 690, 770];
    const colLab = ['t-2', 't-1', 't', 't+1', 't+2', 't+3'];
    const rowsA = [125, 155, 185], barA = 232, labA = 250;
    const OFF = 150; // vertical offset of the edited branch
    const rowsB = rowsA.map(y => y + OFF), barB = barA + OFF, labB = labA + OFF;
    cols.slice(0, 3).forEach((x, i) => {
      const g = svgEl('g');
      g.appendChild(svgEl('image', { href: `assets/img/frames/hero_${i}.jpg?v=2`, x: x - 31, y: 40, width: 62, height: 44, preserveAspectRatio: 'xMidYMid slice', 'clip-path': 'inset(0 round 6px)' }));
      g.appendChild(svgEl('text', { x, y: 97, class: 'lab-s', 'text-anchor': 'middle' }, obsActs[i]));
      add(svg, g, B.observe);
    });
    const rowG = svgEl('g');
    concepts.forEach((c, j) => rowG.appendChild(svgEl('text', { x: 344, y: rowsA[j] + 4, class: 'lab', 'text-anchor': 'end' }, c)));
    rowG.appendChild(svgEl('text', { x: 344, y: 222, class: 'lab-s', 'text-anchor': 'end' }, 'activity'));
    rowG.appendChild(svgEl('text', { x: 344, y: 233, class: 'lab-s', 'text-anchor': 'end' }, 'distribution'));
    add(svg, rowG, B.observe);
    const defs = svgEl('defs');
    const mk = svgEl('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
    mk.appendChild(svgEl('path', { d: 'M0 0 L10 5 L0 10 z', fill: 'var(--text-3)' }));
    const mk2 = svgEl('marker', { id: 'arrF', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
    mk2.appendChild(svgEl('path', { d: 'M0 0 L10 5 L0 10 z', fill: 'var(--forecast)' }));
    defs.appendChild(mk); defs.appendChild(mk2); svg.appendChild(defs);

    const radiiA = [[6, 7, 7, 6, 6, 6], [7, 6, 8, 8, 8, 8], [9, 7, 8, 8, 7, 7]];      // unedited
    const radiiB = [[6, 7, 7, 7, 7, 7], [7, 6, 8, 6, 5, 5], [9, 7, 3, 3, 3, 3]];      // after lowering coffee grinder at t
    const barsA = [[3, 10, 5, 2], [4, 12, 3, 2], [2, 6, 13, 3], [3, 4, 9, 11], [4, 2, 5, 12], [8, 4, 6, 10]];
    const barsB = [[3, 10, 5, 2], [4, 12, 3, 2], [2, 6, 13, 3], [3, 4, 12, 5], [4, 2, 11, 6], [10, 4, 9, 5]];
    const colShow = i => i < 3 ? B.observe : B.roll[i - 3];
    // one column of the trajectory; `variant` picks weights, `rows/barY/labY` place it
    const column = (i, variant, rows, barY, labY, withEdges = true) => {
      const x = cols[i], radii = variant === 'A' ? radiiA : radiiB, bars = variant === 'A' ? barsA : barsB;
      const g = svgEl('g');
      if (i > 0 && withEdges) {
        concepts.forEach((_, j) => g.appendChild(svgEl('path', { d: `M${cols[i - 1] + radii[j][i - 1] + 1} ${rows[j]} H${x - radii[j][i] - 1}`, class: 'edge gold' })));
        g.appendChild(svgEl('path', { d: `M${cols[i - 1] + 6} ${rows[2] - 6} L${x - 6} ${rows[1] + 6}`, class: 'edge dash' }));
        if (i % 2 === 0) g.appendChild(svgEl('path', { d: `M${x + 4} ${rows[0] + 7} Q${x + 16} ${rows[1]} ${x + 4} ${rows[1] - 7}`, class: 'edge', 'marker-end': 'url(#arr)' }));
      }
      concepts.forEach((_, j) => g.appendChild(svgEl('circle', { cx: x, cy: rows[j], r: radii[j][i], class: `node ${cls[j]}` })));
      bars[i].forEach((h, k) => g.appendChild(svgEl('rect', { x: x - 14 + k * 8, y: barY - h, width: 6, height: h, rx: 1, class: 'bar' + (i < 3 ? ' obs' : '') })));
      g.appendChild(svgEl('line', { x1: x - 16, y1: barY + 0.5, x2: x + 18, y2: barY + 0.5, class: 'edge', 'stroke-width': 1 }));
      if (rows === rowsA) g.appendChild(svgEl('text', { x, y: 112, class: 'lab-s', 'text-anchor': 'middle' }, colLab[i]));
      if (i >= 2) {
        const label = i === 2 ? 'Insert portafilter' : (variant === 'A' ? futActs : futActsEdited)[i - 3];
        g.appendChild(svgEl('text', { x, y: labY, class: 'lab', 'text-anchor': 'middle', 'font-weight': 600, fill: variant === 'B' && i > 2 ? 'var(--forecast)' : null }, label));
      }
      return g;
    };
    // observed columns
    for (let i = 0; i < 3; i++) add(svg, column(i, 'A', rowsA, barA, labA), colShow(i));
    // original forecast: stays, dims once the branch opens
    for (let i = 3; i < 6; i++) add(svg, column(i, 'A', rowsA, barA, labA), colShow(i), Infinity, B.branch);
    add(svg, svgEl('rect', { x: cols[3] - 24, y: 104, width: cols[5] - cols[3] + 48, height: 154, class: 'cursor' }), B.roll[0], Infinity, B.branch);
    const origLab = svgEl('text', { x: cols[4], y: 100, class: 'lab-s', 'text-anchor': 'middle' }, 'original forecast');
    add(svg, origLab, B.branch);

    // the edit: ring + shrunken node at t
    const edT = svgEl('g');
    edT.appendChild(svgEl('circle', { cx: cols[2], cy: rowsA[2], r: 15, fill: 'var(--surface)' }));
    edT.appendChild(svgEl('circle', { cx: cols[2], cy: rowsA[2], r: 3, class: 'node c2' }));
    edT.appendChild(svgEl('circle', { cx: cols[2], cy: rowsA[2], r: 13, fill: 'none', stroke: 'var(--forecast)', 'stroke-width': 2, 'stroke-dasharray': '3 3' }));
    add(svg, edT, B.edit);
    const ed = svgEl('g');
    ed.appendChild(svgEl('rect', { x: 800, y: 168, width: 182, height: 34, rx: 10, class: 'edit' }));
    ed.appendChild(svgEl('text', { x: 891, y: 182, class: 'edit-text', 'text-anchor': 'middle' }, '✎ node intervention'));
    ed.appendChild(svgEl('text', { x: 891, y: 195, class: 'lab-s', 'text-anchor': 'middle' }, 'coffee grinder: 0.8 → 0.1'));
    ed.appendChild(svgEl('path', { d: `M800 185 C 760 185, 700 ${rowsA[2] + 30}, ${cols[2] + 12} ${rowsA[2] + 10}`, class: 'edge dash', 'marker-end': 'url(#arr)' }));
    add(svg, ed, B.edit);

    // the branch: fork connectors from the edited t state down to the branch's t+1 column
    const fork = svgEl('g');
    concepts.forEach((_, j) => {
      const r0 = j === 2 ? 3 : radiiA[j][2];
      fork.appendChild(svgEl('path', { d: `M${cols[2] + r0 + 1} ${rowsA[j]} C ${cols[2] + 50} ${rowsA[j]}, ${cols[3] - 60} ${rowsB[j]}, ${cols[3] - radiiB[j][3] - 2} ${rowsB[j]}`, class: 'edge', stroke: 'var(--forecast)', 'stroke-width': 1.6, 'marker-end': 'url(#arrF)' }));
    });
    add(svg, fork, B.branch);
    for (let i = 3; i < 6; i++) add(svg, column(i, 'B', rowsB, barB, labB, i > 3), B.branch + (i - 3) * 0.4);
    add(svg, svgEl('rect', { x: cols[3] - 24, y: rowsB[0] - 21, width: cols[5] - cols[3] + 48, height: 154, class: 'cursor', stroke: 'var(--forecast)' }), B.branch);
    const brLab = svgEl('g');
    brLab.appendChild(svgEl('text', { x: 344, y: rowsB[1] - 2, class: 'lab', 'text-anchor': 'end', fill: 'var(--forecast)', 'font-weight': 600 }, 'edited'));
    brLab.appendChild(svgEl('text', { x: 344, y: rowsB[1] + 11, class: 'lab', 'text-anchor': 'end', fill: 'var(--forecast)', 'font-weight': 600 }, 'trajectory'));
    brLab.appendChild(svgEl('text', { x: 344, y: rowsB[1] + 25, class: 'lab-s', 'text-anchor': 'end' }, 'same relations,'));
    brLab.appendChild(svgEl('text', { x: 344, y: rowsB[1] + 36, class: 'lab-s', 'text-anchor': 'end' }, 'no retraining'));
    add(svg, brLab, B.branch);
    const changed = svgEl('g');
    changed.appendChild(svgEl('rect', { x: 800, y: rowsB[1] - 17, width: 182, height: 34, rx: 10, class: 'edit' }));
    changed.appendChild(svgEl('text', { x: 891, y: rowsB[1] - 3, class: 'edit-text', 'text-anchor': 'middle' }, 'forecast changes at t+1…t+3'));
    changed.appendChild(svgEl('text', { x: 891, y: rowsB[1] + 10, class: 'lab-s', 'text-anchor': 'middle' }, 'compare both trajectories'));
    add(svg, changed, B.branch + 1.2);
    const pills = svgEl('g');
    [['inspect concepts', 820], ['intervene during forecasting', 820], ['explain future predictions', 820]].forEach(([t, x], k) => {
      const w = t.length * 5.6 + 22, y = 48 + k * 26;
      pills.appendChild(svgEl('rect', { x, y, width: w, height: 20, rx: 10, class: 'pill' }));
      pills.appendChild(svgEl('text', { x: x + 11, y: y + 14, class: 'pill-text' }, '✓ ' + t));
    });
    add(svg, pills, B.pills);
    host.appendChild(svg);

    // ---- clock ----
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const btn = $('#animToggle');
    let playing = !reduce, t0 = performance.now(), paused = 0, raf = 0;
    const render = (t) => {
      for (const it of timed) {
        const on = t >= it.show && t < it.hide;
        it.el.style.opacity = on ? (t >= it.dim ? 0.3 : 1) : 0;
        it.el.style.transition = 'opacity .35s';
      }
    };
    const tick = (now) => {
      const t = ((now - t0) / 1000) % T;
      render(t < B.end ? t : -1); // brief blank before the loop restarts
      raf = requestAnimationFrame(tick);
    };
    const setPlaying = (on) => {
      playing = on; btn.setAttribute('aria-pressed', on); btn.textContent = on ? 'Pause animation' : 'Play animation';
      host.classList.toggle('animate', on);
      if (on) { t0 = performance.now() - paused; raf = requestAnimationFrame(tick); }
      else cancelAnimationFrame(raf);
    };
    btn.addEventListener('click', () => { if (playing) { paused = (performance.now() - t0) % (T * 1000); setPlaying(false); render(paused / 1000); } else setPlaying(true); });
    if (playing) setPlaying(true); else render(B.end - 0.1);
  }

  /* ---------------- architecture explorer ---------------- */
  const STAGES = [
    {
      n: 'Stage i', t: 'Concept extraction & normalization',
      body: 'Each 32-frame window and each of the C = 128 natural-language concepts are embedded by a frozen CLIP-like video encoder (PE-L/14). Their cosine similarity is the raw concept score; a learned per-concept threshold and scale map it to a bounded state in (0, 1), so every coordinate stays a named, human-readable concept.',
      eq: '\\mathbf{z}_t=\\sigma\\big((\\mathbf{c}_t-\\boldsymbol{\\tau})\\odot\\mathbf{s}\\big)\\in(0,1)^C',
      bullets: ['Vocabulary proposed by Qwen3.5-27B from label-blind storyboards, visually verified, deduplicated, filtered for label overlap.', 'Normalization is learned on the training split and frozen when producing future concept targets.']
    },
    {
      n: 'Stage ii', t: 'Graph-refined concept states',
      body: 'A node (t, j) is concept j at window t. One message-passing layer adds relational messages in logit space and squashes back to (0, 1), keeping an unpruned identity path. Three relation families: same-time edges between concepts in one window, self-temporal edges carrying a concept from t−1 to t, and cross-temporal edges from one concept at t−1 to another at t.',
      eq: '\\mathbf{r}^{(l)}_t=\\sigma\\!\\Big(\\phi(\\mathbf{r}^{(l-1)}_t)+\\boldsymbol{\\lambda}_{\\mathcal G}\\odot\\big[\\mathbf{r}^{(l-1)}_t\\mathbf{A}_{\\mathrm{same}}+\\mathbf{r}^{(l-1)}_{t-1}\\odot\\mathbf{a}_{\\mathrm{self}}+\\mathbf{r}^{(l-1)}_{t-1}\\mathbf{A}_{\\mathrm{cross}}\\big]\\Big)',
      bullets: ['Top-q pruning keeps each target\'s strongest incoming sources (q = 20 by default); budgets are ramped up during training.', 'Three stacks share this layer type: G_shared refines observations, G_act specializes for recognition, G_fut is the recursive forecast transition.', 'Sparse rollout keeps ~5 effective H3 parents per concept versus ~87 for a dense rollout.']
    },
    {
      n: 'Stage iii', t: 'Recognition & activity-conditioned forecasting',
      body: 'A single linear activity head decodes every state. Its predictions feed back into the concept space through a sparse activity-to-concept matrix, so each forecast transition is conditioned on the activity beliefs of the preceding windows. The forecast stack is applied recursively for h = 1…H, and each new state is decoded with the same head.',
      eq: '\\widetilde{\\mathbf{r}}^{\\,\\mathrm{fut}}_{t+1}=\\sigma\\!\\Big(\\phi\\big(\\mathcal G_{\\mathrm{fut}}(\\widetilde{\\mathbf{r}}^{\\,\\mathrm{fut}}_{1:t})\\big)+\\mathbf{d}^{\\mathrm{hist}}_{t+1}\\Big),\\qquad \\hat{\\mathbf y}_t=\\operatorname{softmax}\\big(\\phi(\\widetilde{\\mathbf r}_t)\\mathbf W^{\\top}+\\mathbf b\\big)',
      bullets: ['d_hist aggregates ŷ A_act over the last L_hist windows with learned lag gates.', 'Because forecasts stay in concept space, an edit at t propagates through the same rollout to t+1 … t+H without retraining.', 'Trained with activity, forecast, concept-alignment and ℓ1 head-sparsity losses.']
    },
    {
      n: 'Edges', t: 'Four relation types, three interventions',
      body: 'The legend of the figure doubles as the intervention menu. Node edits clamp a concept value, edge edits modify one effective relation while leaving the residual path intact, and activity edits set one class probability and redistribute the rest before it flows back through A_act.',
      eq: '\\mathbf{A}_{\\mathrm{act}}=\\mathbf{A}\\odot\\sigma(\\boldsymbol{\\lambda}_A)\\odot\\mathbf{M}_{q_{\\mathrm{fb}}}\\in\\mathbb{R}^{K\\times C}',
      bullets: ['Same-time (grey, solid): i_t → j_t within a window.', 'Self-temporal (gold): k_{t−1} → k_t persistence or decay.', 'Cross-temporal (dashed): i_{t−1} → j_t ordered transitions.', 'Activity-to-concept feedback (purple): class k → concept j.']
    }
  ];
  function buildArch() {
    const tabs = $('#archTabs'), detail = $('#archDetail');
    STAGES.forEach((s, i) => {
      const b = document.createElement('button');
      b.className = 'arch-tab'; b.type = 'button'; b.role = 'tab'; b.dataset.stage = i; b.setAttribute('aria-selected', i === 1);
      b.innerHTML = `<div class="n">${s.n}</div><div class="t">${s.t}</div>`;
      tabs.appendChild(b);
    });
    const select = (i) => {
      $$('.arch-tab').forEach(b => b.setAttribute('aria-selected', +b.dataset.stage === i));
      $$('.hotspot').forEach(b => b.setAttribute('aria-pressed', +b.dataset.stage === i));
      const s = STAGES[i];
      detail.innerHTML = `<h3>${s.n} · ${s.t}</h3><p>${s.body}</p><div class="eq" id="archEq"></div><ul>${s.bullets.map(x => `<li>${x}</li>`).join('')}</ul>`;
      const eq = $('#archEq');
      if (window.katex) { try { katex.render(s.eq, eq, { displayMode: true, throwOnError: false }); } catch (e) { eq.textContent = s.eq; } }
      else { eq.innerHTML = `<code>${s.eq}</code>`; window.addEventListener('load', () => { if (window.katex) katex.render(s.eq, eq, { displayMode: true, throwOnError: false }); }, { once: true }); }
    };
    tabs.addEventListener('click', e => { const b = e.target.closest('.arch-tab'); if (b) select(+b.dataset.stage); });
    $$('.hotspot').forEach(h => {
      h.addEventListener('click', () => { select(+h.dataset.stage); if (innerWidth < 860) detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); });
      h.addEventListener('mouseenter', () => select(+h.dataset.stage));
    });
    select(1);
  }

  /* ---------------- results explorer ---------------- */
  const FAMILY_COLOR = { 'Zero-shot': 'var(--gray-mark)', 'Black-box': 'var(--orange)', 'Concept': 'var(--aqua)', 'ours': 'var(--accent)' };
  const SERIES = ['var(--accent)', 'var(--orange)', 'var(--aqua)', 'var(--gold)'];
  let R = null, rxState = { tab: 'main', dataset: 0, metric: 0, horizon: 'H3', k: 'Top-1', table: false };

  function hbarChart(rows, opts) {
    // rows: [{label, value, std, color, ours, group}] ; opts: {min,max,unit,groups:boolean}
    const W = 760, rowH = 30, left = 190, right = 70, top = opts.groups ? 10 : 6;
    const groups = []; rows.forEach(r => { if (opts.groups && (!groups.length || groups[groups.length - 1] !== r.group)) groups.push(r.group); });
    const H = top + rows.length * rowH + (opts.groups ? groups.length * 18 : 0) + 30;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opts.aria || 'bar chart' });
    const min = opts.min ?? 0, max = opts.max;
    const x = v => left + (v - min) / (max - min) * (W - left - right);
    // ticks
    const ticks = opts.ticks || [];
    ticks.forEach(t => {
      svg.appendChild(svgEl('line', { x1: x(t), x2: x(t), y1: top, y2: H - 26, class: t === 0 ? 'zero' : 'axis' }));
      svg.appendChild(svgEl('text', { x: x(t), y: H - 10, class: 'tick', 'text-anchor': 'middle' }, t + (opts.unit || '')));
    });
    let y = top, lastGroup = null;
    rows.forEach(r => {
      if (opts.groups && r.group !== lastGroup) { lastGroup = r.group; svg.appendChild(svgEl('text', { x: 0, y: y + 12, class: 'group-lab' }, r.group.toUpperCase())); y += 18; }
      const cy = y + rowH / 2;
      svg.appendChild(svgEl('text', { x: left - 10, y: cy + 4, class: 'cat' + (r.ours ? ' ours' : ''), 'text-anchor': 'end' }, r.label));
      if (r.value === null || r.value === undefined) {
        svg.appendChild(svgEl('text', { x: x(Math.max(min, 0)) + 6, y: cy + 4, class: 'tick' }, 'n/a'));
      } else {
        const x0 = x(Math.max(min, 0)), x1 = x(r.value);
        const bx = Math.min(x0, x1), bw = Math.max(1, Math.abs(x1 - x0));
        const g = svgEl('g');
        const rect = svgEl('rect', { x: bx, y: cy - 10, width: bw, height: 20, fill: r.color, class: 'bar' });
        // rounded data end only
        const path = r.value >= 0
          ? `M${bx} ${cy - 10} H${bx + bw - 4} a4 4 0 0 1 4 4 v12 a4 4 0 0 1 -4 4 H${bx} z`
          : `M${bx + bw} ${cy - 10} H${bx + 4} a4 4 0 0 0 -4 4 v12 a4 4 0 0 0 4 4 H${bx + bw} z`;
        rect.remove();
        g.appendChild(svgEl('path', { d: path, fill: r.color, opacity: r.dim ? .45 : 1 }));
        if (r.std) {
          g.appendChild(svgEl('line', { x1: x(r.value - r.std), x2: x(r.value + r.std), y1: cy, y2: cy, class: 'whisker' }));
          g.appendChild(svgEl('line', { x1: x(r.value - r.std), x2: x(r.value - r.std), y1: cy - 4, y2: cy + 4, class: 'whisker' }));
          g.appendChild(svgEl('line', { x1: x(r.value + r.std), x2: x(r.value + r.std), y1: cy - 4, y2: cy + 4, class: 'whisker' }));
        }
        const lx = r.value >= 0 ? x(r.value + (r.std || 0)) + 6 : x(r.value - (r.std || 0)) - 6;
        g.appendChild(svgEl('text', { x: lx, y: cy + 4, class: 'val', 'text-anchor': r.value >= 0 ? 'start' : 'end' }, (r.value > 0 && opts.signed ? '+' : '') + fmt(r.value, r.dec ?? 1)));
        const hit = svgEl('rect', { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: 'transparent' });
        const tipHtml = `<b>${r.label}</b>${r.group ? ' · ' + r.group : ''}<br>${(r.value > 0 && opts.signed ? '+' : '') + fmt(r.value, r.dec ?? 1)}${r.std !== null && r.std !== undefined ? ' ± ' + fmt(r.std, r.dec ?? 1) : ''}${opts.unit || ''}${opts.measure ? ' · ' + opts.measure : ''}`;
        hit.addEventListener('mousemove', e => showTip(tipHtml, e.clientX, e.clientY));
        hit.addEventListener('mouseleave', hideTip);
        g.appendChild(hit);
        svg.appendChild(g);
      }
      y += rowH;
    });
    return svg;
  }

  function legend(items) {
    const d = document.createElement('div'); d.className = 'rx-legend';
    d.innerHTML = items.map(([c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join('');
    return d;
  }
  function seg(options, value, onChange, label) {
    const wrap = document.createElement('div'); wrap.style.display = 'inline-flex'; wrap.style.alignItems = 'center'; wrap.style.gap = '8px';
    if (label) { const l = document.createElement('span'); l.className = 'lbl'; l.textContent = label; wrap.appendChild(l); }
    const s = document.createElement('div'); s.className = 'seg';
    options.forEach(o => {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = o; b.setAttribute('aria-pressed', o === value);
      b.addEventListener('click', () => onChange(o)); s.appendChild(b);
    });
    wrap.appendChild(s); return wrap;
  }
  const pm = (v, s, d = 1) => v === null ? 'n/a' : `${fmt(v, d)}${s !== null && s !== undefined ? ` <span class="sd">± ${fmt(s, d)}</span>` : ''}`;
  function tableBlock(headers, rows) {
    const det = document.createElement('details'); det.className = 'tbl'; det.open = rxState.table;
    det.addEventListener('toggle', () => rxState.table = det.open);
    det.innerHTML = `<summary>${det.open ? 'Hide' : 'Show'} table</summary><div class="rx-table"><table><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr class="${r.ours ? 'ours' : ''}">${r.cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    det.addEventListener('toggle', () => det.querySelector('summary').textContent = (det.open ? 'Hide' : 'Show') + ' table');
    return det;
  }

  const TABS = [['main', 'Main comparison'], ['ablation', 'Component ablation'], ['interventions', 'Interventions'], ['qwen', 'Qwen planning'], ['steering', 'Steering']];
  function renderRx() {
    const host = $('#rx'); host.innerHTML = '';
    $$('#rxTabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === rxState.tab));
    const ctl = document.createElement('div'); ctl.className = 'rx-ctl'; host.appendChild(ctl);
    const chart = document.createElement('div'); chart.className = 'rx-chart'; host.appendChild(chart);
    const cap = document.createElement('p'); cap.className = 'rx-cap';
    const T = R[rxState.tab];
    if (rxState.tab === 'main') {
      ctl.appendChild(seg(T.datasets, T.datasets[rxState.dataset], v => { rxState.dataset = T.datasets.indexOf(v); renderRx(); }, 'Dataset'));
      ctl.appendChild(seg(['Activity', 'H3'], rxState.horizon === 'H3' ? 'H3' : 'Activity', v => { rxState.horizon = v; renderRx(); }, 'Task'));
      ctl.appendChild(seg(['Top-1', 'Top-3'], rxState.k, v => { rxState.k = v; renderRx(); }, 'Metric'));
      const mi = T.metrics.indexOf(`${rxState.horizon} ${rxState.k}`);
      const rows = T.methods.map(m => ({ label: m.name, group: m.family, value: m.values[rxState.dataset][mi], std: m.std[rxState.dataset][mi], ours: !!m.ours, color: m.ours ? FAMILY_COLOR.ours : FAMILY_COLOR[m.family] }));
      chart.appendChild(hbarChart(rows, { min: 0, max: 100, ticks: [0, 20, 40, 60, 80, 100], unit: '', groups: true, measure: `${T.metrics[mi]} accuracy (%)`, aria: 'Main comparison' }));
      chart.appendChild(legend([[FAMILY_COLOR.ours, 'TRACE (ours)'], [FAMILY_COLOR['Concept'], 'Concept-based baselines'], [FAMILY_COLOR['Black-box'], 'Black-box feature baselines'], [FAMILY_COLOR['Zero-shot'], 'Zero-shot']]));
      cap.textContent = T.caption; host.appendChild(cap);
      host.appendChild(tableBlock(['Method', ...T.metrics], T.methods.map(m => ({ ours: m.ours, cells: [m.name, ...T.metrics.map((_, i) => pm(m.values[rxState.dataset][i], m.std[rxState.dataset][i]))] }))));
    } else if (rxState.tab === 'ablation') {
      ctl.appendChild(seg(T.datasets, T.datasets[rxState.dataset], v => { rxState.dataset = T.datasets.indexOf(v); renderRx(); }, 'Dataset'));
      const rows = T.variants.map(v => ({ label: v.name, value: v.values[rxState.dataset], std: v.std[rxState.dataset], ours: !!v.ours, color: v.ours ? FAMILY_COLOR.ours : FAMILY_COLOR['Concept'] }));
      chart.appendChild(hbarChart(rows, { min: 0, max: 100, ticks: [0, 20, 40, 60, 80, 100], measure: 'H3 Top-1 (%)', aria: 'Component ablation' }));
      cap.textContent = T.caption; host.appendChild(cap);
      host.appendChild(tableBlock(['Variant', ...T.datasets], T.variants.map(v => ({ ours: v.ours, cells: [v.name, ...T.datasets.map((_, i) => pm(v.values[i], v.std[i]))] }))));
    } else {
      // grouped: datasets x policies/interfaces, one measure at a time
      const groupsKey = T.interfaces ? 'interfaces' : 'policies';
      const measures = T.measures;
      const mIdx = Math.min(rxState.metric, measures.length - 1);
      ctl.appendChild(seg(measures, measures[mIdx], v => { rxState.metric = measures.indexOf(v); renderRx(); }, 'Measure'));
      const rows = [];
      T.datasets.forEach(ds => T[groupsKey].forEach((g, gi) => rows.push({ label: g, group: ds, value: T.values[ds][gi][mIdx], std: T.std[ds][gi][mIdx], color: SERIES[gi % SERIES.length], dec: Math.abs(T.values[ds][gi][mIdx]) < 0.1 && T.values[ds][gi][mIdx] !== 0 ? 2 : 1 })));
      const vals = rows.map(r => r.value); const lo = Math.min(0, ...vals), hi = Math.max(...vals);
      const signed = measures[mIdx].startsWith('Δ');
      const max = signed ? Math.ceil((hi + 8) / 10) * 10 : 100, min = signed ? Math.floor((lo - 5) / 10) * 10 : 0;
      const ticks = []; for (let t = min; t <= max; t += (max - min) > 60 ? 20 : 10) ticks.push(t);
      const unit = signed ? ' pp' : '%';
      chart.appendChild(hbarChart(rows, { min, max, ticks, unit, groups: true, signed, measure: measures[mIdx], aria: TABS.find(t => t[0] === rxState.tab)[1] }));
      chart.appendChild(legend(T[groupsKey].map((g, i) => [SERIES[i % SERIES.length], g])));
      cap.textContent = T.caption; host.appendChild(cap);
      const headers = ['Dataset', groupsKey === 'interfaces' ? 'Interface' : 'Policy', ...measures];
      const trows = []; T.datasets.forEach(ds => T[groupsKey].forEach((g, gi) => trows.push({ cells: [ds, g, ...measures.map((_, mi) => pm(T.values[ds][gi][mi], T.std[ds][gi][mi], Math.abs(T.values[ds][gi][mi]) < 0.1 && T.values[ds][gi][mi] !== 0 ? 2 : 1))] })));
      host.appendChild(tableBlock(headers, trows));
    }
  }
  function buildRx() {
    const tabs = $('#rxTabs');
    TABS.forEach(([k, l]) => { const b = document.createElement('button'); b.type = 'button'; b.role = 'tab'; b.dataset.tab = k; b.textContent = l; b.addEventListener('click', () => { rxState.tab = k; rxState.metric = 0; renderRx(); }); tabs.appendChild(b); });
    fetch('assets/data/results.json?v=2').then(r => r.json()).then(data => { R = data; renderRx(); }).catch(err => { $('#rx').innerHTML = `<p class="muted">Could not load results.json (${err}). When opened from the file system, serve the folder with a local web server.</p>`; });
  }

  /* ---------------- bibtex copy ---------------- */
  $('#copyBib').addEventListener('click', async () => {
    const btn = $('#copyBib');
    try { await navigator.clipboard.writeText($('#bibText').textContent); btn.textContent = 'Copied'; } catch (e) { btn.textContent = 'Select & copy'; }
    setTimeout(() => btn.textContent = 'Copy', 1600);
  });

  buildHero(); buildArch(); buildRx();
})();
