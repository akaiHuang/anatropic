/**
 * Rotation Curve Interactive Viewer
 * Plots SPARC observed data vs Khronon (1 param) vs NFW (3 params)
 */

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────
  let DATA = null;
  let currentGalaxy = null;
  let filter = 'all';

  const canvas = document.getElementById('chart-canvas');
  const ctx = canvas.getContext('2d');

  // ── Load data ──────────────────────────────────────────────────────
  async function init() {
    try {
      const resp = await fetch('data/sparc_rotation_curves.json');
      DATA = await resp.json();
    } catch (e) {
      document.getElementById('loading-status').textContent =
        'Error loading data. Run: python examples/build_rotation_curves.py';
      return;
    }

    document.getElementById('loading-status').style.display = 'none';
    document.getElementById('rc-layout').style.display = 'flex';

    renderScoreSummary();
    buildGalaxyList();
    setupEvents();

    // Select first featured galaxy
    const featured = DATA.metadata.featured;
    const first = DATA.galaxies.find(g => featured.includes(g.name)) || DATA.galaxies[0];
    selectGalaxy(first.name);
  }

  // ── Score summary ──────────────────────────────────────────────────
  function renderScoreSummary() {
    const s = DATA.metadata.summary;
    const el = document.getElementById('score-summary');
    el.innerHTML = `
      <div class="score-box khronon-bg">
        <div class="score-num">${s.khronon_preferred}</div>
        <div class="score-label">Khronon more accurate<br>(1 free param/galaxy)</div>
      </div>
      <div class="score-box tie-bg">
        <div class="score-num">${s.inconclusive}</div>
        <div class="score-label">Comparable<br>(|ΔBIC| ≤ 2)</div>
      </div>
      <div class="score-box nfw-bg">
        <div class="score-num">${s.nfw_preferred}</div>
        <div class="score-label">NFW more accurate<br>(3 free params/galaxy)</div>
      </div>
    `;
  }

  // ── Galaxy list ────────────────────────────────────────────────────
  function buildGalaxyList() {
    const list = document.getElementById('galaxy-list');
    list.innerHTML = '';

    const search = document.getElementById('galaxy-search').value.toLowerCase();
    const featured = DATA.metadata.featured;

    let galaxies = DATA.galaxies;
    if (filter === 'khronon') galaxies = galaxies.filter(g => g.winner === 'Khronon');
    else if (filter === 'nfw') galaxies = galaxies.filter(g => g.winner === 'NFW');
    else if (filter === 'featured') galaxies = galaxies.filter(g => featured.includes(g.name));

    if (search) {
      galaxies = galaxies.filter(g => g.name.toLowerCase().includes(search));
    }

    for (const g of galaxies) {
      const div = document.createElement('div');
      div.className = 'galaxy-item' + (currentGalaxy === g.name ? ' active' : '');
      div.dataset.name = g.name;

      const winClass = g.winner === 'Khronon' ? 'khronon' :
                        g.winner === 'NFW' ? 'nfw' : 'tie';

      div.innerHTML = `
        <div>
          <span class="name">${g.name}</span>
          <span class="meta">${g.type} · ${g.n_pts}pts</span>
        </div>
        <div class="winner-dot ${winClass}" title="ΔBIC=${g.delta_BIC}"></div>
      `;
      div.addEventListener('click', () => selectGalaxy(g.name));
      list.appendChild(div);
    }
  }

  function selectGalaxy(name) {
    currentGalaxy = name;
    // Update active state
    document.querySelectorAll('.galaxy-item').forEach(el => {
      el.classList.toggle('active', el.dataset.name === name);
    });
    // Scroll into view
    const active = document.querySelector('.galaxy-item.active');
    if (active) active.scrollIntoView({ block: 'nearest' });

    drawChart();
    updateInfo();
  }

  // ── Setup events ───────────────────────────────────────────────────
  function setupEvents() {
    document.getElementById('galaxy-search').addEventListener('input', buildGalaxyList);

    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        filter = btn.dataset.filter;
        buildGalaxyList();
      });
    });

    ['show-errorbars', 'show-baryonic', 'show-khronon', 'show-nfw'].forEach(id => {
      document.getElementById(id).addEventListener('change', drawChart);
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      const galaxies = getVisibleGalaxies();
      const idx = galaxies.findIndex(g => g.name === currentGalaxy);
      if (e.key === 'ArrowDown' || e.key === 'j') {
        e.preventDefault();
        if (idx < galaxies.length - 1) selectGalaxy(galaxies[idx + 1].name);
      } else if (e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault();
        if (idx > 0) selectGalaxy(galaxies[idx - 1].name);
      }
    });

    window.addEventListener('resize', drawChart);
  }

  function getVisibleGalaxies() {
    const search = document.getElementById('galaxy-search').value.toLowerCase();
    const featured = DATA.metadata.featured;
    let galaxies = DATA.galaxies;
    if (filter === 'khronon') galaxies = galaxies.filter(g => g.winner === 'Khronon');
    else if (filter === 'nfw') galaxies = galaxies.filter(g => g.winner === 'NFW');
    else if (filter === 'featured') galaxies = galaxies.filter(g => featured.includes(g.name));
    if (search) galaxies = galaxies.filter(g => g.name.toLowerCase().includes(search));
    return galaxies;
  }

  // ── Draw chart ─────────────────────────────────────────────────────
  function drawChart() {
    if (!currentGalaxy || !DATA) return;

    const g = DATA.galaxies.find(d => d.name === currentGalaxy);
    if (!g) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width;
    const H = rect.height;

    // Margins
    const ml = 65, mr = 20, mt = 45, mb = 50;
    const pw = W - ml - mr;
    const ph = H - mt - mb;

    // Data ranges
    const R = g.R;
    const Vobs = g.Vobs;
    const e_Vobs = g.e_Vobs;
    const Vbar = g.Vbar;
    const V_k = g.V_khronon;
    const V_n = g.V_nfw;

    const rMax = Math.max(...R) * 1.08;
    const allV = [...Vobs, ...V_k, ...V_n, ...Vbar];
    const vMax = Math.max(...allV) * 1.2;
    const vMin = 0;

    function xPos(r) { return ml + (r / rMax) * pw; }
    function yPos(v) { return mt + ph - ((v - vMin) / (vMax - vMin)) * ph; }

    // Clear
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1;
    const nGridX = 5, nGridY = 5;
    for (let i = 0; i <= nGridY; i++) {
      const v = vMin + (vMax - vMin) * i / nGridY;
      const y = yPos(v);
      ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(ml + pw, y); ctx.stroke();
    }
    for (let i = 0; i <= nGridX; i++) {
      const r = rMax * i / nGridX;
      const x = xPos(r);
      ctx.beginPath(); ctx.moveTo(x, mt); ctx.lineTo(x, mt + ph); ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = '#888';
    ctx.font = '12px Inter, Helvetica, sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i <= nGridX; i++) {
      const r = rMax * i / nGridX;
      ctx.fillText(r.toFixed(1), xPos(r), mt + ph + 20);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= nGridY; i++) {
      const v = vMin + (vMax - vMin) * i / nGridY;
      ctx.fillText(v.toFixed(0), ml - 8, yPos(v) + 4);
    }

    // Axis titles
    ctx.fillStyle = '#aaa';
    ctx.font = '13px Inter, Helvetica, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Radius (kpc)', ml + pw / 2, H - 8);
    ctx.save();
    ctx.translate(16, mt + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('V (km/s)', 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = '#e0e0e0';
    ctx.font = 'bold 15px Inter, Helvetica, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(g.name, ml, mt - 14);

    ctx.font = '12px Inter, Helvetica, sans-serif';
    ctx.fillStyle = '#888';
    const winColor = g.winner === 'Khronon' ? '#4fc3f7' :
                     g.winner === 'NFW' ? '#ef5350' : '#888';
    ctx.fillStyle = winColor;
    ctx.textAlign = 'right';
    const winText = g.winner === 'Khronon' ? 'Khronon more accurate' :
                    g.winner === 'NFW' ? 'NFW more accurate' : 'Comparable';
    ctx.fillText(`${winText} (ΔBIC = ${g.delta_BIC > 0 ? '+' : ''}${g.delta_BIC})`, ml + pw, mt - 14);

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(ml, mt, pw, ph);
    ctx.clip();

    // Baryonic (dashed gray)
    if (document.getElementById('show-baryonic').checked) {
      ctx.strokeStyle = '#555';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i < R.length; i++) {
        const x = xPos(R[i]), y = yPos(Vbar[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // NFW line (red)
    if (document.getElementById('show-nfw').checked) {
      ctx.strokeStyle = '#ef5350';
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      for (let i = 0; i < R.length; i++) {
        const x = xPos(R[i]), y = yPos(V_n[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Khronon line (blue)
    if (document.getElementById('show-khronon').checked) {
      ctx.strokeStyle = '#4fc3f7';
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let i = 0; i < R.length; i++) {
        const x = xPos(R[i]), y = yPos(V_k[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Error bars
    if (document.getElementById('show-errorbars').checked) {
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      for (let i = 0; i < R.length; i++) {
        const x = xPos(R[i]);
        const y1 = yPos(Vobs[i] - e_Vobs[i]);
        const y2 = yPos(Vobs[i] + e_Vobs[i]);
        ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke();
        // caps
        ctx.beginPath(); ctx.moveTo(x - 2, y1); ctx.lineTo(x + 2, y1); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 2, y2); ctx.lineTo(x + 2, y2); ctx.stroke();
      }
    }

    // Observed data points (white dots)
    for (let i = 0; i < R.length; i++) {
      const x = xPos(R[i]), y = yPos(Vobs[i]);
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    ctx.restore(); // unclip
  }

  // ── Info panel ─────────────────────────────────────────────────────
  function updateInfo() {
    const g = DATA.galaxies.find(d => d.name === currentGalaxy);
    if (!g) return;

    const el = document.getElementById('galaxy-info');
    el.innerHTML = `
      <div class="info-card">
        <div class="label">Type / Distance</div>
        <div class="value">${g.type} · ${g.D_Mpc} Mpc</div>
      </div>
      <div class="info-card">
        <div class="label">Khronon χ²/dof (1 param)</div>
        <div class="value khronon">${g.chi2_khronon.toFixed(2)}</div>
      </div>
      <div class="info-card">
        <div class="label">NFW χ²/dof (3 params)</div>
        <div class="value nfw">${g.chi2_nfw.toFixed(2)}</div>
      </div>
      <div class="info-card">
        <div class="label">M/L (Khronon → NFW)</div>
        <div class="value">${g.ml_khronon.toFixed(2)} → ${g.ml_nfw.toFixed(2)}</div>
      </div>
      <div class="info-card">
        <div class="label">NFW: log₁₀M₂₀₀, log₁₀c</div>
        <div class="value nfw">${g.nfw_log10_M200}, ${g.nfw_log10_c}</div>
      </div>
      <div class="info-card">
        <div class="label">ΔBIC (>0 = Khronon better)</div>
        <div class="value ${g.delta_BIC > 2 ? 'win' : g.delta_BIC < -2 ? 'nfw' : ''}">${g.delta_BIC > 0 ? '+' : ''}${g.delta_BIC}</div>
      </div>
    `;
  }

  // ── Start ──────────────────────────────────────────────────────────
  init();
})();
