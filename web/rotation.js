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

    // Select first featured galaxy — show honestly, let the data speak
    const featured = DATA.metadata.featured;
    const first = DATA.galaxies.find(g => featured.includes(g.name)) || DATA.galaxies[0];
    selectGalaxy(first.name);
  }

  // ── Score summary ──────────────────────────────────────────────────
  function renderScoreSummary() {
    // Count by Khronon chi2 quality (the meaningful metric)
    const good = DATA.galaxies.filter(g => g.chi2_khronon < 2).length;
    const moderate = DATA.galaxies.filter(g => g.chi2_khronon >= 2 && g.chi2_khronon < 5).length;
    const poor = DATA.galaxies.filter(g => g.chi2_khronon >= 5).length;
    const el = document.getElementById('score-summary');
    el.innerHTML = `
      <div class="score-box khronon-bg">
        <div class="score-num">175 &rarr; 1</div>
        <div class="score-label">Khronon: 1 param per galaxy<br>175 total free parameters</div>
      </div>
      <div class="score-box tie-bg">
        <div class="score-num" style="font-size:1.2rem;">vs</div>
        <div class="score-label">&nbsp;</div>
      </div>
      <div class="score-box nfw-bg">
        <div class="score-num">175 &rarr; 3</div>
        <div class="score-label">NFW: 3 params per galaxy<br>525 total free parameters</div>
      </div>
      <div class="score-box" style="background:rgba(255,255,255,0.03); border:1px solid #222; min-width:280px;">
        <div class="score-label" style="margin-bottom:6px;">Khronon prediction quality (1 param)</div>
        <div style="display:flex; gap:12px; justify-content:center; font-size:0.85rem;">
          <span style="color:#66bb6a;"><strong>${good}</strong> good</span>
          <span style="color:#e8860c;"><strong>${moderate}</strong> moderate</span>
          <span style="color:#ef5350;"><strong>${poor}</strong> poor</span>
        </div>
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
    if (filter === 'good') galaxies = galaxies.filter(g => g.chi2_khronon < 2);
    else if (filter === 'moderate') galaxies = galaxies.filter(g => g.chi2_khronon >= 2 && g.chi2_khronon < 5);
    else if (filter === 'poor') galaxies = galaxies.filter(g => g.chi2_khronon >= 5);
    else if (filter === 'featured') galaxies = galaxies.filter(g => featured.includes(g.name));

    if (search) {
      galaxies = galaxies.filter(g => g.name.toLowerCase().includes(search));
    }

    for (const g of galaxies) {
      const div = document.createElement('div');
      div.className = 'galaxy-item' + (currentGalaxy === g.name ? ' active' : '');
      div.dataset.name = g.name;

      // Color dot by Khronon prediction quality
      const qualClass = g.chi2_khronon < 2 ? 'khronon' :
                         g.chi2_khronon < 5 ? 'tie' : 'nfw';
      const qualLabel = g.chi2_khronon < 2 ? 'Good' :
                         g.chi2_khronon < 5 ? 'Moderate' : 'Poor';

      div.innerHTML = `
        <div>
          <span class="name">${g.name}</span>
          <span class="meta">${g.type} · ${g.n_pts}pts · &chi;&sup2;=${g.chi2_khronon.toFixed(1)}</span>
        </div>
        <div class="winner-dot ${qualClass}" title="${qualLabel}: &chi;&sup2;/dof=${g.chi2_khronon.toFixed(2)}"></div>
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
    if (filter === 'good') galaxies = galaxies.filter(g => g.chi2_khronon < 2);
    else if (filter === 'moderate') galaxies = galaxies.filter(g => g.chi2_khronon >= 2 && g.chi2_khronon < 5);
    else if (filter === 'poor') galaxies = galaxies.filter(g => g.chi2_khronon >= 5);
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
    // Show both fits honestly — the point is parameter count, not winning
    const qualColor = g.chi2_khronon < 2 ? '#66bb6a' :
                      g.chi2_khronon < 5 ? '#e8860c' : '#ef5350';
    ctx.fillStyle = qualColor;
    ctx.textAlign = 'right';
    ctx.fillText(`Khronon \u03C7\u00B2/dof = ${g.chi2_khronon.toFixed(2)} (1 param)  \u00B7  NFW = ${g.chi2_nfw.toFixed(2)} (3 params)`, ml + pw, mt - 14);

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(ml, mt, pw, ph);
    ctx.clip();

    // Shaded gap between baryonic and observed (the "missing mass" problem)
    if (document.getElementById('show-baryonic').checked) {
      ctx.fillStyle = 'rgba(255,255,255,0.04)';
      ctx.beginPath();
      // Forward along observed
      for (let i = 0; i < R.length; i++) {
        const x = xPos(R[i]), y = yPos(Vobs[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      // Backward along baryonic
      for (let i = R.length - 1; i >= 0; i--) {
        ctx.lineTo(xPos(R[i]), yPos(Vbar[i]));
      }
      ctx.closePath();
      ctx.fill();

      // Baryonic line (dashed gray)
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
    const qualColor = g.chi2_khronon < 2 ? '#66bb6a' :
                      g.chi2_khronon < 5 ? '#e8860c' : '#ef5350';
    const qualText = g.chi2_khronon < 2 ? 'Good' :
                     g.chi2_khronon < 5 ? 'Moderate' : 'Poor';
    el.innerHTML = `
      <div class="info-card">
        <div class="label">Type / Distance</div>
        <div class="value">${g.type} · ${g.D_Mpc} Mpc</div>
      </div>
      <div class="info-card">
        <div class="label">Prediction quality</div>
        <div class="value" style="color:${qualColor};">${qualText}</div>
      </div>
      <div class="info-card">
        <div class="label">Khronon &chi;&sup2;/dof</div>
        <div class="value khronon">${g.chi2_khronon.toFixed(2)}</div>
        <div class="label" style="margin-top:2px;">1 free param (M/L only)</div>
      </div>
      <div class="info-card">
        <div class="label">NFW &chi;&sup2;/dof</div>
        <div class="value nfw">${g.chi2_nfw.toFixed(2)}</div>
        <div class="label" style="margin-top:2px;">3 free params (M/L + M&sub;200&sub; + c)</div>
      </div>
      <div class="info-card">
        <div class="label">M/L (Khronon &rarr; NFW)</div>
        <div class="value">${g.ml_khronon.toFixed(2)} &rarr; ${g.ml_nfw.toFixed(2)}</div>
      </div>
      <div class="info-card">
        <div class="label">NFW extra params: log₁₀M₂₀₀, log₁₀c</div>
        <div class="value nfw">${g.nfw_log10_M200}, ${g.nfw_log10_c}</div>
        <div class="label" style="margin-top:2px;">These 2 params are not predicted</div>
      </div>
    `;
  }

  // ── Start ──────────────────────────────────────────────────────────
  init();
})();
