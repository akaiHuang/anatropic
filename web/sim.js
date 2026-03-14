// ═══════════════════════════════════════════════════════════════════════════
//  Anatropic — Simulation Viewer with τ Field Overlay (Three.js + WebGL)
//  Loads binary simulation data and renders density + temporal asymmetry
//  Sheng-Kai Huang, 2026
// ═══════════════════════════════════════════════════════════════════════════

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── Constants ────────────────────────────────────────────────────────────
const DATA_DIR = "data";

// ── Shader: dual-channel volume ray marching ─────────────────────────────
// uVol = density (R channel), uTau = τ field (R channel)
// uViewMode: 0 = density only, 1 = τ only, 2 = combined

const vertexShader = `
varying vec3 vOrigin;
varying vec3 vDirection;
uniform vec3 uCamPos;
void main() {
  vDirection = position - uCamPos;
  vOrigin = uCamPos;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const fragmentShader = `
precision highp float;
precision highp sampler3D;
varying vec3 vOrigin;
varying vec3 vDirection;
uniform sampler3D uVol;   // density
uniform sampler3D uTau;   // τ field
uniform float uThr;       // density threshold
uniform float uOpa;       // opacity multiplier
uniform float uBri;       // brightness
uniform int uViewMode;    // 0=density, 1=tau, 2=combined

// Inferno colormap for density
vec3 inferno(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.001, 0.0, 0.014);
  vec3 c1 = vec3(0.119, 0.047, 0.284);
  vec3 c2 = vec3(0.530, 0.065, 0.380);
  vec3 c3 = vec3(0.735, 0.130, 0.240);
  vec3 c4 = vec3(0.890, 0.290, 0.100);
  vec3 c5 = vec3(0.975, 0.550, 0.040);
  vec3 c6 = vec3(0.993, 0.906, 0.144);
  if (t < 0.167) return mix(c0, c1, t / 0.167);
  if (t < 0.333) return mix(c1, c2, (t - 0.167) / 0.167);
  if (t < 0.500) return mix(c2, c3, (t - 0.333) / 0.167);
  if (t < 0.667) return mix(c3, c4, (t - 0.500) / 0.167);
  if (t < 0.833) return mix(c4, c5, (t - 0.667) / 0.167);
  return mix(c5, c6, (t - 0.833) / 0.167);
}

// Blue-white-red diverging colormap for τ
vec3 tauColor(float t) {
  t = clamp(t, 0.0, 1.0);
  // Blue (τ≈0) → White (τ≈0.5) → Red (τ→1)
  vec3 blue = vec3(0.05, 0.25, 0.85);
  vec3 white = vec3(0.95, 0.95, 0.95);
  vec3 red = vec3(0.85, 0.12, 0.08);
  if (t < 0.5) return mix(blue, white, t * 2.0);
  return mix(white, red, (t - 0.5) * 2.0);
}

vec2 boxHit(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t1 = min(-o * inv, (1.0 - o) * inv);
  vec3 t2 = max(-o * inv, (1.0 - o) * inv);
  return vec2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

void main() {
  vec3 rd = normalize(vDirection);
  vec3 ro = vOrigin + 0.5;  // shift to [0,1]³
  vec2 th = boxHit(ro, rd);
  if (th.x > th.y) discard;
  th.x = max(th.x, 0.0);

  float dt = 1.0 / 96.0;  // step size (higher = better quality)
  vec3 acc = vec3(0.0);
  float aa = 0.0;

  for (float t = th.x; t < th.y; t += dt) {
    vec3 p = ro + rd * t;

    float d = texture(uVol, p).r;   // density [0,1]
    float tau = texture(uTau, p).r; // τ [0,1]

    if (d > uThr) {
      float v = clamp((d - uThr) / (1.0 - uThr + 0.001), 0.0, 1.0);

      vec3 col;
      if (uViewMode == 0) {
        // Density only: inferno colormap
        col = inferno(v) * uBri;
      } else if (uViewMode == 1) {
        // τ only: blue-white-red, density as opacity
        col = tauColor(tau) * uBri;
      } else {
        // Combined: density brightness, τ hue
        // High density + high τ = bright red
        // High density + low τ = bright blue
        // Low density = dark
        vec3 tauC = tauColor(tau);
        col = tauC * v * uBri * 1.5;
      }

      float a = v * uOpa * dt * 18.0;
      acc += (1.0 - aa) * a * col;
      aa += (1.0 - aa) * a;
      if (aa > 0.98) break;
    }
  }

  gl_FragColor = vec4(acc + (1.0 - aa) * vec3(0.015), 1.0);
}`;

// ── Data loader ──────────────────────────────────────────────────────────

async function loadManifest() {
  const res = await fetch(`${DATA_DIR}/manifest.json`);
  if (!res.ok) throw new Error(`No simulation data found at ${DATA_DIR}/manifest.json`);
  return res.json();
}

async function loadBinary(filename, expectedSize) {
  const res = await fetch(`${DATA_DIR}/${filename}`);
  if (!res.ok) throw new Error(`Failed to load ${filename}`);
  const buf = await res.arrayBuffer();
  if (buf.byteLength !== expectedSize) {
    console.warn(`${filename}: expected ${expectedSize} bytes, got ${buf.byteLength}`);
  }
  return new Uint8Array(buf);
}

// ── Scene ────────────────────────────────────────────────────────────────

class SimViewer {
  constructor(container) {
    this.container = container;
    this.manifest = null;
    this.snapshots = [];  // [{density: Uint8Array, tau: Uint8Array}]
    this.currentSnap = 0;
    this.viewMode = 0; // 0=density, 1=tau, 2=combined
    this.threshold = 0.20;
    this.opacity = 0.85;

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 800;

    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(0x050505);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.01, 20);
    this.camera.position.set(1.8, 1.3, 1.8);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.5;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 5;

    // Bounding box
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    this.scene.add(new THREE.LineSegments(edges,
      new THREE.LineBasicMaterial({ color: 0x333333 })));

    // Axis labels
    this._addAxisLabels();

    // Volume mesh (created after data loads)
    this.volMesh = null;
    this.densTex = null;
    this.tauTex = null;

    window.addEventListener("resize", () => this._resize());
  }

  _addAxisLabels() {
    // Simple axis indicator lines
    const mat = (c) => new THREE.LineBasicMaterial({ color: c });
    const pts = (a, b) => new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(...a), new THREE.Vector3(...b)]);
    this.scene.add(new THREE.Line(pts([-0.55, -0.55, -0.55], [-0.35, -0.55, -0.55]), mat(0xff4444)));
    this.scene.add(new THREE.Line(pts([-0.55, -0.55, -0.55], [-0.55, -0.35, -0.55]), mat(0x44ff44)));
    this.scene.add(new THREE.Line(pts([-0.55, -0.55, -0.55], [-0.55, -0.55, -0.35]), mat(0x4444ff)));
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _createTexture(data, N) {
    const tex = new THREE.Data3DTexture(data, N, N, N);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }

  setSnapshot(index) {
    if (index < 0 || index >= this.snapshots.length) return;
    this.currentSnap = index;

    const snap = this.snapshots[index];
    const N = this.manifest.grid[0];

    // Update density texture
    if (this.densTex) this.densTex.dispose();
    this.densTex = this._createTexture(snap.density, N);

    // Update τ texture
    if (this.tauTex) this.tauTex.dispose();
    this.tauTex = this._createTexture(snap.tau, N);

    if (!this.volMesh) {
      // First time: create volume mesh
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uVol: { value: this.densTex },
          uTau: { value: this.tauTex },
          uThr: { value: this.threshold },
          uOpa: { value: this.opacity },
          uBri: { value: 2.2 },
          uCamPos: { value: new THREE.Vector3() },
          uViewMode: { value: this.viewMode },
        },
        vertexShader,
        fragmentShader,
        side: THREE.BackSide,
      });
      this.volMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
      this.scene.add(this.volMesh);
    } else {
      this.volMesh.material.uniforms.uVol.value = this.densTex;
      this.volMesh.material.uniforms.uTau.value = this.tauTex;
    }
  }

  setViewMode(mode) {
    this.viewMode = mode;
    if (this.volMesh) {
      this.volMesh.material.uniforms.uViewMode.value = mode;
    }
  }

  setThreshold(t) {
    this.threshold = t;
    if (this.volMesh) this.volMesh.material.uniforms.uThr.value = t;
  }

  setOpacity(o) {
    this.opacity = o;
    if (this.volMesh) this.volMesh.material.uniforms.uOpa.value = o;
  }

  render() {
    this.controls.update();
    if (this.volMesh) {
      this.volMesh.material.uniforms.uCamPos.value.copy(this.camera.position);
    }
    this.renderer.render(this.scene, this.camera);
  }
}

// ── Main ─────────────────────────────────────────────────────────────────

const statusEl = document.getElementById("loading-status");
const containerEl = document.getElementById("sim-container");
const controlsEl = document.getElementById("sim-controls");
const timeDisplayEl = document.getElementById("time-display");
const statsEl = document.getElementById("stats-panel");

async function main() {
  let manifest;
  try {
    manifest = await loadManifest();
  } catch (e) {
    statusEl.innerHTML = `
      <p style="color:#e8860c;">No simulation data found.</p>
      <p style="font-size:0.85rem;">Run the simulation first:</p>
      <code style="display:block;margin:12px auto;padding:8px;background:#1a1a1a;border-radius:4px;max-width:500px;">
        cd anatropic && python examples/run_and_export_3d.py
      </code>
      <p style="font-size:0.85rem;">Then serve locally:</p>
      <code style="display:block;margin:12px auto;padding:8px;background:#1a1a1a;border-radius:4px;max-width:500px;">
        cd web && python -m http.server 8000
      </code>`;
    return;
  }

  const N = manifest.grid[0];
  const N3 = N * N * N;
  const nSnaps = manifest.snapshots.length;

  statusEl.innerHTML = `<span class="spinner"></span> Loading ${nSnaps} snapshots (${N}³ grid)...`;

  // Load all snapshots
  const viewer = new SimViewer(containerEl);
  viewer.manifest = manifest;

  for (let i = 0; i < nSnaps; i++) {
    const s = manifest.snapshots[i];
    statusEl.innerHTML = `<span class="spinner"></span> Loading snapshot ${i + 1}/${nSnaps} (t = ${s.time_ff.toFixed(1)} t<sub>ff</sub>)...`;

    const [density, tau] = await Promise.all([
      loadBinary(s.density_file, N3),
      loadBinary(s.tau_file, N3),
    ]);
    viewer.snapshots.push({ density, tau, meta: s });
  }

  // Show viewer
  statusEl.style.display = "none";
  containerEl.style.display = "block";
  controlsEl.style.display = "flex";
  timeDisplayEl.style.display = "block";

  // Set initial snapshot (last one = most structure)
  const lastIdx = nSnaps - 1;
  viewer.setSnapshot(lastIdx);

  // ── Time slider ──────────────────────────────────────────────────────
  const timeSlider = document.getElementById("time-slider");
  const tValue = document.getElementById("t-value");
  timeSlider.max = lastIdx;
  timeSlider.value = lastIdx;

  function updateTime(idx) {
    viewer.setSnapshot(idx);
    const meta = viewer.snapshots[idx].meta;
    tValue.textContent = meta.time_ff.toFixed(2);
    statsEl.textContent =
      `ρ ∈ [${meta.density_stats.min.toFixed(3)}, ${meta.density_stats.max.toFixed(3)}]` +
      ` | contrast: ${meta.density_stats.contrast.toFixed(1)}×` +
      ` | τ ∈ [${meta.tau_stats.min.toFixed(4)}, ${meta.tau_stats.max.toFixed(4)}]`;
  }
  updateTime(lastIdx);

  timeSlider.addEventListener("input", () => {
    updateTime(parseInt(timeSlider.value, 10));
  });

  // ── Auto-play ────────────────────────────────────────────────────────
  const playToggle = document.getElementById("play-toggle");
  let playInterval = null;

  playToggle.addEventListener("change", () => {
    if (playToggle.checked) {
      let idx = parseInt(timeSlider.value, 10);
      playInterval = setInterval(() => {
        idx = (idx + 1) % nSnaps;
        timeSlider.value = idx;
        updateTime(idx);
      }, 400);
    } else {
      clearInterval(playInterval);
    }
  });

  // ── View mode ────────────────────────────────────────────────────────
  const viewBtns = document.querySelectorAll(".view-btn");
  const legendDens = document.getElementById("legend-density");
  const legendTau = document.getElementById("legend-tau");

  viewBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      viewBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const view = btn.dataset.view;
      const mode = view === "density" ? 0 : view === "tau" ? 1 : 2;
      viewer.setViewMode(mode);

      legendDens.style.display = (mode === 0 || mode === 2) ? "flex" : "none";
      legendTau.style.display = (mode === 1 || mode === 2) ? "flex" : "none";
    });
  });

  // ── Threshold / Opacity ──────────────────────────────────────────────
  const thrSlider = document.getElementById("sim-threshold");
  const thrVal = document.getElementById("sim-thr-val");
  const opaSlider = document.getElementById("sim-opacity");
  const opaVal = document.getElementById("sim-opa-val");

  thrSlider.addEventListener("input", () => {
    const t = parseInt(thrSlider.value, 10) / 100;
    thrVal.textContent = t.toFixed(2);
    viewer.setThreshold(t);
  });

  opaSlider.addEventListener("input", () => {
    const o = parseInt(opaSlider.value, 10) / 100;
    opaVal.textContent = o.toFixed(2);
    viewer.setOpacity(o);
  });

  // ── Animation loop ───────────────────────────────────────────────────
  (function animate() {
    requestAnimationFrame(animate);
    viewer.render();
  })();
}

main().catch(err => {
  console.error(err);
  statusEl.innerHTML = `<p style="color:#ff4444;">Error: ${err.message}</p>`;
});
