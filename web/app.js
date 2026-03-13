// ═══════════════════════════════════════════════════════════════════════════
//  Anatropic — Three Dark Matter Morphologies  (Three.js + WebGL)
//  Sheng-Kai Huang, 2026
// ═══════════════════════════════════════════════════════════════════════════

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── Constants ───────────────────────────────────────────────────────────────
const N = 64;                // grid resolution per axis
const N3 = N * N * N;
const HALF = N / 2;

// ── Inferno-like colormap (sampled at 8 stops) ─────────────────────────────
const INFERNO_STOPS = [
  [0.00, 0.001, 0.014, 0.071],
  [0.14, 0.119, 0.047, 0.284],
  [0.29, 0.320, 0.060, 0.430],
  [0.43, 0.530, 0.065, 0.380],
  [0.57, 0.735, 0.130, 0.240],
  [0.71, 0.890, 0.290, 0.100],
  [0.86, 0.975, 0.550, 0.040],
  [1.00, 0.993, 0.906, 0.144],
];

function infernoColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < INFERNO_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = INFERNO_STOPS[i];
    const [t1, r1, g1, b1] = INFERNO_STOPS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0)];
    }
  }
  const last = INFERNO_STOPS[INFERNO_STOPS.length - 1];
  return [last[1], last[2], last[3]];
}

// ── 3D Simplex-style noise (value noise with smooth interpolation) ──────────
// Lightweight: no external dependency. Uses hash-based value noise.

function hashNoise3D(ix, iy, iz) {
  // Simple integer hash → float in [0,1)
  let h = ix * 374761393 + iy * 668265263 + iz * 1274126177;
  h = (h ^ (h >> 13)) * 1103515245;
  h = h ^ (h >> 16);
  return (h & 0x7fffffff) / 0x7fffffff;
}

function smoothstep(t) {
  return t * t * t * (t * (t * 6 - 15) + 10); // quintic
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function valueNoise3D(x, y, z) {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = smoothstep(x - ix), fy = smoothstep(y - iy), fz = smoothstep(z - iz);

  const c000 = hashNoise3D(ix,     iy,     iz);
  const c100 = hashNoise3D(ix + 1, iy,     iz);
  const c010 = hashNoise3D(ix,     iy + 1, iz);
  const c110 = hashNoise3D(ix + 1, iy + 1, iz);
  const c001 = hashNoise3D(ix,     iy,     iz + 1);
  const c101 = hashNoise3D(ix + 1, iy,     iz + 1);
  const c011 = hashNoise3D(ix,     iy + 1, iz + 1);
  const c111 = hashNoise3D(ix + 1, iy + 1, iz + 1);

  return lerp(
    lerp(lerp(c000, c100, fx), lerp(c010, c110, fx), fy),
    lerp(lerp(c001, c101, fx), lerp(c011, c111, fx), fy),
    fz
  );
}

function fbmNoise3D(x, y, z, octaves = 4, lacunarity = 2.0, gain = 0.5) {
  let val = 0, amp = 1, freq = 1, maxAmp = 0;
  for (let o = 0; o < octaves; o++) {
    val += amp * valueNoise3D(x * freq, y * freq, z * freq);
    maxAmp += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return val / maxAmp;
}

// ── Density field generators ────────────────────────────────────────────────

function generatePsiDM(seed = 42) {
  const grid = new Float32Array(N3);
  const numWaves = 18;
  const rng = mulberry32(seed);

  // Pre-compute wave vectors & phases
  const waves = [];
  for (let w = 0; w < numWaves; w++) {
    const theta = Math.acos(2 * rng() - 1);
    const phi = 2 * Math.PI * rng();
    const kMag = 2 + rng() * 6; // k magnitude range
    waves.push({
      kx: kMag * Math.sin(theta) * Math.cos(phi),
      ky: kMag * Math.sin(theta) * Math.sin(phi),
      kz: kMag * Math.cos(theta),
      amp: 0.5 + 0.5 * rng(),
      phase: 2 * Math.PI * rng(),
    });
  }

  for (let iz = 0; iz < N; iz++) {
    const z = (iz - HALF) / N;
    for (let iy = 0; iy < N; iy++) {
      const y = (iy - HALF) / N;
      for (let ix = 0; ix < N; ix++) {
        const x = (ix - HALF) / N;
        let reSum = 0, imSum = 0;
        for (const w of waves) {
          const dot = w.kx * x + w.ky * y + w.kz * z;
          const arg = 2 * Math.PI * dot + w.phase;
          reSum += w.amp * Math.cos(arg);
          imSum += w.amp * Math.sin(arg);
        }
        grid[iz * N * N + iy * N + ix] = reSum * reSum + imSum * imSum;
      }
    }
  }
  return normalizeGrid(grid);
}

function generateKhronon(seed = 137) {
  const grid = new Float32Array(N3);
  const rng = mulberry32(seed);
  // Offset noise space randomly so each regeneration looks different
  const ox = rng() * 1000, oy = rng() * 1000, oz = rng() * 1000;
  const scale = 3.5; // controls filament spacing

  for (let iz = 0; iz < N; iz++) {
    const z = iz / N * scale + oz;
    for (let iy = 0; iy < N; iy++) {
      const y = iy / N * scale + oy;
      for (let ix = 0; ix < N; ix++) {
        const x = ix / N * scale + ox;

        // Base FBM noise
        const n = fbmNoise3D(x, y, z, 4, 2.2, 0.45);

        // Create filamentary structure by using ridged noise:
        // |2*noise - 1| inverted → peaks at 0.5 crossing become ridges
        const ridge = 1.0 - Math.abs(2 * n - 1);
        // Sharpen ridges to get thin filaments
        const filament = Math.pow(ridge, 3.0);

        // Add smaller-scale detail along filaments
        const detail = fbmNoise3D(x * 2.5 + 50, y * 2.5 + 50, z * 2.5 + 50, 2, 2.0, 0.3);
        const combined = filament * (0.7 + 0.3 * detail);

        // Create voids: suppress low-density regions further
        const voided = combined > 0.15 ? combined : combined * 0.1;

        grid[iz * N * N + iy * N + ix] = voided;
      }
    }
  }
  return normalizeGrid(grid);
}

function generateCDM(seed = 271) {
  const grid = new Float32Array(N3);
  const rng = mulberry32(seed);
  const numHalos = 15;

  const halos = [];
  for (let h = 0; h < numHalos; h++) {
    halos.push({
      cx: rng() * N,
      cy: rng() * N,
      cz: rng() * N,
      mass: 0.4 + rng() * 0.6,
      sigma: 1.5 + rng() * 3.5,
    });
  }

  for (let iz = 0; iz < N; iz++) {
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        let rho = 0;
        for (const h of halos) {
          const dx = ix - h.cx, dy = iy - h.cy, dz = iz - h.cz;
          const r2 = dx * dx + dy * dy + dz * dz;
          rho += h.mass * Math.exp(-r2 / (2 * h.sigma * h.sigma));
        }
        grid[iz * N * N + iy * N + ix] = rho;
      }
    }
  }
  return normalizeGrid(grid);
}

// ── Utilities ───────────────────────────────────────────────────────────────

function normalizeGrid(grid) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] < mn) mn = grid[i];
    if (grid[i] > mx) mx = grid[i];
  }
  const range = mx - mn || 1;
  for (let i = 0; i < grid.length; i++) {
    grid[i] = (grid[i] - mn) / range;
  }
  return grid;
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Three.js scene builder ──────────────────────────────────────────────────

class VoxelScene {
  constructor(container, label) {
    this.container = container;
    this.label = label;
    this.grid = null;
    this.mesh = null;
    this.threshold = 0.30;
    this.opacity = 0.85;

    // Loading overlay
    this.overlay = document.createElement("div");
    this.overlay.className = "loading-overlay";
    this.overlay.textContent = "Generating...";
    this.container.appendChild(this.overlay);

    const w = container.clientWidth || 400;
    const h = container.clientHeight || 400;

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(0x050505);
    container.appendChild(this.renderer.domElement);

    // Scene
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 200);
    this.camera.position.set(1.8, 1.4, 1.8);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.8;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 6;

    // Lighting (subtle)
    const amb = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(amb);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(3, 5, 4);
    this.scene.add(dir);

    // Bounding box wireframe
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x333333 })
    );
    this.scene.add(line);

    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  setGrid(grid) {
    this.grid = grid;
    this.rebuild();
    this.overlay.classList.add("hidden");
  }

  rebuild() {
    if (!this.grid) return;

    // Remove old mesh
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }

    // Count voxels above threshold
    const positions = [];
    const colors = [];
    const threshold = this.threshold;

    for (let iz = 0; iz < N; iz++) {
      for (let iy = 0; iy < N; iy++) {
        for (let ix = 0; ix < N; ix++) {
          const val = this.grid[iz * N * N + iy * N + ix];
          if (val < threshold) continue;

          // Position: map grid to [-0.5, 0.5]
          const px = (ix / N) - 0.5;
          const py = (iy / N) - 0.5;
          const pz = (iz / N) - 0.5;
          positions.push(px, py, pz);

          // Color: remap [threshold..1] → [0..1] for colormap
          const t = (val - threshold) / (1 - threshold + 1e-8);
          const [r, g, b] = infernoColor(t);
          colors.push(r, g, b);
        }
      }
    }

    const count = positions.length / 3;
    if (count === 0) return;

    // Instanced mesh with small cubes
    const voxelSize = 1.0 / N;
    const baseGeo = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);
    const mat = new THREE.MeshLambertMaterial({
      vertexColors: false,
      transparent: true,
      opacity: this.opacity,
    });

    const instGeo = new THREE.InstancedBufferGeometry();
    instGeo.index = baseGeo.index;
    instGeo.attributes.position = baseGeo.attributes.position;
    instGeo.attributes.normal = baseGeo.attributes.normal;

    // Instance attributes
    const offsetAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(positions), 3
    );
    instGeo.setAttribute("instanceOffset", offsetAttr);

    const colorAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(colors), 3
    );
    instGeo.setAttribute("instanceColor", colorAttr);

    instGeo.instanceCount = count;

    // Custom shader material for instanced coloring
    const shaderMat = new THREE.ShaderMaterial({
      transparent: true,
      uniforms: {
        uOpacity: { value: this.opacity },
        uLightDir: { value: new THREE.Vector3(0.4, 0.7, 0.5).normalize() },
      },
      vertexShader: `
        attribute vec3 instanceOffset;
        attribute vec3 instanceColor;
        varying vec3 vColor;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        void main() {
          vColor = instanceColor;
          vNormal = normalMatrix * normal;
          vec3 transformed = position + instanceOffset;
          vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uOpacity;
        uniform vec3 uLightDir;
        varying vec3 vColor;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        void main() {
          vec3 n = normalize(vNormal);
          float diff = max(dot(n, uLightDir), 0.0);
          float ambient = 0.45;
          vec3 lit = vColor * (ambient + (1.0 - ambient) * diff);
          gl_FragColor = vec4(lit, uOpacity);
        }
      `,
    });

    this.mesh = new THREE.Mesh(instGeo, shaderMat);
    this.scene.add(this.mesh);
  }

  setThreshold(t) {
    this.threshold = t;
    this.rebuild();
  }

  setOpacity(o) {
    this.opacity = o;
    if (this.mesh && this.mesh.material.uniforms) {
      this.mesh.material.uniforms.uOpacity.value = o;
    }
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    window.removeEventListener("resize", this._onResize);
    this.controls.dispose();
    this.renderer.dispose();
  }
}

// ── Main application ────────────────────────────────────────────────────────

const containerPsi = document.getElementById("container-psiDM");
const containerKhr = document.getElementById("container-khronon");
const containerCDM = document.getElementById("container-CDM");

const scenePsi = new VoxelScene(containerPsi, "psiDM");
const sceneKhr = new VoxelScene(containerKhr, "Khronon");
const sceneCDM = new VoxelScene(containerCDM, "CDM");
const scenes = [scenePsi, sceneKhr, sceneCDM];

// Generate density fields asynchronously (via setTimeout to avoid blocking)
let globalSeed = Date.now();

function generateAll() {
  scenes.forEach((s) => {
    s.overlay.classList.remove("hidden");
    s.overlay.textContent = "Generating...";
  });

  setTimeout(() => {
    scenePsi.setGrid(generatePsiDM(globalSeed));
    setTimeout(() => {
      sceneKhr.setGrid(generateKhronon(globalSeed + 100));
      setTimeout(() => {
        sceneCDM.setGrid(generateCDM(globalSeed + 200));
      }, 30);
    }, 30);
  }, 30);
}

generateAll();

// ── Controls ────────────────────────────────────────────────────────────────
const thresholdSlider = document.getElementById("threshold-slider");
const thresholdValue = document.getElementById("threshold-value");
const opacitySlider = document.getElementById("opacity-slider");
const opacityValue = document.getElementById("opacity-value");
const btnRegenerate = document.getElementById("btn-regenerate");

thresholdSlider.addEventListener("input", () => {
  const t = parseInt(thresholdSlider.value, 10) / 100;
  thresholdValue.textContent = t.toFixed(2);
  scenes.forEach((s) => s.setThreshold(t));
});

opacitySlider.addEventListener("input", () => {
  const o = parseInt(opacitySlider.value, 10) / 100;
  opacityValue.textContent = o.toFixed(2);
  scenes.forEach((s) => s.setOpacity(o));
});

btnRegenerate.addEventListener("click", () => {
  globalSeed = Date.now();
  generateAll();
});

// ── Animation loop ──────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  for (const s of scenes) s.render();
}
animate();

// ── Handle resize ───────────────────────────────────────────────────────────
window.addEventListener("resize", () => {
  for (const s of scenes) s.resize();
});
