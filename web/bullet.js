/**
 * bullet.js — Bullet Cluster: ΛCDM vs Khronon τ visualisation
 *
 * Procedural 3D scene showing two colliding galaxy clusters with
 * gas (center), galaxies (left/right), and lensing mass contours.
 * Three modes: ΛCDM, Khronon, Observed.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/* ================================================================
   0. CONSTANTS & STATE
   ================================================================ */

const MODES = ["lcdm", "khronon", "observed"];
let currentMode = "lcdm";

// Cluster geometry
const CLUSTER_OFFSET = 3.5;     // x-offset of galaxy clusters from center
const GAS_SCALE = { x: 2.2, y: 1.1, z: 1.1 };  // elongated gas ellipsoid
const N_GALAXIES = 80;          // per cluster
const GALAXY_SPREAD = 1.2;      // Gaussian sigma

// Contour shell params
const CONTOUR_INNER_R = 0.6;
const CONTOUR_OUTER_R = 2.0;
const CONTOUR_SEGMENTS = 64;

// Ray grid
const RAY_GRID = 7;             // 7x7 grid of light rays
const RAY_LENGTH = 14;

/* ================================================================
   1. SCENE SETUP
   ================================================================ */

const container = document.getElementById("bullet-container");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x050505);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, 2, 0.1, 100);
camera.position.set(0, 4, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = false;
controls.autoRotateSpeed = 0.6;
controls.minDistance = 4;
controls.maxDistance = 25;
controls.target.set(0, 0, 0);

// Lights
scene.add(new THREE.AmbientLight(0x444466, 1.0));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(5, 8, 6);
scene.add(dirLight);

/* ================================================================
   2. BUILD SCENE ELEMENTS
   ================================================================ */

// ── 2a. Gas cloud (center) ──────────────────────────────────────

const gasGroup = new THREE.Group();
scene.add(gasGroup);

function createGasCloud() {
  // Main ellipsoid shell
  const gasGeom = new THREE.SphereGeometry(1, 48, 32);
  const gasMat = new THREE.MeshPhongMaterial({
    color: 0xff4411,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const gasMesh = new THREE.Mesh(gasGeom, gasMat);
  gasMesh.scale.set(GAS_SCALE.x, GAS_SCALE.y, GAS_SCALE.z);
  gasGroup.add(gasMesh);

  // Inner brighter core
  const coreGeom = new THREE.SphereGeometry(1, 32, 24);
  const coreMat = new THREE.MeshPhongMaterial({
    color: 0xff7733,
    transparent: true,
    opacity: 0.25,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const coreMesh = new THREE.Mesh(coreGeom, coreMat);
  coreMesh.scale.set(GAS_SCALE.x * 0.6, GAS_SCALE.y * 0.6, GAS_SCALE.z * 0.6);
  gasGroup.add(coreMesh);

  // Gas particle cloud for volumetric feel
  const particleCount = 600;
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    // Gaussian in ellipsoidal coordinates
    const u = gaussianRandom() * GAS_SCALE.x * 0.8;
    const v = gaussianRandom() * GAS_SCALE.y * 0.8;
    const w = gaussianRandom() * GAS_SCALE.z * 0.8;
    positions[i * 3]     = u;
    positions[i * 3 + 1] = v;
    positions[i * 3 + 2] = w;

    // Color: orange-red gradient
    const r2 = (u / GAS_SCALE.x) ** 2 + (v / GAS_SCALE.y) ** 2 + (w / GAS_SCALE.z) ** 2;
    const t = Math.min(r2, 1);
    colors[i * 3]     = 1.0;
    colors[i * 3 + 1] = 0.3 + 0.4 * (1 - t);
    colors[i * 3 + 2] = 0.1 * (1 - t);

    sizes[i] = 3.0 + Math.random() * 4.0;
  }

  const particleGeom = new THREE.BufferGeometry();
  particleGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  particleGeom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  particleGeom.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

  const particleMat = new THREE.PointsMaterial({
    size: 0.06,
    vertexColors: true,
    transparent: true,
    opacity: 0.6,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });

  gasGroup.add(new THREE.Points(particleGeom, particleMat));
}

createGasCloud();

// ── 2b. Galaxy clusters (left & right) ─────────────────────────

const galaxyGroups = [new THREE.Group(), new THREE.Group()];
scene.add(galaxyGroups[0]);
scene.add(galaxyGroups[1]);

function createGalaxies(group, centerX) {
  const positions = new Float32Array(N_GALAXIES * 3);
  const colors = new Float32Array(N_GALAXIES * 3);

  for (let i = 0; i < N_GALAXIES; i++) {
    positions[i * 3]     = centerX + gaussianRandom() * GALAXY_SPREAD;
    positions[i * 3 + 1] = gaussianRandom() * GALAXY_SPREAD * 0.8;
    positions[i * 3 + 2] = gaussianRandom() * GALAXY_SPREAD * 0.8;

    // Slight color variation: warm white to pale yellow
    const warmth = 0.85 + Math.random() * 0.15;
    colors[i * 3]     = 1.0;
    colors[i * 3 + 1] = warmth;
    colors[i * 3 + 2] = warmth * 0.85;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.1,
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });

  group.add(new THREE.Points(geom, mat));

  // Add a few brighter "BCG" (brightest cluster galaxy) points
  const bcgGeom = new THREE.SphereGeometry(0.06, 8, 8);
  const bcgMat = new THREE.MeshBasicMaterial({ color: 0xffeedd });
  for (let i = 0; i < 3; i++) {
    const bcg = new THREE.Mesh(bcgGeom, bcgMat);
    bcg.position.set(
      centerX + gaussianRandom() * 0.4,
      gaussianRandom() * 0.3,
      gaussianRandom() * 0.3
    );
    group.add(bcg);
  }
}

createGalaxies(galaxyGroups[0], -CLUSTER_OFFSET);
createGalaxies(galaxyGroups[1],  CLUSTER_OFFSET);

// ── 2c. Lensing mass contours ──────────────────────────────────

// We create three sets of contours: one per mode
const contourSets = {
  lcdm: { left: new THREE.Group(), right: new THREE.Group() },
  khronon: { left: new THREE.Group(), right: new THREE.Group() },
  observed: { left: new THREE.Group(), right: new THREE.Group() },
};

function nfwProfile(r, rs) {
  // NFW: rho ~ 1 / (r/rs * (1 + r/rs)^2)
  const x = r / rs;
  if (x < 0.01) return 1.0;
  return 1.0 / (x * (1 + x) * (1 + x));
}

function exponentialProfile(r, rs) {
  // Khronon: exp(-rs/r) — exponential metric
  if (r < 0.01) return 0.0;
  return Math.exp(-rs / r);
}

function createContourShells(group, centerX, profileFn, rs, color, nShells) {
  const radii = [];
  for (let i = 0; i < nShells; i++) {
    radii.push(CONTOUR_INNER_R + (CONTOUR_OUTER_R - CONTOUR_INNER_R) * (i / (nShells - 1)));
  }

  radii.forEach((r, idx) => {
    const val = profileFn(r, rs);
    const opacity = 0.08 + 0.28 * val;

    // Sphere shell
    const geom = new THREE.SphereGeometry(r, CONTOUR_SEGMENTS, CONTOUR_SEGMENTS / 2);
    const mat = new THREE.MeshPhongMaterial({
      color: color,
      transparent: true,
      opacity: Math.min(opacity, 0.35),
      depthWrite: false,
      side: THREE.DoubleSide,
      wireframe: false,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(centerX, 0, 0);
    group.add(mesh);

    // Wireframe contour ring (equatorial)
    const ringGeom = new THREE.RingGeometry(r - 0.01, r + 0.01, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.15 + 0.3 * val,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.position.set(centerX, 0, 0);
    ring.rotation.x = Math.PI / 2;
    group.add(ring);
  });
}

// ΛCDM: NFW halos — blue-cyan
const CDM_COLOR = 0x3399ff;
createContourShells(contourSets.lcdm.left,  -CLUSTER_OFFSET, nfwProfile,         0.8, CDM_COLOR, 5);
createContourShells(contourSets.lcdm.right,  CLUSTER_OFFSET, nfwProfile,         0.8, CDM_COLOR, 5);

// Khronon: exponential metric — cyan-teal
const KHRONON_COLOR = 0x22ccbb;
createContourShells(contourSets.khronon.left,  -CLUSTER_OFFSET, exponentialProfile, 0.6, KHRONON_COLOR, 5);
createContourShells(contourSets.khronon.right,  CLUSTER_OFFSET, exponentialProfile, 0.6, KHRONON_COLOR, 5);

// Observed: reconstruction contours — blue (similar to CDM positioning)
const OBS_COLOR = 0x5588ff;
createContourShells(contourSets.observed.left,  -CLUSTER_OFFSET, (r, rs) => {
  // Observed profile: empirical, between NFW and exponential
  const x = r / rs;
  if (x < 0.01) return 0.8;
  return 0.8 / (1 + x * x);
}, 0.7, OBS_COLOR, 5);
createContourShells(contourSets.observed.right,  CLUSTER_OFFSET, (r, rs) => {
  const x = r / rs;
  if (x < 0.01) return 0.8;
  return 0.8 / (1 + x * x);
}, 0.7, OBS_COLOR, 5);

// Add all to scene
for (const mode of MODES) {
  scene.add(contourSets[mode].left);
  scene.add(contourSets[mode].right);
}

// ── 2d. Photon light rays ──────────────────────────────────────

const raysGroup = new THREE.Group();
scene.add(raysGroup);

function computeDeflection(x, y, mode) {
  // Simplified lensing deflection angle
  // Each cluster deflects light toward its center
  let dx = 0, dy = 0;

  const clusters = [-CLUSTER_OFFSET, CLUSTER_OFFSET];
  for (const cx of clusters) {
    const relX = x - cx;
    const relY = y;
    const r = Math.sqrt(relX * relX + relY * relY);
    if (r < 0.15) continue;

    let strength;
    if (mode === "khronon") {
      // Exponential: sharper deflection close, faster falloff
      strength = 0.5 * Math.exp(-0.6 / Math.max(r, 0.2)) / r;
    } else {
      // NFW: broader deflection, slower falloff
      const x_nfw = r / 0.8;
      strength = 0.4 * Math.log(1 + x_nfw) / (r * r + 0.3);
    }

    dx -= relX / r * strength;
    dy -= relY / r * strength;
  }
  return { dx, dy };
}

function createRays(mode) {
  // Clear existing
  while (raysGroup.children.length > 0) {
    const child = raysGroup.children[0];
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
    raysGroup.remove(child);
  }

  const halfGrid = (RAY_GRID - 1) / 2;
  const spacing = 1.2;

  for (let ix = 0; ix < RAY_GRID; ix++) {
    for (let iy = 0; iy < RAY_GRID; iy++) {
      const x0 = (ix - halfGrid) * spacing;
      const y0 = (iy - halfGrid) * spacing;

      const deflection = computeDeflection(x0, y0, mode);
      const deflAngle = Math.sqrt(deflection.dx ** 2 + deflection.dy ** 2);

      // Build a bent ray: come from behind, deflect in middle, exit in front
      const points = [];
      const zBack = -RAY_LENGTH / 2;
      const zFront = RAY_LENGTH / 2;
      const nSegments = 20;

      for (let s = 0; s <= nSegments; s++) {
        const t = s / nSegments;
        const z = zBack + (zFront - zBack) * t;

        // Smooth deflection ramp: sigmoid-like transition
        const ramp = 1 / (1 + Math.exp(-8 * (t - 0.5)));

        const px = x0 + deflection.dx * ramp * 2.0;
        const py = y0 + deflection.dy * ramp * 2.0;

        points.push(new THREE.Vector3(px, py, z));
      }

      const curve = new THREE.CatmullRomCurve3(points);
      const curvePoints = curve.getPoints(40);
      const lineGeom = new THREE.BufferGeometry().setFromPoints(curvePoints);

      // Color by deflection angle
      const hue = 0.14 - Math.min(deflAngle * 0.3, 0.14); // yellow -> red
      const saturation = 0.6 + Math.min(deflAngle * 0.5, 0.4);
      const color = new THREE.Color().setHSL(hue, saturation, 0.55);

      const lineMat = new THREE.LineBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.3 + Math.min(deflAngle * 0.8, 0.5),
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });

      raysGroup.add(new THREE.Line(lineGeom, lineMat));
    }
  }
}

createRays(currentMode);

// ── 2e. Collision direction indicator ──────────────────────────

// Arrow showing collision axis
const arrowDir = new THREE.Vector3(1, 0, 0);
const arrowOrigin = new THREE.Vector3(-6, -2.2, 0);
const arrowHelper = new THREE.ArrowHelper(arrowDir, arrowOrigin, 4, 0x555555, 0.3, 0.15);
scene.add(arrowHelper);

// Reverse arrow
const arrowDir2 = new THREE.Vector3(-1, 0, 0);
const arrowOrigin2 = new THREE.Vector3(6, -2.2, 0);
const arrowHelper2 = new THREE.ArrowHelper(arrowDir2, arrowOrigin2, 4, 0x555555, 0.3, 0.15);
scene.add(arrowHelper2);

// ── 2f. Background reference grid ─────────────────────────────

const gridHelper = new THREE.GridHelper(16, 16, 0x111111, 0x0a0a0a);
gridHelper.position.y = -3;
scene.add(gridHelper);

/* ================================================================
   3. MODE SWITCHING
   ================================================================ */

function setMode(mode) {
  currentMode = mode;

  // Toggle contour visibility
  for (const m of MODES) {
    const visible = m === mode;
    contourSets[m].left.visible = visible;
    contourSets[m].right.visible = visible;
  }

  // Rebuild rays
  createRays(mode);

  // Toggle info text
  for (const m of MODES) {
    const el = document.getElementById(`info-${m}`);
    if (el) el.classList.toggle("active", m === mode);
  }

  // Toggle button active class
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
}

// Initial
setMode("lcdm");

// Button listeners
document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

/* ================================================================
   4. CONTROLS BINDING
   ================================================================ */

// Gas opacity
const gasOpacitySlider = document.getElementById("gas-opacity");
const gasOpaVal = document.getElementById("gas-opa-val");
gasOpacitySlider.addEventListener("input", () => {
  const val = parseInt(gasOpacitySlider.value) / 100;
  gasOpaVal.textContent = val.toFixed(2);
  gasGroup.traverse(child => {
    if (child.material && child.material.opacity !== undefined) {
      if (child instanceof THREE.Points) {
        child.material.opacity = val * 0.8;
      } else {
        child.material.opacity = val * 0.7;
      }
    }
  });
});

// Contour opacity
const contourOpacitySlider = document.getElementById("contour-opacity");
const contourOpaVal = document.getElementById("contour-opa-val");
contourOpacitySlider.addEventListener("input", () => {
  const val = parseInt(contourOpacitySlider.value) / 100;
  contourOpaVal.textContent = val.toFixed(2);
  for (const mode of MODES) {
    [contourSets[mode].left, contourSets[mode].right].forEach(grp => {
      grp.traverse(child => {
        if (child.material && child.material.opacity !== undefined) {
          // Scale original opacity proportionally
          child.material.opacity = Math.min(child.material.opacity, val);
        }
      });
    });
  }
});

// Show/hide rays
const showRaysCheckbox = document.getElementById("show-rays");
showRaysCheckbox.addEventListener("change", () => {
  raysGroup.visible = showRaysCheckbox.checked;
});

// Auto-rotate
const autoRotateCheckbox = document.getElementById("auto-rotate");
autoRotateCheckbox.addEventListener("change", () => {
  controls.autoRotate = autoRotateCheckbox.checked;
});

/* ================================================================
   5. ANNOTATIONS OVERLAY
   ================================================================ */

const annotationOverlay = document.getElementById("annotation-overlay");

const annotations = [
  { label: "Hot X-ray gas", pos: new THREE.Vector3(0, 1.4, 0), color: "#ff6633" },
  { label: "Cluster 1 (galaxies)", pos: new THREE.Vector3(-CLUSTER_OFFSET, 1.8, 0), color: "#dddddd" },
  { label: "Cluster 2 (galaxies)", pos: new THREE.Vector3(CLUSTER_OFFSET, 1.8, 0), color: "#dddddd" },
  { label: "Collision axis", pos: new THREE.Vector3(0, -2.2, 0), color: "#666666" },
];

const annotationEls = annotations.map(a => {
  const el = document.createElement("div");
  el.className = "annotation-label";
  el.textContent = a.label;
  el.style.color = a.color;
  annotationOverlay.appendChild(el);
  return { el, pos: a.pos };
});

function updateAnnotations() {
  const rect = renderer.domElement.getBoundingClientRect();
  const widthHalf = rect.width / 2;
  const heightHalf = rect.height / 2;

  annotationEls.forEach(({ el, pos }) => {
    const projected = pos.clone().project(camera);
    if (projected.z > 1) {
      el.style.display = "none";
      return;
    }
    el.style.display = "block";
    const x = (projected.x * widthHalf) + widthHalf;
    const y = -(projected.y * heightHalf) + heightHalf;
    el.style.left = x + "px";
    el.style.top = y + "px";
    el.style.transform = "translate(-50%, -100%)";
  });
}

/* ================================================================
   6. ANIMATION LOOP
   ================================================================ */

let time = 0;

function animate() {
  requestAnimationFrame(animate);
  time += 0.005;

  controls.update();

  // Gentle gas pulsation
  const pulse = 1.0 + 0.02 * Math.sin(time * 2);
  gasGroup.children.forEach(child => {
    if (child instanceof THREE.Mesh) {
      child.scale.x = child.userData.baseScaleX !== undefined
        ? child.userData.baseScaleX * pulse
        : child.scale.x;
    }
  });

  // Store base scales on first frame
  if (time < 0.01) {
    gasGroup.children.forEach(child => {
      if (child instanceof THREE.Mesh) {
        child.userData.baseScaleX = child.scale.x;
        child.userData.baseScaleY = child.scale.y;
        child.userData.baseScaleZ = child.scale.z;
      }
    });
  }

  updateAnnotations();
  renderer.render(scene, camera);
}

/* ================================================================
   7. RESIZE HANDLING
   ================================================================ */

function onResize() {
  const rect = container.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

window.addEventListener("resize", onResize);

// Use ResizeObserver for container size changes
const resizeObserver = new ResizeObserver(onResize);
resizeObserver.observe(container);

onResize();
animate();

/* ================================================================
   UTILITIES
   ================================================================ */

function gaussianRandom() {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
