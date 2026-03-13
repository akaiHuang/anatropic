// ═══════════════════════════════════════════════════════════════════════════
//  Anatropic — Three Dark Matter Morphologies  (Three.js + WebGL)
//  Dual renderer: Volume ray marching + Voxel instanced cubes
//  Sheng-Kai Huang, 2026
// ═══════════════════════════════════════════════════════════════════════════

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── Constants ───────────────────────────────────────────────────────────────
const N = 64;
const N3 = N * N * N;
const HALF = N / 2;

// ── PRNG ────────────────────────────────────────────────────────────────────
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Inferno colormap ────────────────────────────────────────────────────────
const INFERNO = [
  [0.00, 0.001, 0.014, 0.071], [0.14, 0.119, 0.047, 0.284],
  [0.29, 0.320, 0.060, 0.430], [0.43, 0.530, 0.065, 0.380],
  [0.57, 0.735, 0.130, 0.240], [0.71, 0.890, 0.290, 0.100],
  [0.86, 0.975, 0.550, 0.040], [1.00, 0.993, 0.906, 0.144],
];
function infernoJS(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < INFERNO.length - 1; i++) {
    const [t0,r0,g0,b0] = INFERNO[i], [t1,r1,g1,b1] = INFERNO[i+1];
    if (t <= t1) { const f=(t-t0)/(t1-t0); return [r0+f*(r1-r0),g0+f*(g1-g0),b0+f*(b1-b0)]; }
  }
  return [INFERNO[7][1], INFERNO[7][2], INFERNO[7][3]];
}

// ── 3D Value noise ──────────────────────────────────────────────────────────
function hashNoise3D(ix,iy,iz) {
  let h = ix*374761393 + iy*668265263 + iz*1274126177;
  h = (h^(h>>13))*1103515245; h = h^(h>>16);
  return (h & 0x7fffffff) / 0x7fffffff;
}
function sstep(t) { return t*t*t*(t*(t*6-15)+10); }
function lerp(a,b,t) { return a+(b-a)*t; }
function valueNoise3D(x,y,z) {
  const ix=Math.floor(x), iy=Math.floor(y), iz=Math.floor(z);
  const fx=sstep(x-ix), fy=sstep(y-iy), fz=sstep(z-iz);
  return lerp(
    lerp(lerp(hashNoise3D(ix,iy,iz),hashNoise3D(ix+1,iy,iz),fx),
         lerp(hashNoise3D(ix,iy+1,iz),hashNoise3D(ix+1,iy+1,iz),fx),fy),
    lerp(lerp(hashNoise3D(ix,iy,iz+1),hashNoise3D(ix+1,iy,iz+1),fx),
         lerp(hashNoise3D(ix,iy+1,iz+1),hashNoise3D(ix+1,iy+1,iz+1),fx),fy),fz);
}
function fbm(x,y,z,oct=4,lac=2.0,g=0.5) {
  let v=0,a=1,f=1,m=0;
  for(let o=0;o<oct;o++){v+=a*valueNoise3D(x*f,y*f,z*f);m+=a;a*=g;f*=lac;}
  return v/m;
}

// ── Density generators ──────────────────────────────────────────────────────
function generatePsiDM(seed=42) {
  const grid=new Float32Array(N3), rng=mulberry32(seed), waves=[];
  for(let w=0;w<18;w++){
    const th=Math.acos(2*rng()-1),ph=2*Math.PI*rng(),km=2+rng()*6;
    waves.push({kx:km*Math.sin(th)*Math.cos(ph),ky:km*Math.sin(th)*Math.sin(ph),
      kz:km*Math.cos(th),amp:0.5+0.5*rng(),phase:2*Math.PI*rng()});
  }
  for(let iz=0;iz<N;iz++){const z=(iz-HALF)/N;
    for(let iy=0;iy<N;iy++){const y=(iy-HALF)/N;
      for(let ix=0;ix<N;ix++){const x=(ix-HALF)/N;
        let re=0,im=0;
        for(const w of waves){const a=2*Math.PI*(w.kx*x+w.ky*y+w.kz*z)+w.phase;re+=w.amp*Math.cos(a);im+=w.amp*Math.sin(a);}
        grid[iz*N*N+iy*N+ix]=re*re+im*im;
  }}} return normalizeGrid(grid);
}
function generateKhronon(seed=137) {
  const grid=new Float32Array(N3), rng=mulberry32(seed);
  const ox=rng()*1000,oy=rng()*1000,oz=rng()*1000, sc=3.5;
  for(let iz=0;iz<N;iz++){const z=iz/N*sc+oz;
    for(let iy=0;iy<N;iy++){const y=iy/N*sc+oy;
      for(let ix=0;ix<N;ix++){const x=ix/N*sc+ox;
        const n=fbm(x,y,z,4,2.2,0.45), ridge=1-Math.abs(2*n-1), fil=Math.pow(ridge,3);
        const det=fbm(x*2.5+50,y*2.5+50,z*2.5+50,2,2,0.3), comb=fil*(0.7+0.3*det);
        grid[iz*N*N+iy*N+ix]=comb>0.15?comb:comb*0.1;
  }}} return normalizeGrid(grid);
}
function generateCDM(seed=271) {
  const grid=new Float32Array(N3), rng=mulberry32(seed), halos=[];
  for(let h=0;h<15;h++) halos.push({cx:rng()*N,cy:rng()*N,cz:rng()*N,mass:0.4+rng()*0.6,sigma:1.5+rng()*3.5});
  for(let iz=0;iz<N;iz++) for(let iy=0;iy<N;iy++) for(let ix=0;ix<N;ix++){
    let rho=0;
    for(const h of halos){const dx=ix-h.cx,dy=iy-h.cy,dz=iz-h.cz;rho+=h.mass*Math.exp(-(dx*dx+dy*dy+dz*dz)/(2*h.sigma*h.sigma));}
    grid[iz*N*N+iy*N+ix]=rho;
  } return normalizeGrid(grid);
}
function normalizeGrid(g) {
  let mn=Infinity,mx=-Infinity;
  for(let i=0;i<g.length;i++){if(g[i]<mn)mn=g[i];if(g[i]>mx)mx=g[i];}
  const r=mx-mn||1; for(let i=0;i<g.length;i++)g[i]=(g[i]-mn)/r; return g;
}

// ═══════════════════════════════════════════════════════════════════════════
//  VOLUME RAY MARCHING RENDERER
// ═══════════════════════════════════════════════════════════════════════════

const volVS = `
varying vec3 vOrigin; varying vec3 vDirection; uniform vec3 uCamPos;
void main(){vDirection=position-uCamPos;vOrigin=uCamPos;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;

const volFS = `
precision highp float; precision highp sampler3D;
varying vec3 vOrigin; varying vec3 vDirection;
uniform sampler3D uVol; uniform float uThr; uniform float uOpa; uniform float uBri;
vec3 inferno(float t){t=clamp(t,0.,1.);
  vec3 c0=vec3(.001,.0,.014),c1=vec3(.119,.047,.284),c2=vec3(.53,.065,.38),
       c3=vec3(.735,.13,.24),c4=vec3(.89,.29,.1),c5=vec3(.975,.55,.04),c6=vec3(.993,.906,.144);
  if(t<.167)return mix(c0,c1,t/.167);if(t<.333)return mix(c1,c2,(t-.167)/.167);
  if(t<.5)return mix(c2,c3,(t-.333)/.167);if(t<.667)return mix(c3,c4,(t-.5)/.167);
  if(t<.833)return mix(c4,c5,(t-.667)/.167);return mix(c5,c6,(t-.833)/.167);}
vec2 boxHit(vec3 o,vec3 d){vec3 inv=1./d,t1=min(-o*inv,(1.-o)*inv),t2=max(-o*inv,(1.-o)*inv);
  return vec2(max(max(t1.x,t1.y),t1.z),min(min(t2.x,t2.y),t2.z));}
void main(){vec3 rd=normalize(vDirection),ro=vOrigin+.5;vec2 th=boxHit(ro,rd);
  if(th.x>th.y)discard;th.x=max(th.x,0.);float dt=1./80.;vec3 acc=vec3(0.);float aa=0.;
  for(float t=th.x;t<th.y;t+=dt){vec3 p=ro+rd*t;float d=texture(uVol,p).r;
    if(d>uThr){float v=clamp((d-uThr)/(1.-uThr+.001),0.,1.);vec3 c=inferno(v)*uBri;
      float a=v*uOpa*dt*15.;acc+=(1.-aa)*a*c;aa+=(1.-aa)*a;if(aa>.98)break;}}
  gl_FragColor=vec4(acc+(1.-aa)*vec3(.02),1.);}`;

// ═══════════════════════════════════════════════════════════════════════════
//  VOXEL INSTANCED RENDERER
// ═══════════════════════════════════════════════════════════════════════════

const voxVS = `
attribute vec3 instanceOffset; attribute vec3 instanceColor;
varying vec3 vColor; varying vec3 vNormal;
void main(){vColor=instanceColor;vNormal=normalMatrix*normal;
  gl_Position=projectionMatrix*modelViewMatrix*vec4(position+instanceOffset,1.);}`;

const voxFS = `
uniform float uOpa; uniform vec3 uLight;
varying vec3 vColor; varying vec3 vNormal;
void main(){float d=max(dot(normalize(vNormal),uLight),0.);
  gl_FragColor=vec4(vColor*(.45+.55*d),uOpa);}`;

// ═══════════════════════════════════════════════════════════════════════════
//  DUAL-MODE SCENE
// ═══════════════════════════════════════════════════════════════════════════

class DualScene {
  constructor(container) {
    this.container = container;
    this.grid = null;
    this.mode = "volume"; // "volume" | "voxel"
    this.threshold = 0.30;
    this.opacity = 0.85;

    this.overlay = document.createElement("div");
    this.overlay.className = "loading-overlay";
    this.overlay.textContent = "Generating...";
    container.appendChild(this.overlay);

    const w = container.clientWidth || 400, h = container.clientHeight || 400;
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(0x050505);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, w/h, 0.01, 20);
    this.camera.position.set(1.6, 1.2, 1.6);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.6;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 5;

    // Bounding box
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1));
    this.bbox = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({color:0x222222}));
    this.scene.add(this.bbox);

    // Lighting for voxel mode
    this.ambLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.dirLight.position.set(3,5,4);

    // Meshes (created on setGrid)
    this.volMesh = null;
    this.volTex = null;
    this.voxMesh = null;

    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
  }

  resize() {
    const w=this.container.clientWidth, h=this.container.clientHeight;
    if(!w||!h) return;
    this.camera.aspect = w/h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w,h);
  }

  setGrid(grid) {
    this.grid = grid;
    this._buildVolume();
    this._buildVoxels();
    this._applyMode();
    this.overlay.classList.add("hidden");
  }

  _buildVolume() {
    if (this.volMesh) { this.scene.remove(this.volMesh); this.volMesh.geometry.dispose(); this.volMesh.material.dispose(); }
    if (this.volTex) this.volTex.dispose();

    const data = new Uint8Array(N3);
    for (let i=0;i<N3;i++) data[i] = Math.floor(this.grid[i]*255);
    this.volTex = new THREE.Data3DTexture(data,N,N,N);
    this.volTex.format = THREE.RedFormat;
    this.volTex.type = THREE.UnsignedByteType;
    this.volTex.minFilter = THREE.LinearFilter;
    this.volTex.magFilter = THREE.LinearFilter;
    this.volTex.wrapS = this.volTex.wrapT = this.volTex.wrapR = THREE.ClampToEdgeWrapping;
    this.volTex.needsUpdate = true;

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uVol:{value:this.volTex}, uThr:{value:this.threshold},
        uOpa:{value:this.opacity}, uBri:{value:2.0}, uCamPos:{value:new THREE.Vector3()},
      },
      vertexShader: volVS, fragmentShader: volFS, side: THREE.BackSide,
    });
    this.volMesh = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), mat);
  }

  _buildVoxels() {
    if (this.voxMesh) { this.scene.remove(this.voxMesh); this.voxMesh.geometry.dispose(); this.voxMesh.material.dispose(); }

    const pos=[], cols=[];
    for(let iz=0;iz<N;iz++) for(let iy=0;iy<N;iy++) for(let ix=0;ix<N;ix++){
      const val = this.grid[iz*N*N+iy*N+ix];
      if(val < this.threshold) continue;
      pos.push(ix/N-0.5, iy/N-0.5, iz/N-0.5);
      const t=(val-this.threshold)/(1-this.threshold+1e-8);
      const [r,g,b] = infernoJS(Math.min(t,1));
      cols.push(r,g,b);
    }
    const count = pos.length/3;
    if (!count) return;

    const sz = 1.0/N;
    const baseGeo = new THREE.BoxGeometry(sz,sz,sz);
    const instGeo = new THREE.InstancedBufferGeometry();
    instGeo.index = baseGeo.index;
    instGeo.attributes.position = baseGeo.attributes.position;
    instGeo.attributes.normal = baseGeo.attributes.normal;
    instGeo.setAttribute("instanceOffset", new THREE.InstancedBufferAttribute(new Float32Array(pos),3));
    instGeo.setAttribute("instanceColor", new THREE.InstancedBufferAttribute(new Float32Array(cols),3));
    instGeo.instanceCount = count;

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      uniforms: { uOpa:{value:this.opacity}, uLight:{value:new THREE.Vector3(0.4,0.7,0.5).normalize()} },
      vertexShader: voxVS, fragmentShader: voxFS,
    });
    this.voxMesh = new THREE.Mesh(instGeo, mat);
  }

  setMode(mode) {
    this.mode = mode;
    this._applyMode();
  }

  _applyMode() {
    // Remove both
    if (this.volMesh) this.scene.remove(this.volMesh);
    if (this.voxMesh) this.scene.remove(this.voxMesh);
    this.scene.remove(this.ambLight);
    this.scene.remove(this.dirLight);

    if (this.mode === "volume" && this.volMesh) {
      this.scene.add(this.volMesh);
    } else if (this.mode === "voxel" && this.voxMesh) {
      this.scene.add(this.voxMesh);
      this.scene.add(this.ambLight);
      this.scene.add(this.dirLight);
    }
  }

  setThreshold(t) {
    this.threshold = t;
    if (this.volMesh) this.volMesh.material.uniforms.uThr.value = t;
    // Rebuild voxels for new threshold
    if (this.grid) { this._buildVoxels(); this._applyMode(); }
  }

  setOpacity(o) {
    this.opacity = o;
    if (this.volMesh) this.volMesh.material.uniforms.uOpa.value = o;
    if (this.voxMesh) this.voxMesh.material.uniforms.uOpa.value = o;
  }

  render() {
    this.controls.update();
    if (this.volMesh) this.volMesh.material.uniforms.uCamPos.value.copy(this.camera.position);
    this.renderer.render(this.scene, this.camera);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════

const sPsi = new DualScene(document.getElementById("container-psiDM"));
const sKhr = new DualScene(document.getElementById("container-khronon"));
const sCDM = new DualScene(document.getElementById("container-CDM"));
const scenes = [sPsi, sKhr, sCDM];

let globalSeed = Date.now();

function generateAll() {
  scenes.forEach(s => { s.overlay.classList.remove("hidden"); s.overlay.textContent = "Generating..."; });
  setTimeout(() => { sPsi.setGrid(generatePsiDM(globalSeed));
    setTimeout(() => { sKhr.setGrid(generateKhronon(globalSeed+100));
      setTimeout(() => { sCDM.setGrid(generateCDM(globalSeed+200)); }, 30);
    }, 30);
  }, 30);
}
generateAll();

// ── Tab switching ───────────────────────────────────────────────────────────
const tabs = document.querySelectorAll(".mode-tab");
const caveats = document.querySelectorAll(".caveat-text");

tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    const mode = tab.dataset.mode;
    tabs.forEach(t => t.classList.toggle("active", t === tab));
    caveats.forEach(c => c.classList.toggle("active", c.id === `caveat-${mode}`));
    scenes.forEach(s => s.setMode(mode));
  });
});

// ── Controls ────────────────────────────────────────────────────────────────
const thrSlider = document.getElementById("threshold-slider");
const thrVal = document.getElementById("threshold-value");
const opaSlider = document.getElementById("opacity-slider");
const opaVal = document.getElementById("opacity-value");

thrSlider.addEventListener("input", () => {
  const t = parseInt(thrSlider.value,10)/100;
  thrVal.textContent = t.toFixed(2);
  scenes.forEach(s => s.setThreshold(t));
});
opaSlider.addEventListener("input", () => {
  const o = parseInt(opaSlider.value,10)/100;
  opaVal.textContent = o.toFixed(2);
  scenes.forEach(s => s.setOpacity(o));
});
document.getElementById("btn-regenerate").addEventListener("click", () => {
  globalSeed = Date.now();
  generateAll();
});

// ── Animate ─────────────────────────────────────────────────────────────────
(function animate() { requestAnimationFrame(animate); for(const s of scenes) s.render(); })();
window.addEventListener("resize", () => { for(const s of scenes) s.resize(); });
