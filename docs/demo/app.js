/* CANOPI Studio — front end. Talks to server.py; all physics happens there. */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

/* ── state ────────────────────────────────────────────────────────────── */
const S = {
  mode: "jbs",
  // stand
  p_lai: 0.5, month: "Jul", max_leaves: 1200000, seed: 42,
  scene_xy: 4, scene_z: 1, lai_eff: 2, leaf_radius: 0.02, radius_std_mul: 0.3, ala: "UNI", pixel_m: 2,
  // spherical crowns (the paper's heterogeneous stand)
  n_spheres: 15, r_sph: 10, lai_per_sphere: 5,
  // optics
  band: "O08", rho: 0.0632, tau: 0.0213, rsoil: 0.0274,
  sun_zenith_deg: 41, sun_azimuth_deg: 180,
  view_zenith_deg: 0, view_azimuth_deg: 0,
  obs_zenith_max_deg: 60, obs_step_deg: 5,
  // sampling
  img: 32, n_spp: 8, n_photons: 200000, max_depth: 99,
  // inversion
  max_iter: 12, lm_lambda: 0.001, k_max: 3, tol: 0.0001,
  species_shadow: true, uniform: true, free: [], p_true: {}, p_init: {}, p_true_scalar: 0.5, p_init_scalar: 0.25,
};
let CAPS = null, STAND = null, RENDER = null, EST = "opt", TIMING = {};

/* ── tiny helpers ─────────────────────────────────────────────────────── */
const fmt = (x, n = 4) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(n);
const sci = (x) => (x === null || x === undefined) ? "—" : Number(x).toExponential(2);
const human = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + " M" : n >= 1e3 ? (n / 1e3).toFixed(0) + " k" : String(n);

async function api(path, body) {
  const opt = body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (j && j.error) throw new Error(j.error);
  return j;
}

function watchJob(id, onTick) {
  return new Promise((resolve, reject) => {
    const t = setInterval(async () => {
      let j;
      try { j = await api("/api/job/" + id); } catch (e) { clearInterval(t); reject(e); return; }
      onTick(j);
      if (j.status === "done") { clearInterval(t); resolve(j); }
      else if (j.status === "error") { clearInterval(t); reject(new Error(j.error)); }
      else if (j.status === "cancelled") { clearInterval(t); reject(new Error("cancelled")); }
    }, 350);
  });
}

function bar(sel, frac, busy) {
  const el = $(sel);
  el.classList.toggle("busy", !!busy);
  el.querySelector("i").style.width = (busy && frac < 0.02 ? 100 : frac * 100).toFixed(1) + "%";
}
function alertBox(sel, msg, kind) {
  const el = $(sel);
  if (!msg) { el.hidden = true; return; }
  el.hidden = false;
  el.className = "alert " + (kind || "bad");
  el.querySelector("span:last-child").textContent = msg;
}

/* ── colour ramps (scientific, deliberately theme-independent) ─────────── */
const BRF_STOPS = ["#2E2417", "#6B5228", "#AE8F3E", "#DCD292"];
const DIV_STOPS = ["#0E5C63", "#6FA8AB", "#ECEDE6", "#D98BBE", "#B23B86"];
const hex2rgb = (c) => [1, 3, 5].map((i) => parseInt(c.substr(i, 2), 16));
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function rampBar(id, stops) { $(id).style.background = "linear-gradient(90deg," + stops.join(",") + ")"; }

function paint(canvasId, grid, stops, diverging) {
  const rows = grid.length, cols = grid[0].length;
  const cv = $(canvasId); cv.width = cols; cv.height = rows;
  const g = cv.getContext("2d"), im = g.createImageData(cols, rows), d = im.data;
  let lo = Infinity, hi = -Infinity;
  for (const row of grid) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; }
  const span = (hi - lo) || 1;
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const c = ramp(stops, (grid[y][x] - lo) / span), k = (y * cols + x) * 4;
    d[k] = c[0]; d[k + 1] = c[1]; d[k + 2] = c[2]; d[k + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return [lo, hi];
}

/* ── generic input binding ────────────────────────────────────────────── */
function bindInputs() {
  $$("[data-k]").forEach((el) => {
    const k = el.dataset.k;
    if (S[k] !== undefined) el.value = S[k];
    el.addEventListener("change", () => {
      S[k] = (el.type === "number") ? Number(el.value) : el.value;
      if (k === "seed") $("#tb-seed").textContent = S.seed;
      if (["rho", "tau", "rsoil"].includes(k)) updateOmega();
      if (["sun_zenith_deg", "sun_azimuth_deg"].includes(k)) drawSunDial();
      if (["img", "n_spp", "n_photons", "pixel_m", "scene_xy"].includes(k)) updateCostNote();
      syncRail(); checkStand();
    });
  });
}

const MODE_LABEL = { homogeneous: "homogeneous slab", spheres: "spherical crowns",
                     jbs: "RAMI-5 HET09_JBS_SUM" };

// Each stand comes with the geometry, optics and illumination it was published with.
function applyModeDefaults(mode) {
  if (mode === "spheres") {
    Object.assign(S, { scene_xy: 100, scene_z: 40, leaf_radius: 0.05, max_leaves: 0,
                       rho: 0.5, tau: 0.5, rsoil: 0.1, pixel_m: 2,
                       sun_zenith_deg: 50, sun_azimuth_deg: 180 });
  } else if (mode === "homogeneous") {
    Object.assign(S, { scene_xy: 4, scene_z: 1, leaf_radius: 0.02, lai_eff: 2, pixel_m: 0.1,
                       sun_zenith_deg: 41, sun_azimuth_deg: 180 });
  } else {
    Object.assign(S, { max_leaves: 1200000, pixel_m: 2, uniform: false,
                       sun_zenith_deg: 41, sun_azimuth_deg: 180 });
  }
  $$("#seg-retrieve button[data-uniform]").forEach((b) =>
    b.setAttribute("aria-pressed", String((b.dataset.uniform === "1") === S.uniform)));
  $$("[data-k]").forEach((el) => { if (S[el.dataset.k] !== undefined) el.value = S[el.dataset.k]; });
  updateOmega(); drawSunDial();
}

function applyMode() {
  $$("[data-only]").forEach((el) => { el.hidden = !el.dataset.only.split(",").includes(S.mode); });
  $$("#seg-mode button[data-mode]").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.mode === S.mode)));
  $("#img-sub").textContent = `${imagePixels()} × ${imagePixels()} px · ${S.pixel_m} m/px`;
  $("#stand-src").textContent = MODE_LABEL[S.mode];
  syncRail(); updateCostNote(); buildFreeTable();
}

// Does the stand these settings describe already exist on the server? Anything
// that changes the geometry — radius_std_mul included — makes the built one stale.
let staleTimer = 0;
function checkStand() {
  clearTimeout(staleTimer);
  staleTimer = setTimeout(async () => {
    try {
      const s = await api("/api/stand_status", standParams());
      const chip = $("#tb-stand"), note = $("#stale-note");
      if (s.cached) {
        chip.className = "chip ok";
        chip.innerHTML = `stand <b>${human(s.n_leaf)}</b> leaves`;
        note.hidden = true;
      } else if (STAND) {
        chip.className = "chip warn";
        chip.textContent = "stand stale — rebuild";
        note.hidden = false;
        note.querySelector("span:last-child").innerHTML =
          `<b>These settings describe a different stand from the one in memory.</b> Renders and
           derivatives still come from the stand built earlier. Press <b>Build stand</b> to apply
           the change.`;
      } else {
        chip.className = "chip";
        chip.textContent = "no stand built";
        note.hidden = true;
      }
    } catch (e) { /* leave the chip as it is */ }
  }, 250);
}

function syncRail() {
  $("#rail-optics").textContent = `${S.band} · θs ${S.sun_zenith_deg}°`;
  $("#tb-path").textContent =
    S.mode === "jbs" ? `RAMI-5 JBS · ${S.month} · ${S.band}`
    : S.mode === "spheres" ? `${S.n_spheres} crowns · r ${S.r_sph} m · LAI ${S.lai_per_sphere}/crown`
    : `homogeneous · LAI_eff ${S.lai_eff} · ${S.band}`;
  $("#ro-band").textContent = S.band;
}

/* ── P_LAI slider ─────────────────────────────────────────────────────── */
function updateP() {
  const p = S.p_lai;
  $("#ro-p").textContent = p.toFixed(2);
  const el = $("#p-alert"), txt = el.querySelector("span:last-child");
  if (p > 0.9) {
    el.className = "alert warn";
    txt.innerHTML = `<b>Few fictive leaves left at P_LAI = ${p.toFixed(2)}.</b> The structural
      estimator scores the inactive contacts, so ∂BRF/∂LAI loses precision and tends to zero as
      P → 1. Keep P_LAI below about 0.9 when you need the derivative.`;
  } else {
    el.className = "alert info";
    const lm = STAND ? ` LAI_max = ${STAND.lai_max.toFixed(3)}.` : "";
    txt.innerHTML = `<b>LAI_eff = P_LAI × LAI_max.</b>${lm} ${Math.round((1 - p) * 100)} % of leaves
      are fictive on any given path — those are the contacts the structural estimator scores.`;
  }
}

/* ── optics ───────────────────────────────────────────────────────────── */
function updateOmega() {
  const w = S.rho + S.tau;
  $("#ro-omega").textContent = w.toFixed(4);
  if (w > 0.9 && S.max_depth < 99) {
    alertBox("#omega-alert",
      `Nearly conservative scattering (ω = ${w.toFixed(3)}): paths lose almost no energy, so ` +
      `max_depth = ${S.max_depth} truncates them and biases the BRF low. Use 99.`, "warn");
  } else alertBox("#omega-alert", "");
}

function buildBandSeg() {
  const seg = $("#seg-band"); seg.innerHTML = "";
  Object.entries(CAPS.bands).forEach(([name, bp]) => {
    const b = document.createElement("button");
    b.textContent = `${name} · ${Math.round(bp.wavelength_nm || 0)} nm`;
    b.setAttribute("aria-pressed", String(name === S.band));
    b.addEventListener("click", () => {
      S.band = name; S.rho = bp.rho; S.tau = bp.tau; S.rsoil = bp.rsoil;
      $$("[data-k=rho]").forEach((e) => (e.value = S.rho));
      $$("[data-k=tau]").forEach((e) => (e.value = S.tau));
      $$("[data-k=rsoil]").forEach((e) => (e.value = S.rsoil));
      $$("#seg-band button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      updateOmega(); syncRail();
    });
    seg.appendChild(b);
  });
}

function drawSunDial() {
  const z = S.sun_zenith_deg, a = S.sun_azimuth_deg, r = 58 * Math.min(1, z / 90);
  const rad = (a - 90) * Math.PI / 180;
  const x = 70 + r * Math.cos(rad), y = 70 + r * Math.sin(rad);
  $("#sun-dial").innerHTML = `
    <circle cx="70" cy="70" r="58" fill="none" stroke="var(--line)"/>
    <circle cx="70" cy="70" r="38" fill="none" stroke="var(--line-soft)"/>
    <circle cx="70" cy="70" r="19" fill="none" stroke="var(--line-soft)"/>
    <line x1="12" y1="70" x2="128" y2="70" stroke="var(--line-soft)"/>
    <line x1="70" y1="12" x2="70" y2="128" stroke="var(--line-soft)"/>
    <line x1="70" y1="70" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--warn)" stroke-width="2"/>
    <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="var(--warn)"/>
    <text x="70" y="9" text-anchor="middle" font-family="IBM Plex Mono" font-size="8" fill="var(--ink-3)">N</text>
    <text x="70" y="136" text-anchor="middle" font-family="IBM Plex Mono" font-size="8" fill="var(--ink-3)">S</text>`;
  $("#sun-sub").textContent = `θs ${z}° · φs ${a}°`;
}

/* ── cost note (what a run will cost, before you press it) ────────────── */
// Pixels across the scene for the image pass: the control is a ground resolution.
function imagePixels() {
  // the scene footprint: the JBS stand is ~106 m wide whatever scene_xy says
  const L = S.mode === "jbs" ? (STAND ? STAND.scene_xy[0] : 105.5) : S.scene_xy;
  return S.pixel_m > 0 ? Math.max(8, Math.min(192, Math.round(L / S.pixel_m))) : 0;
}
function rayCount() {
  const px = imagePixels();
  return S.mode === "jbs" ? px * px * S.n_spp : S.n_photons + px * px * S.n_spp;
}

function updateCostNote() {
  const rays = rayCount(), px = imagePixels();
  const t = TIMING[S.mode];
  let est = "no timing yet — the first render also compiles the CUDA kernel (~20–60 s, once).";
  if (t) est = `≈ ${(t.sec * rays / t.rays).toFixed(1)} s per render (measured ${t.sec.toFixed(1)} s at ${human(t.rays)} rays).`;
  $("#cost-note").textContent =
    `${human(rays)} rays per render` +
    (S.mode !== "jbs" && px ? ` (angular + ${px}x${px} image)` : "") + `.\n${est}` +
    (S.mode === "jbs" ? `\ninversion ≈ ${S.max_iter} × (1 + up to ${S.k_max + 1}) renders.` : "");
}

/* ── 1 · stand ────────────────────────────────────────────────────────── */
async function buildStand() {
  const btn = $("#btn-stand"); btn.disabled = true;
  $("#log-stand").hidden = false; $("#log-stand").textContent = "";
  bar("#pg-stand", 0.02, true);
  try {
    const { job_id } = await api("/api/stand", standParams());
    const job = await watchJob(job_id, (j) => {
      bar("#pg-stand", j.progress, j.status === "running");
      $("#log-stand").textContent = j.log.join("\n");
    });
    STAND = job.result;
    fillStand(STAND, job.elapsed);
  } catch (e) {
    $("#log-stand").textContent = "error: " + e.message;
  } finally {
    btn.disabled = false; bar("#pg-stand", 1, false);
  }
}

function standParams() {
  return {
    mode: S.mode, p_lai: S.p_lai, seed: S.seed, month: S.month,
    max_leaves: S.max_leaves, band: S.band,
    scene_xy: S.scene_xy, scene_z: S.scene_z, lai_eff: S.lai_eff,
    n_spheres: S.n_spheres, r_sph: S.r_sph, lai_per_sphere: S.lai_per_sphere,
    leaf_radius: S.leaf_radius, radius_std_mul: S.radius_std_mul, ala: S.ala,
    rho: S.rho, tau: S.tau, rsoil: S.rsoil,
    sun_zenith_deg: S.sun_zenith_deg, sun_azimuth_deg: S.sun_azimuth_deg,
    obs_zenith_max_deg: S.obs_zenith_max_deg, obs_step_deg: S.obs_step_deg,
    n_photons: S.n_photons, max_depth: S.max_depth,
  };
}
function renderParams() {
  return Object.assign(standParams(), {
    img: S.img, n_spp: S.n_spp, species_shadow: S.species_shadow, pixel_m: S.pixel_m,
    view_zenith_deg: S.view_zenith_deg, view_azimuth_deg: S.view_azimuth_deg,
  });
}

function fillStand(m, secs) {
  $("#ro-laieff").textContent = m.lai_eff.toFixed(3);
  $("#ro-laimax").textContent = m.lai_max.toFixed(3);
  $("#ro-leaves").textContent = human(m.n_leaf);
  $("#ro-extent").textContent = `${m.scene_xy[0].toFixed(0)}²m`;
  $("#stand-time").textContent = m.cached ? "from memory" : `built in ${m.build_s}s`;
  if (m.source) $("#stand-src").textContent = m.source;
  $("#tb-stand").innerHTML = `stand <b>${human(m.n_leaf)}</b> leaves`;
  $("#tb-stand").className = "chip ok";
  $("#rail-stand").textContent = `${human(m.n_leaf)} leaves · LAI ${m.lai_eff.toFixed(2)}`;
  $$(".step")[0].classList.add("done");
  const tb = $("#tb-species");
  if (m.species && m.species.length) {
    tb.innerHTML = m.species.map((s) =>
      `<tr><td class="ink">${s.name}</td><td>${human(s.n_leaf)}</td>
       <td>${s.lai_max.toFixed(4)}</td><td class="ink">${s.lai_eff.toFixed(4)}</td></tr>`).join("");
  }
  if (m.crowns && m.crowns.length) drawCrowns(m);
  load3D();
  updateP(); buildFreeTable();
}

/* ── 3-D view ─────────────────────────────────────────────────────────── */
let N3D = 40000;
async function load3D() {
  const sub = $("#v3-sub");
  if (!View3D.available) { sub.textContent = "WebGL unavailable in this browser"; return; }
  sub.textContent = "loading…";
  try {
    const d = await api("/api/points", Object.assign(standParams(), { n_points: N3D }));
    if (d.error) { sub.textContent = d.error; return; }
    View3D.load(d);
    const grp = d.group_by === "none" ? "" : ` · ${d.names.length} ${d.group_by}${d.names.length > 1 ? "s" : ""}`;
    sub.textContent = `${human(d.n)} of ${human(d.n_total)} leaves${grp} · ` +
      `${d.box[0].toFixed(0)} × ${d.box[1].toFixed(0)} × ${d.box[2].toFixed(0)} m`;
  } catch (e) { sub.textContent = "error: " + e.message; }
}

function drawCrowns(m) {
  const cv = $("#cv-crowns"); if (!cv) return;
  const g = cv.getContext("2d"), W = cv.width, L = m.scene_xy[0], s = W / L;
  const css = getComputedStyle(document.body);
  g.fillStyle = css.getPropertyValue("--surface-3") || "#e9ede5";
  g.fillRect(0, 0, W, W);
  // depth order: deeper crowns first, so the upper ones read as being on top
  const cr = m.crowns.slice().sort((a, b) => a.z - b.z);
  for (const c of cr) {
    const depth = Math.min(1, Math.max(0, (-c.z) / (m.height || 40)));
    g.globalAlpha = 0.9 - 0.35 * depth;
    g.fillStyle = "#2E7D53";
    g.beginPath(); g.arc(c.x * s, (L - c.y) * s, c.r * s, 0, 6.2832); g.fill();
    g.globalAlpha = 1; g.strokeStyle = "#16452e"; g.lineWidth = 1;
    g.beginPath(); g.arc(c.x * s, (L - c.y) * s, c.r * s, 0, 6.2832); g.stroke();
  }
  const cover = m.crowns.reduce((a, c) => a + Math.PI * c.r * c.r, 0) / (L * L);
  $("#crown-sub").textContent =
    `${m.crowns.length} crowns · ${L.toFixed(0)} × ${L.toFixed(0)} m · cover ${(100 * cover).toFixed(0)} %`;
}

/* ── 3 · render ───────────────────────────────────────────────────────── */
let renderJobId = null;
async function doRender() {
  const btn = $("#btn-render"); btn.disabled = true; $("#btn-render-stop").hidden = false;
  alertBox("#render-alert", "");
  bar("#pg-render", 0.02, true);
  $("#render-est").textContent = "running…";
  const t0 = performance.now();
  try {
    const { job_id } = await api("/api/render", renderParams());
    renderJobId = job_id;
    const job = await watchJob(job_id, (j) => {
      bar("#pg-render", j.progress, j.status === "running");
      $("#render-est").textContent = `${j.stage} · ${j.elapsed.toFixed(1)} s`;
    });
    RENDER = job.result;
    TIMING[S.mode] = { sec: RENDER.seconds, rays: rayCount() };
    showRender();
    $("#render-est").innerHTML =
      `render <b>${RENDER.seconds.toFixed(2)} s</b> · ${human(rayCount())} rays · ` +
      `total ${job.elapsed.toFixed(1)} s`;
    if (!STAND) { STAND = null; }
    updateCostNote();
    $$(".step")[2].classList.add("done");
  } catch (e) {
    alertBox("#render-alert", e.message, "bad");
    $("#render-est").textContent = "failed after " + ((performance.now() - t0) / 1000).toFixed(1) + " s";
  } finally {
    btn.disabled = false; $("#btn-render-stop").hidden = true; bar("#pg-render", 1, false);
  }
}

function showRender() {
  const r = RENDER;
  rampBar("#ramp-brf", BRF_STOPS); rampBar("#ramp-der", DIV_STOPS);
  const [blo, bhi] = paint("#cv-brf", r.brf_image, BRF_STOPS, false);
  $("#cap-brf").textContent = `${blo.toFixed(4)} – ${bhi.toFixed(4)}`;
  $("#ro-secs").textContent = r.seconds.toFixed(2);

  const derCell = $("#cv-der").parentElement;
  derCell.hidden = !r.deriv_image;
  $("#cv-brf").parentElement.hidden = !r.brf_image;
  if (r.deriv_image) drawDerImage();
  if (r.brf) drawAngular();
  if (r.species) {
    $("#tb-sens").innerHTML = r.species.map((s) =>
      `<tr><td class="ink">${s.name}</td><td>${sci(s.dfic)}</td><td>${sci(s.dedg)}</td>
       <td class="ink">${sci(s.dopt)}</td></tr>`).join("");
  }
  $("#img-sub").textContent = r.img
    ? `${r.img} × ${r.img} px · ${r.pixel_m} m/px` : "no image pass";
  setEstimator(EST);
}

function drawDerImage() {
  const grid = RENDER.deriv_image[EST];
  const [lo, hi] = paint("#cv-der", grid, DIV_STOPS, true);
  $("#cap-der").textContent = "∂BRF/∂P  ·  D_" + EST;
  $("#cap-der-rng").textContent = `${lo.toExponential(1)} … ${hi.toExponential(1)}`;
}

function drawAngular() {
  const r = RENDER, W = 460, H = 220, L = 52, R = 54, T = 14, B = 32;
  const n = r.n_pp, zen = r.zenith.slice(0, n), brf = r.brf.slice(0, n);
  const der = r.deriv[EST].slice(0, n);
  const xr = [Math.min(...zen), Math.max(...zen)];
  const brange = [Math.min(...brf), Math.max(...brf)];
  const drange = [Math.min(...der), Math.max(...der)];
  const pad = (a) => { const d = (a[1] - a[0]) || Math.abs(a[1]) || 1; return [a[0] - .12 * d, a[1] + .12 * d]; };
  const bR = pad(brange), dR = pad(drange);
  const px = (v) => L + (v - xr[0]) / ((xr[1] - xr[0]) || 1) * (W - L - R);
  const py = (v, rg) => T + (1 - (v - rg[0]) / ((rg[1] - rg[0]) || 1)) * (H - T - B);
  const out = [`<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="none" stroke="var(--line-soft)"/>`];
  for (let a = Math.ceil(xr[0] / 20) * 20; a <= xr[1]; a += 20) {
    out.push(`<line x1="${px(a).toFixed(1)}" y1="${T}" x2="${px(a).toFixed(1)}" y2="${H - B}" stroke="var(--line-soft)"/>`);
    out.push(`<text x="${px(a).toFixed(1)}" y="${H - B + 15}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="var(--ink-3)">${a}°</text>`);
  }
  // the retro-reflection angle comes from the server, which knows the sign
  // convention of the observation directions (phi = 0 is +zenith, phi = 180 is −)
  const hot = (r.hotspot === null || r.hotspot === undefined) ? null : r.hotspot;
  if (hot !== null && hot > xr[0] && hot < xr[1]) {
    out.push(`<line x1="${px(hot).toFixed(1)}" y1="${T}" x2="${px(hot).toFixed(1)}" y2="${H - B}" stroke="var(--warn)" stroke-dasharray="3 3"/>`);
    out.push(`<text x="${(px(hot) + 4).toFixed(1)}" y="${T + 11}" font-family="IBM Plex Mono" font-size="9" fill="var(--warn)">hotspot</text>`);
  }
  const line = (vals, rg) => vals.map((v, i) => (i ? "L" : "M") + px(zen[i]).toFixed(1) + " " + py(v, rg).toFixed(1)).join(" ");
  out.push(`<path d="${line(brf, bR)}" fill="none" stroke="var(--red-band)" stroke-width="2"/>`);
  out.push(`<path d="${line(der, dR)}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="5 3"/>`);
  const lab = (x, y, t, c, anchor) => `<text x="${x}" y="${y}" ${anchor ? 'text-anchor="' + anchor + '"' : ""} font-family="IBM Plex Mono" font-size="9" fill="${c}">${t}</text>`;
  out.push(lab(6, T + 9, "BRF", "var(--red-band)"));
  out.push(lab(6, T + 21, bR[1].toFixed(4), "var(--red-band)"));
  out.push(lab(6, H - B, bR[0].toFixed(4), "var(--red-band)"));
  out.push(lab(W - R + 6, T + 9, "D_" + EST, "var(--accent)"));
  out.push(lab(W - R + 6, T + 21, dR[1].toExponential(1), "var(--accent)"));
  out.push(lab(W - R + 6, H - B, dR[0].toExponential(1), "var(--accent)"));
  out.push(lab((L + W - R) / 2, H - 3,
    "view zenith  (+ = azimuth 0°, − = azimuth 180°)", "var(--ink-3)", "middle"));
  $("#sv-brf").innerHTML = out.join("");
}

function setEstimator(kind) {
  EST = kind;
  $$("#seg-est button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.est === kind)));
  if (!RENDER) return;
  const note = $("#est-note"), txt = note.querySelector("span:last-child");
  let mean = null;
  if (RENDER.deriv_image) { drawDerImage(); mean = RENDER.deriv_stats[kind].mean; }
  if (RENDER.deriv) {
    drawAngular();
    const arr = RENDER.deriv[kind];
    mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  }
  $("#ro-w").textContent = (RENDER.weight === null || RENDER.weight === undefined)
    ? (RENDER.std_lai > 0 ? "per-path" : "1.000") : fmt(RENDER.weight, 3);
  $("#ro-dmean").textContent = sci(mean);
  // std_lai comes back from the render, so this describes the stand that was
  // actually traced — not whatever is currently typed in the panel.
  const degenerate = RENDER.std_lai <= 0;
  const mul = RENDER.lai_r ? (RENDER.std_lai / RENDER.lai_r) : 0;
  if (degenerate && kind === "edg") {
    note.className = "alert warn";
    txt.innerHTML = `The geometric estimator is <b>degenerate in the stand that was rendered</b>:
      its leaf radius is fixed (std_LAI = 0), so dq = 0 at every contact and Var(D_edg) = 0. The
      optimal weight collapses to w = 1 and D_opt is the structural estimator alone. Set
      radius_std_mul above 0 in step 1 <b>and rebuild the stand</b> to make D_edg meaningful.`;
  } else if (degenerate) {
    note.className = "alert info";
    txt.innerHTML = `The rendered stand has a fixed leaf radius, so the geometric term carries no
      variance and the minimum-variance blend collapses to <b>w = 1</b> — D_opt = D_fic. That is
      reported rather than silently returned as zero.`;
  } else {
    note.className = "alert info";
    txt.innerHTML = `Both estimators are live: the rendered stand has std_LAI =
      ${RENDER.std_lai.toExponential(2)} (radius_std_mul ≈ ${mul.toFixed(2)}), and w is the
      closed-form minimum-variance weight between them.`;
  }
}

/* ── 4 · inversion ────────────────────────────────────────────────────── */
function retrieveNote() {
  const el = $("#retrieve-note"), txt = el.querySelector("span:last-child");
  if (S.mode === "jbs" && !S.uniform && !S.species_shadow) {
    el.className = "alert warn";
    txt.innerHTML = `<b>Per-species retrieval is approximate with the species-blind shadow ray.</b>
      The camera path is species-aware, but the sun-visibility test reads one scalar activation
      probability for every leaf, so the model's response to a single species is only partly
      described by that species' Jacobian column — the retrieval stalls after one step. Switch
      <b>species-aware shadow</b> on in the inspector to close the gap.`;
  } else if (S.mode === "jbs" && !S.uniform) {
    el.className = "alert info";
    txt.innerHTML = `Species-aware shadow rays are on, so the sun-visibility test reads
      <code>P_LAI_sp[species]</code> and the model responds to each species exactly as its Jacobian
      column says. Central differences match the analytic columns to within Monte Carlo noise.`;
  } else {
    el.className = "alert info";
    txt.innerHTML = `One parameter scales every species together, and the kernel's activation
      probability follows it everywhere — model and Jacobian describe the same thing, checked
      against central differences to about 1.5 %.`;
  }
}

function buildFreeTable() {
  const tb = $("#tb-free");
  retrieveNote();
  if (S.mode !== "jbs" || S.uniform) {
    S.free = ["LAI"];
    const label = S.mode === "jbs" ? "LAI (all species)" : "LAI_eff (via P_LAI)";
    tb.innerHTML = `<tr><td class="ink">${label}</td><td><input type="checkbox" checked disabled></td>
      <td><input type="number" id="pt-scalar" step="0.05" min="0.05" max="0.98" value="${S.p_true_scalar}"></td>
      <td><input type="number" id="pi-scalar" step="0.05" min="0.05" max="0.98" value="${S.p_init_scalar}"></td>
      <td style="text-align:right">${STAND ? STAND.lai_max.toFixed(3) : "—"}</td></tr>`;
    $("#pt-scalar").addEventListener("change", (e) => (S.p_true_scalar = Number(e.target.value)));
    $("#pi-scalar").addEventListener("change", (e) => (S.p_init_scalar = Number(e.target.value)));
    $("#ro-npar").textContent = "1";
    $("#rail-invert").textContent = "1 free parameter";
    return;
  }
  const sp = (STAND && STAND.species) ? STAND.species : (CAPS ? CAPS.species.map((n) => ({ name: n, lai_max: NaN })) : []);
  if (!S.free.length) S.free = sp.map((s) => s.name);
  tb.innerHTML = sp.map((s) => {
    const pt = S.p_true[s.name] !== undefined ? S.p_true[s.name] : S.p_lai;
    const pi = S.p_init[s.name] !== undefined ? S.p_init[s.name] : 0.25;
    return `<tr data-sp="${s.name}"><td>${s.name}</td>
      <td><input type="checkbox" class="f-free" ${S.free.includes(s.name) ? "checked" : ""} aria-label="${s.name} free"></td>
      <td><input type="number" class="f-true" step="0.05" min="0.05" max="0.98" value="${pt}" aria-label="${s.name} true P"></td>
      <td><input type="number" class="f-init" step="0.05" min="0.05" max="0.98" value="${pi}" aria-label="${s.name} start P"></td>
      <td style="text-align:right">${Number.isFinite(s.lai_max) ? s.lai_max.toFixed(4) : "—"}</td></tr>`;
  }).join("");
  tb.addEventListener("change", readFreeTable, { once: true });
  readFreeTable();
}

function readFreeTable() {
  const rows = $$("#tb-free tr[data-sp]");
  if (!rows.length) return;
  S.free = []; S.p_true = {}; S.p_init = {};
  rows.forEach((tr) => {
    const n = tr.dataset.sp;
    if (tr.querySelector(".f-free").checked) S.free.push(n);
    S.p_true[n] = Number(tr.querySelector(".f-true").value);
    S.p_init[n] = Number(tr.querySelector(".f-init").value);
  });
  $("#ro-npar").textContent = S.free.length;
  $("#rail-invert").textContent = `${S.free.length} free parameter${S.free.length === 1 ? "" : "s"}`;
  $$("#tb-free input").forEach((el) => el.addEventListener("change", readFreeTable, { once: true }));
}

let invertJobId = null;
async function doInvert() {
  readFreeTable();
  const btn = $("#btn-invert"); btn.disabled = true; $("#btn-invert-stop").hidden = false;
  alertBox("#invert-alert", "");
  $("#log-invert").textContent = ""; $("#tb-inv").innerHTML = ""; drawCost([]);
  bar("#pg-invert", 0.02, true);
  try {
    const body = Object.assign(renderParams(), {
      uniform: S.uniform, free: S.free, p_true: S.p_true, p_init: S.p_init,
      p_true_scalar: S.p_true_scalar, p_init_scalar: S.p_init_scalar,
      max_iter: S.max_iter, lm_lambda: S.lm_lambda, k_max: S.k_max, tol: S.tol,
    });
    const { job_id } = await api("/api/invert", body);
    invertJobId = job_id;
    const job = await watchJob(job_id, (j) => {
      bar("#pg-invert", j.progress, j.status === "running");
      $("#log-invert").textContent = j.log.join("\n");
      $("#log-invert").scrollTop = 1e6;
      $("#invert-est").innerHTML = `${j.stage} · <b>${j.elapsed.toFixed(1)} s</b>`;
      if (j.result && j.result.iterations) showInvert(j.result);
    });
    showInvert(job.result);
    $("#invert-est").innerHTML =
      `${job.result.iterations.length - 1} iterations · <b>${job.result.seconds.toFixed(1)} s</b> · cost ${sci(job.result.cost)}`;
    $$(".step")[3].classList.add("done");
  } catch (e) {
    alertBox("#invert-alert", e.message, "bad");
  } finally {
    btn.disabled = false; $("#btn-invert-stop").hidden = true; bar("#pg-invert", 1, false);
  }
}

function showInvert(res) {
  drawCost(res.iterations.map((h) => h.cost));
  $("#inv-secs").textContent = `${res.seconds.toFixed(1)} s`;
  $("#tb-inv").innerHTML = res.rows.map((r) => {
    const ok = Math.abs(r.p - r.p_true) < 5e-3;
    return `<tr><td class="ink">${r.name}</td><td>${r.p_true.toFixed(4)}</td>
      <td class="ink">${r.p.toFixed(4)}</td><td>${r.lai_true.toFixed(4)}</td>
      <td class="ink">${r.lai.toFixed(4)}</td>
      <td style="color:${ok ? "var(--ok)" : "var(--ink-3)"}">${r.d >= 0 ? "+" : ""}${r.d.toFixed(4)}</td>
      <td>${ok ? '<span class="pill ok">converged</span>' : '<span class="pill">iterating</span>'}</td></tr>`;
  }).join("");
}

function drawCost(costs) {
  const W = 460, H = 190, L = 52, R = 12, T = 12, B = 28, sv = $("#sv-cost");
  if (!costs.length) { sv.innerHTML = ""; return; }
  const safe = costs.map((c) => Math.max(c, 1e-16));
  const hiE = Math.ceil(Math.log10(Math.max(...safe)));
  const loE = Math.floor(Math.log10(Math.min(...safe)));
  const span = Math.max(1, hiE - loE);
  const px = (i) => L + (costs.length === 1 ? 0.5 : i / (costs.length - 1)) * (W - L - R);
  const py = (c) => T + (hiE - Math.log10(c)) / span * (H - T - B);
  const out = [`<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="none" stroke="var(--line-soft)"/>`];
  for (let e = hiE; e >= loE; e--) {
    const y = py(Math.pow(10, e));
    out.push(`<line x1="${L}" y1="${y.toFixed(1)}" x2="${W - R}" y2="${y.toFixed(1)}" stroke="var(--line-soft)"/>`);
    out.push(`<text x="${L - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-family="IBM Plex Mono" font-size="9" fill="var(--ink-3)">1e${e}</text>`);
  }
  out.push(`<path d="${safe.map((c, i) => (i ? "L" : "M") + px(i).toFixed(1) + " " + py(c).toFixed(1)).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="2"/>`);
  out.push(safe.map((c, i) => `<circle cx="${px(i).toFixed(1)}" cy="${py(c).toFixed(1)}" r="2.6" fill="var(--accent)"/>`).join(""));
  out.push(`<text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="var(--ink-3)">iteration · one render each</text>`);
  sv.innerHTML = out.join("");
}

/* ── 5 · export ───────────────────────────────────────────────────────── */
let SCRIPT = "";
async function genScript() {
  try {
    const j = await api("/api/script", renderParams());
    SCRIPT = j.code;
    $("#script-out").textContent = SCRIPT;
    $("#script-sub").textContent = `${SCRIPT.split("\n").length} lines · ${S.mode}`;
  } catch (e) { $("#script-out").textContent = "error: " + e.message; }
}

/* ── wiring ───────────────────────────────────────────────────────────── */
function wire() {
  $$(".step").forEach((btn) => btn.addEventListener("click", () => {
    $$(".step").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    $$("section.pane").forEach((p) => (p.hidden = p.dataset.pane !== btn.dataset.pane));
    $$("[data-only]").forEach((el) => { el.hidden = !el.dataset.only.split(",").includes(S.mode); });
  }));
  $("#seg-mode").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]"); if (!b) return;
    S.mode = b.dataset.mode; STAND = null; RENDER = null;
    applyModeDefaults(S.mode);
    $("#stale-note").hidden = true;
    $("#tb-stand").textContent = "no stand built"; $("#tb-stand").className = "chip";
    ["#ro-laieff", "#ro-laimax", "#ro-leaves", "#ro-extent"].forEach((s) => ($(s).textContent = "—"));
    applyMode();
  });
  $("#seg-retrieve").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-uniform]"); if (!b) return;
    S.uniform = b.dataset.uniform === "1";
    $$("#seg-retrieve button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    buildFreeTable();
  });
  $("#seg-est").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (b) setEstimator(b.dataset.est);
  });
  $("#cb-shadow").addEventListener("change", (e) => {
    S.species_shadow = e.target.checked; retrieveNote();
  });
  $("#sl-p").addEventListener("input", (e) => { S.p_lai = Number(e.target.value); updateP(); });
  $("#sl-p").addEventListener("change", checkStand);
  $("#btn-stand").addEventListener("click", buildStand);
  if (View3D.init($("#cv-3d"))) {
    $("#seg-3dcol").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-col]"); if (!b) return;
      $$("#seg-3dcol button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      View3D.setColourBy(b.dataset.col);
    });
    $("#sl-3dsize").addEventListener("input", (e) => {
      $("#v3-size").textContent = e.target.value; View3D.setSize(+e.target.value);
    });
    $("#sl-3dn").addEventListener("change", (e) => {
      N3D = +e.target.value; $("#v3-n").textContent = human(N3D); if (STAND) load3D();
    });
    $("#sl-3dn").addEventListener("input", (e) => { $("#v3-n").textContent = human(+e.target.value); });
    $("#btn-3dreset").addEventListener("click", () => View3D.reset());
  }
  $("#btn-render").addEventListener("click", doRender);
  $("#btn-invert").addEventListener("click", doInvert);
  $("#btn-render-stop").addEventListener("click", () => renderJobId && api("/api/cancel/" + renderJobId, {}));
  $("#btn-invert-stop").addEventListener("click", () => invertJobId && api("/api/cancel/" + invertJobId, {}));
  $("#btn-script").addEventListener("click", genScript);
  $("#btn-copy").addEventListener("click", () => SCRIPT && navigator.clipboard.writeText(SCRIPT));
  $("#btn-download").addEventListener("click", () => {
    if (!SCRIPT) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([SCRIPT], { type: "text/x-python" }));
    a.download = "canopi_session.py"; a.click(); URL.revokeObjectURL(a.href);
  });
}

async function boot() {
  bindInputs(); wire(); drawSunDial(); updateOmega(); updateP();
  try {
    CAPS = await api("/api/capabilities");
    const dev = $("#tb-device");
    if (CAPS.gpu) {
      dev.className = "chip ok";
      dev.innerHTML = `${CAPS.gpu_name} · <b>${CAPS.vram_gb} GB</b>`;
    } else {
      dev.className = "chip bad"; dev.textContent = "no CUDA device";
    }
    if (!CAPS.jbs_data) {
      S.mode = "homogeneous";
      $$("#seg-mode button[data-mode=jbs]").forEach((b) => {
        b.disabled = true;
        b.title = "RAMI-5 tree locations not found — run scripts/download_jbs_data.py";
      });
      $("#rail-note").textContent = "RAMI-5 tree locations not found; homogeneous mode only.";
    }
    const msel = $("[data-k=month]");
    msel.innerHTML = CAPS.months.map((m) => `<option${m === S.month ? " selected" : ""}>${m}</option>`).join("");
    buildBandSeg();
    $("#foot-right").textContent = CAPS.scene_dir;
  } catch (e) {
    $("#tb-device").className = "chip bad";
    $("#tb-device").textContent = "server unreachable";
  }
  applyModeDefaults(S.mode);   // the opening stand gets its own defaults too
  applyMode();
  checkStand();
}
boot();
