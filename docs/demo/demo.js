/* demo.js — replays a recorded session so the interface runs with no server.
 *
 * Loaded before app.js, it replaces window.fetch with a small in-page stand-in
 * for the API: the same endpoints, the same job-polling shape, answered from the
 * recording in demo-data.js. app.js is byte-identical to the live version — it
 * cannot tell the difference, which is the point: what a reviewer clicks through
 * is the real interface, driven by real results.
 *
 * Jobs are replayed over the wall-clock time the original run took, capped so
 * nobody waits a minute for a screenshot, and the log and the inversion trace
 * fill in progressively exactly as they did live.
 */
"use strict";

(() => {
  const D = window.DEMO;
  if (!D) return;

  const MAX_REPLAY_MS = 4000;      // never make the reviewer wait longer than this
  const MIN_REPLAY_MS = 700;
  const built = new Set();         // stands "built" in this session
  const jobs = new Map();
  let seq = 0;

  const base = (mode) => D.stands[mode] || D.stands[Object.keys(D.stands)[0]];

  // Some stands were recorded twice under one changed setting (the spherical stand
  // with and without a leaf-size spread). Pick the recording that matches what the
  // reviewer typed, so both regimes of the estimator are reachable.
  function rec(mode, body) {
    const b = base(mode), v = b.variants;
    if (!v) return b;
    for (const [value, variant] of Object.entries(v)) {
      const asked = Number(body[variant.field]);
      if (Number.isFinite(asked) && Math.abs(asked - Number(value)) < 1e-9) {
        return Object.assign({}, b, { stand: variant.stand, render: variant.render,
                                      variant: value });
      }
    }
    return b;
  }

  const json = (obj) => new Response(JSON.stringify(obj),
    { status: 200, headers: { "Content-Type": "application/json" } });

  // ── progressive view of a finished job ────────────────────────────────
  function snapshot(j) {
    const t = Math.min(1, (performance.now() - j.t0) / j.ms);
    const status = t >= 1 ? "done" : "running";
    const log = j.log.slice(0, Math.max(1, Math.ceil(j.log.length * t)));
    let result = t >= 1 ? j.result : {};
    if (t < 1 && j.kind === "invert" && j.result.iterations) {
      // replay the descent: the cost curve and the table fill in per iteration
      const k = Math.max(1, Math.floor(j.result.iterations.length * t));
      const its = j.result.iterations.slice(0, k);
      const last = its[its.length - 1];
      result = Object.assign({}, j.result, {
        iterations: its, done: false,
        cost: last.cost,
        rows: j.result.rows.map((r) => {
          const p = last.p[r.name];
          if (p === undefined) return r;
          const scale = r.p_true ? r.lai_true / r.p_true : 0;
          return Object.assign({}, r, { p, lai: p * scale, d: (p - r.p_true) * scale });
        }),
      });
    }
    return {
      id: j.id, kind: j.kind, status, stage: t >= 1 ? "done" : j.stage,
      progress: t, log, result, error: "",
      elapsed: Math.round(j.seconds * t * 10) / 10,
    };
  }

  function start(kind, mode, r) {
    const id = "demo" + (++seq);
    const ms = Math.min(MAX_REPLAY_MS, Math.max(MIN_REPLAY_MS, (r.seconds || 1) * 250));
    jobs.set(id, { id, kind, mode, t0: performance.now(), ms,
                   log: r.log || [], result: r.result || {}, seconds: r.seconds || 0,
                   stage: kind === "invert" ? "iterating" : "tracing" });
    return id;
  }

  // ── the stand-in API ──────────────────────────────────────────────────
  const realFetch = window.fetch.bind(window);
  window.fetch = async (url, opt) => {
    const path = String(url).split("?")[0];
    if (!path.startsWith("/api/")) return realFetch(url, opt);
    const body = opt && opt.body ? JSON.parse(opt.body) : {};
    const mode = body.mode || "jbs";
    const R = rec(mode, body);

    if (path === "/api/capabilities") {
      return json({
        gpu: true, gpu_name: D.recorded_on + " (recorded)", vram_gb: 46,
        jbs_data: true, scene_dir: "data/JBS_treelocations",
        species: engineSpecies(), bands: bands(), months: ["Jul"], cached: [],
      });
    }
    if (path.startsWith("/api/job/")) {
      const j = jobs.get(path.split("/").pop());
      return json(j ? snapshot(j) : { error: "no such job" });
    }
    if (path === "/api/stand") {
      built.add(mode + (R.variant ? "@" + R.variant : ""));
      return json({ job_id: start("stand", mode, R.stand) });
    }
    if (path === "/api/render") return json({ job_id: start("render", mode, R.render) });
    if (path === "/api/invert") {
      const key = body.uniform === false && R.inversions.species ? "species" : "uniform";
      const inv = R.inversions[key] || Object.values(R.inversions)[0];
      if (!inv) return json({ error: "no inversion was recorded for this stand" });
      return json({ job_id: start("invert", mode, inv) });
    }
    if (path === "/api/points") return json(R.points);
    if (path === "/api/script") return json({ code: R.script });
    if (path === "/api/stand_status") {
      const vkey = mode + (R.variant ? "@" + R.variant : "");
      return json({ key: "demo|" + vkey, cached: built.has(vkey),
                    n_leaf: R.stand.result.n_leaf, std_lai: R.stand.result.std_lai || 0 });
    }
    if (path.startsWith("/api/cancel/")) return json({ cancelled: true });
    return json({ error: "not part of the recording" });
  };

  function engineSpecies() {
    const s = D.stands.jbs && D.stands.jbs.stand.result.species;
    return s ? s.map((x) => x.name) : [];
  }
  function bands() {
    const out = {};
    for (const [m, r] of Object.entries(D.stands)) {
      const p = r.params;
      out[p.band] = { rho: p.rho, tau: p.tau, rsoil: p.rsoil,
                      wavelength_nm: p.band === "O08" ? 665.25 : p.band === "O17" ? 865.6 : 0 };
    }
    return out;
  }

  // ── the panels must show the settings that were recorded ──────────────
  // Otherwise a field could read 0.3 while the replayed stand was built at 0, which
  // is exactly the mismatch the live app warns about. app.js keeps its state in a
  // top-level `S`, reachable from here because this script shares its global scope.
  function seedFromRecording() {
    if (typeof S === "undefined") return;
    const p = base(S.mode).params || {};
    for (const [k, v] of Object.entries(p)) if (k in S && v !== null) S[k] = v;
    document.querySelectorAll("[data-k]").forEach((el) => {
      const v = S[el.dataset.k];
      if (v === undefined) return;
      if (el.tagName === "SELECT" &&
          ![...el.options].some((o) => String(o.value || o.text) === String(v))) {
        // the recording used a value this dropdown does not list (a 48 px image,
        // say): add it rather than leaving the control blank
        const o = document.createElement("option");
        o.value = v; o.textContent = `${v} × ${v}`;
        el.appendChild(o);
      }
      el.value = v;
    });
    const sl = document.querySelector("#sl-p");
    if (sl) sl.value = S.p_lai;
    try { updateP(); updateOmega(); drawSunDial(); applyMode(); checkStand(); } catch (e) {}
  }

  // ── say plainly when a control shows a recorded run, not a new one ─────
  window.addEventListener("DOMContentLoaded", () => {
    setTimeout(seedFromRecording, 0);
    const seg = document.querySelector("#seg-mode");
    if (seg) seg.addEventListener("click", () => setTimeout(seedFromRecording, 0));
    const mark = (sel, text) => {
      const el = document.querySelector(sel);
      if (el) el.title = text;
    };
    mark("#btn-render", "Demo: replays the recorded render for this stand");
    mark("#btn-invert", "Demo: replays the recorded inversion for this stand");
    mark("#btn-stand", "Demo: replays the recorded stand build");
    const note = document.createElement("p");
    note.style.cssText = "font-family:var(--mono);font-size:10.5px;color:var(--ink-3);margin:0";
    note.textContent = "Recorded run — parameter changes are not recomputed here.";
    document.querySelectorAll(".runbar").forEach((rb) => rb.appendChild(note.cloneNode(true)));
  });
})();
