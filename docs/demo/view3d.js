/* view3d.js — orbitable 3-D view of the stand.
 *
 * Draws the real leaf positions (a random subsample) as depth-tested points in
 * raw WebGL: no library, no CDN, so it works over a tunnel with no internet.
 * Falls back to a painter's-algorithm 2-D canvas where WebGL is unavailable.
 */
"use strict";

const View3D = (() => {
  // ── small matrix helpers (column-major, WebGL order) ──────────────────
  const mul = (a, b) => {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  };
  const perspective = (fovy, aspect, near, far) => {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0,
      0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  };
  const lookAt = (eye, ctr, up) => {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const norm = (v) => { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = norm(sub(eye, ctr)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  };

  const VS = `
    attribute vec3 aPos; attribute vec3 aCol;
    uniform mat4 uMVP;
    uniform mediump float uIsPoint;   // shared with the fragment stage: same precision there
    uniform float uPx;     // pixels per world unit at unit distance
    uniform float uDiam;   // drawn leaf diameter [m] — see note in render()
    uniform float uR;      // camera distance to the scene centre, for the depth cue
    varying vec3 vCol; varying float vFog;
    void main() {
      gl_Position = uMVP * vec4(aPos, 1.0);
      float w = max(gl_Position.w, 0.001);
      gl_PointSize = uIsPoint * max(1.0, uPx * uDiam / w);
      vCol = aCol;
      vFog = clamp((w / uR - 0.75) * 0.7, 0.0, 0.45);   // far side recedes
    }`;
  const FS = `
    precision mediump float;
    varying vec3 vCol; varying float vFog;
    uniform vec3 uBg; uniform float uIsPoint;
    void main() {
      if (uIsPoint > 0.5) {
        vec2 d = gl_PointCoord - vec2(0.5);
        float r2 = dot(d, d);
        if (r2 > 0.25) discard;                    // round splat, not a square
        float shade = 0.72 + 0.55 * (0.25 - r2);   // cheap spherical shading
        gl_FragColor = vec4(mix(vCol * shade, uBg, vFog), 1.0);
      } else {
        gl_FragColor = vec4(mix(vCol, uBg, vFog * 0.5), 1.0);
      }
    }`;

  // 12 hues, distinguishable and stable across ids
  const PALETTE = [
    [0.18, 0.49, 0.33], [0.05, 0.36, 0.39], [0.48, 0.55, 0.18], [0.71, 0.26, 0.18],
    [0.54, 0.35, 0.17], [0.25, 0.49, 0.61], [0.70, 0.23, 0.53], [0.35, 0.30, 0.60],
    [0.85, 0.55, 0.15], [0.14, 0.58, 0.53], [0.60, 0.62, 0.30], [0.44, 0.20, 0.35],
  ];

  let gl, prog, cv, data = null, buf = {}, box = [1, 1, 1];
  let az = 0.7, el = 0.35, dist = 2.0, size = 26, colourBy = "group";
  let dragging = false, lastX = 0, lastY = 0, raf = 0;

  function compile(src, type) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }

  function bgColour() {
    const c = getComputedStyle(document.body).backgroundColor.match(/\d+/g) || [240, 242, 237];
    return [c[0] / 255, c[1] / 255, c[2] / 255];
  }

  function init(canvas) {
    cv = canvas;
    gl = cv.getContext("webgl", { antialias: true, alpha: false, depth: true })
      || cv.getContext("experimental-webgl");
    if (!gl) return false;
    prog = gl.createProgram();
    gl.attachShader(prog, compile(VS, gl.VERTEX_SHADER));
    gl.attachShader(prog, compile(FS, gl.FRAGMENT_SHADER));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error("shader link failed: " + gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    gl.enable(gl.DEPTH_TEST);
    buf.pos = gl.createBuffer(); buf.col = gl.createBuffer();
    clearOnly();          // an empty canvas should read as empty, not as broken
    buf.lpos = gl.createBuffer(); buf.lcol = gl.createBuffer();

    cv.addEventListener("pointerdown", (e) => {
      dragging = true; lastX = e.clientX; lastY = e.clientY; cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener("pointerup", (e) => {
      dragging = false; try { cv.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    cv.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      az -= (e.clientX - lastX) * 0.008;
      el = Math.max(-1.45, Math.min(1.45, el + (e.clientY - lastY) * 0.006));
      lastX = e.clientX; lastY = e.clientY; draw();
    });
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      dist = Math.max(0.35, Math.min(6, dist * (e.deltaY > 0 ? 1.12 : 0.89)));
      draw();
    }, { passive: false });
    window.addEventListener("resize", () => draw());
    return true;
  }

  // ── ground grid + box edges, as line segments ─────────────────────────
  function frameLines(Lx, Ly, H) {
    const v = [], c = [], grey = [0.55, 0.58, 0.55], edge = [0.35, 0.40, 0.36];
    const push = (a, b, col) => { v.push(...a, ...b); c.push(...col, ...col); };
    const N = 10;
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      push([t * Lx, 0, -H], [t * Lx, Ly, -H], grey);
      push([0, t * Ly, -H], [Lx, t * Ly, -H], grey);
    }
    const cor = [[0, 0], [Lx, 0], [Lx, Ly], [0, Ly]];
    for (let i = 0; i < 4; i++) {
      const a = cor[i], b = cor[(i + 1) % 4];
      push([a[0], a[1], -H], [b[0], b[1], -H], edge);
      push([a[0], a[1], 0], [b[0], b[1], 0], edge);
      push([a[0], a[1], -H], [a[0], a[1], 0], edge);
    }
    return { v: new Float32Array(v), c: new Float32Array(c), n: v.length / 3 };
  }

  function load(d) {
    if (!gl || d.error) return;
    data = d;
    box = d.box;
    const n = d.n, xyz = new Float32Array(d.xyz), col = new Float32Array(n * 3);
    recolour(col, xyz, d);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.pos); gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.col); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    data.count = n; data.colBuf = col; data.xyzBuf = xyz;

    const f = frameLines(box[0], box[1], box[2]);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.lpos); gl.bufferData(gl.ARRAY_BUFFER, f.v, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.lcol); gl.bufferData(gl.ARRAY_BUFFER, f.c, gl.STATIC_DRAW);
    data.lineCount = f.n;
    draw();
  }

  function recolour(col, xyz, d) {
    const H = d.box[2];
    for (let i = 0; i < d.n; i++) {
      let c;
      if (colourBy === "group" && d.group_by !== "none") {
        c = PALETTE[d.gid[i] % PALETTE.length];
      } else {                                   // height: canopy top pale, base dark
        const t = Math.min(1, Math.max(0, 1 + xyz[i * 3 + 2] / H));
        c = [0.13 + 0.55 * t, 0.30 + 0.42 * t, 0.18 + 0.30 * t];
      }
      col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
    }
  }

  function setColourBy(mode) {
    colourBy = mode;
    if (!data) return;
    recolour(data.colBuf, data.xyzBuf, data);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.col);
    gl.bufferData(gl.ARRAY_BUFFER, data.colBuf, gl.STATIC_DRAW);
    draw();
  }

  function setSize(px) { size = px; draw(); }
  function reset() { az = 0.7; el = 0.35; dist = 2.0; draw(); }

  function clearOnly() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.max(1, Math.round(cv.clientWidth * dpr));
    cv.height = Math.max(1, Math.round((cv.clientHeight || 340) * dpr));
    gl.viewport(0, 0, cv.width, cv.height);
    const bg = bgColour();
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  function draw() {
    if (!gl) return;
    if (!data) { clearOnly(); return; }
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; render(); });
  }

  function render() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(cv.clientWidth * dpr));
    const h = Math.max(1, Math.round((cv.clientHeight || 340) * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    gl.viewport(0, 0, w, h);
    const bg = bgColour();
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const [Lx, Ly, H] = box;
    const ctr = [Lx / 2, Ly / 2, -H / 2];
    const R = Math.max(Lx, Ly) * dist;
    const eye = [ctr[0] + R * Math.cos(el) * Math.cos(az),
                 ctr[1] + R * Math.cos(el) * Math.sin(az),
                 ctr[2] + R * Math.sin(el)];
    const mvp = mul(perspective(0.72, w / h, Math.max(0.5, R * 0.02), R * 4),
                    lookAt(eye, ctr, [0, 0, 1]));

    const aPos = gl.getAttribLocation(prog, "aPos"), aCol = gl.getAttribLocation(prog, "aCol");
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMVP"), false, mvp);
    gl.uniform3fv(gl.getUniformLocation(prog, "uBg"), bg);
    // A 5 cm leaf 200 m away covers half a pixel, so a faithful size would draw
    // nothing. Points are given an effective diameter that scales with the scene
    // (about 0.6 m in the 100 m stand at the default setting) — the positions are
    // real, the disk size on screen is not.
    const fpx = 1 / Math.tan(0.72 / 2);
    gl.uniform1f(gl.getUniformLocation(prog, "uPx"), (h / 2) * fpx);
    gl.uniform1f(gl.getUniformLocation(prog, "uDiam"), (size / 26) * Math.max(Lx, Ly) * 0.006);
    gl.uniform1f(gl.getUniformLocation(prog, "uR"), R);

    const bind = (bp, bc) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, bp);
      gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, bc);
      gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
    };
    gl.uniform1f(gl.getUniformLocation(prog, "uIsPoint"), 0);
    bind(buf.lpos, buf.lcol);
    gl.drawArrays(gl.LINES, 0, data.lineCount);

    gl.uniform1f(gl.getUniformLocation(prog, "uIsPoint"), 1);
    bind(buf.pos, buf.col);
    gl.drawArrays(gl.POINTS, 0, data.count);
  }

  return { init, load, setColourBy, setSize, reset,
           get available() { return !!gl; } };
})();
