/* Canonical vision preprocessing, ported line for line from heads/vision.py center_by_mass (D-011).
   Ink bbox (no padding) -> longer side to 20 px (aspect kept, area resampling) -> centre of mass to (14, 14)
   in a 28x28 frame, placement rounded half up. Python round()/np.rint are half-to-even; so is roundHalfEven.
   Parity with the server is checked by tests/test_ui_centering.py under Node. */
(function (root) {
  "use strict";
  var SIDE = 28, BOX = 20;

  function roundHalfEven(x) {
    var f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  // (m, n) area-averaging matrix taking a length-n axis to length m (rows sum to 1), as rows of Float64Array.
  function resample(m, n) {
    var step = n / m, rows = [];
    for (var i = 0; i < m; i++) {
      var lo = i * step, hi = (i + 1) * step, row = new Float64Array(n);
      for (var j = 0; j < n; j++) {
        var ov = Math.min(hi, j + 1) - Math.max(lo, j);
        row[j] = (ov > 0 ? ov : 0) / step;
      }
      rows.push(row);
    }
    return rows;
  }

  // img: row-major array of h*w intensities (0..255). Returns Array of 784 ints 0..255.
  function centerByMass(img, h, w) {
    if (h === undefined) { h = SIDE; w = SIDE; }
    var y0 = h, y1 = -1, x0 = w, x1 = -1, y, x, i, j, k;
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) if (img[y * w + x] !== 0) {
      if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x;
    }
    var out = new Array(SIDE * SIDE).fill(0);
    if (y1 < 0) return out;
    var ch = y1 - y0 + 1, cw = x1 - x0 + 1;
    var scale = BOX / Math.max(ch, cw);
    var nh = Math.max(1, roundHalfEven(ch * scale)), nw = Math.max(1, roundHalfEven(cw * scale));
    var rh = resample(nh, ch), rw = resample(nw, cw);
    // tmp = rh @ crop  (nh x cw), then small = tmp @ rw.T  (nh x nw)
    var tmp = [], small = [];
    for (i = 0; i < nh; i++) {
      var t = new Float64Array(cw), r = rh[i];
      for (k = 0; k < ch; k++) {
        var a = r[k];
        if (a === 0) continue;
        var base = (y0 + k) * w + x0;
        for (j = 0; j < cw; j++) t[j] += a * img[base + j];
      }
      tmp.push(t);
    }
    for (i = 0; i < nh; i++) {
      var s = new Float64Array(nw);
      for (j = 0; j < nw; j++) {
        var acc = 0, q = rw[j], ti = tmp[i];
        for (k = 0; k < cw; k++) acc += ti[k] * q[k];
        s[j] = acc;
      }
      small.push(s);
    }
    var mass = 0, my = 0, mx = 0;
    for (i = 0; i < nh; i++) {
      var rs = 0;
      for (j = 0; j < nw; j++) { rs += small[i][j]; mx += small[i][j] * j; mass += small[i][j]; }
      my += rs * i;
    }
    var cy = my / mass, cx = mx / mass, centre = SIDE / 2;
    var top = Math.min(Math.max(Math.floor(centre - cy + 0.5), 0), SIDE - nh);
    var left = Math.min(Math.max(Math.floor(centre - cx + 0.5), 0), SIDE - nw);
    for (i = 0; i < nh; i++) for (j = 0; j < nw; j++) {
      var v = roundHalfEven(small[i][j]);
      out[(top + i) * SIDE + left + j] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    return out;
  }

  // D-023: raw ink bbox whose longer side is under 3/28 of the canvas side (under 3 px at 28x28 scale), or no ink.
  // The client then posts 784 zeros so the server answers with its canonical blank message.
  var MIN_EXTENT = 3;
  function isTinyStroke(img, h, w) {
    var y0 = h, y1 = -1, x0 = w, x1 = -1;
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) if (img[y * w + x] !== 0) {
      if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x;
    }
    if (y1 < 0) return true;
    return Math.max(y1 - y0 + 1, x1 - x0 + 1) < MIN_EXTENT * Math.max(h, w) / SIDE;
  }

  var api = { centerByMass: centerByMass, isTinyStroke: isTinyStroke, SIDE: SIDE };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Centering = api;
})(this);
