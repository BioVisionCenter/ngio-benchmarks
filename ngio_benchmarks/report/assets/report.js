/* ══════════════════════════════════════════════════════════════════════════
   ngio benchmarks — the report page, for all three suites

   No dependencies, no network. The charts are SVG built through the DOM API,
   which also settles the escaping question: every label here came out of a CSV
   an adapter wrote, so all of it goes in through textContent and none of it
   through innerHTML.

   Three things are load-bearing in the design:

   * Colour identifies the series — the implementation, or the environment —
     and nothing else. It never encodes a value, and the slot is assigned by
     name in `_model.colours`, so hiding a series in the filter row never
     repaints the survivors.
   * Colour is never the only channel. Every bar is directly labelled, every
     chart has a table view, and the texture toggle carries identity without
     hue for readers who need that.
   * The audit column is drawn on the chart, whichever one the suite has. A
     writer whose pyramid differs from the one it was asked for, or a reader
     whose checksum differs from what the rest of its cell returned, is hatched
     and marked — a bar that timed a different artefact is not comparable with
     the bars beside it.

   Nothing below names a suite. What differs between the three arrives in
   `DATA.profile`: which columns to show, what to call them, which memory
   figure was recorded, and every word of prose the page prints. See
   `_profile.py`, where all of it is written down once.
   ══════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("report-data").textContent);
  var P = DATA.profile;
  var SVGNS = "http://www.w3.org/2000/svg";

  /* ── formatting, mirroring core/output.py so the page and the terminal
        never disagree about the same number ─────────────────────────────── */

  function duration(seconds) {
    if (seconds == null) return "";
    if (seconds >= 60) return seconds.toFixed(1) + " s";
    return (seconds * 1000).toFixed(1) + " ms";
  }

  function megabytes(value) {
    return value == null ? "n/a" : value.toFixed(1) + " MB";
  }

  function bytesize(value) {
    if (value == null) return "";
    var units = ["B", "KB", "MB", "GB", "TB"];
    var n = value;
    var i = 0;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }
    return (i === 0 ? n.toFixed(0) : n.toFixed(1)) + " " + units[i];
  }

  // The three spread cases from core.output.spread: one run has no spread to
  // report, two runs give a half-range, three or more give the deviation.
  function spread(row) {
    if (!row.repeats || row.repeats <= 1) return "n=1";
    if (row.repeats === 2) return "± " + duration((row.high - row.low) / 2);
    var text = "± " + duration(row.mad);
    var mid = (row.low + row.high) / 2;
    if (mid > 0 && row.mad / mid > 0.05) {
      text += " (" + Math.round((row.mad / mid) * 100) + "%)";
    }
    return text;
  }

  function times(value) {
    if (value == null) return "";
    if (value >= 100) return value.toFixed(0) + "×";
    if (value >= 10) return value.toFixed(1) + "×";
    return value.toFixed(2) + "×";
  }

  // The palette slot this series holds, assigned once in Python over the whole
  // file. Never derived from position here: a name the profile did not pin
  // still keeps its slot when the filter row hides its neighbours.
  function slotOf(name) {
    var slot = P.colours[name];
    return slot == null ? -1 : slot;
  }

  function seriesColour(name) {
    var slot = slotOf(name);
    return slot < 0
      ? "var(--s-other)"
      : "var(--s-slot-" + slot + ", var(--s-other))";
  }

  // The tooltip and table columns are named by the profile, so the value they
  // pull out of a record has to be formatted by name too. Every one of these
  // is a formatter above, which is what keeps a number in the browser reading
  // exactly as `core/output.py` would print it in the terminal.
  function cell(row, key, format) {
    if (format === "spread") return row.status === "ok" ? spread(row) : "";
    var value = key ? row[key] : null;
    if (format === "duration") return duration(value);
    if (format === "megabytes") return megabytes(value);
    // tracemalloc sees nothing at all for an adapter allocating in C++ or Rust
    // buffers, and that is a different claim from "it used no memory".
    if (format === "peak") return value == null ? "n/a — native buffers" : megabytes(value);
    if (format === "bytes") return bytesize(value);
    if (format === "times") return times(value);
    return value == null ? "" : String(value);
  }

  /* ── state ────────────────────────────────────────────────────────────── */

  var AXES = DATA.axes;
  var AXIS_BY_FIELD = {};
  AXES.forEach(function (a) {
    AXIS_BY_FIELD[a.field] = a;
  });

  var state = {
    view: "timing",
    facet: DATA.roles.facet,
    group: DATA.roles.group,
    scale: "linear",
    values: "absolute",
    baseline: null,
    sort: "fixed",
    texture: false,
    theme: "system",
    filters: {},
    pinned: null,
  };
  AXES.forEach(function (a) {
    state.filters[a.field] = a.values.slice();
  });

  /* ── derived structure ───────────────────────────────────────────────── */

  // Every axis not spent on a facet or a group identifies a series. `impl`
  // always lands here unless the reader moves it, which is what makes the
  // implementation the thing bars are coloured by.
  function seriesFields() {
    return AXES.map(function (a) {
      return a.field;
    }).filter(function (f) {
      return f !== state.facet && f !== state.group;
    });
  }

  function seriesKey(row) {
    return seriesFields()
      .map(function (f) {
        return row.axes[f] || "";
      })
      .join("");
  }

  // "ngio · dask", "iohub · zarrs-python" — the implementation, then whatever
  // of its own options this file varied. Blank axis cells are dropped: `mode`
  // is ngio's alone, and printing "bioio · " for every other writer would be
  // noise standing in for an option they do not have.
  function seriesLabel(row) {
    var parts = seriesFields()
      .map(function (f) {
        var value = row.axes[f];
        if (!value) return "";
        // "true" and "8" name nothing on their own, so an axis of booleans or
        // bare numbers keeps its field name. `dask` and `zarrs-python` do not
        // need one, and "ngio · mode=dask" would only be longer.
        return AXIS_BY_FIELD[f] && AXIS_BY_FIELD[f].prefixed ? f + "=" + value : value;
      })
      .filter(Boolean);
    return parts.length ? parts.join(" · ") : row.column;
  }

  function passesFilters(row) {
    return AXES.every(function (a) {
      var value = row.axes[a.field];
      // A blank means the axis does not apply to this implementation, so it is
      // never filtered out by a value it could not have had.
      return !value || state.filters[a.field].indexOf(value) !== -1;
    });
  }

  function visibleRows() {
    return DATA.rows.filter(passesFilters);
  }

  function distinct(rows, field) {
    if (!field) return [""];
    var order = AXIS_BY_FIELD[field] ? AXIS_BY_FIELD[field].values : [];
    var present = {};
    rows.forEach(function (r) {
      present[r.axes[field] || ""] = true;
    });
    var found = order.filter(function (v) {
      return present[v];
    });
    if (present[""]) found.push("");
    return found.length ? found : [""];
  }

  // Series order is fixed by the axis value order, which is itself fixed by
  // the implementation name — so it is stable across filters and across files.
  function seriesList(rows) {
    var seen = {};
    var out = [];
    rows.forEach(function (r) {
      var key = seriesKey(r);
      if (seen[key]) return;
      seen[key] = true;
      out.push({ key: key, label: seriesLabel(r), column: r.column, row: r });
    });
    var fields = seriesFields();
    out.sort(function (a, b) {
      for (var i = 0; i < fields.length; i++) {
        var axis = AXIS_BY_FIELD[fields[i]];
        var av = a.row.axes[fields[i]] || "";
        var bv = b.row.axes[fields[i]] || "";
        if (av !== bv) return axis.values.indexOf(av) - axis.values.indexOf(bv);
      }
      return 0;
    });
    return out;
  }

  /* ── tiny DOM helpers ────────────────────────────────────────────────── */

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    apply(node, attrs, children);
    return node;
  }

  function svgEl(tag, attrs, children) {
    var node = document.createElementNS(SVGNS, tag);
    apply(node, attrs, children);
    return node;
  }

  function apply(node, attrs, children) {
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value == null || value === false) return;
        if (key === "text") node.textContent = String(value);
        else if (key === "class") node.setAttribute("class", value);
        else if (key.slice(0, 2) === "on") node[key] = value;
        else node.setAttribute(key, String(value));
      });
    }
    (children || []).forEach(function (child) {
      if (child) node.appendChild(child);
    });
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  /* ── scales ──────────────────────────────────────────────────────────── */

  function niceMax(value) {
    if (!(value > 0)) return 1;
    var exp = Math.pow(10, Math.floor(Math.log10(value)));
    var f = value / exp;
    var step = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
    return step * exp;
  }

  // Round numbers, not max/count — an axis reading 0.63× / 1.25× / 1.88× makes
  // the reader do arithmetic to place a bar.
  function linearTicks(max, count) {
    var raw = max / Math.max(count, 1);
    var exp = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / exp;
    var step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * exp;
    var out = [];
    for (var v = 0; v <= max * 1.0001; v += step) out.push(v);
    return out;
  }

  function logTicks(min, max) {
    var out = [];
    var start = Math.floor(Math.log10(min));
    var end = Math.ceil(Math.log10(max));
    for (var e = start; e <= end; e++) {
      [1, 2, 5].forEach(function (m) {
        var v = m * Math.pow(10, e);
        if (v >= min * 0.999 && v <= max * 1.001) out.push(v);
      });
    }
    return out;
  }

  var textRuler = document.createElement("canvas").getContext("2d");
  function textWidth(text, font) {
    textRuler.font = font || "11.5px ui-sans-serif, system-ui, sans-serif";
    return textRuler.measureText(text).width;
  }

  /* ── tooltip ─────────────────────────────────────────────────────────── */

  var tooltip = el("div", { class: "tooltip", role: "status" });
  document.body.appendChild(tooltip);

  // Which lines a suite shows is the profile's decision; a record carries the
  // same keys either way, so a column the schema does not have arrives blank
  // and drops out here alongside the ones that are merely empty on this row.
  function detailPairs(row) {
    if (row.status !== "ok") return [];
    return P.details
      .map(function (d) {
        return [d[0], cell(row, d[1], d[2])];
      })
      .filter(function (p) {
        return p[1] !== "" && p[1] != null;
      });
  }

  function tooltipBody(row, headline) {
    var nodes = [
      el("div", { class: "tip-value", text: headline }),
      el("div", { class: "tip-series" }, [
        el("span", {
          class: "key",
          style: "background:" + seriesColour(row.column),
        }),
        el("span", { text: seriesLabel(row) }),
      ]),
    ];
    var pairs = detailPairs(row);
    if (pairs.length) {
      var dl = el("dl");
      pairs.forEach(function (pair) {
        dl.appendChild(el("dt", { text: pair[0] }));
        dl.appendChild(el("dd", { text: String(pair[1]) }));
      });
      nodes.push(dl);
    }
    if (row.note) nodes.push(el("p", { class: "tip-note", text: row.note }));
    return nodes;
  }

  function headlineFor(row) {
    if (row.status !== "ok") return row.status;
    return duration(row.seconds);
  }

  function showTip(event, row, headline) {
    clear(tooltip);
    tooltipBody(row, headline || headlineFor(row)).forEach(function (n) {
      tooltip.appendChild(n);
    });
    tooltip.setAttribute("data-show", "true");
    positionTip(event);
  }

  function positionTip(event) {
    var pad = 14;
    var box = tooltip.getBoundingClientRect();
    var x = event.clientX + pad;
    var y = event.clientY + pad;
    if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - pad;
    if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - pad;
    tooltip.style.left = Math.max(8, x) + "px";
    tooltip.style.top = Math.max(8, y) + "px";
  }

  function hideTip() {
    tooltip.setAttribute("data-show", "false");
  }

  /* ── texture patterns: identity without hue ──────────────────────────── */

  function defsFor(names) {
    var defs = svgEl("defs");
    // The "did not measure the same thing" hatch, whichever audit column the
    // suite has. One neutral texture, so it reads as a caveat laid over a bar
    // rather than as another series.
    var caveat = svgEl("pattern", {
      id: "hatch-caveat",
      width: 6,
      height: 6,
      patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    caveat.appendChild(
      svgEl("rect", { width: 6, height: 6, fill: "var(--surface)", "fill-opacity": 0.001 })
    );
    caveat.appendChild(
      svgEl("line", {
        x1: 0,
        y1: 0,
        x2: 0,
        y2: 6,
        stroke: "var(--surface)",
        "stroke-width": 2.4,
        "stroke-opacity": 0.85,
      })
    );
    defs.appendChild(caveat);

    // One ordered texture per series, 45° and its 135° mirror only, inked
    // tone-on-tone. Off by default; the toggle is the accessibility channel,
    // not decoration.
    //
    // Keyed on the palette slot rather than on the name, for the same reason
    // the colour is: the slot is stable across filters, and an `internal`
    // environment label is free text that could otherwise collide once
    // sanitised into an element id.
    names.forEach(function (name) {
      var slot = slotOf(name);
      if (slot < 0) return;
      var pattern = svgEl("pattern", {
        id: "tex-slot-" + slot,
        width: 8,
        height: 8,
        patternUnits: "userSpaceOnUse",
        patternTransform: "rotate(" + (slot % 2 ? 135 : 45) + ")",
      });
      pattern.appendChild(
        svgEl("rect", { width: 8, height: 8, fill: seriesColour(name) })
      );
      var gap = 3 + Math.floor(slot / 2) * 2;
      for (var x = 0; x < 8; x += gap) {
        pattern.appendChild(
          svgEl("line", {
            x1: x,
            y1: 0,
            x2: x,
            y2: 8,
            stroke: "var(--ink)",
            "stroke-width": 1.4,
            "stroke-opacity": 0.4,
          })
        );
      }
      defs.appendChild(pattern);
    });
    return defs;
  }

  function fillFor(name) {
    var slot = slotOf(name);
    return state.texture && slot >= 0
      ? "url(#tex-slot-" + slot + ")"
      : seriesColour(name);
  }

  /* ── the timing view ─────────────────────────────────────────────────── */

  var BAR = 15; // thickness cap; the band's leftover is air
  var ROW = 21; // bar + the 2px surface gap the marks spec asks for
  var GROUP_GAP = 26;
  var PAD_TOP = 26;
  var PAD_BOTTOM = 34;

  function baselineSeries(all) {
    if (state.baseline) {
      var found = all.filter(function (s) {
        return s.key === state.baseline;
      });
      if (found.length) return found[0];
    }
    // The suite's own library leads where there is one to lead: a ratio in a
    // comparison report is "against ngio" unless the reader says otherwise.
    // `internal` names no baseline, because every series there is ngio.
    var preferred = !P.baseline
      ? []
      : all.filter(function (s) {
          return s.column === P.baseline;
        });
    return preferred.length ? preferred[0] : all[0];
  }

  function timingValue(row, base) {
    if (!row || row.status !== "ok" || row.seconds == null) return null;
    if (state.values === "absolute") return row.seconds;
    if (!base || base.seconds == null || !base.seconds) return null;
    return row.seconds / base.seconds;
  }

  function renderTiming(root, rows) {
    var facets = distinct(rows, state.facet);
    var all = seriesList(rows);
    var base = baselineSeries(all);

    facets.forEach(function (facetValue) {
      var facetRows = rows.filter(function (r) {
        return !state.facet || (r.axes[state.facet] || "") === facetValue;
      });
      if (!facetRows.length) return;
      root.appendChild(timingCard(facetValue, facetRows, all, base));
    });

    root.appendChild(timingLegend(all));
  }

  function timingCard(facetValue, rows, allSeries, base) {
    var groups = distinct(rows, state.group);
    var series = seriesList(rows);
    var byKey = {};
    rows.forEach(function (r) {
      byKey[(r.axes[state.group] || "") + "" + seriesKey(r)] = r;
    });

    // The baseline for a ratio is taken inside the same group, so a "2.4×"
    // always compares two rows that asked for the same downsampling filter.
    function baseRowFor(groupValue) {
      return base ? byKey[groupValue + "" + base.key] : null;
    }

    var values = [];
    groups.forEach(function (g) {
      series.forEach(function (s) {
        var v = timingValue(byKey[g + "" + s.key], baseRowFor(g));
        if (v != null && v > 0) values.push(v);
      });
    });

    var gutter = Math.min(
      210,
      Math.max(
        96,
        Math.ceil(
          Math.max.apply(
            null,
            series
              .map(function (s) {
                return textWidth(s.label);
              })
              .concat([60])
          )
        ) + 14
      )
    );
    var plotWidth = 660;
    var valueGutter = 92;
    var width = gutter + plotWidth + valueGutter;
    var height =
      PAD_TOP +
      PAD_BOTTOM +
      groups.length * GROUP_GAP +
      groups.length * series.length * ROW;

    var maxValue = values.length ? Math.max.apply(null, values) : 1;
    var minValue = values.length ? Math.min.apply(null, values) : 0.001;
    var logMode = state.scale === "log";
    var ratioMode = state.values === "ratio";
    var top = logMode ? Math.pow(10, Math.ceil(Math.log10(maxValue))) : niceMax(maxValue);
    var bottom = logMode
      ? Math.pow(10, Math.floor(Math.log10(Math.max(minValue, 1e-6))))
      : 0;
    // In ratio mode a bar grows from the 1× line — right for slower, left for
    // faster — so the direction carries the sign rather than a second colour.
    var pivot = ratioMode && !logMode ? Math.min(1, top) : bottom;

    function x(value) {
      if (logMode) {
        var lo = Math.log10(bottom);
        var hi = Math.log10(top);
        var v = Math.log10(Math.max(value, bottom));
        return gutter + ((v - lo) / (hi - lo)) * plotWidth;
      }
      return gutter + ((value - bottom) / (top - bottom)) * plotWidth;
    }

    var svg = svgEl("svg", {
      class: "chart",
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      role: "img",
      "aria-label":
        "Median wall-clock per " + P.columnLabel +
        (state.facet ? ", " + state.facet + " " + facetValue : "") +
        ". The table below carries the same numbers.",
    });
    svg.appendChild(defsFor(DATA.columns));

    var tickValues = logMode ? logTicks(bottom, top) : linearTicks(top, 5);
    tickValues.forEach(function (t) {
      svg.appendChild(
        svgEl("line", {
          class: "gridline",
          x1: x(t),
          x2: x(t),
          y1: PAD_TOP - 8,
          y2: height - PAD_BOTTOM + 4,
        })
      );
      svg.appendChild(
        svgEl("text", {
          class: "tick",
          x: x(t),
          y: height - PAD_BOTTOM + 18,
          "text-anchor": "middle",
          text: ratioMode ? times(t) : duration(t),
        })
      );
    });

    svg.appendChild(
      svgEl("line", {
        class: "baseline",
        x1: x(pivot),
        x2: x(pivot),
        y1: PAD_TOP - 8,
        y2: height - PAD_BOTTOM + 4,
      })
    );

    var y = PAD_TOP;
    groups.forEach(function (groupValue) {
      if (state.group) {
        svg.appendChild(
          svgEl("text", {
            class: "group-label",
            x: 0,
            y: y + 2,
            text: state.group + " = " + (groupValue || "—"),
          })
        );
        y += GROUP_GAP - 8;
      }
      var ordered = series.slice();
      if (state.sort === "value") {
        ordered.sort(function (a, b) {
          var av = timingValue(byKey[groupValue + "" + a.key], baseRowFor(groupValue));
          var bv = timingValue(byKey[groupValue + "" + b.key], baseRowFor(groupValue));
          if (av == null) return 1;
          if (bv == null) return -1;
          return av - bv;
        });
      }
      ordered.forEach(function (s) {
        var row = byKey[groupValue + "" + s.key];
        svg.appendChild(
          timingRow(row, s, y, x, pivot, gutter, plotWidth, baseRowFor(groupValue))
        );
        y += ROW;
      });
      y += 8;
    });

    var subhead =
      (ratioMode ? "median wall-clock, relative to " + base.label : "median wall-clock") +
      (logMode ? ", log scale" : "") +
      " · scale is per " +
      (state.facet || "chart");

    return card(
      state.facet ? state.facet + " = " + facetValue : "all rows",
      subhead,
      svg,
      rows,
      "timing"
    );
  }

  function timingRow(row, s, y, x, pivot, gutter, plotWidth, baseRow) {
    var g = svgEl("g", { class: "row" });
    var mid = y + ROW / 2;

    g.appendChild(
      svgEl("text", {
        class: "series-label",
        x: gutter - 10,
        y: mid + 4,
        "text-anchor": "end",
        text: s.label,
      })
    );

    // A ratio needs the baseline measured in this same group, and it often is
    // not: no writer here has a gaussian filter except ngff-zarr, so a
    // gaussian row divided by an ngio baseline has nothing to divide by. That
    // is a stub saying so, never a blank space.
    var measured = row && row.status === "ok" && row.seconds != null;
    if (measured && timingValue(row, baseRow) == null) {
      g.appendChild(
        svgEl("rect", {
          class: "ghost-bar",
          x: x(pivot),
          y: mid - 3,
          width: 26,
          height: 6,
          rx: 3,
        })
      );
      g.appendChild(
        svgEl("text", {
          class: "value-label",
          x: x(pivot) + 34,
          y: mid + 4,
          text: "no baseline here",
        })
      );
      attachTip(g, row, gutter, plotWidth, y, duration(row.seconds));
      return g;
    }

    if (!row || row.status !== "ok" || row.seconds == null) {
      // The slot survives. A capability gap that simply removed its bar would
      // read as "not run" instead of "cannot".
      var stub = svgEl("rect", {
        class: "ghost-bar",
        x: x(pivot),
        y: mid - 3,
        width: 26,
        height: 6,
        rx: 3,
      });
      g.appendChild(stub);
      g.appendChild(
        svgEl("text", {
          class: "value-label",
          x: x(pivot) + 34,
          y: mid + 4,
          text: row ? row.status : "not run",
        })
      );
      if (row) attachTip(g, row, gutter, plotWidth, y, row.status);
      return g;
    }

    var value = timingValue(row, baseRow);
    if (value == null) return g;

    // On a log axis a bar's length is not proportional to its value, so the
    // mark stops claiming to be one: position carries the number and the row
    // becomes a dot plot. The guide rule is a leader back to the label, drawn
    // in the gridline tone so it never reads as a magnitude.
    if (state.scale === "log") {
      g.appendChild(
        svgEl("line", {
          class: "gridline",
          x1: gutter,
          x2: gutter + plotWidth,
          y1: mid,
          y2: mid,
        })
      );
      g.appendChild(
        svgEl("circle", {
          class: "bar",
          cx: x(value),
          cy: mid,
          r: 5,
          fill: fillFor(row.column),
          stroke: "var(--surface)",
          "stroke-width": 2,
        })
      );
      if (row.fair === false) {
        g.appendChild(
          svgEl("circle", {
            cx: x(value),
            cy: mid,
            r: 8,
            fill: "none",
            stroke: "var(--st-warn)",
            "stroke-width": 1.5,
          })
        );
      }
      g.appendChild(
        svgEl("text", {
          class: "value-label",
          x: x(value) + 14,
          y: mid + 4,
          text: valueLabel(row, value),
        })
      );
      attachTip(g, row, gutter, plotWidth, y);
      return g;
    }

    var from = Math.min(x(pivot), x(value));
    var to = Math.max(x(pivot), x(value));
    var grows = x(value) >= x(pivot);

    // 4px rounded data-end, square at the baseline: the mark says which end is
    // the value and which end is the origin.
    var w = Math.max(to - from, 2);
    var r = Math.min(4, w / 2);
    var path = grows
      ? "M" + from + "," + (mid - BAR / 2) +
        "h" + (w - r) +
        "a" + r + "," + r + " 0 0 1 " + r + "," + r +
        "v" + (BAR - 2 * r) +
        "a" + r + "," + r + " 0 0 1 " + -r + "," + r +
        "H" + from + "z"
      : "M" + to + "," + (mid - BAR / 2) +
        "H" + (from + r) +
        "a" + r + "," + r + " 0 0 0 " + -r + "," + r +
        "v" + (BAR - 2 * r) +
        "a" + r + "," + r + " 0 0 0 " + r + "," + r +
        "H" + to + "z";

    g.appendChild(svgEl("path", { class: "bar", d: path, fill: fillFor(row.column) }));

    if (row.fair === false) {
      g.appendChild(
        svgEl("path", { class: "bar-caveat", d: path, fill: "url(#hatch-caveat)" })
      );
    }

    // Whiskers only where there is a spread to draw. At n=1 min == max, and a
    // zero-width whisker would assert a precision nobody measured.
    if (row.repeats > 1 && state.values === "absolute" && row.low != null) {
      var lo = x(row.low);
      var hi = x(row.high);
      g.appendChild(
        svgEl("line", { class: "whisker", x1: lo, x2: hi, y1: mid, y2: mid })
      );
      g.appendChild(
        svgEl("line", { class: "whisker", x1: lo, x2: lo, y1: mid - 4, y2: mid + 4 })
      );
      g.appendChild(
        svgEl("line", { class: "whisker", x1: hi, x2: hi, y1: mid - 4, y2: mid + 4 })
      );
    }

    g.appendChild(
      svgEl("text", {
        class: "value-label",
        x: to + 8,
        y: mid + 4,
        text: valueLabel(row, value),
      })
    );

    attachTip(g, row, gutter, plotWidth, y);
    return g;
  }

  // Every row in a `repeats = 1` file is n=1, and stamping that on 88 bars says
  // it 88 times when the banner above already said it once. The mark earns its
  // place only when the file mixes repeat counts, where it distinguishes a bar
  // that has a spread from one that never could.
  var MIXED_REPEATS =
    DATA.provenance.minRepeats !== DATA.provenance.maxRepeats;

  function valueLabel(row, value) {
    return (
      (state.values === "ratio" ? times(value) : duration(row.seconds)) +
      (row.fair === false ? " " + P.caveat.mark : "") +
      (MIXED_REPEATS && row.repeats === 1 ? "  n=1" : "")
    );
  }

  function attachTip(g, row, gutter, plotWidth, y, headline) {
    // The hit target is the whole band, not the painted pixels — a 15px bar is
    // not something to ask anyone to land on.
    var hit = svgEl("rect", {
      class: "hit",
      x: gutter - 200,
      y: y,
      width: plotWidth + 300,
      height: ROW,
    });
    hit.addEventListener("pointerenter", function (e) {
      g.setAttribute("data-active", "true");
      showTip(e, row, headline);
    });
    hit.addEventListener("pointermove", positionTip);
    hit.addEventListener("pointerleave", function () {
      g.removeAttribute("data-active");
      hideTip();
    });
    hit.addEventListener("click", function () {
      pin(row);
    });
    g.appendChild(hit);
  }

  function timingLegend(series) {
    var names = [];
    series.forEach(function (s) {
      if (names.indexOf(s.column) === -1) names.push(s.column);
    });
    var list = el("ul", { class: "legend" });

    // One series in the whole file means colour distinguishes nothing, and a
    // one-item swatch list would dress that up as a key. `internal` with no
    // `[[environments]]` is the ordinary case of this, not an edge one.
    var single = DATA.columns.length === 1;
    if (!single) {
      names.forEach(function (name) {
        list.appendChild(
          el("li", {}, [
            el("span", { class: "swatch", style: "background:" + seriesColour(name) }),
            el("span", { text: name }),
          ])
        );
      });
    }
    list.appendChild(
      el("li", {}, [
        el("span", {
          class: "swatch",
          style: "background:var(--st-unsupported);opacity:.6",
        }),
        el("span", { text: "unsupported or not run" }),
      ])
    );
    if (P.caveat) {
      list.appendChild(
        el("li", {}, [
          el("span", {
            class: "swatch",
            style:
              "background:" +
              "repeating-linear-gradient(45deg,var(--muted) 0 2px,transparent 2px 5px)",
          }),
          el("span", { text: P.caveat.legend }),
        ])
      );
    }

    var wrap = el("div", { class: "card" }, [
      el("h3", { text: "Reading these bars" }),
      list,
    ]);
    if (single) {
      wrap.appendChild(
        el("p", {
          class: "legend-note",
          text:
            "One " + P.columnLabel + " in this file, so every bar is the same " +
            "colour. What tells two of them apart is the label beside each one.",
        })
      );
    }
    // Assembled in Python: the colour paragraph, then the audit column's own
    // explanation when the suite has one.
    P.legend.forEach(function (text) {
      wrap.appendChild(el("p", { class: "legend-note", text: text }));
    });
    return wrap;
  }

  /* ── the coverage view ───────────────────────────────────────────────── */

  var STATE_WORDS = {
    ok: "ok",
    unsupported: "cannot",
    unavailable: "no env",
    failed: "failed",
  };

  function renderCoverage(root, rows) {
    var facets = distinct(rows, state.facet);
    facets.forEach(function (facetValue) {
      var facetRows = rows.filter(function (r) {
        return !state.facet || (r.axes[state.facet] || "") === facetValue;
      });
      if (!facetRows.length) return;
      root.appendChild(coverageCard(facetValue, facetRows));
    });
    root.appendChild(coverageLegend());
  }

  function coverageCard(facetValue, rows) {
    var groups = distinct(rows, state.group);
    var series = seriesList(rows);
    var byKey = {};
    rows.forEach(function (r) {
      byKey[(r.axes[state.group] || "") + "" + seriesKey(r)] = r;
    });

    // With no group axis the matrix has one column and nothing to head it, so
    // it stops being a matrix and becomes a list: no header row, and a natural
    // width rather than a single cell stretched across the card. `compare-io`
    // faceted by operation and `internal` faceted by block both land here.
    var listed = !state.group && groups.length === 1;

    var table = el("table", { class: "matrix" });
    if (listed) table.setAttribute("data-single", "true");
    if (!listed) {
      var head = el("tr", {}, [el("th", { text: "" })]);
      groups.forEach(function (g) {
        head.appendChild(
          el("th", { scope: "col", text: (state.group ? state.group + " = " : "") + (g || "—") })
        );
      });
      table.appendChild(el("thead", {}, [head]));
    }

    var body = el("tbody");
    series.forEach(function (s) {
      var tr = el("tr");
      tr.appendChild(
        el("th", { scope: "row" }, [
          el("div", { class: "row-head" }, [
            el("span", {
              class: "swatch",
              style: "background:" + seriesColour(s.column),
            }),
            el("span", { text: s.label }),
          ]),
        ])
      );
      groups.forEach(function (g) {
        tr.appendChild(el("td", {}, [coverageCell(byKey[g + "" + s.key])]));
      });
      body.appendChild(tr);
    });
    table.appendChild(body);
    var scroll = el("div", { class: "table-scroll" }, [table]);
    return card(
      state.facet ? state.facet + " = " + facetValue : "all rows",
      P.coverageSubhead,
      scroll,
      rows,
      "coverage"
    );
  }

  function coverageCell(row) {
    if (!row) {
      return el("div", { class: "cell", "data-status": "none" }, [
        el("span", { class: "state", text: "not run" }),
      ]);
    }
    var button = el("button", {
      class: "cell",
      type: "button",
      "data-status": row.status,
      title: row.note || "",
    });
    button.appendChild(
      el("span", { class: "state", text: STATE_WORDS[row.status] || row.status })
    );
    if (row.status === "ok") {
      button.appendChild(el("span", { class: "metric", text: duration(row.seconds) }));
      if (row.fair === false) {
        button.appendChild(el("span", { class: "caveat", text: P.caveat.chip }));
      } else if (P.coverageChip && row[P.coverageChip]) {
        button.appendChild(
          el("span", {
            class: "caveat",
            style: "color:var(--faint)",
            text: row[P.coverageChip],
          })
        );
      }
    }
    button.addEventListener("pointerenter", function (e) {
      showTip(e, row);
    });
    button.addEventListener("pointermove", positionTip);
    button.addEventListener("pointerleave", hideTip);
    button.addEventListener("focus", function () {
      var box = button.getBoundingClientRect();
      showTip({ clientX: box.left, clientY: box.bottom }, row);
    });
    button.addEventListener("blur", hideTip);
    button.addEventListener("click", function () {
      pin(row);
    });
    return button;
  }

  function coverageLegend() {
    var list = el("ul", { class: "legend" });
    [
      ["ok", "measured"],
      ["unsupported", "the writer declares it cannot express this"],
      ["unavailable", "the environment failed to install or import"],
      ["failed", "it ran and raised"],
      ["none", "not in this file"],
    ].forEach(function (pair) {
      var swatch = el("span", { class: "swatch" });
      swatch.setAttribute("data-status", pair[0]);
      swatch.style.background =
        pair[0] === "ok"
          ? "var(--accent-soft)"
          : pair[0] === "unsupported"
            ? "var(--sunk)"
            : pair[0] === "none"
              ? "transparent"
              : "var(--st-" + pair[0] + ")";
      if (pair[0] === "none") swatch.style.border = "1px dashed var(--line-strong)";
      list.appendChild(el("li", {}, [swatch, el("span", { text: pair[1] })]));
    });
    return el("div", { class: "card" }, [
      el("h3", { text: "Reading this matrix" }),
      list,
      el("p", {
        class: "legend-note",
        text:
          "A blank cell is a claim, so the four reasons a case has no number stay " +
          "apart: the writer declared it cannot do this, its environment never " +
          "installed, it ran and raised, or the combination is not in this file at " +
          "all. Hover or focus any cell for the reason the adapter gave.",
      }),
    ]);
  }

  /* ── the memory and CPU view ─────────────────────────────────────────── */

  function renderMemory(root, rows) {
    var facets = distinct(rows, state.facet);
    facets.forEach(function (facetValue) {
      var facetRows = rows.filter(function (r) {
        return !state.facet || (r.axes[state.facet] || "") === facetValue;
      });
      var measured = facetRows.filter(function (r) {
        return r.status === "ok";
      });
      if (!measured.length) return;
      // A suite that recorded no memory figure at all would otherwise draw a
      // card of empty bars. `internal` records tracemalloc's, the comparison
      // suites record the RSS split; neither draws when the column is blank.
      if (!measured.some(memoryValue)) return;
      // The whole facet goes in, not just the measured rows: a writer that
      // cannot express this filter should say so in its slot, exactly as it
      // does on the timing chart, rather than reading as "not run".
      root.appendChild(memoryCard(facetValue, facetRows, measured));
    });
    root.appendChild(cpuCard(rows.filter(function (r) {
      return r.status === "ok" && r.parallelism != null;
    })));
    root.appendChild(memoryLegend());
  }

  // Which memory figure this suite actually recorded. The comparison runners
  // take a baseline before each case (`compare/_run.py`), so their rows carry
  // the import/case split; `internal`'s blocks share one interpreter and take
  // none, so its process high-water mark is a run-wide number wearing a row's
  // label and the honest per-case column is tracemalloc's.
  var SPLIT = P.memory === "split";

  function memoryValue(row) {
    if (!row) return null;
    if (SPLIT) return row.rssBaseMb == null ? null : row.rssBaseMb + (row.caseMb || 0);
    return row.peakMb;
  }

  function memoryCard(facetValue, rows, measured) {
    var groups = distinct(rows, state.group);
    var series = seriesList(rows);
    var byKey = {};
    rows.forEach(function (r) {
      byKey[(r.axes[state.group] || "") + "" + seriesKey(r)] = r;
    });

    var totals = measured.map(function (r) {
      return memoryValue(r) || 0;
    });
    var top = niceMax(Math.max.apply(null, totals.concat([1])));

    var gutter = 200;
    var plotWidth = 560;
    var valueGutter = 270;
    var width = gutter + plotWidth + valueGutter;
    var height =
      PAD_TOP + PAD_BOTTOM + groups.length * GROUP_GAP + groups.length * series.length * ROW;

    function x(v) {
      return gutter + (v / top) * plotWidth;
    }

    var svg = svgEl("svg", {
      class: "chart",
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      role: "img",
      "aria-label":
        (SPLIT
          ? "Peak resident memory split into import cost and case cost. "
          : "Peak memory Python's allocator accounted for, per case. ") +
        "The table below carries the same numbers.",
    });
    svg.appendChild(defsFor(DATA.columns));

    linearTicks(top, 5).forEach(function (t) {
      svg.appendChild(
        svgEl("line", {
          class: "gridline",
          x1: x(t),
          x2: x(t),
          y1: PAD_TOP - 8,
          y2: height - PAD_BOTTOM + 4,
        })
      );
      svg.appendChild(
        svgEl("text", {
          class: "tick",
          x: x(t),
          y: height - PAD_BOTTOM + 18,
          "text-anchor": "middle",
          // Enough places for the axis to have distinct labels. Whole numbers
          // suit the hundreds of MB a writer moves; `internal` measures single
          // megabytes per case, where five ticks rounded to integers would read
          // "0, 1, 1, 2, 2 MB" and claim the scale repeats itself.
          text: (top < 10 ? t.toFixed(1) : t.toFixed(0)) + " MB",
        })
      );
    });

    var y = PAD_TOP;
    groups.forEach(function (groupValue) {
      if (state.group) {
        svg.appendChild(
          svgEl("text", {
            class: "group-label",
            x: 0,
            y: y + 2,
            text: state.group + " = " + (groupValue || "—"),
          })
        );
        y += GROUP_GAP - 8;
      }
      series.forEach(function (s) {
        var row = byKey[groupValue + "" + s.key];
        svg.appendChild(memoryRow(row, s, y, x, gutter, plotWidth));
        y += ROW;
      });
      y += 8;
    });

    return card(
      state.facet ? state.facet + " = " + facetValue : "all rows",
      P.memorySubhead,
      svg,
      rows,
      "memory"
    );
  }

  function memoryRow(row, s, y, x, gutter, plotWidth) {
    var g = svgEl("g", { class: "row" });
    var mid = y + ROW / 2;
    g.appendChild(
      svgEl("text", {
        class: "series-label",
        x: gutter - 10,
        y: mid + 4,
        "text-anchor": "end",
        text: s.label,
      })
    );
    if (memoryValue(row) == null) {
      g.appendChild(
        svgEl("rect", {
          class: "ghost-bar",
          x: x(0),
          y: mid - 3,
          width: 26,
          height: 6,
          rx: 3,
        })
      );
      // A measured row with no figure is not a row that did not run. Under
      // tracemalloc that is an adapter allocating in native buffers Python
      // never saw, which is a fact about the library, not a gap in the file.
      var why = !row
        ? "not run"
        : row.status !== "ok"
          ? row.status
          : "n/a — native buffers";
      g.appendChild(
        svgEl("text", {
          class: "value-label",
          x: x(0) + 34,
          y: mid + 4,
          text: why,
        })
      );
      if (row) attachTip(g, row, gutter, plotWidth, y, why);
      return g;
    }

    // One bar, one claim: what Python's allocator accounted for inside the
    // case. There is no import cost to lay beneath it -- the blocks share an
    // interpreter, so nothing was imported per case.
    if (!SPLIT) {
      g.appendChild(
        svgEl("rect", {
          class: "bar",
          x: x(0),
          y: mid - BAR / 2,
          width: Math.max(x(row.peakMb) - x(0), 1),
          height: BAR,
          rx: 3,
          fill: fillFor(row.column),
        })
      );
      g.appendChild(
        svgEl("text", {
          class: "value-label",
          x: x(row.peakMb) + 8,
          y: mid + 4,
          text: megabytes(row.peakMb),
        })
      );
      attachTip(g, row, gutter, plotWidth, y, megabytes(row.peakMb) + " for the case");
      return g;
    }

    var baseEnd = x(row.rssBaseMb);
    var caseEnd = x(row.rssBaseMb + (row.caseMb || 0));

    // Import cost, in a recessive step of the series hue: it is the price of
    // admission, not what the case did.
    g.appendChild(
      svgEl("rect", {
        class: "bar",
        x: x(0),
        y: mid - BAR / 2,
        width: Math.max(baseEnd - x(0), 1),
        height: BAR,
        fill: fillFor(row.column),
        "fill-opacity": 0.32,
      })
    );
    // The 2px surface gap does the separating, never a stroke.
    g.appendChild(
      svgEl("rect", {
        class: "bar",
        x: baseEnd + 2,
        y: mid - BAR / 2,
        width: Math.max(caseEnd - baseEnd - 2, 1),
        height: BAR,
        rx: 3,
        fill: fillFor(row.column),
      })
    );

    // tracemalloc's own figure, where the adapter is not native. It is a
    // different measurement from the RSS bar, so it rides as a tick rather
    // than as a second bar on the same row.
    if (row.peakMb != null) {
      var px = x(Math.min(row.rssBaseMb + row.peakMb, 1e9));
      g.appendChild(
        svgEl("line", {
          class: "peak-mark",
          x1: px,
          x2: px,
          y1: mid - BAR / 2 - 3,
          y2: mid + BAR / 2 + 3,
        })
      );
    }

    g.appendChild(
      svgEl("text", {
        class: "value-label",
        x: caseEnd + 8,
        y: mid + 4,
        // Named, not just added: the two segments are different claims, and
        // "199.1 + 86.9 MB" invites reading them as one total.
        text:
          "case " +
          megabytes(row.caseMb) +
          " · import " +
          megabytes(row.rssBaseMb) +
          (row.peakMb == null ? " · tracemalloc n/a" : ""),
      })
    );
    attachTip(g, row, gutter, plotWidth, y, megabytes(row.caseMb) + " for the case");
    return g;
  }

  function cpuCard(rows) {
    var series = seriesList(rows);
    var byKey = {};
    var counts = {};
    // One bar per series: cpu/wall barely moves with the filter, so this panel
    // shows the median across whatever is visible and says so.
    rows.forEach(function (r) {
      var k = seriesKey(r);
      (counts[k] = counts[k] || []).push(r.parallelism);
      if (!byKey[k]) byKey[k] = r;
    });

    var medians = {};
    Object.keys(counts).forEach(function (k) {
      var v = counts[k].slice().sort(function (a, b) {
        return a - b;
      });
      medians[k] = v[Math.floor(v.length / 2)];
    });

    var top = niceMax(Math.max.apply(null, Object.keys(medians).map(function (k) {
      return medians[k];
    }).concat([1.2])));

    var gutter = 200;
    var plotWidth = 620;
    var width = gutter + plotWidth + 90;
    var height = PAD_TOP + PAD_BOTTOM + series.length * ROW;

    function x(v) {
      return gutter + (v / top) * plotWidth;
    }

    var svg = svgEl("svg", {
      class: "chart",
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      role: "img",
      "aria-label":
        "CPU seconds divided by wall seconds, per " + P.columnLabel + ". Above one " +
        "means the library used more than one thread.",
    });
    svg.appendChild(defsFor(DATA.columns));

    linearTicks(top, 4).forEach(function (t) {
      svg.appendChild(
        svgEl("line", {
          class: "gridline",
          x1: x(t),
          x2: x(t),
          y1: PAD_TOP - 8,
          y2: height - PAD_BOTTOM + 4,
        })
      );
      svg.appendChild(
        svgEl("text", {
          class: "tick",
          x: x(t),
          y: height - PAD_BOTTOM + 18,
          "text-anchor": "middle",
          text: times(t),
        })
      );
    });

    svg.appendChild(
      svgEl("line", {
        class: "reference",
        x1: x(1),
        x2: x(1),
        y1: PAD_TOP - 8,
        y2: height - PAD_BOTTOM + 4,
      })
    );
    svg.appendChild(
      svgEl("text", {
        class: "tick",
        x: x(1),
        y: PAD_TOP - 14,
        "text-anchor": "middle",
        text: "single-threaded",
      })
    );

    var y = PAD_TOP;
    series.forEach(function (s) {
      var g = svgEl("g", { class: "row" });
      var mid = y + ROW / 2;
      g.appendChild(
        svgEl("text", {
          class: "series-label",
          x: gutter - 10,
          y: mid + 4,
          "text-anchor": "end",
          text: s.label,
        })
      );
      var value = medians[s.key];
      var w = Math.max(x(value) - x(0), 2);
      g.appendChild(
        svgEl("rect", {
          class: "bar",
          x: x(0),
          y: mid - BAR / 2,
          width: w,
          height: BAR,
          rx: 3,
          fill: fillFor(s.column),
        })
      );
      g.appendChild(
        svgEl("text", {
          class: "value-label",
          x: x(value) + 8,
          y: mid + 4,
          text: times(value),
        })
      );
      attachTip(g, byKey[s.key], gutter, plotWidth, y, times(value) + " cpu / wall");
      svg.appendChild(g);
      y += ROW;
    });

    return card(
      "cpu / wall",
      "median across every visible case — above 1× the library used threads",
      svg,
      rows,
      "cpu"
    );
  }

  function memoryLegend() {
    if (!SPLIT) {
      return el("div", { class: "card" }, [
        el("h3", { text: "Reading these bars" }),
        el("ul", { class: "legend" }, [
          el("li", {}, [
            el("span", { class: "swatch", style: "background:var(--s-slot-0)" }),
            el("span", { text: "peak memory tracemalloc accounted for, per case" }),
          ]),
        ]),
        el("p", {
          class: "legend-note",
          text:
            "Python's allocator, not the operating system's. It is the per-case " +
            "column because the blocks share one interpreter: the process " +
            "high-water mark is a single number for the whole run, and repeating " +
            "it on every row would say nothing about any of them. It is in the " +
            "tooltip as `process peak RSS`, named for what it is.",
        }),
        el("p", {
          class: "legend-note",
          text:
            "A row reading n/a has no figure rather than a figure of nothing — " +
            "an allocation Python's allocator never saw.",
        }),
      ]);
    }
    return el("div", { class: "card" }, [
      el("h3", { text: "Reading these bars" }),
      el("ul", { class: "legend" }, [
        el("li", {}, [
          el("span", {
            class: "swatch",
            style: "background:var(--s-slot-0);opacity:.32",
          }),
          el("span", { text: "import cost — the process before the case ran" }),
        ]),
        el("li", {}, [
          el("span", { class: "swatch", style: "background:var(--s-slot-0)" }),
          el("span", { text: "case cost — peak RSS above that baseline" }),
        ]),
        el("li", {}, [
          el("span", { class: "swatch line", style: "background:var(--ink);width:2px;height:14px" }),
          el("span", { text: "tracemalloc's own peak, where it can see one" }),
        ]),
      ]),
      el("p", {
        class: "legend-note",
        text:
          "The two segments are different claims. Import cost is what merely " +
          "loading the library costs any program that imports it; case cost is the " +
          "high-water mark the pyramid build added on top. A writer with a small " +
          "case cost and a large import cost is not a frugal writer.",
      }),
      el("p", {
        class: "legend-note",
        text:
          "Where the tracemalloc tick is missing the row reads n/a, not zero: that " +
          "adapter allocates in C++ or Rust buffers Python's allocator never sees, " +
          "so there is no figure to report rather than a figure of nothing.",
      }),
    ]);
  }

  /* ── cards, tables, detail ───────────────────────────────────────────── */

  function card(title, subhead, content, rows, kind) {
    var node = el("section", { class: "card" }, [
      el("header", {}, [
        el("div", {}, [
          el("h2", { text: title }),
          el("p", { class: "lede", text: subhead }),
        ]),
      ]),
      el("div", { class: "plot" }, [content]),
      tableView(rows, kind),
    ]);
    return node;
  }

  function tableView(rows, kind) {
    var details = el("details", { class: "table-view" });
    details.appendChild(
      el("summary", { text: "Table view — the same numbers, without the chart" })
    );
    var columns = [
      ["series", function (r) { return seriesLabel(r); }, false],
      [state.group || "case", function (r) { return state.group ? r.axes[state.group] : r.case; }, false],
      ["status", function (r) { return r.status; }, false],
    ];
    // The columns every suite has are here; the trailing ones a suite has
    // because its schema has them come from the profile, so a table never
    // carries a header for a column the file could not fill.
    function fromProfile(which) {
      P.tables[which].forEach(function (c) {
        columns.push([
          c[0],
          function (r) {
            return cell(r, c[1], c[2]);
          },
          c[3],
        ]);
      });
    }

    if (kind === "cpu") {
      columns.push(["cpu / wall", function (r) { return times(r.parallelism); }, true]);
    } else if (kind === "memory") {
      if (SPLIT) {
        columns.push(["import MB", function (r) { return megabytes(r.rssBaseMb); }, true]);
        columns.push(["case MB", function (r) { return megabytes(r.caseMb); }, true]);
        columns.push(["tracemalloc", function (r) { return r.peakMb == null ? "n/a" : megabytes(r.peakMb); }, true]);
      } else {
        columns.push(["tracemalloc", function (r) { return r.peakMb == null ? "n/a" : megabytes(r.peakMb); }, true]);
        columns.push(["process peak MB", function (r) { return megabytes(r.procPeakMb); }, true]);
      }
    } else if (kind === "coverage") {
      columns.push(["median", function (r) { return duration(r.seconds); }, true]);
      fromProfile("coverage");
    } else {
      columns.push(["median", function (r) { return duration(r.seconds); }, true]);
      columns.push(["spread", function (r) { return r.status === "ok" ? spread(r) : ""; }, true]);
      fromProfile("timing");
    }

    var table = el("table", { class: "data-table" });
    var head = el("tr");
    columns.forEach(function (c) {
      head.appendChild(el("th", { scope: "col", text: c[0] }));
    });
    table.appendChild(el("thead", {}, [head]));
    var body = el("tbody");
    rows.forEach(function (r) {
      var tr = el("tr");
      columns.forEach(function (c, i) {
        var td = el("td", { class: c[2] ? "num" : "" });
        if (i === 0) {
          td.appendChild(
            el("span", { class: "swatch", style: "background:" + seriesColour(r.column) })
          );
        }
        td.appendChild(document.createTextNode(String(c[1](r) || "")));
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.appendChild(body);
    details.appendChild(el("div", { class: "table-scroll" }, [table]));
    return details;
  }

  function pin(row) {
    state.pinned = row;
    renderDetail();
    var panel = document.getElementById("detail");
    if (panel && panel.firstChild) panel.scrollIntoView({ block: "nearest" });
  }

  function renderDetail() {
    var panel = clear(document.getElementById("detail"));
    var row = state.pinned;
    if (!row) return;
    var dl = el("dl");
    var pairs = [["case", row.case], ["status", row.status]].concat(detailPairs(row));
    if (row.syncSeconds != null) pairs.push(["os.sync after", duration(row.syncSeconds)]);
    if (row.zarrFormat) pairs.push(["zarr format", "v" + row.zarrFormat]);
    if (row.python) pairs.push(["python", row.python]);
    if (row.zarr) pairs.push(["zarr", row.zarr]);
    if (row.platform) pairs.push(["platform", row.platform]);
    pairs.forEach(function (pair) {
      dl.appendChild(el("dt", { text: pair[0] }));
      dl.appendChild(el("dd", { text: String(pair[1]) }));
    });
    var close = el("button", { class: "close", type: "button", text: "close" });
    close.addEventListener("click", function () {
      state.pinned = null;
      renderDetail();
    });
    panel.appendChild(
      el("section", { class: "card" }, [
        el("header", {}, [
          el("div", {}, [
            el("span", { class: "eyebrow", text: "pinned row" }),
            el("h2", { text: seriesLabel(row) }),
          ]),
          close,
        ]),
        row.note ? el("p", { class: "notice", text: row.note }) : null,
        dl,
      ])
    );
  }

  /* ── chrome ──────────────────────────────────────────────────────────── */

  function masthead() {
    var p = DATA.provenance;
    var head = el("header", { class: "masthead" }, [
      el("div", { class: "title" }, [
        el("span", { class: "eyebrow", text: P.eyebrow }),
        el("h1", {}, [document.createTextNode(P.title)]),
        el("p", {
          class: "source",
          text:
            p.rows +
            " rows · " +
            Object.keys(p.statuses)
              .map(function (s) {
                return p.statuses[s] + " " + s;
              })
              .join(", "),
        }),
      ]),
      el("div", { class: "tools" }, [themeToggle(), textureToggle()]),
    ]);

    var facts = el("ul", { class: "provenance" });
    [
      [P.groupLabel, p.groups.join(", ")],
      ["platform", p.platform.join(", ")],
      ["python", p.python.join(", ")],
      ["zarr", p.zarr.join(", ")],
      ["zarr format", p.zarrFormat.map(function (v) { return "v" + v; }).join(", ")],
      [
        "timed runs",
        p.minRepeats == null
          ? "—"
          : p.minRepeats === p.maxRepeats
            ? String(p.minRepeats)
            : p.minRepeats + "–" + p.maxRepeats,
      ],
      [
        P.versionsLabel,
        Object.keys(p.versions)
          .map(function (name) {
            return name + " " + p.versions[name].join("/");
          })
          .join(" · "),
      ],
    ]
      .filter(function (pair) {
        return pair[1];
      })
      .forEach(function (pair) {
        facts.appendChild(
          el("li", {}, [
            el("dt", { text: pair[0] }),
            el("dd", { text: pair[1] }),
          ])
        );
      });

    return el("div", {}, [head, facts]);
  }

  function themeToggle() {
    var wrap = el("div", { class: "segmented", role: "group", "aria-label": "theme" });
    [["system", "auto"], ["light", "light"], ["dark", "dark"]].forEach(function (pair) {
      var button = el("button", { type: "button", text: pair[1] });
      button.setAttribute("aria-pressed", String(state.theme === pair[0]));
      button.addEventListener("click", function () {
        state.theme = pair[0];
        if (pair[0] === "system") document.documentElement.removeAttribute("data-theme");
        else document.documentElement.setAttribute("data-theme", pair[0]);
        render();
      });
      wrap.appendChild(button);
    });
    return wrap;
  }

  function textureToggle() {
    var button = el("button", {
      class: "ghost",
      type: "button",
      text: state.texture ? "texture on" : "texture off",
      title:
        "Carry series identity with 45°/135° hatching as well as hue, for " +
        "colour-vision deficiency, greyscale print, or forced colours.",
    });
    button.addEventListener("click", function () {
      state.texture = !state.texture;
      document.body.setAttribute("data-texture", state.texture ? "on" : "off");
      render();
    });
    return button;
  }

  function notices() {
    if (!DATA.notices.length) return null;
    var wrap = el("div", { class: "notices" });
    DATA.notices.forEach(function (text) {
      wrap.appendChild(el("p", { class: "notice", text: text }));
    });
    return wrap;
  }

  function controls() {
    var bar = el("div", { class: "controls" });

    bar.appendChild(
      axisSelect("facet", "facet by", state.facet, function (value) {
        state.facet = value || null;
        if (state.group === state.facet) state.group = null;
      })
    );
    bar.appendChild(
      axisSelect("group", "group by", state.group, function (value) {
        state.group = value || null;
        if (state.facet === state.group) state.facet = null;
      })
    );

    if (state.view === "timing") {
      bar.appendChild(
        segmented("scale", [["linear", "linear"], ["log", "log"]], state.scale, function (v) {
          state.scale = v;
        })
      );
      bar.appendChild(
        segmented(
          "values",
          [["absolute", "absolute"], ["ratio", "vs baseline"]],
          state.values,
          function (v) {
            state.values = v;
          }
        )
      );
      if (state.values === "ratio") bar.appendChild(baselineSelect());
      bar.appendChild(
        segmented("order", [["fixed", "fixed"], ["value", "fastest first"]], state.sort, function (v) {
          state.sort = v;
        })
      );
    }

    AXES.forEach(function (axis) {
      bar.appendChild(filterMenu(axis));
    });

    var reset = el("button", { class: "ghost", type: "button", text: "reset filters" });
    reset.addEventListener("click", function () {
      AXES.forEach(function (a) {
        state.filters[a.field] = a.values.slice();
      });
      render();
    });
    bar.appendChild(el("div", { class: "control" }, [el("span", { class: "label", text: " " }), reset]));
    return bar;
  }

  function axisSelect(id, label, value, onchange) {
    var select = el("select", { id: "axis-" + id });
    var none = el("option", { value: "", text: "— none —" });
    if (!value) none.selected = true;
    select.appendChild(none);
    AXES.forEach(function (axis) {
      var option = el("option", { value: axis.field, text: axis.field });
      if (axis.field === value) option.selected = true;
      select.appendChild(option);
    });
    select.addEventListener("change", function () {
      onchange(select.value);
      render();
    });
    return el("div", { class: "control" }, [
      el("label", { for: "axis-" + id, text: label }),
      select,
    ]);
  }

  function segmented(label, options, current, onchange) {
    var wrap = el("div", { class: "segmented", role: "group", "aria-label": label });
    options.forEach(function (pair) {
      var button = el("button", { type: "button", text: pair[1] });
      button.setAttribute("aria-pressed", String(current === pair[0]));
      button.addEventListener("click", function () {
        onchange(pair[0]);
        render();
      });
      wrap.appendChild(button);
    });
    return el("div", { class: "control" }, [
      el("span", { class: "label", text: label }),
      wrap,
    ]);
  }

  function baselineSelect() {
    var all = seriesList(visibleRows());
    var base = baselineSeries(all);
    var select = el("select", { id: "baseline" });
    all.forEach(function (s) {
      var option = el("option", { value: s.key, text: s.label });
      if (s.key === base.key) option.selected = true;
      select.appendChild(option);
    });
    select.addEventListener("change", function () {
      state.baseline = select.value;
      render();
    });
    return el("div", { class: "control" }, [
      el("label", { for: "baseline", text: "baseline" }),
      select,
    ]);
  }

  function filterMenu(axis) {
    var chosen = state.filters[axis.field];
    var details = el("details", { class: "filter" });
    var summary = el("summary", {
      text:
        axis.field +
        (chosen.length === axis.values.length
          ? " · all"
          : " · " + chosen.length + "/" + axis.values.length),
    });
    details.appendChild(summary);
    var menu = el("div", { class: "menu" });
    axis.values.forEach(function (value) {
      var input = el("input", { type: "checkbox" });
      input.checked = chosen.indexOf(value) !== -1;
      input.addEventListener("change", function () {
        var next = state.filters[axis.field].slice();
        var at = next.indexOf(value);
        if (input.checked && at === -1) next.push(value);
        if (!input.checked && at !== -1) next.splice(at, 1);
        state.filters[axis.field] = axis.values.filter(function (v) {
          return next.indexOf(v) !== -1;
        });
        render(true);
      });
      menu.appendChild(el("label", {}, [input, el("span", { text: value })]));
    });
    var actions = el("div", { class: "menu-actions" });
    var allButton = el("button", { type: "button", text: "all" });
    allButton.addEventListener("click", function () {
      state.filters[axis.field] = axis.values.slice();
      render(true);
    });
    var noneButton = el("button", { type: "button", text: "none" });
    noneButton.addEventListener("click", function () {
      state.filters[axis.field] = [];
      render(true);
    });
    actions.appendChild(allButton);
    actions.appendChild(noneButton);
    menu.appendChild(actions);
    details.appendChild(menu);
    return el("div", { class: "control" }, [
      el("span", { class: "label", text: "filter" }),
      details,
    ]);
  }

  function tabs() {
    var nav = el("div", { class: "views", role: "tablist" });
    [
      ["timing", "Timing"],
      ["coverage", "Coverage"],
      ["memory", "Memory & CPU"],
    ].forEach(function (pair) {
      var button = el("button", { type: "button", role: "tab", text: pair[1] });
      button.setAttribute("aria-selected", String(state.view === pair[0]));
      button.addEventListener("click", function () {
        state.view = pair[0];
        render();
      });
      nav.appendChild(button);
    });
    return nav;
  }

  /* ── render ──────────────────────────────────────────────────────────── */

  var openFilters = {};

  function render(keepFilters) {
    if (keepFilters) {
      openFilters = {};
      document.querySelectorAll("details.filter[open]").forEach(function (node) {
        openFilters[node.querySelector("summary").textContent.split(" ·")[0]] = true;
      });
    }
    var app = clear(document.getElementById("app"));
    app.removeAttribute("aria-busy");
    app.appendChild(masthead());
    var note = notices();
    if (note) app.appendChild(note);
    app.appendChild(controls());
    app.appendChild(tabs());

    var main = el("main");
    var rows = visibleRows();
    if (!rows.length) {
      main.appendChild(
        el("p", {
          class: "empty",
          text: "Nothing selected. Widen a filter, or reset them.",
        })
      );
    } else if (state.view === "timing") {
      renderTiming(main, rows);
    } else if (state.view === "coverage") {
      renderCoverage(main, rows);
    } else {
      renderMemory(main, rows);
    }
    app.appendChild(main);
    app.appendChild(el("div", { class: "detail", id: "detail" }));
    renderDetail();

    if (keepFilters) {
      document.querySelectorAll("details.filter").forEach(function (node) {
        var field = node.querySelector("summary").textContent.split(" ·")[0];
        if (openFilters[field]) node.open = true;
      });
    }
  }

  document.body.setAttribute("data-texture", "off");
  render();
})();
