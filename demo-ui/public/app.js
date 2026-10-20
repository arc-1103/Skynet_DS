// Skynet_DS review console. Vanilla JS over three endpoints; every number on
// the page comes from their payloads, and every panel says where it came from.

const SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"];
const SEV_VAR = { CRITICAL: "crit", HIGH: "high", MEDIUM: "med", LOW: "low", INFO: "info" };
const BLOCKING = new Set(["CRITICAL", "HIGH"]); // what the synthesizer fails a PR on

// Mirrors src/drift/methods.py: KS and MMD flag drift when p < threshold;
// PSI, Euclidean, Cosine distance and the domain classifier when
// statistic >= threshold; Page-Hinkley only when its change point falls in
// the current window.
const METHODS = [
  { id: "ks", short: "KS", full: "Kolmogorov–Smirnov test", family: "test", rule: "p" },
  { id: "mmd", short: "MMD", full: "Maximum Mean Discrepancy", family: "test", rule: "p" },
  { id: "psi", short: "PSI", full: "Population Stability Index", family: "distance", rule: "ge" },
  { id: "euclidean", short: "Euclidean", full: "Euclidean distance (binned)", family: "distance", rule: "ge" },
  { id: "cosine", short: "Cosine", full: "Cosine distance (binned)", family: "distance", rule: "ge" },
  { id: "page_hinkley", short: "Page-Hinkley", full: "Page-Hinkley change-point", family: "sequential", rule: "ph" },
  { id: "model_based", short: "Domain clf", full: "Domain classifier (ROC AUC)", family: "classifier", rule: "ge" },
];
const FAMILIES = [
  { id: "test", name: "Hypothesis tests" },
  { id: "distance", name: "Distance & divergence" },
  { id: "sequential", name: "Change-point" },
  { id: "classifier", name: "Classifier" },
];
const RULE_TEXT = {
  p: "drift when the p-value is below the threshold",
  ge: "drift when the statistic is at or above the threshold",
  ph: "drift when a change point is detected inside the current window",
};
const PANEL_NAME = { "pr-review": "PR review", drift: "Drift report", "root-cause": "Root cause" };

const ICON = {
  check: "M3 8.5l3 3 7-7",
  x: "M4 4l8 8M12 4l-8 8",
  circleCheck: "M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5zM5.25 8.25l1.9 1.9 3.6-3.9",
  circleX: "M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5zM5.75 5.75l4.5 4.5M10.25 5.75l-4.5 4.5",
  alert: "M8 2.5l6 10.5H2zM8 6.5v3M8 11.5h.01",
  chevron: "M6 4l4 4-4 4",
  sort: "M4 6l4 4 4-4",
  info: "M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5zM8 7.25v4M8 4.75h.01",
  slash: "M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5zM3.6 12.4l8.8-8.8",
  lock: "M4 7.5h8v6H4zM5.75 7.5V5.25a2.25 2.25 0 0 1 4.5 0V7.5",
  wrench: "M10.5 2.5a3 3 0 0 0-3.4 4L2.75 10.85a1.2 1.2 0 0 0 1.7 1.7L8.8 8.2a3 3 0 0 0 4-3.4l-1.8 1.8-1.6-.4-.4-1.6z",
  up: "M8 13V3M4 7l4-4 4 4",
  down: "M8 3v10M4 9l4 4 4-4",
  both: "M2 8h12M5 5L2 8l3 3M11 5l3 3-3 3",
};
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);

// DOM builder. Text always goes in as text nodes, never markup: finding
// titles and file paths come from PR contents.
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) node.append(kid);
  return node;
}

// replaceChildren() would print a literal "null" for a skipped optional part.
function put(node, ...kids) {
  node.replaceChildren(...kids.flat(Infinity).filter((k) => k != null && k !== false));
}

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", ICON[name]);
  svg.append(path);
  return svg;
}

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const pct = (x) => `${Math.round(x * 100)}%`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
function num(n) {
  if (n == null || Number.isNaN(n)) return "—";
  const a = Math.abs(n);
  return a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(2) : n.toFixed(3);
}
function pval(p) {
  return p < 0.001 ? "<0.001" : p.toFixed(3);
}
// exact form for comparisons: "p 8.0e-4 < 0.05", never "p <0.001 < 0.05"
function pexact(p) {
  return p < 0.001 ? p.toExponential(1) : p.toFixed(3);
}
function ago(date) {
  const s = Math.round((Date.now() - date.getTime()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (Math.abs(s) < 60) return rtf.format(-s, "second");
  if (Math.abs(s) < 3600) return rtf.format(-Math.round(s / 60), "minute");
  if (Math.abs(s) < 86400) return rtf.format(-Math.round(s / 3600), "hour");
  return rtf.format(-Math.round(s / 86400), "day");
}

// ---------- provenance ----------

const prov = {}; // key -> { state: live|stale|sample|unavailable, updated?, error? }
const sampled = []; // keys served from the bundled sample
let dataMode = null;
let dataVia = null; // "api": the backend's /results API; "disk": the shared data dir

const VALID = {
  "pr-review": (d) => d && Array.isArray(d.inline_findings) && typeof d.overall_passed === "boolean" && Array.isArray(d.labels),
  drift: (d) => d && Array.isArray(d.columns) && typeof d.share_threshold === "number",
  "root-cause": (d) => d && typeof d.surrogate_auc === "number" && typeof d.auc_threshold === "number" &&
    typeof d.separable === "boolean" && Array.isArray(d.attributions),
};

function unavailable(key, message, status) {
  prov[key] = { state: "unavailable", error: { status, message } };
  return null;
}

// A panel is LIVE only when the server says so; anything else that is not
// an explicit SAMPLE becomes UNAVAILABLE rather than a guess.
async function load(key, url) {
  let res;
  try {
    res = await fetch(url, { cache: "no-store" });
  } catch {
    return unavailable(key, "The dashboard server did not respond.");
  }
  dataMode = res.headers.get("X-Data-Mode") || dataMode;
  dataVia = res.headers.get("X-Data-Via") || dataVia;
  if (!res.ok) {
    let message = `The API answered HTTP ${res.status}.`;
    try {
      const body = await res.json();
      if (body && body.message) message = String(body.message);
    } catch { /* keep the status message */ }
    return unavailable(key, message, res.status);
  }
  let data;
  try {
    data = await res.json();
  } catch {
    return unavailable(key, "The API response was not valid JSON.", res.status);
  }
  const source = res.headers.get("X-Data-Source");
  if (source !== "live" && source !== "sample") return unavailable(key, "The server did not state where this data came from.");
  if (!VALID[key](data)) return unavailable(key, "The response does not match the expected schema.");
  const updated = res.headers.get("X-Data-Updated");
  prov[key] = {
    state: source === "sample" ? "sample" : res.headers.get("X-Data-Stale") ? "stale" : "live",
    updated: updated ? new Date(updated) : null,
  };
  if (source === "sample") sampled.push(key);
  return data;
}

function provTip(key) {
  const p = prov[key];
  const name = PANEL_NAME[key];
  switch (p.state) {
    case "live":
      return `LIVE: ${name} read from the Python core's latest output.` +
        (p.updated ? `\nUpdated ${p.updated.toLocaleString()} (${ago(p.updated)}).` : "");
    case "stale":
      return `STALE: ${name} is live output, but older than the freshness window.` +
        (p.updated ? `\nLast updated ${p.updated.toLocaleString()} (${ago(p.updated)}).` : "") +
        "\nThe monitoring sweep may have stopped.";
    case "sample":
      return `SAMPLE: ${name} is a bundled demonstration fixture.\nIt was not computed from any live review or model.`;
    default:
      return `UNAVAILABLE: ${p.error.message}\nNothing is shown in its place.`;
  }
}

function provChip(key, compact) {
  const p = prov[key];
  const label = { live: "Live", stale: "Stale", sample: "Sample", unavailable: "Unavailable" }[p.state];
  const when = !compact && p.updated && p.state !== "sample" ? ` · ${ago(p.updated)}` : "";
  return el("span", { class: `chip source ${p.state}`, tabindex: "0", "data-tip": provTip(key) }, label + when);
}

function paintProvenance() {
  for (const key of Object.keys(PANEL_NAME)) $(`source-${key}`).replaceWith(Object.assign(provChip(key), { id: `source-${key}` }));

  const by = {};
  for (const [key, p] of Object.entries(prov)) (by[p.state] ||= []).push(PANEL_NAME[key]);
  const order = ["live", "stale", "sample", "unavailable"];
  $("source-summary").replaceChildren(...order.filter((s) => by[s]).map((s) =>
    el("span", {
      class: `chip source ${s}`, tabindex: "0",
      "data-tip": `${s.toUpperCase()}: ${by[s].join(", ")}`,
    }, `${by[s].length} ${s}`)));

  const modeText = {
    auto: "Data mode: auto (live output where it exists, the bundled sample elsewhere).",
    live: "Data mode: live (samples are never shown).",
    sample: "Data mode: sample (every panel shows the bundled fixture).",
  }[dataMode];
  const viaText = dataMode === "sample" ? "" : { api: " Live data via the backend results API.", disk: " Live data read from the shared data directory." }[dataVia] || "";
  $("data-mode").textContent = (modeText || "") + viaText;
  $("data-source").textContent = sampled.length ? `Sample data shown for: ${sampled.join(", ")}.` : "";
}

// ---------- tooltip ----------

const tipEl = $("tooltip");
let tipFor = null;

function showTip(target) {
  tipEl.textContent = target.dataset.tip;
  tipEl.hidden = false;
  target.setAttribute("aria-describedby", "tooltip");
  tipFor = target;
  const r = target.getBoundingClientRect();
  const w = tipEl.offsetWidth;
  const h = tipEl.offsetHeight;
  const left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
  const below = r.bottom + 8;
  tipEl.style.left = `${left}px`;
  tipEl.style.top = `${below + h > window.innerHeight - 8 ? r.top - h - 8 : below}px`;
}

function hideTip() {
  if (tipFor) tipFor.removeAttribute("aria-describedby");
  tipEl.hidden = true;
  tipFor = null;
}

document.addEventListener("pointerover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (t && t !== tipFor) showTip(t);
  else if (!t && tipFor) hideTip();
});
document.addEventListener("focusin", (e) => {
  const t = e.target.closest("[data-tip]");
  t ? showTip(t) : hideTip();
});
document.addEventListener("keydown", (e) => e.key === "Escape" && hideTip());
window.addEventListener("scroll", hideTip, { passive: true });

function infoButton(text, label) {
  return el("button", { class: "info", type: "button", "aria-label": label, "data-tip": text }, icon("info"));
}

// ---------- shared pieces ----------

// Threshold bar: marker label above (pinned to the marker, flipped near an
// edge so it never runs off the tile), range ends below.
function gauge({ fill, mark, tone, lo, hi, th }) {
  const at = clamp01(mark) * 100;
  const shift = at < 15 ? "0" : at > 85 ? "-100%" : "-50%";
  return el("div", { class: "gauge" },
    el("div", { class: "scale mark-row", "aria-hidden": "true" },
      el("span", { class: "th", style: `left:${at}%;transform:translateX(${shift})` }, th)),
    el("div", { class: `bar ${tone || ""}` },
      el("span", { class: "fill", style: `width:${clamp01(fill) * 100}%` }),
      el("span", { class: "mark", style: `left:${at}%` })),
    el("div", { class: "scale", "aria-hidden": "true" },
      el("span", { class: "lo" }, lo),
      el("span", { class: "hi" }, hi)));
}

function stateBox(kind, iconName, title, body) {
  return el("div", { class: `state-box ${kind}` }, icon(iconName), el("div", {}, el("h3", {}, title), body ? el("p", {}, body) : null));
}

function unavailableBox(key) {
  const err = prov[key].error;
  return err.status === 404
    ? stateBox("muted", "info", `No live ${key === "pr-review" ? "PR review" : PANEL_NAME[key].toLowerCase()} yet`, `${err.message} Sample data is off in live mode.`)
    : stateBox("err", "alert", `${PANEL_NAME[key]} unavailable`, `${err.message} No data is shown in its place.`);
}

// ---------- header + KPIs ----------

function renderHeader(review) {
  if (!review) {
    $("pr-title").textContent = "PR review unavailable";
    $("crumb-repo").hidden = $("crumb-repo").nextElementSibling.hidden = true;
    return;
  }
  // PR identity is joined in by the server from the cached PR; absent when that file is missing.
  const id = review.pr_id != null ? `PR #${review.pr_id}` : "Latest PR review";
  $("pr-title").textContent = review.pr_title ? `${id} · ${review.pr_title}` : id;
  $("crumb-pr").textContent = id;
  $("crumb-repo").textContent = review.repo_full_name || "";
  $("crumb-repo").hidden = $("crumb-repo").nextElementSibling.hidden = !review.repo_full_name;
  put($("pr-meta"),
    review.repo_full_name ? el("span", { class: "mono" }, review.repo_full_name) : null,
    review.commit_sha ? el("span", {}, "head ", el("span", { class: "mono" }, review.commit_sha.slice(0, 7))) : null,
    review.base_ref ? el("span", {}, "into ", el("span", { class: "mono" }, review.base_ref)) : null);
  // Live labels are full enum strings ("security: Patches, auth, ..."); the chip wants the name.
  $("pr-labels").replaceChildren(...review.labels.map((l) => el("span", { class: "chip plain", "data-tip": l, tabindex: "0" }, l.split(":")[0])));
}

function tile({ label, key, about, tone, value, body, foot }) {
  return el("div", { class: `kpi ${tone || ""}` },
    el("div", { class: "kpi-head" },
      el("span", { class: "kpi-label" }, label, about ? infoButton(about, `About ${label}`) : null),
      key ? provChip(key, true) : null),
    el("div", { class: "kpi-value" }, value),
    body,
    foot ? el("div", { class: "kpi-foot" }, foot) : null);
}

function unavailableTile(label, key) {
  return tile({ label, key, tone: "is-unavailable", value: "Unavailable", foot: prov[key].error.message });
}

function renderKpis(review, drift, rc) {
  const kpis = $("kpis");
  kpis.removeAttribute("aria-busy");
  kpis.replaceChildren();

  if (review) {
    const findings = review.inline_findings;
    const counts = Object.fromEntries(SEVERITIES.map((s) => [s, findings.filter((f) => f.severity === s).length]));
    const blocking = findings.filter((f) => BLOCKING.has(f.severity)).length;
    const passed = review.overall_passed;
    kpis.append(
      tile({
        label: "Review verdict", key: "pr-review",
        about: "The synthesizer fails a PR when any finding is HIGH or CRITICAL. Lower severities are reported without blocking.",
        tone: passed ? "is-good" : "is-bad",
        value: [icon(passed ? "circleCheck" : "circleX"), passed ? "Passed" : "Changes requested"],
        foot: [
          el("span", {}, `${plural(findings.length, "finding")} · ${blocking} blocking`),
          review.pr_id != null ? el("span", { class: "ctx" }, `PR #${review.pr_id}${review.repo_full_name ? ` · ${review.repo_full_name}` : ""}`) : null,
        ],
      }),
      tile({
        label: "Findings by severity", key: "pr-review",
        about: "Inline findings from the review, counted by severity. HIGH and CRITICAL block the PR.",
        value: [String(findings.length), el("small", {}, findings.length === 1 ? "finding" : "findings")],
        body: [
          findings.length
            ? el("div", { class: "bar seg", role: "img", "aria-label": SEVERITIES.filter((s) => counts[s]).map((s) => `${counts[s]} ${s}`).join(", ") },
                SEVERITIES.filter((s) => counts[s]).map((s) => el("span", { style: `flex:${counts[s]};background:var(--${SEV_VAR[s]})` })))
            : el("div", { class: "bar seg empty", role: "img", "aria-label": "No findings" }),
          el("div", { class: "sev-counts" }, SEVERITIES.map((s) =>
            el("div", { class: `sev-count ${s} ${counts[s] ? "" : "zero"}` }, el("b", {}, String(counts[s])), el("span", {}, s === "CRITICAL" ? "CRIT" : s === "MEDIUM" ? "MED" : s)))),
        ],
      }));
  } else {
    kpis.append(unavailableTile("Review verdict", "pr-review"), unavailableTile("Findings by severity", "pr-review"));
  }

  if (drift) {
    const nCols = drift.columns.length;
    const nDrifted = drift.columns.filter((c) => c.drifted).length;
    const share = nCols ? nDrifted / nCols : 0;
    const exceeded = nCols > 0 && share > drift.share_threshold; // strict, as detector.py decides
    const vote = drift.columns[0]?.vote;
    kpis.append(tile({
      label: "Drifted columns", key: "drift",
      about: `${nDrifted} drifted ÷ ${nCols} columns = ${pct(share)}.\nThe dataset counts as drifted only when this share is above ${pct(drift.share_threshold)}.` +
        (vote ? `\nEach column's verdict is a ${vote} vote of its detectors.` : ""),
      tone: exceeded ? "is-bad" : "",
      value: [`${nDrifted} / ${nCols}`, el("small", {}, pct(share))],
      body: gauge({ fill: share, mark: drift.share_threshold, tone: exceeded ? "bad" : "good", lo: "0%", hi: "100%", th: `${pct(drift.share_threshold)} threshold` }),
      foot: [
        el("span", { class: `verdict-line ${exceeded ? "met" : "unmet"}` }, icon(exceeded ? "alert" : "check"),
          exceeded ? "Dataset drift threshold exceeded" : "Dataset drift threshold not exceeded"),
        el("span", {}, `Requires more than ${pct(drift.share_threshold)} of columns`),
      ],
    }));
  } else {
    kpis.append(unavailableTile("Drifted columns", "drift"));
  }

  if (rc) {
    const scale = (x) => (x - 0.5) / 0.5; // 0.5 (chance) .. 1.0
    const hasP = typeof rc.auc_p_value === "number" && typeof rc.alpha === "number";
    kpis.append(tile({
      label: "Surrogate AUC", key: "root-cause",
      about: "How well a surrogate classifier tells reference rows from current rows (held-out ROC AUC). 0.5 means the two are indistinguishable.\n" +
        `Attribution is shown only when AUC ≥ ${rc.auc_threshold} and the AUC is significant for the test-set size (Mann-Whitney p < α).`,
      value: [rc.surrogate_auc.toFixed(2), el("small", {}, "ROC AUC")],
      body: gauge({ fill: scale(rc.surrogate_auc), mark: scale(rc.auc_threshold), lo: "0.50 chance", hi: "1.00", th: `${rc.auc_threshold} gate` }),
      foot: [
        el("span", { class: `verdict-line ${rc.separable ? "unmet" : "met"}` }, icon(rc.separable ? "check" : "lock"),
          rc.separable ? "Gate passed · attribution shown" : "Gate failed · attribution suppressed"),
        el("span", {}, hasP ? `Significance p ${pexact(rc.auc_p_value)} (α ${rc.alpha})` : "Significance not reported by this source"),
      ],
    }));
  } else {
    kpis.append(unavailableTile("Surrogate AUC", "root-cause"));
  }
}

// ---------- findings ----------

// The synthesizer's markdown summary lists each specialist's section; on a
// clean review that is the evidence of what ran.
function parseChecks(markdown) {
  const lines = String(markdown || "").split("\n").map((l) => l.trim());
  const checks = [];
  lines.forEach((line, i) => {
    const m = line.match(/^###\s+(.+)$/);
    if (!m) return;
    const result = lines.slice(i + 1).find((l) => l && !l.startsWith("#"));
    if (result && !result.startsWith("**")) checks.push({ name: m[1], result });
  });
  return checks;
}

const findingsState = { active: new Set(), sort: { key: "severity", dir: 1 }, open: new Set() };

function renderFindings(review) {
  const body = $("findings-body");
  const toolbar = $("findings-toolbar");
  const panel = $("findings");
  if (!review) {
    $("findings-count").textContent = "–";
    toolbar.hidden = true;
    body.replaceChildren(unavailableBox("pr-review"));
    return;
  }

  const all = review.inline_findings.map((f, i) => ({ ...f, _id: i }));
  $("findings-count").textContent = all.length;
  panel.classList.toggle("is-empty", all.length === 0);

  if (!all.length) {
    toolbar.hidden = true;
    const who = review.pr_id != null ? `PR #${review.pr_id}` : "this PR";
    const checks = parseChecks(review.markdown_summary);
    put(body,
      stateBox("ok", "circleCheck", `Review completed: no findings for ${who}`,
        "The multi-agent review finished and none of its checks reported an issue. Findings from later reviews appear here automatically."),
      checks.length ? el("ul", { class: "checks", "aria-label": "Checks run" }, checks.map((c) =>
        el("li", {}, icon("check"), el("span", { class: "name" }, c.name), el("span", { class: "res" }, c.result)))) : null);
    return;
  }

  const counts = Object.fromEntries(SEVERITIES.map((s) => [s, all.filter((f) => f.severity === s).length]));
  toolbar.hidden = false;
  $("severity-filters").replaceChildren(...SEVERITIES.map((s) =>
    el("button", {
      class: `filter ${s}`, type: "button", "aria-pressed": String(findingsState.active.has(s)), disabled: counts[s] === 0,
      onclick: (e) => {
        findingsState.active.has(s) ? findingsState.active.delete(s) : findingsState.active.add(s);
        e.currentTarget.setAttribute("aria-pressed", String(findingsState.active.has(s)));
        drawTable();
      },
    }, s.charAt(0) + s.slice(1).toLowerCase(), el("span", { class: "n" }, String(counts[s])))));

  const cmp = {
    severity: (a, b) => SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity),
    title: (a, b) => a.title.localeCompare(b.title),
    location: (a, b) => a.file_path.localeCompare(b.file_path) || (a.line_number ?? 0) - (b.line_number ?? 0),
  };

  function header(key, label, cls) {
    const { sort } = findingsState;
    const active = sort.key === key;
    return el("th", { scope: "col", class: cls, "aria-sort": active ? (sort.dir === 1 ? "ascending" : "descending") : "none" },
      el("button", {
        class: "sort", type: "button",
        onclick: () => {
          findingsState.sort = { key, dir: active ? -sort.dir : 1 };
          drawTable();
        },
      }, label, icon("sort")));
  }

  function drawTable() {
    const { active, sort, open } = findingsState;
    const shown = all.filter((f) => !active.size || active.has(f.severity))
      .sort((a, b) => sort.dir * cmp[sort.key](a, b) || cmp.severity(a, b) || cmp.location(a, b));
    if (!shown.length) {
      body.replaceChildren(el("p", { class: "empty-filter" }, "No findings match the selected severities."));
      return;
    }
    const rows = shown.flatMap((f) => {
      const detailId = `finding-detail-${f._id}`;
      const loc = f.line_number ? `${f.file_path}:${f.line_number}` : f.file_path;
      const isOpen = open.has(f._id);
      const detail = el("tr", { class: "detail", id: detailId, hidden: !isOpen },
        el("td", { colspan: 3 },
          el("div", { class: "detail-body" },
            el("dl", { class: "kv" },
              el("dt", {}, "Location"), el("dd", {}, loc),
              el("dt", {}, "Severity"), el("dd", {}, `${f.severity}${BLOCKING.has(f.severity) ? " (blocking)" : ""}`)),
            el("p", {}, f.description),
            f.suggestion ? el("div", { class: "fix" }, el("b", {}, icon("wrench"), "Suggested fix"), f.suggestion) : null)));
      const row = el("tr", { class: `finding ${isOpen ? "open" : ""}` },
        el("td", {}, el("span", { class: `chip ${f.severity}` }, f.severity)),
        el("td", {}, el("button", {
          class: "row-toggle", type: "button", "aria-expanded": String(isOpen), "aria-controls": detailId,
          onclick: (e) => {
            const next = !open.has(f._id);
            next ? open.add(f._id) : open.delete(f._id);
            e.currentTarget.setAttribute("aria-expanded", String(next));
            row.classList.toggle("open", next);
            detail.hidden = !next;
          },
        }, icon("chevron"), f.title)),
        el("td", { class: "loc", title: loc }, loc));
      return [row, detail];
    });
    body.replaceChildren(el("div", { class: "table-wrap" },
      el("table", { class: "data" },
        el("thead", {}, el("tr", {}, header("severity", "Severity", "col-sev"), header("title", "Finding"), header("location", "Location", "col-loc"))),
        el("tbody", {}, rows))));
  }

  drawTable();
}

// ---------- drift detector matrix ----------

function cellState(col, m) {
  const result = col.results.find((r) => r.method === m.id);
  if (result) return { kind: result.drifted ? "hit" : "ok", result };
  if (m.id in col.skipped) return { kind: "skip", reason: col.skipped[m.id] };
  return { kind: "none" };
}

const STATE_LABEL = { hit: "Triggered", ok: "Not triggered", skip: "Not applicable", none: "Not evaluated" };

function comparison(m, r) {
  if (m.rule === "p" && r.p_value != null) return `p ${pexact(r.p_value)} ${r.p_value < r.threshold ? "<" : "≥"} ${r.threshold}`;
  if (m.rule === "ge") return `${num(r.statistic)} ${r.statistic >= r.threshold ? "≥" : "<"} ${num(r.threshold)}`;
  return `peak ${num(r.statistic)} vs ${num(r.threshold)}`;
}

function cellTip(col, m, s) {
  const head = `${col.column} · ${m.full}\nState: ${STATE_LABEL[s.kind]}`;
  if (s.kind === "skip") return `${head}\nReason: ${s.reason}`;
  if (s.kind === "none") return `${head}\nThis detector has no recorded result for this column.`;
  const r = s.result;
  return `${head}\nStatistic: ${num(r.statistic)}${r.p_value != null ? ` · p = ${pexact(r.p_value)}` : ""}` +
    `\nThreshold: ${num(r.threshold)} (${RULE_TEXT[m.rule]})\nComparison: ${comparison(m, r)}` +
    (r.detail ? `\n${r.detail}` : "");
}

const driftState = { triggeredOnly: false, families: new Set(FAMILIES.map((f) => f.id)), query: "", selected: null };

function renderDrift(report) {
  const body = $("drift-body");
  if (!report) {
    body.replaceChildren(unavailableBox("drift"));
    return;
  }
  $("drift-label").textContent = report.label || "";

  const cols = report.columns;
  const states = cols.flatMap((c) => METHODS.map((m) => cellState(c, m)));
  const evaluated = states.filter((s) => s.kind === "hit" || s.kind === "ok").length;
  const triggered = states.filter((s) => s.kind === "hit").length;
  const nDrifted = cols.filter((c) => c.drifted).length;
  const share = cols.length ? nDrifted / cols.length : 0;
  const datasetDrifted = cols.length > 0 && share > report.share_threshold;

  const stats = el("div", { class: "stats" },
    el("div", { class: "stat" }, el("b", {}, String(cols.length)), el("span", {}, "features")),
    el("div", { class: "stat" }, el("b", {}, String(METHODS.length)), el("span", {}, "detector methods")),
    el("div", { class: "stat" }, el("b", {}, `${evaluated} / ${cols.length * METHODS.length}`), el("span", {}, "checks evaluated")),
    el("div", { class: `stat ${triggered ? "hot" : ""}` }, el("b", {}, String(triggered)), el("span", {}, "checks triggered")),
    el("div", { class: `stat ${nDrifted ? "hot" : ""}` }, el("b", {}, `${nDrifted} / ${cols.length}`), el("span", {}, "features drifted")),
    el("div", { class: `stat ${datasetDrifted ? "hot" : ""}` }, el("b", {}, datasetDrifted ? "Drifted" : "Not drifted"),
      el("span", {}, `dataset · ${pct(share)} drifted, needs > ${pct(report.share_threshold)}`)));

  const toolbar = el("div", { class: "toolbar" },
    el("div", { class: "filters", role: "group", "aria-label": "Matrix filters" },
      el("button", {
        class: "filter accent", type: "button", "aria-pressed": String(driftState.triggeredOnly),
        onclick: (e) => {
          driftState.triggeredOnly = !driftState.triggeredOnly;
          e.currentTarget.setAttribute("aria-pressed", String(driftState.triggeredOnly));
          drawMatrix();
        },
      }, "Triggered only"),
      FAMILIES.map((f) => el("button", {
        class: "filter nodot", type: "button", "aria-pressed": String(driftState.families.has(f.id)),
        onclick: (e) => {
          const fam = driftState.families;
          if (fam.has(f.id) && fam.size === 1) return; // keep at least one family visible
          fam.has(f.id) ? fam.delete(f.id) : fam.add(f.id);
          e.currentTarget.setAttribute("aria-pressed", String(fam.has(f.id)));
          drawMatrix();
        },
      }, f.name))),
    el("input", {
      class: "search", type: "search", placeholder: "Filter features", "aria-label": "Filter features by name",
      value: driftState.query,
      oninput: (e) => {
        driftState.query = e.currentTarget.value.trim().toLowerCase();
        drawMatrix();
      },
    }));

  const matrixHost = el("div", { class: "table-wrap" });
  const inspector = el("div", { class: "inspector", hidden: true, "aria-live": "polite" });
  const legend = el("p", { class: "legend" },
    el("span", {}, el("span", { class: "cell hit" }, icon("alert"), "0.00"), "Triggered"),
    el("span", {}, el("span", { class: "cell ok" }, icon("check"), "0.00"), "Not triggered"),
    el("span", {}, el("span", { class: "cell skip" }, icon("slash"), "n/a"), "Not applicable (with reason)"),
    el("span", {}, el("span", { class: "cell none" }, "—"), "Not evaluated"),
    el("span", {}, "Values are each detector's statistic; KS and MMD show their p-value. Hover or select a cell for its threshold."));

  function inspect(col, m, s) {
    driftState.selected = `${col.column}|${m.id}`;
    const applicable = col.results.length;
    const fired = col.results.filter((r) => r.drifted).length;
    const r = s.result;
    inspector.hidden = false;
    put(inspector,
      el("div", { class: "inspector-head" },
        el("h3", {}, el("span", { class: "mono" }, col.column), " · ", m.full, " ", el("span", { class: `chip ${s.kind === "hit" ? "fail" : s.kind === "ok" ? "pass" : "plain"}` }, STATE_LABEL[s.kind])),
        el("button", { class: "close", type: "button", "aria-label": "Close detector details", onclick: () => { driftState.selected = null; inspector.hidden = true; drawMatrix(); } }, icon("x"))),
      el("dl", { class: "kv" },
        r ? [el("dt", {}, "Statistic"), el("dd", {}, num(r.statistic))] : null,
        r && r.p_value != null ? [el("dt", {}, "p-value"), el("dd", {}, pexact(r.p_value))] : null,
        r ? [el("dt", {}, "Threshold"), el("dd", {}, `${num(r.threshold)}  (${RULE_TEXT[m.rule]})`)] : null,
        r ? [el("dt", {}, "Comparison"), el("dd", {}, comparison(m, r))] : null,
        r && r.detail ? [el("dt", {}, "Detail"), el("dd", {}, r.detail)] : null,
        s.kind === "skip" ? [el("dt", {}, "Reason"), el("dd", {}, s.reason)] : null,
        el("dt", {}, "Column"), el("dd", {}, `${col.kind}, ${col.n_reference.toLocaleString()} reference / ${col.n_current.toLocaleString()} current rows`),
        el("dt", {}, "Column verdict"), el("dd", {}, `${col.drifted ? "Drifted" : "Stable"} by ${col.vote} vote: ${fired} of ${applicable} evaluated detectors triggered`)),
      s.kind === "none" ? el("p", {}, "No result for this detector was recorded on this column, so nothing is claimed either way.") : null);
  }

  function drawMatrix() {
    const methods = METHODS.filter((m) => driftState.families.has(m.family));
    const rows = cols.filter((c) =>
      (!driftState.query || c.column.toLowerCase().includes(driftState.query)) &&
      (!driftState.triggeredOnly || c.results.some((r) => r.drifted)));
    const fams = FAMILIES.filter((f) => methods.some((m) => m.family === f.id));
    const firstOfFamily = new Set(fams.map((f) => methods.find((m) => m.family === f.id).id));

    if (!rows.length) {
      matrixHost.replaceChildren(el("p", { class: "empty-filter" }, "No features match these filters."));
      return;
    }

    matrixHost.replaceChildren(el("table", { class: "data matrix" },
      el("thead", {},
        el("tr", { class: "fam" },
          el("th", { class: "sticky" }, ""), el("th", { class: "col-aux" }, ""), el("th", { class: "col-aux" }, ""),
          fams.map((f) => el("th", { class: "f", colspan: methods.filter((m) => m.family === f.id).length, scope: "colgroup" }, f.name)),
          el("th", {}, "")),
        el("tr", {},
          el("th", { scope: "col", class: "sticky" }, "Feature"),
          el("th", { scope: "col", class: "col-aux" }, "Kind"),
          el("th", { scope: "col", class: "col-aux" }, "Rows ref / cur"),
          methods.map((m) => el("th", { scope: "col", class: `m ${firstOfFamily.has(m.id) ? "f" : ""}` }, el("abbr", { title: m.full }, m.short))),
          el("th", { scope: "col" }, "Column verdict"))),
      el("tbody", {}, rows.map((c) =>
        el("tr", { class: c.drifted ? "is-drifted" : "" },
          el("th", { scope: "row", class: "name sticky" }, c.column),
          el("td", { class: "col-aux" }, el("span", { class: "chip plain" }, c.kind)),
          el("td", { class: "n col-aux" }, `${c.n_reference.toLocaleString()} / ${c.n_current.toLocaleString()}`),
          methods.map((m) => {
            const s = cellState(c, m);
            const tip = cellTip(c, m, s);
            const td = (inner) => el("td", { class: `m ${firstOfFamily.has(m.id) ? "f" : ""}` }, inner);
            if (s.kind === "none") return td(el("span", { class: "cell none", "aria-label": tip.replace(/\n/g, ". "), "data-tip": tip, tabindex: "0" }, "—"));
            const text = s.kind === "skip" ? "n/a" : m.rule === "p" && s.result.p_value != null ? `p ${pval(s.result.p_value).replace("0.", ".")}` : num(s.result.statistic);
            const cellKey = `${c.column}|${m.id}`;
            return td(el("button", {
              class: `cell ${s.kind}`, type: "button", "data-tip": tip, "aria-label": tip.replace(/\n/g, ". "),
              "data-key": cellKey, "aria-pressed": String(driftState.selected === cellKey),
              onclick: () => {
                inspect(c, m, s);
                drawMatrix();
                // the redraw replaced the button; keep keyboard focus where it was
                [...matrixHost.querySelectorAll("[data-key]")].find((b) => b.dataset.key === cellKey)?.focus();
              },
            }, icon(s.kind === "hit" ? "alert" : s.kind === "ok" ? "check" : "slash"), text));
          }),
          el("td", {}, el("span", { class: `chip ${c.drifted ? "fail" : "pass"}`, tabindex: "0",
            "data-tip": `${c.results.filter((r) => r.drifted).length} of ${c.results.length} evaluated detectors triggered (${c.vote} vote)` }, c.drifted ? "Drifted" : "Stable")))))));
  }

  body.replaceChildren(stats, toolbar, matrixHost, inspector, legend);
  drawMatrix();
}

// ---------- root cause ----------

const DIRECTION = {
  increased: ["up", "Increased"],
  decreased: ["down", "Decreased"],
  reshaped: ["both", "Reshaped"],
};

function gateFacts(rc) {
  const hasP = typeof rc.auc_p_value === "number" && typeof rc.alpha === "number";
  const aucOk = rc.surrogate_auc >= rc.auc_threshold;
  const pOk = hasP && rc.auc_p_value < rc.alpha;
  return { hasP, aucOk, pOk };
}

function gateChecks(rc) {
  const { hasP, aucOk, pOk } = gateFacts(rc);
  return el("div", { class: "gate-checks" },
    el("span", { class: `gate-check ${aucOk ? "ok" : "no"}` }, icon(aucOk ? "check" : "x"), `AUC ${rc.surrogate_auc.toFixed(2)} ${aucOk ? "≥" : "<"} ${rc.auc_threshold}`),
    hasP
      ? el("span", { class: `gate-check ${pOk ? "ok" : "no"}` }, icon(pOk ? "check" : "x"), `p ${pexact(rc.auc_p_value)} ${pOk ? "<" : "≥"} α ${rc.alpha}`)
      : el("span", { class: "gate-check na" }, icon("info"), "significance not reported"));
}

function gateChip(rc) {
  return el("span", { class: `chip ${rc.separable ? "pass" : "fail"}` }, rc.separable ? "Passed" : "Suppressed");
}

function rcScope(rc) {
  return [rc.label ? `comparison: ${rc.label}` : null,
    rc.n_reference != null && rc.n_current != null ? `${rc.n_reference.toLocaleString()} reference / ${rc.n_current.toLocaleString()} current rows` : null]
    .filter(Boolean).join(" · ");
}

function renderRootCause(rc) {
  const gateBody = $("rc-gate-body");
  const body = $("rc-body");
  $("root-cause").hidden = !rc;
  if (!rc) {
    gateBody.replaceChildren(unavailableBox("root-cause"));
    return;
  }
  const { hasP, aucOk, pOk } = gateFacts(rc);
  const scale = (x) => (x - 0.5) / 0.5; // 0.5 (chance) .. 1.0
  // the ranking card carries its own provenance chip, same source as the gate
  const rankHead = $("root-cause").querySelector(".panel-head");
  rankHead.querySelector(".chip")?.remove();
  rankHead.append(provChip("root-cause"));

  put(gateBody,
    el("div", { class: "gate-card" },
      el("div", { class: "gate-head" },
        el("span", { class: "gate-value" }, rc.surrogate_auc.toFixed(2), el("small", {}, "surrogate ROC AUC")),
        gateChip(rc)),
      gauge({ fill: scale(rc.surrogate_auc), mark: scale(rc.auc_threshold), lo: "0.50 chance", hi: "1.00", th: `${rc.auc_threshold} gate` }),
      gateChecks(rc),
      el("p", {}, "The surrogate is a classifier trained to tell reference rows from current rows. " +
        `Attribution is shown only when its AUC is at least ${rc.auc_threshold} and that lead over chance is significant for the test-set size (Mann-Whitney p < α).`),
      rcScope(rc) ? el("p", { class: "mono scope" }, rcScope(rc)) : null));

  if (!rc.separable) {
    const why = [!aucOk ? `AUC ${rc.surrogate_auc.toFixed(2)} is below the ${rc.auc_threshold} gate` : null,
      hasP && !pOk ? `its lead over chance is not significant (p ${pexact(rc.auc_p_value)} ≥ α ${rc.alpha})` : null]
      .filter(Boolean).join(" and ");
    body.replaceChildren(stateBox("gate", "lock", "Attribution suppressed",
      `Reference and current samples are not separable enough under the configured gate${why ? `: ${why}` : ""}. ` +
      "A feature ranking now would only order noise, so none is shown. Clearer separation (more data or a larger shift) is needed first."));
    return;
  }

  if (!rc.attributions.length) {
    body.replaceChildren(stateBox("muted", "info", "No attributions reported", "The gate passed but the source listed no features."));
    return;
  }

  const note = el("p", { class: "rc-note" },
    "Ranked by mean |SHAP| in the surrogate. A high rank means the feature shifted most: a root-cause candidate, not proof it caused a failure.");

  const list = el("ol", { class: "attributions" }, rc.attributions.map((a, i) => {
    const [dirIcon, dirText] = DIRECTION[a.direction] || ["both", a.direction];
    const moved = a.reference_mean != null && a.current_mean != null;
    const moreId = `attr-more-${i}`;
    const more = el("div", { class: "attr-more", id: moreId, hidden: true },
      el("dl", { class: "kv" },
        el("dt", {}, "Mean |SHAP|"), el("dd", {}, num(a.mean_abs_shap)),
        el("dt", {}, "Share of attribution"), el("dd", {}, `${(a.share * 100).toFixed(1)}%`),
        moved ? [el("dt", {}, "Reference mean"), el("dd", {}, String(a.reference_mean))] : null,
        moved ? [el("dt", {}, "Current mean"), el("dd", {}, String(a.current_mean))] : null,
        moved && a.reference_mean !== 0 ? [el("dt", {}, "Change"),
          el("dd", {}, `${a.current_mean - a.reference_mean >= 0 ? "+" : ""}${num(a.current_mean - a.reference_mean)} (${((a.current_mean / a.reference_mean - 1) * 100).toFixed(1)}%)`)] : null,
        !moved ? [el("dt", {}, "Means"), el("dd", {}, "Not reported (non-numeric feature)")] : null));
    return el("li", { class: "attr" },
      el("button", {
        class: "attr-btn", type: "button", "aria-expanded": "false", "aria-controls": moreId,
        onclick: (e) => {
          const open = e.currentTarget.getAttribute("aria-expanded") !== "true";
          e.currentTarget.setAttribute("aria-expanded", String(open));
          more.hidden = !open;
        },
      },
        el("span", { class: "attr-rank" }, `#${i + 1}`),
        el("span", { class: "attr-name" }, a.feature),
        el("span", { class: "attr-share" }, pct(a.share), el("small", {}, "contribution")),
        el("span", { class: "attr-detail" },
          el("span", { class: `dir dir-${a.direction}` }, icon(dirIcon), dirText),
          moved ? el("span", { class: "mono" }, `${a.reference_mean.toFixed(1)} → ${a.current_mean.toFixed(1)}`) : null),
        el("div", { class: "bar", "aria-hidden": "true" }, el("span", { class: "fill", style: `width:${clamp01(a.share) * 100}%` }))),
      more);
  }));

  body.replaceChildren(note, list);
}

// ---------- overview summary cards ----------

function card({ title, key, href, linkText, body }) {
  return el("section", { class: "panel card", "aria-label": title },
    el("div", { class: "panel-head" }, el("h2", {}, title), key ? provChip(key, true) : null),
    el("div", { class: "card-body" }, body),
    el("a", { class: "card-link", href }, linkText, icon("chevron")));
}

function compactUnavailable(key) {
  return el("p", { class: "card-note err" }, icon("alert"), prov[key].error.message);
}

function findingsCard(review) {
  if (!review) return card({ title: "Findings", key: "pr-review", href: "#/findings", linkText: "Open findings", body: compactUnavailable("pr-review") });
  const all = [...review.inline_findings].sort((a, b) => SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity));
  const blocking = all.filter((f) => BLOCKING.has(f.severity)).length;
  if (!all.length) {
    const checks = parseChecks(review.markdown_summary);
    return card({
      title: "Findings", key: "pr-review", href: "#/findings", linkText: "Open review",
      body: [
        el("p", { class: "card-note ok" }, icon("circleCheck"), checks.length ? `No findings. ${plural(checks.length, "check")} passed.` : "No findings."),
        checks.length ? el("ul", { class: "mini-list" }, checks.map((c) =>
          el("li", {}, icon("check"), el("span", { class: "mini-main" }, c.name)))) : null,
      ],
    });
  }
  const shown = all.slice(0, 4);
  return card({
    title: "Findings", key: "pr-review", href: "#/findings", linkText: `View all ${plural(all.length, "finding")}`,
    body: [
      el("p", { class: "card-sum" }, `${plural(all.length, "finding")} · ${blocking} blocking`),
      el("ul", { class: "mini-list" }, shown.map((f) =>
        el("li", {},
          el("span", { class: `chip ${f.severity}` }, f.severity),
          el("span", { class: "mini-main" }, f.title),
          el("span", { class: "mini-side mono" }, f.line_number ? `${f.file_path.split("/").pop()}:${f.line_number}` : f.file_path.split("/").pop())))),
      all.length > shown.length ? el("p", { class: "card-more" }, `+ ${all.length - shown.length} more`) : null,
    ],
  });
}

function rootCauseCard(rc) {
  if (!rc) return card({ title: "Root cause", key: "root-cause", href: "#/root-cause", linkText: "Open root cause", body: compactUnavailable("root-cause") });
  const top = rc.attributions.slice(0, 3);
  return card({
    title: "Root cause", key: "root-cause", href: "#/root-cause", linkText: rc.separable ? "Open full ranking" : "Open gate details",
    body: [
      el("div", { class: "card-gate" }, el("span", { class: "card-gate-label" }, "Gate"), gateChip(rc), gateChecks(rc)),
      rc.separable && top.length
        ? el("ol", { class: "mini-list ranked" }, top.map((a, i) =>
            el("li", {},
              el("span", { class: "attr-rank" }, `#${i + 1}`),
              el("span", { class: "mini-main mono" }, a.feature),
              el("span", { class: "mini-bar", "aria-hidden": "true" }, el("span", { style: `width:${clamp01(a.share) * 100}%` })),
              el("span", { class: "mini-side" }, pct(a.share)))))
        : el("p", { class: "card-note gate" }, icon("lock"), rc.separable ? "No attributions reported." : "Ranking suppressed: samples are not separable enough to attribute."),
    ],
  });
}

function driftCard(report) {
  if (!report) return card({ title: "Drift", key: "drift", href: "#/drift", linkText: "Open drift matrix", body: compactUnavailable("drift") });
  const cols = report.columns;
  const triggered = cols.reduce((n, c) => n + c.results.filter((r) => r.drifted).length, 0);
  const evaluated = cols.reduce((n, c) => n + c.results.length, 0);
  const nDrifted = cols.filter((c) => c.drifted).length;
  const share = cols.length ? nDrifted / cols.length : 0;
  const datasetDrifted = cols.length > 0 && share > report.share_threshold;
  const ordered = [...cols].sort((a, b) => b.drifted - a.drifted || b.results.filter((r) => r.drifted).length - a.results.filter((r) => r.drifted).length);
  const shown = ordered.slice(0, 4);
  return card({
    title: "Drift", key: "drift", href: "#/drift", linkText: "Open drift matrix",
    body: [
      el("div", { class: "mini-stats" },
        el("span", {}, el("b", { class: triggered ? "hot" : "" }, `${triggered}/${evaluated}`), "checks triggered"),
        el("span", {}, el("b", { class: nDrifted ? "hot" : "" }, `${nDrifted}/${cols.length}`), "features drifted"),
        el("span", {}, el("b", { class: datasetDrifted ? "hot" : "" }, datasetDrifted ? "Drifted" : "Not drifted"), "dataset")),
      el("ul", { class: "mini-list" }, shown.map((c) =>
        el("li", {},
          el("span", { class: "mini-main mono" }, c.column),
          el("span", { class: "mini-side" }, `${c.results.filter((r) => r.drifted).length}/${c.results.length} detectors`),
          el("span", { class: `chip ${c.drifted ? "fail" : "pass"}` }, c.drifted ? "Drifted" : "Stable")))),
      cols.length > shown.length ? el("p", { class: "card-more" }, `+ ${cols.length - shown.length} more`) : null,
    ],
  });
}

function renderOverviewCards(review, drift, rc) {
  $("overview-cards").replaceChildren(findingsCard(review), driftCard(drift), rootCauseCard(rc));
}

// Each sub-page names its own source: drift and root cause describe
// monitoring output (or the sample), never the PR under review.
function renderPageContext(review, drift, rc) {
  const ctx = (page, text) => { document.querySelector(`.page[data-page="${page}"] .page-context`).textContent = text; };
  ctx("findings", review
    ? [review.pr_id != null ? `PR #${review.pr_id}` : "Latest review", review.pr_title, review.repo_full_name].filter(Boolean).join(" · ")
    : "Review unavailable");
  ctx("drift", drift ? `${drift.label || "Unlabelled report"} · ${prov.drift.state === "sample" ? "bundled sample" : "monitoring output"}` : "Drift report unavailable");
  ctx("root-cause", rc ? `${rc.label || "Unlabelled report"} · ${prov["root-cause"].state === "sample" ? "bundled sample" : "monitoring output"}` : "Root cause unavailable");
}

// ---------- chrome ----------

function setupTheme() {
  const root = document.documentElement;
  const btn = $("theme-toggle");
  const paint = () => {
    const next = root.dataset.theme === "light" ? "dark" : "light";
    $("theme-label").textContent = `${next[0].toUpperCase()}${next.slice(1)} theme`;
    btn.setAttribute("aria-label", `Switch to ${next} theme`);
  };
  paint();
  btn.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    try { localStorage.setItem("skynet-theme", root.dataset.theme); } catch { /* storage blocked: theme just won't persist */ }
    paint();
  });
}

const PAGES = { overview: "Overview", findings: "Findings", "root-cause": "Root cause", drift: "Drift matrix" };

// Hash routes (#/findings) so the static server needs no routes and the
// back button works. Old single-page anchors (#findings) still resolve.
function route(moveFocus) {
  const id = (location.hash.match(/^#\/?([\w-]+)/) || [])[1];
  const page = id in PAGES ? id : "overview";
  document.querySelectorAll(".page").forEach((p) => { p.hidden = p.dataset.page !== page; });
  document.querySelectorAll(".rail-nav a").forEach((a) => {
    if (a.dataset.page === page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title = `${PAGES[page]} · Skynet_DS`;
  hideTip();
  window.scrollTo(0, 0);
  if (moveFocus) document.querySelector(`.page[data-page="${page}"] h1`)?.focus({ preventScroll: true });
}

async function init() {
  setupTheme();
  route(false);
  window.addEventListener("hashchange", () => route(true));

  const [review, drift, rc] = await Promise.all([
    load("pr-review", "/api/pr-review"),
    load("drift", "/api/drift"),
    load("root-cause", "/api/root-cause"),
  ]);

  renderHeader(review);
  renderKpis(review, drift, rc);
  renderFindings(review);
  renderRootCause(rc);
  renderDrift(drift);
  renderOverviewCards(review, drift, rc);
  renderPageContext(review, drift, rc);
  paintProvenance();
}

init();
