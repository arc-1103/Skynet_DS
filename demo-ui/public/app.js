async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function renderPrReview(review) {
  document.getElementById("pr-title").textContent = `PR #${review.pr_id}: ${review.pr_title}`;
  document.getElementById("pr-meta").textContent = review.repo_full_name;

  const labelsEl = document.getElementById("pr-labels");
  labelsEl.innerHTML = "";
  review.labels.forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "label-chip";
    chip.textContent = label;
    labelsEl.appendChild(chip);
  });

  const statusEl = document.getElementById("pr-status");
  statusEl.textContent = review.overall_passed ? "Passed" : "Changes requested";
  statusEl.classList.add(review.overall_passed ? "pass" : "fail");

  document.getElementById("findings-count").textContent = review.inline_findings.length;

  const listEl = document.getElementById("findings-list");
  listEl.innerHTML = "";
  review.inline_findings.forEach((finding) => {
    const card = document.createElement("div");
    card.className = "finding";
    card.dataset.severity = finding.severity;

    const loc = finding.line_number
      ? `${finding.file_path}:${finding.line_number}`
      : finding.file_path;

    card.innerHTML = `
      <div class="finding-top">
        <div>
          <span class="finding-title">${finding.title}</span>
          <div class="finding-loc">${loc}</div>
        </div>
        <span class="sev-badge ${finding.severity}">${finding.severity}</span>
      </div>
      <p class="finding-desc">${finding.description}</p>
      ${finding.suggestion ? `<div class="finding-suggestion">💡 ${finding.suggestion}</div>` : ""}
    `;
    listEl.appendChild(card);
  });
}

function renderDrift(report) {
  document.getElementById("drift-label").textContent = report.label;
  const nDrifted = report.columns.filter((c) => c.drifted).length;
  document.getElementById("drift-summary").textContent =
    `${nDrifted}/${report.columns.length} column(s) flagged (share threshold ${Math.round(report.share_threshold * 100)}%)`;

  const tbody = document.getElementById("drift-tbody");
  tbody.innerHTML = "";
  report.columns.forEach((col) => {
    const row = document.createElement("tr");
    const methodChips = col.results
      .map(
        (r) =>
          `<span class="method-chip ${r.drifted ? "hit" : ""}">${r.method}${
            r.p_value != null ? ` p=${r.p_value.toFixed(3)}` : ` ${r.statistic.toFixed(2)}`
          }</span>`
      )
      .join("");

    row.innerHTML = `
      <td><strong>${col.column}</strong></td>
      <td>${col.kind}</td>
      <td><span class="verdict ${col.drifted ? "drifted" : "stable"}">${col.drifted ? "Drifted" : "Stable"}</span></td>
      <td>${methodChips}</td>
    `;
    tbody.appendChild(row);
  });
}

function renderRootCause(rootCause) {
  document.getElementById("rootcause-summary").textContent = rootCause.separable
    ? `Surrogate AUC ${rootCause.surrogate_auc.toFixed(2)} (threshold ${rootCause.auc_threshold}) — samples are separable, attribution below is meaningful.`
    : `Surrogate AUC ${rootCause.surrogate_auc.toFixed(2)} — samples not reliably separable; attribution suppressed.`;

  const listEl = document.getElementById("attribution-list");
  listEl.innerHTML = "";
  rootCause.attributions.forEach((attr) => {
    const row = document.createElement("div");
    row.className = "attribution-row";
    const movement =
      attr.reference_mean != null && attr.current_mean != null
        ? ` (${attr.reference_mean.toFixed(1)} → ${attr.current_mean.toFixed(1)})`
        : "";
    row.innerHTML = `
      <div class="attribution-feature">${attr.feature}<div style="font-weight:400;font-size:11px;color:var(--brown-muted)">${attr.direction}${movement}</div></div>
      <div class="attribution-bar-track"><div class="attribution-bar-fill" style="width:${Math.round(attr.share * 100)}%"></div></div>
      <div class="attribution-share">${Math.round(attr.share * 100)}%</div>
    `;
    listEl.appendChild(row);
  });
}

async function init() {
  const [review, drift, rootCause] = await Promise.all([
    fetchJson("/api/pr-review"),
    fetchJson("/api/drift"),
    fetchJson("/api/root-cause"),
  ]);
  renderPrReview(review);
  renderDrift(drift);
  renderRootCause(rootCause);
}

init().catch((err) => {
  document.getElementById("pr-title").textContent = "Failed to load demo data";
  // eslint-disable-next-line no-console
  console.error(err);
});
