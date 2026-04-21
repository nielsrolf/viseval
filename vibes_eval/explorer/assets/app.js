// VisEval HTML Explorer - vanilla JS, no build step.

const state = {
    header: null,
    rows: [],
    filtered: [],
    sortKey: null,
    sortAsc: true,
    selectedIdx: null,
    filters: {
        group: "__all__",
        model: "__all__",
        question_id: "__all__",
        text: "",
        metric: {}, // { metric: {min, max} }
    },
};

async function main() {
    const resp = await fetch("data.json");
    const data = await resp.json();
    state.header = data.header;
    state.rows = data.rows;

    document.getElementById("eval-title").textContent =
        `VisEval Explorer — ${data.header.name}`;
    document.getElementById("header-info").textContent =
        `${data.header.n_rows} rows · primary metric: ${data.header.primary_metric}`;

    state.sortKey = data.header.primary_metric;
    state.sortAsc = true;

    buildTabs();
    buildFilters();
    buildTableHeader();
    buildPlots();
    setupDivider();
    applyFiltersAndRender();
}

function buildTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
        });
    });
}

function uniqueValues(col) {
    const seen = new Set();
    for (const row of state.rows) {
        if (row[col] !== undefined && row[col] !== null) seen.add(row[col]);
    }
    return Array.from(seen).sort();
}

function buildFilters() {
    const container = document.getElementById("filters");
    container.innerHTML = "";

    const groups = uniqueValues("group");
    const models = uniqueValues("model");
    const questionIds = uniqueValues("question_id");

    container.appendChild(makeSelect("Group", "group", ["__all__", ...groups]));
    container.appendChild(makeSelect("Model", "model", ["__all__", ...models]));
    container.appendChild(makeSelect("Question", "question_id", ["__all__", ...questionIds]));

    const textLabel = document.createElement("label");
    textLabel.innerHTML = `Search: `;
    const textInput = document.createElement("input");
    textInput.type = "text";
    textInput.placeholder = "substring of question/answer";
    textInput.addEventListener("input", e => {
        state.filters.text = e.target.value.toLowerCase();
        applyFiltersAndRender();
    });
    textLabel.appendChild(textInput);
    container.appendChild(textLabel);

    for (const metric of state.header.metrics) {
        const wrap = document.createElement("label");
        wrap.innerHTML = `${metric}: `;
        const minInp = document.createElement("input");
        minInp.type = "number";
        minInp.placeholder = "min";
        minInp.addEventListener("input", e => {
            state.filters.metric[metric] = state.filters.metric[metric] || {};
            state.filters.metric[metric].min = e.target.value === "" ? null : parseFloat(e.target.value);
            applyFiltersAndRender();
        });
        const maxInp = document.createElement("input");
        maxInp.type = "number";
        maxInp.placeholder = "max";
        maxInp.addEventListener("input", e => {
            state.filters.metric[metric] = state.filters.metric[metric] || {};
            state.filters.metric[metric].max = e.target.value === "" ? null : parseFloat(e.target.value);
            applyFiltersAndRender();
        });
        wrap.appendChild(minInp);
        wrap.appendChild(document.createTextNode(" – "));
        wrap.appendChild(maxInp);
        container.appendChild(wrap);
    }

    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear filters";
    clearBtn.addEventListener("click", resetFilters);
    container.appendChild(clearBtn);
}

function makeSelect(labelText, field, values) {
    const label = document.createElement("label");
    label.textContent = `${labelText}: `;
    const sel = document.createElement("select");
    sel.dataset.field = field;
    for (const v of values) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v === "__all__" ? "(any)" : String(v);
        sel.appendChild(opt);
    }
    sel.addEventListener("change", e => {
        state.filters[field] = e.target.value;
        applyFiltersAndRender();
    });
    label.appendChild(sel);
    return label;
}

function resetFilters() {
    state.filters = { group: "__all__", model: "__all__", question_id: "__all__", text: "", metric: {} };
    document.querySelectorAll("#filters select").forEach(sel => { sel.value = "__all__"; });
    document.querySelectorAll("#filters input").forEach(inp => { inp.value = ""; });
    applyFiltersAndRender();
}

function setFilterValue(field, value) {
    state.filters[field] = value;
    const sel = document.querySelector(`#filters select[data-field="${field}"]`);
    if (sel) sel.value = value;
    applyFiltersAndRender();
}

function buildTableHeader() {
    const row = document.getElementById("thead-row");
    row.innerHTML = "";
    const cols = tableColumns();
    for (const col of cols) {
        const th = document.createElement("th");
        th.dataset.key = col.key;
        th.textContent = col.label;
        const indicator = document.createElement("span");
        indicator.className = "sort-indicator";
        th.appendChild(indicator);
        th.addEventListener("click", () => {
            if (state.sortKey === col.key) {
                state.sortAsc = !state.sortAsc;
            } else {
                state.sortKey = col.key;
                state.sortAsc = true;
            }
            applyFiltersAndRender();
        });
        row.appendChild(th);
    }
}

function tableColumns() {
    const cols = [
        { key: "model", label: "Model", kind: "str" },
        { key: "group", label: "Group", kind: "str" },
        { key: "question_id", label: "Question", kind: "str" },
    ];
    for (const m of state.header.metrics) {
        cols.push({ key: m, label: m, kind: "num" });
    }
    cols.push({ key: "answer", label: "Answer", kind: "str", truncate: true });
    return cols;
}

function applyFiltersAndRender() {
    const f = state.filters;
    let out = state.rows.filter(r => {
        if (f.group !== "__all__" && r.group !== f.group) return false;
        if (f.model !== "__all__" && r.model !== f.model) return false;
        if (f.question_id !== "__all__" && r.question_id !== f.question_id) return false;
        if (f.text) {
            const q = String(r.question || "").toLowerCase();
            const a = String(r.answer || "").toLowerCase();
            if (!q.includes(f.text) && !a.includes(f.text)) return false;
        }
        for (const [metric, range] of Object.entries(f.metric)) {
            if (!range) continue;
            const v = r[metric];
            if (v === null || v === undefined) return false;
            if (range.min !== null && range.min !== undefined && v < range.min) return false;
            if (range.max !== null && range.max !== undefined && v > range.max) return false;
        }
        return true;
    });

    const key = state.sortKey;
    const asc = state.sortAsc ? 1 : -1;
    out.sort((a, b) => {
        const av = a[key], bv = b[key];
        if (av === undefined || av === null) return 1;
        if (bv === undefined || bv === null) return -1;
        if (typeof av === "number" && typeof bv === "number") return asc * (av - bv);
        return asc * String(av).localeCompare(String(bv));
    });

    state.filtered = out;
    renderTable();
    renderStatus();
    renderSortIndicators();
}

function renderSortIndicators() {
    document.querySelectorAll("#examples-table thead th").forEach(th => {
        const indicator = th.querySelector(".sort-indicator");
        if (th.dataset.key === state.sortKey) {
            indicator.textContent = state.sortAsc ? " ▲" : " ▼";
        } else {
            indicator.textContent = "";
        }
    });
}

function renderStatus() {
    const n = state.filtered.length;
    const total = state.rows.length;
    document.getElementById("status-bar").textContent =
        `${n} / ${total} rows` + (n === total ? "" : " (filtered)");
}

function renderTable() {
    const tbody = document.getElementById("tbody");
    tbody.innerHTML = "";
    const cols = tableColumns();
    const frag = document.createDocumentFragment();
    // Cap to 2000 visible rows at a time to keep the DOM light.
    const MAX_RENDER = 2000;
    const rows = state.filtered.slice(0, MAX_RENDER);

    rows.forEach((row, idx) => {
        const tr = document.createElement("tr");
        tr.dataset.rowId = row.__explorer_id ?? row._viseval_idx ?? state.rows.indexOf(row);
        for (const col of cols) {
            const td = document.createElement("td");
            let v = row[col.key];
            if (v === null || v === undefined) {
                td.textContent = "";
            } else if (col.kind === "num") {
                td.textContent = formatNumber(v);
                td.className = "metric-cell";
            } else if (col.truncate) {
                td.className = "truncate";
                td.textContent = String(v);
                td.title = String(v);
            } else {
                td.textContent = String(v);
            }
            tr.appendChild(td);
        }
        tr.addEventListener("click", () => selectRow(row, tr));
        frag.appendChild(tr);
    });

    tbody.appendChild(frag);

    if (state.filtered.length > MAX_RENDER) {
        const note = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = cols.length;
        td.style.textAlign = "center";
        td.style.color = "#888";
        td.style.padding = "0.5em";
        td.textContent = `Showing first ${MAX_RENDER} of ${state.filtered.length} rows — add filters to narrow further.`;
        note.appendChild(td);
        tbody.appendChild(note);
    }
}

function formatNumber(v) {
    if (typeof v !== "number") return String(v);
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(2);
}

function selectRow(row, trEl) {
    document.querySelectorAll("#examples-table tbody tr.selected").forEach(el => el.classList.remove("selected"));
    if (trEl) trEl.classList.add("selected");
    state.selectedIdx = row;
    renderChat(row);
}

function renderChat(row) {
    const container = document.getElementById("chat-view");
    container.innerHTML = "";

    // Meta block
    const meta = document.createElement("div");
    meta.className = "chat-meta";

    const metaRow = document.createElement("div");
    metaRow.className = "chat-meta-row";
    metaRow.innerHTML = `<span><strong>Model:</strong> ${escapeHtml(row.model ?? "")}</span>` +
        `<span><strong>Group:</strong> ${escapeHtml(row.group ?? "")}</span>` +
        `<span><strong>Question:</strong> ${escapeHtml(row.question_id ?? "")}</span>`;
    meta.appendChild(metaRow);

    const scores = document.createElement("div");
    scores.className = "chat-scores";
    for (const m of state.header.metrics) {
        const v = row[m];
        const pill = document.createElement("span");
        pill.className = "score-pill" + (m === state.header.primary_metric ? " primary" : "");
        pill.textContent = `${m}: ${v === null || v === undefined ? "—" : formatNumber(v)}`;
        scores.appendChild(pill);
    }
    meta.appendChild(scores);

    // Meta columns (any extras)
    if (state.header.meta_columns && state.header.meta_columns.length) {
        const extra = document.createElement("div");
        extra.className = "chat-meta-row";
        extra.style.marginTop = "0.4em";
        for (const c of state.header.meta_columns) {
            const v = row[c];
            if (v === null || v === undefined || v === "") continue;
            const s = document.createElement("span");
            s.innerHTML = `<strong>${escapeHtml(c)}:</strong> ${escapeHtml(String(v))}`;
            extra.appendChild(s);
        }
        if (extra.children.length) meta.appendChild(extra);
    }

    container.appendChild(meta);

    // Actions
    const actions = document.createElement("div");
    actions.className = "chat-actions";

    const allResponses = document.createElement("button");
    allResponses.textContent = "All responses to this question";
    allResponses.addEventListener("click", () => {
        setFilterValue("question_id", row.question_id);
        setFilterValue("model", "__all__");
        setFilterValue("group", "__all__");
    });
    actions.appendChild(allResponses);

    const thisModel = document.createElement("button");
    thisModel.textContent = `Only ${row.model}`;
    thisModel.addEventListener("click", () => {
        setFilterValue("model", row.model);
    });
    actions.appendChild(thisModel);

    const thisGroup = document.createElement("button");
    thisGroup.textContent = `Only group: ${row.group}`;
    thisGroup.addEventListener("click", () => {
        setFilterValue("group", row.group);
        setFilterValue("model", "__all__");
    });
    actions.appendChild(thisGroup);

    container.appendChild(actions);

    // System bubble
    if (row.system) {
        container.appendChild(makeBubble("System", row.system, "bubble-system"));
    }
    // User bubble
    container.appendChild(makeBubble("User", row.question ?? "", "bubble-user"));
    // Assistant bubble
    container.appendChild(makeBubble("Assistant", row.answer ?? "", "bubble-assistant"));
}

function makeBubble(label, text, className) {
    const div = document.createElement("div");
    div.className = `bubble ${className}`;
    const lbl = document.createElement("span");
    lbl.className = "bubble-label";
    lbl.textContent = label;
    div.appendChild(lbl);
    div.appendChild(document.createTextNode(String(text)));
    return div;
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function buildPlots() {
    const container = document.getElementById("plots-container");
    container.innerHTML = "";
    const plots = state.header.plots || [];
    if (plots.length === 0) {
        container.innerHTML = "<p class='empty-hint'>No plots were rendered.</p>";
        return;
    }
    for (const p of plots) {
        const card = document.createElement("div");
        card.className = "plot-card";
        const h = document.createElement("h3");
        h.textContent = p.title;
        card.appendChild(h);
        const img = document.createElement("img");
        img.src = p.filename;
        img.alt = p.title;
        img.loading = "lazy";
        card.appendChild(img);
        container.appendChild(card);
    }
}

function setupDivider() {
    const divider = document.getElementById("divider");
    const left = document.getElementById("left-pane");
    let dragging = false;
    divider.addEventListener("mousedown", e => {
        dragging = true;
        document.body.style.cursor = "col-resize";
        e.preventDefault();
    });
    document.addEventListener("mousemove", e => {
        if (!dragging) return;
        const split = document.getElementById("split");
        const rect = split.getBoundingClientRect();
        const pct = Math.max(20, Math.min(80, ((e.clientX - rect.left) / rect.width) * 100));
        left.style.width = `${pct}%`;
    });
    document.addEventListener("mouseup", () => {
        dragging = false;
        document.body.style.cursor = "";
    });
}

main().catch(err => {
    console.error(err);
    document.body.innerHTML = `<pre style="color:red;padding:1em;">Failed to load explorer: ${err}</pre>`;
});
