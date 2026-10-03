// Arbitration page: one disputed line at a time, decided from the keyboard.
"use strict";

const $ = (s) => document.querySelector(s);
const state = { pages: [], page: null, queue: null, idx: 0, history: [], editing: false };

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

// ---- page list ---------------------------------------------------------------
async function loadPages() {
  state.pages = await api("/api/pages");
  const ul = $("#pages");
  ul.innerHTML = "";
  for (const p of state.pages) {
    const li = document.createElement("li");
    li.dataset.page = p.page;
    li.className = (p.done ? "done " : "") + (p.page === state.page ? "cur" : "");
    li.innerHTML = `<span>${p.page}</span><span class="n">${p.decided}/${p.items}</span>`;
    li.onclick = () => openPage(p.page);
    ul.appendChild(li);
  }
  showAllDone();
}

// "all pages done" when every queue is decided
function showAllDone() {
  const all = state.pages.length > 0 && state.pages.every((p) => p.done);
  $("#alldone").hidden = !all;
  return all;
}

// history is kept across pages, so undo can go back over a page jump
async function openPage(pid) {
  await pending;
  state.page = pid;
  state.queue = await api(`/api/queue/${encodeURIComponent(pid)}`);
  const first = state.queue.items.findIndex((it) => !decided(it));
  state.idx = first >= 0 ? first : 0;
  try { localStorage.setItem("arb.page", pid); } catch (e) { /* ignore */ }
  await loadPages();
  render();
}

// ---- helpers -------------------------------------------------------------------
const CHOICES = ["A", "B", "neither", "either", "unknown"];
const LABEL = { either: "either: it is one of these two", unknown: "unknown: neither reading is confirmed" };
// who decided, as shown beside a decision not made by Carson
const BY_LABEL = { reviewer: "review model (whole-book pass)" };
const decided = (it) => !!(it.decision && CHOICES.includes(it.decision.choice));
const chars = (s) => {
  if (s == null) return [];
  if (window.Intl && Intl.Segmenter) {
    return Array.from(new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(s), (x) => x.segment);
  }
  return Array.from(s);
};

// character-level LCS diff: returns [markA[], markB[]] booleans (true = differs)
function charDiff(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ma = new Array(n).fill(true), mb = new Array(m).fill(true);
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ma[i] = mb[j] = false; i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  return [ma, mb];
}

const esc = (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] || c;
// long s gets a serif face (in most monospace fonts it is hard to tell from f)
const glyph = (c) => c === "ſ" ? '<span class="ls">ſ</span>' : esc(c);

function renderMarked(el, cs, marks) {
  let html = "", on = false;
  cs.forEach((c, k) => {
    if (marks[k] !== on) { html += marks[k] ? "<mark>" : "</mark>"; on = marks[k]; }
    // make a differing space or line break visible
    html += marks[k] && c === " " ? "·" : (marks[k] && c === "\n" ? "⏎\n" : glyph(c));
  });
  if (on) html += "</mark>";
  el.innerHTML = html;
}

const stripUrl = (p) => p ? "/" + p.replace(/^pages\/strips\//, "strips/").replace(/^pages\/read\//, "read/") : null;

// ---- rendering -----------------------------------------------------------------
function render() {
  const q = state.queue;
  $("#empty").hidden = !!q;
  if (!q) return;
  const items = q.items;
  const nDec = items.filter(decided).length;
  $("#pageid").textContent = q.page;
  $("#progress").textContent = `${nDec} of ${items.length} decided`;
  $("#done").hidden = !(items.length && nDec === items.length);
  if (!items.length) {
    $("#item").hidden = true; $("#empty").hidden = false;
    $("#empty").textContent = "No disagreements on this page."; $("#done").hidden = false;
    return;
  }
  $("#item").hidden = false;
  const it = items[state.idx];
  $("#pos").textContent = `item ${state.idx + 1} / ${items.length}`;

  const img = $("#crop");
  img.src = it.crop ? "/" + it.crop : "";
  $("#cropbox").dataset.strip = stripUrl(it.strip) || "";

  let hint = `${it.kind}`;
  if (it.key != null) hint += ` · note ${it.key}` + (it.line != null ? ` line ${it.line + 1}` : " (whole note)");
  if (it.kind === "unmatched") hint += ` · only ${it.side} has this` + (it.whole_note ? " note" : " line");
  if (it.where_a || it.where_b) hint += ` · A ${it.where_a || "—"} · B ${it.where_b || "—"}`;
  $("#hint").textContent = hint + (it.strip ? " · click the image for the native strip" : "");
  $("#ctxb").textContent = it.context_before || "";
  $("#ctxa").textContent = it.context_after || "";

  const ta = $("#ra .txt"), tb = $("#rb .txt");
  const ca = chars(it.a), cb = chars(it.b);
  const [ma, mb] = charDiff(ca, cb);
  if (it.a == null) { ta.innerHTML = '<span class="none">(no line)</span>'; } else renderMarked(ta, ca, ma);
  if (it.b == null) { tb.innerHTML = '<span class="none">(no line)</span>'; } else renderMarked(tb, cb, mb);

  let extra = "";
  if (it.kind === "structural" && it.field !== "blocks") extra = it.text;
  if (it.kind === "note-structure") extra = it.hint || "";
  if (it.kind === "flagged") extra = it.hint || ("Both readers agree on this line. Flagged:\n" + (it.flags || []).join("\n"));
  $("#extra").textContent = extra;
  renderBlocks(it);
  const isBlocks = it.kind === "structural" && it.field === "blocks";
  $("#ra").hidden = $("#rb").hidden = isBlocks;

  const d = it.decision;
  const st = $("#state");
  st.className = d && decided(it) ? d.choice : "";
  st.textContent = d ? `decided: ${LABEL[d.choice] || d.choice}` + (d.choice === "neither" ? ` → ${JSON.stringify(d.text)}` : "")
    + (d.by && d.by !== "carson" ? ` [${BY_LABEL[d.by] || d.by}]` : "") : "";
  closeEditor();
}

// structural "blocks" item: only the blocks that differ, side by side, with one block of
// context; the full lists sit behind a toggle
function renderBlocks(it) {
  const box = $("#blocks");
  if (it.kind !== "structural" || it.field !== "blocks") { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  const cell = (t, other, side) => {
    if (t == null) return '<td class="none">(no block)</td>';
    if (other == null) return `<td class="d${side}"><mark>${Array.from(t).map(glyph).join("")}</mark></td>`;
    const ca = chars(t), cb = chars(other);
    const [m] = charDiff(ca, cb);
    let h = "", on = false;
    ca.forEach((c, k) => { if (m[k] !== on) { h += m[k] ? "<mark>" : "</mark>"; on = m[k]; } h += glyph(c); });
    if (on) h += "</mark>";
    return `<td class="d${side}">${h}</td>`;
  };
  let rows = "";
  for (const r of it.block_rows || []) {
    if (r.status === "gap") rows += '<tr class="gap"><td>⋯</td><td>⋯</td></tr>';
    else if (r.status === "same") rows += `<tr class="same"><td>${Array.from(r.a).map(glyph).join("")}</td><td>${Array.from(r.b).map(glyph).join("")}</td></tr>`;
    else rows += `<tr class="diff">${cell(r.a, r.b, "a")}${cell(r.b, r.a, "b")}</tr>`;
  }
  const full = (xs) => (xs || []).map((x) => Array.from(x).map(glyph).join("")).join("<br>");
  box.innerHTML = `<div class="small">${esc2(it.text)}</div>
    <table><thead><tr><th>A</th><th>B</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="small">1 = build the page on A's block structure, 2 = on B's.</div>
    <details><summary>show all blocks</summary><table><tr><td>${full(it.a_blocks)}</td><td>${full(it.b_blocks)}</td></tr></table></details>`;
}
const esc2 = (t) => Array.from(t || "").map(esc).join("");

// ---- actions -------------------------------------------------------------------
// POSTs run one after another, in key order; the UI moves on without waiting for them
let pending = Promise.resolve();
function send(body, onError) {
  const p = pending.then(() => api("/api/decide", body)).then((prog) => { updateSide(prog); return prog; })
    .catch((e) => {
      onError && onError();
      alert(`Not saved: ${e.message}`);
      render();
      return null;
    });
  pending = p;
  return p;
}

function decide(choice, text) {
  const it = state.queue.items[state.idx];
  const before = it.decision;
  it.decision = { choice, text, by: "carson" };
  const page = state.page;
  state.history.push({ page, idx: state.idx });
  const sent = send({ page, item: it.id, choice, text }, () => { it.decision = before; });
  advance();
  if (state.queue.items.every(decided)) {
    // the page is finished: once the server confirms, move to the next page with work left
    sent.then((prog) => {
      if (!prog || state.page !== page) return;
      if (prog.next_page) openPage(prog.next_page);
      else loadPages();
    });
  }
}

function advance() {
  // next undecided item after this one; else stay (page done)
  const items = state.queue.items, n = items.length;
  for (let k = 1; k <= n; k++) {
    const j = (state.idx + k) % n;
    if (!decided(items[j])) { state.idx = j; break; }
  }
  render();
}

async function undo() {
  if (!state.history.length) return;
  const { page, idx: j } = state.history.pop();
  if (page !== state.page) await openPage(page);     // back over a page jump
  const it = state.queue.items[j];
  const before = it.decision;
  delete it.decision;
  state.idx = j;
  send({ page: state.page, item: it.id, choice: "clear" }, () => { it.decision = before; })
    .then(() => loadPages());
  render();
}

function updateSide(prog) {
  const li = document.querySelector(`#pages li[data-page="${CSS.escape(prog.page)}"]`);
  if (li) {
    li.querySelector(".n").textContent = `${prog.decided}/${prog.items}`;
    li.classList.toggle("done", prog.done);
  }
  const rec = state.pages.find((p) => p.page === prog.page);
  if (rec) { rec.decided = prog.decided; rec.done = prog.done; }
  showAllDone();
}

function openEditor() {
  const it = state.queue.items[state.idx];
  state.editing = true;
  $("#editor").hidden = false;
  const t = $("#edit");
  t.value = (it.decision && it.decision.choice === "neither") ? it.decision.text : (it.a ?? it.b ?? "");
  t.rows = Math.max(2, t.value.split("\n").length);
  t.focus();
}
function closeEditor() { state.editing = false; $("#editor").hidden = true; }

function move(delta) {
  const n = state.queue.items.length;
  state.idx = (state.idx + delta + n) % n;
  render();
}

// ---- whole-page overlay ---------------------------------------------------------
// pages/read/<id>.jpg fitted to the window height; drag to pan, wheel or +/- to zoom.
// While it is open the decision keys do nothing.
const ov = { open: false, z: 1, x: 0, y: 0, fit: 1, drag: null };

function ovApply() {
  $("#pageimg").style.transform = `translate(${ov.x}px, ${ov.y}px) scale(${ov.z})`;
}
function ovFit() {
  const img = $("#pageimg");
  if (!img.naturalHeight) return;
  ov.fit = ov.z = innerHeight / img.naturalHeight;
  ov.x = (innerWidth - img.naturalWidth * ov.z) / 2;
  ov.y = 0;
  ovApply();
}
function ovZoom(factor, cx = innerWidth / 2, cy = innerHeight / 2) {
  const z = Math.min(ov.fit * 8, Math.max(ov.fit * 0.5, ov.z * factor));
  ov.x = cx - (cx - ov.x) * (z / ov.z);
  ov.y = cy - (cy - ov.y) * (z / ov.z);
  ov.z = z;
  ovApply();
}
function openOverlay() {
  if (!state.page) return;
  ov.open = true;
  $("#overlay").hidden = false;
  const img = $("#pageimg");
  const src = `/read/${encodeURIComponent(state.page)}.jpg`;
  if (img.getAttribute("src") !== src) { img.onload = ovFit; img.src = src; } else ovFit();
}
function closeOverlay() { ov.open = false; $("#overlay").hidden = true; }

const ovEl = $("#overlay");
ovEl.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  ovZoom(ev.deltaY < 0 ? 1.15 : 1 / 1.15, ev.clientX, ev.clientY);
}, { passive: false });
ovEl.addEventListener("mousedown", (ev) => {
  ov.drag = { x: ev.clientX - ov.x, y: ev.clientY - ov.y };
  ovEl.classList.add("drag");
});
window.addEventListener("mousemove", (ev) => {
  if (!ov.drag) return;
  ov.x = ev.clientX - ov.drag.x;
  ov.y = ev.clientY - ov.drag.y;
  ovApply();
});
window.addEventListener("mouseup", () => { ov.drag = null; ovEl.classList.remove("drag"); });
window.addEventListener("resize", () => { if (ov.open) ovFit(); });

document.addEventListener("keydown", (ev) => {
  if (ov.open) {
    // only the overlay's own keys; decisions are off while it is open
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const k = ev.key;
    const act = { Escape: closeOverlay, p: closeOverlay, "+": () => ovZoom(1.25), "=": () => ovZoom(1.25),
                  "-": () => ovZoom(0.8), "_": () => ovZoom(0.8), "0": ovFit }[k];
    if (act) act();
    ev.preventDefault();          // every other key (1, 2, e, arrows...) is swallowed
    return;
  }
  if (!state.queue || !state.queue.items.length) return;
  if (state.editing) {
    if (ev.key === "Escape") { ev.preventDefault(); closeEditor(); }
    else if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); decide("neither", $("#edit").value); }
    return;
  }
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const act = { "1": () => decide("A"), "2": () => decide("B"), "3": () => decide("either"),
                "4": () => decide("unknown"), "e": openEditor, "u": undo, "p": openOverlay,
                "ArrowLeft": () => move(-1), "ArrowRight": () => move(1) }[ev.key];
  if (act) { ev.preventDefault(); act(); }
});

document.querySelectorAll("#buttons button").forEach((b) => {
  b.onclick = () => document.dispatchEvent(new KeyboardEvent("keydown", { key: b.dataset.k }));
});

// crop: click opens the native strip; hover zooms 2x around the pointer
const box = $("#cropbox");
box.onclick = () => { if (box.dataset.strip) window.open(box.dataset.strip, "_blank"); };
box.onmouseenter = () => box.classList.add("zoom");
box.onmouseleave = () => { box.classList.remove("zoom"); $("#crop").style.transformOrigin = "center"; };
box.onmousemove = (ev) => {
  const img = $("#crop");
  const bx = box.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (ev.clientX - bx.left - img.offsetLeft) / img.offsetWidth));
  const y = Math.min(1, Math.max(0, (ev.clientY - bx.top - img.offsetTop) / img.offsetHeight));
  img.style.transformOrigin = `${x * 100}% ${y * 100}%`;
};

(async () => {
  await loadPages();
  let last = null;
  try { last = localStorage.getItem("arb.page"); } catch (e) { /* ignore */ }
  const pick = state.pages.find((p) => p.page === last) || state.pages.find((p) => !p.done) || state.pages[0];
  if (pick) openPage(pick.page);
})();
