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
  if (it.kind === "structural") {
    extra = it.text;
    if (it.a_blocks) extra += `\n\nA blocks:\n  ${it.a_blocks.join("\n  ")}\n\nB blocks:\n  ${it.b_blocks.join("\n  ")}`;
    extra += it.field === "blocks" ? "\n\n1 = build the page on A's block structure, 2 = on B's." : "";
  }
  if (it.kind === "flagged") extra = "Both reads agree. Flagged:\n" + (it.flags || []).join("\n");
  $("#extra").textContent = extra;

  const d = it.decision;
  const st = $("#state");
  st.className = d && decided(it) ? d.choice : "";
  st.textContent = d ? `decided: ${LABEL[d.choice] || d.choice}` + (d.choice === "neither" ? ` → ${JSON.stringify(d.text)}` : "") : "";
  closeEditor();
}

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
  it.decision = { choice, text };
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

document.addEventListener("keydown", (ev) => {
  if (!state.queue || !state.queue.items.length) return;
  if (state.editing) {
    if (ev.key === "Escape") { ev.preventDefault(); closeEditor(); }
    else if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); decide("neither", $("#edit").value); }
    return;
  }
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const act = { "1": () => decide("A"), "2": () => decide("B"), "3": () => decide("either"),
                "4": () => decide("unknown"), "e": openEditor, "u": undo,
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
