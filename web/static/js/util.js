// Tiny DOM + format helpers shared by every page.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    if (Array.isArray(c)) node.append(...c);
    else if (c instanceof Node) node.append(c);
    else node.append(String(c));
  }
  return node;
}

export function fmtTime(seconds) {
  if (seconds == null) return "—";
  const s = Math.floor(seconds);
  const m = Math.floor(s / 60);
  const r = (s % 60).toString().padStart(2, "0");
  return `${m}:${r}`;
}

export function fmtTimestamp(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

export function fmtDateTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

export function fmtFloat(x, n = 3) {
  return typeof x === "number" ? x.toFixed(n) : "—";
}

// Color a speaker label deterministically. Uses HSL so we get visually
// distinct hues without picking from a small palette that wraps.
const _speakerHues = new Map();
export function speakerColor(label) {
  if (!label) return "var(--text-3)";
  if (_speakerHues.has(label)) return _speakerHues.get(label);
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  const isDark = document.documentElement.dataset.theme !== "light";
  const css = `hsl(${hue} 70% ${isDark ? 70 : 45}%)`;
  _speakerHues.set(label, css);
  return css;
}

export function attachProximityGlow(card) {
  card.classList.add("glow");
  card.addEventListener("mousemove", (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  });
}

export function toast(parent, kind, text, ttl = 5000) {
  const node = el("div", { class: `toast ${kind}` }, text);
  parent.append(node);
  if (ttl > 0) setTimeout(() => node.remove(), ttl);
  return node;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Reusable "page intro" callout that explains how to use the page.
// Persists collapsed/expanded state per `key` in localStorage so users
// who already know what the page does don't get nagged on every visit.
export function pageHint(key, title, bodyChildren) {
  const stateKey = `sv_hint_${key}`;
  const collapsed = localStorage.getItem(stateKey) === "1";

  const body = el("div", { class: "page-hint-body" }, ...bodyChildren);
  const toggle = el("button", { class: "icon-btn", type: "button", "aria-label": "toggle help" },
    collapsed ? "▸" : "▾");
  const card = el("section", { class: "page-hint" + (collapsed ? " is-collapsed" : "") },
    el("div", { class: "page-hint-head" },
      el("span", { class: "page-hint-icon" }, "?"),
      el("span", { class: "page-hint-title" }, title),
      toggle,
    ),
    body,
  );

  toggle.addEventListener("click", () => {
    const isNow = card.classList.toggle("is-collapsed");
    toggle.textContent = isNow ? "▸" : "▾";
    localStorage.setItem(stateKey, isNow ? "1" : "0");
  });
  return card;
}

// Wrap a label's <span> with a small `(?)` icon. We attach a custom tooltip
// (250ms hover delay, theme-aware, multi-line) instead of the native `title`
// attribute, which has a 1.5–2s OS-level delay and chops on `\n`.
export function withHint(spanContent, hint) {
  const help = el("span",
    { class: "field-help", role: "button", tabindex: "0", "aria-label": "help" },
    "?");
  attachTooltip(help, hint);
  return el("span", { class: "field-label-row" }, spanContent, help);
}

// One floating tooltip element shared across the page; positioned relative
// to the anchor on hover/focus. Multi-line: `\n` becomes <br>.
let _tipNode = null;
let _tipTimer = null;
let _tipAnchor = null;

function _ensureTipNode() {
  if (_tipNode) return _tipNode;
  _tipNode = el("div", { class: "sv-tooltip", role: "tooltip", hidden: "" });
  document.body.append(_tipNode);
  return _tipNode;
}

function _positionTip(anchor) {
  if (!_tipNode || !anchor) return;
  // Render off-screen to measure, then place above-or-below the anchor
  // with horizontal clamping into the viewport.
  const a = anchor.getBoundingClientRect();
  const t = _tipNode.getBoundingClientRect();
  const margin = 8;
  let top = a.top - t.height - margin;
  let placeBelow = false;
  if (top < 4) {
    top = a.bottom + margin;
    placeBelow = true;
  }
  let left = a.left + a.width / 2 - t.width / 2;
  left = Math.max(4, Math.min(left, window.innerWidth - t.width - 4));
  _tipNode.style.top = `${top + window.scrollY}px`;
  _tipNode.style.left = `${left + window.scrollX}px`;
  _tipNode.dataset.placement = placeBelow ? "below" : "above";
}

function _showTip(anchor, text) {
  const node = _ensureTipNode();
  node.innerHTML = String(text)
    .split("\n")
    .map(line => escapeHtml(line))
    .join("<br>");
  node.hidden = false;
  _tipAnchor = anchor;
  // Two-frame defer so the layout is measured after content lands.
  requestAnimationFrame(() => requestAnimationFrame(() => _positionTip(anchor)));
}

function _hideTip() {
  if (_tipTimer) { clearTimeout(_tipTimer); _tipTimer = null; }
  if (_tipNode) _tipNode.hidden = true;
  _tipAnchor = null;
}

export function attachTooltip(anchor, text, { delay = 250 } = {}) {
  if (!text) return;
  const open = () => {
    if (_tipTimer) clearTimeout(_tipTimer);
    _tipTimer = setTimeout(() => _showTip(anchor, text), delay);
  };
  anchor.addEventListener("mouseenter", open);
  anchor.addEventListener("focus", open);
  anchor.addEventListener("mouseleave", _hideTip);
  anchor.addEventListener("blur", _hideTip);
  // If the anchor is removed mid-hover, kill the pending timer.
  anchor.addEventListener("click", _hideTip);
  // Reposition on scroll/resize while open.
  window.addEventListener("scroll", () => { if (_tipAnchor === anchor) _positionTip(anchor); }, { passive: true });
  window.addEventListener("resize", () => { if (_tipAnchor === anchor) _positionTip(anchor); });
}
