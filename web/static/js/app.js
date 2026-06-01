// App entry: wires sidebar, theme, settings drawer, status pill,
// then mounts the router with all 5 page modules.

import { api, getToken, setToken } from "./api.js";
import { registerRoute, start } from "./router.js";
import { renderLive } from "./pages/live.js";
import { renderSessions } from "./pages/sessions.js";
import { renderRegistry } from "./pages/registry.js";
import { renderJobs } from "./pages/jobs.js";
import { renderHealth } from "./pages/health.js";
import { renderMaintenance } from "./pages/maintenance.js";

// ---- theme persistence ----
const root = document.documentElement;
const savedTheme = localStorage.getItem("sv_theme")
  || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
applyTheme(savedTheme);

document.getElementById("themeToggle").addEventListener("click", () => {
  const next = root.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  localStorage.setItem("sv_theme", next);
});

function applyTheme(t) {
  root.dataset.theme = t;
  document.getElementById("themeIcon").textContent = t === "light" ? "☀" : "☾";
  document.getElementById("themeLabel").textContent = t === "light" ? "Light" : "Dark";
}

// ---- settings drawer ----
// The drawer only carries the Bearer-token field, and only opens when the
// server actually rejects a request with 401 (i.e. SV_API_TOKEN is set on
// the api). For dev (no token) the gear stays out of the way.
const drawer = document.getElementById("settingsDrawer");
const apiTokenInput = document.getElementById("apiToken");

function openSettings() {
  apiTokenInput.value = getToken();
  drawer.hidden = false;
  apiTokenInput.focus();
}
function closeSettings() { drawer.hidden = true; }

document.getElementById("settingsBtn").addEventListener("click", openSettings);
document.getElementById("settingsClose").addEventListener("click", closeSettings);
document.getElementById("settingsClear").addEventListener("click", () => {
  setToken("");
  apiTokenInput.value = "";
});
document.getElementById("settingsSave").addEventListener("click", () => {
  setToken(apiTokenInput.value.trim());
  closeSettings();
  pollHealth();
});
drawer.addEventListener("click", (e) => { if (e.target === drawer) closeSettings(); });

document.addEventListener("sv:auth-required", () => {
  setStatus("err", "auth required");
  openSettings();
});

// ---- live status pill ----
const statusEl = document.getElementById("apiStatus");
const statusText = document.getElementById("apiStatusText");
let healthTimer = null;

function setStatus(kind, text) {
  statusEl.classList.remove("connecting", "warn", "err");
  if (kind === "connecting") statusEl.classList.add("connecting");
  if (kind === "warn") statusEl.classList.add("warn");
  if (kind === "err") statusEl.classList.add("err");
  statusText.textContent = text;
}

const settingsBtn = document.getElementById("settingsBtn");

async function pollHealth() {
  try {
    const h = await api.health();
    const auth = h.auth_enabled ? " · auth" : "";
    setStatus("ok", `online · ${h.workers}w${auth}`);
    // Settings drawer only carries the bearer token field. When the server
    // doesn't enforce auth it has nothing to do — keep the gear out of the
    // way. The `sv:auth-required` listener still pops the drawer if the
    // server's posture changes (e.g. someone restarts with SV_API_TOKEN).
    settingsBtn.hidden = !h.auth_enabled;
  } catch (e) {
    setStatus("err", "offline");
  }
}

setStatus("connecting", "connecting…");
pollHealth();
healthTimer = setInterval(pollHealth, 7000);
window.addEventListener("beforeunload", () => clearInterval(healthTimer));

// ---- routes ----
registerRoute("/live", renderLive);
registerRoute("/sessions", renderSessions);
registerRoute("/registry", renderRegistry);
registerRoute("/jobs", renderJobs);
registerRoute("/health", renderHealth);
registerRoute("/maintenance", renderMaintenance);

start({
  defaultRoute: "/live",
  outlet: document.getElementById("page"),
  titleEl: document.getElementById("pageTitle"),
  navEl: document.getElementById("nav"),
});
