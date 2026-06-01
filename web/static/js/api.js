// Thin fetch wrapper. Reads token from localStorage; pops the settings drawer
// on 401. Exposes a typed-ish surface for each route the app needs.
//
// Base URL is always same-origin — the api server hosts the static UI, so
// there's no cross-host case worth supporting. Removing the override avoids
// the foot-gun where users put `southeastasia` etc. into the field.

const TOKEN_KEY = "sv_token";
const LEGACY_BASE_KEY = "sv_base";
// One-time cleanup of the now-removed override. Old visits may have stored
// a value (often something like a region accidentally typed in) that would
// now break every request — wipe it on first load post-upgrade.
try { localStorage.removeItem(LEGACY_BASE_KEY); } catch {}

export function getToken() { return localStorage.getItem(TOKEN_KEY) || ""; }
export function setToken(t) { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); }

function url(path) {
  if (!path.startsWith("/")) path = "/" + path;
  return path;
}

async function request(method, path, { body, headers, signal } = {}) {
  const opts = {
    method,
    headers: { "Accept": "application/json", ...(headers || {}) },
    signal,
  };
  if (body !== undefined && body !== null) {
    if (body instanceof FormData) {
      opts.body = body;
    } else {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
  }
  const tok = getToken();
  if (tok) opts.headers["Authorization"] = "Bearer " + tok;

  const r = await fetch(url(path), opts);
  if (r.status === 401) {
    document.dispatchEvent(new CustomEvent("sv:auth-required"));
    throw new Error("auth required");
  }
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail || ""; } catch {}
    throw new Error(`HTTP ${r.status}${detail ? ": " + detail : ""}`);
  }
  if (r.status === 204) return null;
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

export const api = {
  health:        () => request("GET", "/api/health"),
  schema:        () => request("GET", "/api/schema"),

  listSessions:  () => request("GET", "/api/sessions"),
  sessionHistory: (id) => request("GET", `/api/sessions/${encodeURIComponent(id)}/history`),
  postEvent:     (id, rec) => request("POST", `/api/sessions/${encodeURIComponent(id)}/events`, { body: rec }),

  listSpeakers:  () => request("GET", "/api/registry/speakers"),
  getSpeaker:    (id) => request("GET", `/api/registry/speakers/${encodeURIComponent(id)}`),
  renameSpeaker: (id, name) => request("PATCH", `/api/registry/speakers/${encodeURIComponent(id)}`, { body: { display_name: name } }),
  deleteSpeaker: (id) => request("DELETE", `/api/registry/speakers/${encodeURIComponent(id)}`),
  bulkDeleteSpeakers: (scope = "all") => request("DELETE", `/api/registry/speakers?scope=${encodeURIComponent(scope)}`),

  listJobs:      () => request("GET", "/api/jobs"),
  getJob:        (id) => request("GET", `/api/jobs/${encodeURIComponent(id)}`),
  submitJob:     (body) => request("POST", "/api/jobs/transcribe", { body }),
  uploadJob:     (form) => request("POST", "/api/jobs/transcribe-upload", { body: form }),
  deleteJob:     (id) => request("DELETE", `/api/jobs/${encodeURIComponent(id)}`),
  deleteSession: (id) => request("DELETE", `/api/sessions/${encodeURIComponent(id)}`),

  maintenanceUsage:   () => request("GET", "/api/maintenance/usage"),
  maintenanceCleanup: (body) => request("POST", "/api/maintenance/cleanup", { body }),

  streamFile:    (id, form) => request("POST", `/api/sessions/${encodeURIComponent(id)}/stream-file`, { body: form }),
  streamStop:    (id) => request("POST", `/api/sessions/${encodeURIComponent(id)}/stream-stop`),
  streamStatus:  (id) => request("GET", `/api/sessions/${encodeURIComponent(id)}/stream-status`),
};

// Build a WebSocket URL for /ws/events with the right base + token. Falls
// back to ?token= since browsers can't add Authorization headers to WS.
export function wsUrlForEvents(session = "default") {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams();
  params.set("session", session);
  const tok = getToken();
  if (tok) params.set("token", tok);
  return `${scheme}//${location.host}/ws/events?${params.toString()}`;
}

// Same shape but for /ws/ingest — pushes 16k mono int16 PCM from the
// browser microphone (resampled by an AudioWorklet) to the api.
export function wsUrlForIngest(session = "default", language = "en-US") {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams();
  params.set("session", session);
  params.set("language", language);
  const tok = getToken();
  if (tok) params.set("token", tok);
  return `${scheme}//${location.host}/ws/ingest?${params.toString()}`;
}
