// Health page: single dashboard of api status + counts.

import { api } from "../api.js";
import { attachProximityGlow, el, escapeHtml, fmtDateTime } from "../util.js";

export async function renderHealth(root) {
  const metricsCard = el("section", { class: "card glow", id: "healthMetrics" });
  const sessionsCard = el("section", { class: "card glow", id: "healthSessions" });
  const jobsCard = el("section", { class: "card glow", id: "healthJobs" });
  attachProximityGlow(metricsCard);
  attachProximityGlow(sessionsCard);
  attachProximityGlow(jobsCard);

  const layout = el("div", { class: "health-layout" }, metricsCard, sessionsCard, jobsCard);
  root.append(layout);

  async function refresh() {
    let h, sessions, jobs;
    try {
      [h, sessions, jobs] = await Promise.all([api.health(), api.listSessions(), api.listJobs()]);
    } catch (e) {
      metricsCard.innerHTML = `<div class="toast err">${escapeHtml(e.message)}</div>`;
      return;
    }
    metricsCard.innerHTML = "";
    metricsCard.append(
      el("div", { class: "card-head" }, el("span", { class: "card-title" }, "API status")),
      el("div", { class: "health-metric-grid" },
        metric("Auth", h.auth_enabled ? "enabled" : "disabled", h.auth_enabled ? "ok" : "muted"),
        metric("Workers", String(h.workers)),
        metric("Sessions", String(h.sessions)),
        metric("Jobs", String(h.jobs)),
      ),
      el("div", { class: "health-row" },
        el("span", { class: "card-sub" }, `Registry: ${escapeHtml(h.registry_path || "(not configured)")}`),
      ),
    );

    sessionsCard.innerHTML = "";
    sessionsCard.append(
      el("div", { class: "card-head" },
        el("span", { class: "card-title" }, "Active sessions"),
        el("span", { class: "card-sub" }, `${sessions.length}`),
      ),
      sessions.length
        ? el("table", { class: "table" },
            el("thead", {}, el("tr", {},
              el("th", {}, "Session"),
              el("th", {}, "Subscribers"),
              el("th", {}, "History"),
            )),
            el("tbody", {}, ...sessions.map(s => el("tr", {},
              el("td", { class: "mono" }, s.session),
              el("td", {}, String(s.clients)),
              el("td", {}, String(s.history)),
            ))),
          )
        : el("div", { class: "empty" }, "No active sessions."),
    );

    jobsCard.innerHTML = "";
    const recent = (jobs || []).slice(0, 5);
    jobsCard.append(
      el("div", { class: "card-head" },
        el("span", { class: "card-title" }, "Recent jobs"),
        el("span", { class: "card-sub" }, `${jobs.length} total`),
      ),
      recent.length
        ? el("table", { class: "table" },
            el("thead", {}, el("tr", {},
              el("th", {}, "ID"),
              el("th", {}, "Mode"),
              el("th", {}, "Status"),
              el("th", {}, "Submitted"),
            )),
            el("tbody", {}, ...recent.map(j => el("tr", {},
              el("td", { class: "mono" }, j.job_id),
              el("td", {}, j.mode),
              el("td", {}, j.status),
              el("td", {}, fmtDateTime(j.submitted_at)),
            ))),
          )
        : el("div", { class: "empty" }, "No jobs yet."),
    );
  }

  function metric(label, value, kind = "muted") {
    return el("div", { class: "metric" },
      el("span", { class: "metric-label" }, label),
      el("span", { class: `metric-value` }, value),
    );
  }

  await refresh();
  const t = setInterval(refresh, 4000);
  return () => clearInterval(t);
}
