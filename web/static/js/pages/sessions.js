// Sessions page: list active sessions; click into one to replay history.

import { api } from "../api.js";
import { attachProximityGlow, el, escapeHtml, fmtFloat, pageHint, speakerColor } from "../util.js";

export async function renderSessions(root) {
  root.append(pageHint("sessions", "Sessions — 全部活跃 / 历史会话", [
    el("p", {}, "每个 ", el("strong", {}, "session"), " 是一路实时流的逻辑标签 —— 由 streaming 进程的 ", el("code", {}, "--session"), " 参数标识。多路源用不同 session 名互不干扰。"),
    el("p", {}, "左边卡片列出当前所有 session(含已结束但仍有历史的);", el("strong", {}, "live"), " 标记表示正在被 Live 页订阅;点 ", el("strong", {}, "View history"), " 在右侧回放最近 ~500 条事件,点 ", el("strong", {}, "Open Live →"), " 跳到 Live 页订阅它。"),
  ]));

  const grid = el("div", { class: "sessions-grid" });
  const detail = el("section", { class: "card glow", id: "sessionDetail" });
  attachProximityGlow(detail);

  const layout = el("div", { class: "sessions-layout" }, grid, detail);
  root.append(layout);

  let pollTimer = null;
  let activeSession = null;

  async function refresh() {
    let list;
    try { list = await api.listSessions(); }
    catch (e) {
      grid.innerHTML = `<div class="toast err">${escapeHtml(e.message)}</div>`;
      return;
    }
    grid.innerHTML = "";
    if (!list.length) {
      grid.append(el("div", { class: "empty" },
        el("div", { class: "empty-icon" }, "▤"),
        el("div", {}, "暂无活跃 session — 启动 streaming 后,会出现在这里。"),
      ));
      return;
    }
    for (const s of list) {
      const card = el("article", { class: "card glow session-card", "data-session": s.session },
        el("div", { class: "session-head" },
          el("h3", { class: "session-name" }, s.session),
          s.clients > 0
            ? el("span", { class: "badge ok" }, el("span", { class: "blink-dot" }), `${s.clients} live`)
            : el("span", { class: "badge muted" }, "idle"),
        ),
        el("div", { class: "session-stats" },
          el("div", { class: "metric" },
            el("span", { class: "metric-label" }, "Events"),
            el("span", { class: "metric-value" }, String(s.history)),
          ),
          el("div", { class: "metric" },
            el("span", { class: "metric-label" }, "Subscribers"),
            el("span", { class: "metric-value" }, String(s.clients)),
          ),
        ),
        el("div", { class: "session-actions" },
          el("button", { class: "btn-ghost", onclick: () => loadDetail(s.session) }, "View history"),
          el("a", { class: "btn-primary", href: `#/live?session=${encodeURIComponent(s.session)}` }, "Open Live →"),
        ),
      );
      attachProximityGlow(card);
      grid.append(card);
    }
  }

  async function loadDetail(sessionId) {
    activeSession = sessionId;
    detail.innerHTML = "";
    detail.append(
      el("div", { class: "card-head" },
        el("span", { class: "card-title" }, `History · ${escapeHtml(sessionId)}`),
        el("span", { class: "spinner" }),
      ),
    );
    let history;
    try { history = await api.sessionHistory(sessionId); }
    catch (e) {
      detail.innerHTML = `<div class="toast err">${escapeHtml(e.message)}</div>`;
      return;
    }
    detail.innerHTML = "";
    detail.append(
      el("div", { class: "card-head" },
        el("span", { class: "card-title" }, `History · ${escapeHtml(sessionId)}`),
        el("span", { class: "card-sub" }, `${history.length} events`),
      ),
    );
    if (!history.length) {
      detail.append(el("div", { class: "empty" }, "(空)"));
      return;
    }
    const list = el("div", { class: "session-history" });
    for (const r of history) {
      list.append(
        el("div", { class: "session-row", "data-event": r.event || "final" },
          el("span", { class: "live-time" }, `${(r.start || 0).toFixed(2)}–${(r.end || 0).toFixed(2)}`),
          el("span", { class: "live-chip", style: { color: speakerColor(r.speaker), borderColor: speakerColor(r.speaker) } }, r.speaker || "—"),
          el("span", { class: "live-conf" }, `(${fmtFloat(r.speaker_confidence, 2)})`),
          el("span", { class: "live-text" }, r.text || ""),
        ),
      );
    }
    detail.append(list);
  }

  detail.append(el("div", { class: "empty" },
    el("div", { class: "empty-icon" }, "←"),
    el("div", {}, "选择左侧一个 session,在这里查看历史回放。"),
  ));

  await refresh();
  pollTimer = setInterval(refresh, 5000);

  return () => clearInterval(pollTimer);
}
