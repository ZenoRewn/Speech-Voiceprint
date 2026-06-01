// Maintenance page: data-dir usage + bulk cleanup with dry-run + confirm.

import { api } from "../api.js";
import { attachProximityGlow, el, pageHint, toast } from "../util.js";

const SCOPE_LABELS = {
  uploads:  { label: "Job uploads",   note: "data/uploads/ — multipart 上传暂存。" },
  stream:   { label: "Stream temp",   note: "data/stream/ — Live 页 stream-file 上传暂存。" },
  outputs:  { label: "Job outputs",   note: "data/outputs/ — 完成的 Job result JSON,Jobs 页 ↓ 下载源。" },
  jobs:     { label: "Job records",   note: "in-memory JobStore + 对应 outputs JSON。清掉等于忘记历史 Job。" },
  sessions: { label: "Session history", note: "SessionHub 历史回放缓冲。清掉只清回放,活跃订阅不断。" },
};

function fmtBytes(n) {
  if (n == null) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 ? 0 : 1)} ${u[i]}`;
}

export async function renderMaintenance(root) {
  root.append(pageHint("maintenance", "Maintenance — 清理临时文件 / 内存状态", [
    el("p", {},
      "数据根目录(", el("code", {}, "SV_DATA_DIR"), ",默认 ",
      el("code", {}, "<repo>/data/"),
      ")的占用情况。可针对每个 scope 一键清理 —— 清的都是 ",
      el("strong", {}, "可重新生成"), " 的临时数据,registry 永远不动。",
    ),
    el("p", {},
      el("strong", {}, "Dry run"), " 先看会删多少;确认无误再实际删。",
      "Older than 是按文件 mtime 过滤(仅文件型 scope 生效)。",
    ),
  ]));

  const card = el("section", { class: "card glow" },
    el("div", { class: "card-head" },
      el("span", { class: "card-title" }, "Disk + memory usage"),
      el("button", { class: "btn-ghost", id: "mntRefresh" }, "↻ Refresh"),
    ),
    el("div", { class: "mnt-rows", id: "mntRows" },
      el("div", { class: "empty" },
        el("span", { class: "spinner" }), " loading…",
      ),
    ),
  );
  attachProximityGlow(card);
  root.append(card);

  card.querySelector("#mntRefresh").addEventListener("click", refresh);

  async function refresh() {
    const rowsEl = card.querySelector("#mntRows");
    rowsEl.innerHTML = "";
    rowsEl.append(el("div", { class: "empty" }, el("span", { class: "spinner" }), " loading…"));
    let usage;
    try { usage = await api.maintenanceUsage(); }
    catch (e) {
      rowsEl.innerHTML = "";
      rowsEl.append(el("div", { class: "toast err" }, e.message));
      return;
    }
    rowsEl.innerHTML = "";
    rowsEl.append(el("div", { class: "mnt-root" },
      el("span", {}, "Data root: "),
      el("code", {}, usage.data_root || "—"),
    ));
    for (const scope of ["uploads", "stream", "outputs", "jobs", "sessions"]) {
      rowsEl.append(buildRow(scope, usage[scope] || {}, refresh));
    }
  }

  function buildRow(scope, info, onChange) {
    const meta = SCOPE_LABELS[scope];
    const olderInput = el("input", {
      type: "number", min: "0", placeholder: "all",
      class: "mnt-older",
      title: "Only apply to files older than this many days. Empty = all.",
    });
    const dryBtn = el("button", { class: "btn-ghost" }, "Dry run");
    const goBtn = el("button", { class: "btn-ghost danger-btn" }, "Clear");

    const isFs = ["uploads", "stream", "outputs"].includes(scope);
    if (!isFs) {
      olderInput.disabled = true;
      olderInput.placeholder = "n/a";
    }

    const sizeText = info.bytes != null
      ? `${info.count} files · ${fmtBytes(info.bytes)}`
      : `${info.count ?? 0} entries`;

    async function run(dry) {
      const days = olderInput.value ? parseInt(olderInput.value, 10) : null;
      if (!dry) {
        const what = days != null
          ? `${meta.label} older than ${days} day(s)`
          : `ALL ${meta.label}`;
        if (!confirm(`Clear ${what}?\n\n这是 ${scope} scope 的不可撤销清理。`)) return;
      }
      const btn = dry ? dryBtn : goBtn;
      btn.disabled = true;
      try {
        const r = await api.maintenanceCleanup({
          scope,
          older_than_days: days,
          dry_run: dry,
        });
        const msg = dry
          ? `[dry-run] would delete ${r.deleted} entries (${fmtBytes(r.bytes_freed)})`
          : `Deleted ${r.deleted} entries (${fmtBytes(r.bytes_freed)} freed)`;
        toast(root, dry ? "warn" : "ok", msg);
        if (!dry) await onChange();
      } catch (e) {
        toast(root, "err", e.message);
      } finally {
        btn.disabled = false;
      }
    }

    dryBtn.addEventListener("click", () => run(true));
    goBtn.addEventListener("click", () => run(false));

    return el("div", { class: "mnt-row" },
      el("div", { class: "mnt-row-text" },
        el("div", { class: "mnt-row-title" }, meta.label, " ",
          el("span", { class: "badge muted" }, scope),
        ),
        el("div", { class: "mnt-row-note" }, meta.note),
        el("div", { class: "mnt-row-size" }, sizeText),
      ),
      el("div", { class: "mnt-row-actions" },
        el("label", { class: "mnt-older-label" },
          "Older than ", olderInput, " days",
        ),
        dryBtn, goBtn,
      ),
    );
  }

  await refresh();
}
