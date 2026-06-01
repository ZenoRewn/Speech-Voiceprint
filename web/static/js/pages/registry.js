// Registry page: speaker grid with rename/delete drawer.

import { api } from "../api.js";
import { attachProximityGlow, el, escapeHtml, fmtDateTime, pageHint, speakerColor, toast } from "../util.js";

export async function renderRegistry(root) {
  root.append(pageHint("registry", "Registry — 跨会话的持久声纹库", [
    el("p", {}, "每个条目是一个 ", el("strong", {}, "speaker identity"), ":display name + 一组 voiceprint(192/512 维 embedding,按模型分桶)。新音频的声纹与库内 centroid 余弦匹配,命中就贴现有名字,未命中就新建一个 (auto-enroll)。"),
    el("p", {}, "怎么往这里入库 → 跑作业时带 ", el("code", {}, "registry_path"), "(Jobs 表单里那一栏),或 streaming 用 ", el("code", {}, "--registry"), "。点卡片打开右侧抽屉可改名 / 删除;rename 后已有历史/Live 的 speaker 标签会自动同步。"),
    el("p", {}, "embeddings 按 ", el("code", {}, "model"), " 标签分(", el("code", {}, "speechbrain-192"), " / ", el("code", {}, "pyannote-512"), ")—— 跨模型的向量从来不互比。"),
  ]));

  const toolbar = el("div", { class: "registry-toolbar" },
    el("div", { class: "registry-toolbar-left" },
      el("span", { class: "registry-count", id: "registryCount" }, ""),
    ),
    el("div", { class: "registry-toolbar-right" },
      el("button", {
        class: "btn-ghost",
        onclick: () => bulkForget("unnamed"),
      }, "Forget unnamed"),
      el("button", {
        class: "btn-ghost danger-btn",
        onclick: () => bulkForget("all"),
      }, "Forget all"),
    ),
  );
  root.append(toolbar);

  const grid = el("div", { class: "registry-grid" });
  const drawer = el("aside", { class: "card glow registry-detail", id: "registryDetail" });
  attachProximityGlow(drawer);
  drawer.append(emptyDetail());

  const layout = el("div", { class: "registry-layout" }, grid, drawer);
  root.append(layout);

  async function bulkForget(scope) {
    const label = scope === "all" ? "ALL speakers" : "all unnamed (sp_*) speakers";
    if (!confirm(
      `Delete ${label}?\n\n这会清空 voiceprint 向量,且不可撤销。`,
    )) return;
    if (scope === "all" && !confirm("Really delete EVERY speaker?")) return;
    try {
      const r = await api.bulkDeleteSpeakers(scope);
      toast(root, "ok", `Deleted ${r.deleted} ${scope === "all" ? "speakers" : "unnamed speakers"}`);
      drawer.innerHTML = "";
      drawer.append(emptyDetail());
      await refresh();
    } catch (e) {
      toast(root, "err", e.message);
    }
  }

  function emptyDetail() {
    return el("div", { class: "empty" },
      el("div", { class: "empty-icon" }, "◐"),
      el("div", {}, "选择左侧一位 speaker 查看详情或改名。"),
    );
  }

  async function refresh() {
    let speakers;
    try { speakers = await api.listSpeakers(); }
    catch (e) {
      grid.innerHTML = "";
      grid.append(el("div", { class: "toast err" }, e.message));
      return;
    }
    grid.innerHTML = "";
    const named = speakers.filter(s => s.display_name).length;
    const counter = document.getElementById("registryCount");
    if (counter) {
      counter.textContent = speakers.length === 0
        ? ""
        : `${speakers.length} speaker${speakers.length === 1 ? "" : "s"} · ${named} named, ${speakers.length - named} unnamed`;
    }
    if (!speakers.length) {
      grid.append(el("div", { class: "empty" },
        el("div", { class: "empty-icon" }, "◐"),
        el("div", {}, "Registry 为空。先跑一轮 fast/realtime + --registry 让 speakers 入库。"),
      ));
      return;
    }
    for (const sp of speakers) {
      const card = el("article", { class: "card glow speaker-card", "data-id": sp.id, onclick: () => openDetail(sp.id) },
        el("div", { class: "speaker-card-head" },
          el("div", {
            class: "speaker-avatar",
            style: { background: speakerColor(sp.display_name || sp.id) },
          }, (sp.display_name || sp.id).slice(0, 2).toUpperCase()),
          el("div", { class: "speaker-card-text" },
            el("h3", { class: "speaker-name" }, sp.display_name || "(unnamed)"),
            el("span", { class: "speaker-id" }, sp.id),
          ),
        ),
        el("div", { class: "speaker-meta" },
          el("span", { class: "badge muted" }, `${sp.voiceprint_count} prints`),
          ...(sp.models || []).map(m => el("span", { class: "badge purple" }, m)),
        ),
        el("div", { class: "speaker-card-foot" },
          el("span", {}, "Updated "), el("strong", {}, fmtDateTime(sp.updated_at)),
        ),
      );
      attachProximityGlow(card);
      grid.append(card);
    }
  }

  async function openDetail(speakerId) {
    drawer.innerHTML = "";
    drawer.append(el("div", { class: "card-head" },
      el("span", { class: "card-title" }, "Loading…"),
      el("span", { class: "spinner" }),
    ));
    let sp;
    try { sp = await api.getSpeaker(speakerId); }
    catch (e) {
      drawer.innerHTML = `<div class="toast err">${escapeHtml(e.message)}</div>`;
      return;
    }
    drawer.innerHTML = "";
    drawer.append(
      el("div", { class: "card-head" },
        el("span", { class: "card-title" }, sp.display_name || sp.id),
        el("span", { class: "card-sub mono" }, sp.id),
      ),
      el("div", { class: "speaker-detail-stats" },
        el("div", { class: "metric" },
          el("span", { class: "metric-label" }, "Voiceprints"),
          el("span", { class: "metric-value" }, String(sp.voiceprint_count)),
        ),
        el("div", { class: "metric" },
          el("span", { class: "metric-label" }, "Models"),
          el("span", { class: "metric-value" }, sp.models.join(" ") || "—"),
        ),
      ),
      el("div", { class: "speaker-detail-section" },
        el("h4", {}, "Rename"),
        renameForm(sp),
      ),
      el("div", { class: "speaker-detail-section danger" },
        el("h4", {}, "Danger zone"),
        el("p", { class: "speaker-warn" }, "删除后,所有声纹向量也会一并清空。"),
        el("button", {
          class: "btn-ghost danger-btn",
          onclick: async () => {
            if (!confirm(`Delete speaker ${sp.display_name || sp.id}?`)) return;
            try {
              await api.deleteSpeaker(sp.id);
              toast(root, "ok", "Deleted");
              drawer.innerHTML = "";
              drawer.append(emptyDetail());
              await refresh();
            } catch (e) { toast(root, "err", e.message); }
          },
        }, "Delete speaker"),
      ),
    );
  }

  function renameForm(sp) {
    const input = el("input", { type: "text", value: sp.display_name || "", autocomplete: "off" });
    const submit = el("button", { class: "btn-primary" }, "Save");
    submit.addEventListener("click", async () => {
      const name = input.value.trim();
      if (!name) return toast(root, "warn", "name required");
      submit.disabled = true;
      try {
        await api.renameSpeaker(sp.id, name);
        toast(root, "ok", "Renamed");
        await refresh();
        await openDetail(sp.id);
      } catch (e) {
        toast(root, "err", e.message);
        submit.disabled = false;
      }
    });
    return el("div", { class: "field" },
      el("span", {}, "Display name"),
      input,
      el("div", { style: { display: "flex", justifyContent: "flex-end", marginTop: "var(--space-2)" } }, submit),
    );
  }

  await refresh();
}
