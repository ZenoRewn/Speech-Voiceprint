// Jobs page: submit fast/batch transcription, watch them run, view rich result.

import { api } from "../api.js";
import {
  attachProximityGlow,
  el,
  escapeHtml,
  fmtDateTime,
  fmtFloat,
  fmtTime,
  pageHint,
  speakerColor,
  toast,
  withHint,
} from "../util.js";

export async function renderJobs(root) {
  root.append(pageHint("jobs", "Jobs — 离线音频转写", [
    el("p", {},
      el("strong", {}, "Fast"), " 走 Azure Fast Transcription REST(同步,≤ 5 分钟 / 500 MB);",
      el("strong", {}, "Batch"), " 走 v3.2 异步,适合长音频,",
      el("strong", {}, "必须"), " 给 SAS URL(本地路径不行)。两种模式都会自动跑声纹分离 + 可选 registry 匹配。",
    ),
    el("p", {},
      "三个音频来源任选其一:① 浏览器上传(最常用)② ", el("code", {}, "audio_path"),
      " 是 ", el("strong", {}, "api 进程能直接读到的本地路径"),
      ",③ ", el("code", {}, "audio_url"), " 是 batch 模式专用的 SAS URL。",
    ),
    el("p", {},
      "字段右边的 ", el("span", { class: "field-help", style: { display: "inline-grid" } }, "?"),
      " 鼠标悬停可看每个字段的详细说明。",
    ),
  ]));

  const submitCard = buildSubmitCard(root, () => listing.refresh());
  const listing = buildListing(root);

  const layout = el("div", { class: "jobs-layout" }, submitCard, listing.node);
  root.append(layout);

  await listing.refresh();
  const timer = setInterval(() => listing.refresh(), 3000);
  return () => clearInterval(timer);
}

const HINTS = {
  mode:        "fast = 同步 REST,≤5min/500MB,本地或上传文件均可。\nbatch = 异步 v3.2,长音频必选,只接 SAS URL。",
  backend:     "speechbrain = ECAPA 192 维,~30× 快(M1 实测),无需 HF token。\npyannote = 512 维 + community-1 diarization,精度更高,需 HF_TOKEN + 接受模型 license。",
  upload:      "浏览器把整个文件 POST 到 /api/jobs/transcribe-upload,服务端落到 SV_UPLOAD_DIR(默认 /tmp/sv-uploads),再走与本地路径一样的流程。",
  audioPath:   "已经存在于 api 服务进程能读到的路径(容器里就是挂载到容器内的位置)。比如 /data/sample.wav。",
  audioUrl:    "Azure Blob 的 SAS URL —— 必须 https,且 Azure 服务端能 GET 到。仅 batch 模式有意义。",
  languages:   "Azure 自动语言识别候选,逗号分隔。例:en-US,zh-CN。空着用 server 默认。",
  registryPath: "可选 SQLite 路径。给了之后,作业结束的 speaker label 会去匹配/写入这个 registry,跨会话保持身份。",
  enroll:      "勾上 = 匹配 + 把新 speaker 写入 registry(跨会话累积身份)。\n不勾 = 只匹配已知 speaker,新声音停在 session-local 的 Speaker_A/B,不污染 registry。\n做实验/单次转写不想留痕的场景关掉。仅当上面 Registry path 有值时才生效。",
  dual:        "跑完主 backend 后,自动用另一个 backend 在同一段音频再嵌一次,把两套向量都写到同一个 speaker_id 下。\n好处:以后任意 backend 都能识别同一个人。\n代价:多一次 diarize(主 backend speechbrain 时,要 ~30s 起 pyannote;反之要 HF_TOKEN 已配)。Compare 模式下自动跳过(已经跑两次了)。",
};

function buildSubmitCard(root, onSubmitted) {
  const upload = el("input", { type: "file", accept: "audio/*" });
  const audioPath = el("input", { type: "text", placeholder: "/data/sample.wav (server-side path)" });
  const audioUrl = el("input", { type: "text", placeholder: "https://...sample.wav?<SAS>" });
  const mode = el("select", {},
    el("option", { value: "fast", selected: "selected" }, "fast"),
    el("option", { value: "batch" }, "batch"),
  );
  const backend = el("select", {},
    el("option", { value: "speechbrain", selected: "selected" }, "speechbrain"),
    el("option", { value: "pyannote" }, "pyannote"),
  );
  const languages = el("input", { type: "text", placeholder: "en-US,zh-CN" });
  const registryPath = el("input", { type: "text", placeholder: "(optional) /registry/speakers.db" });

  const compareBoth = el("input", { type: "checkbox", id: "compareBoth" });
  const enrollIntoRegistry = el("input", { type: "checkbox", id: "enrollIntoRegistry", checked: "checked" });
  const dualEnroll = el("input", { type: "checkbox", id: "dualEnroll" });

  const submit = el("button", { class: "btn-primary" }, "Submit");
  submit.addEventListener("click", async () => {
    submit.disabled = true;
    const file = upload.files[0] || null;
    const dual = compareBoth.checked;
    const backends = dual ? ["speechbrain", "pyannote"] : [backend.value];
    const cmpId = dual ? randomCompareId() : null;
    try {
      const submitted = [];
      for (const be of backends) {
        let res;
        if (file) {
          const fd = new FormData();
          fd.append("upload", file);
          fd.append("mode", mode.value);
          fd.append("backend", be);
          if (languages.value) fd.append("languages", languages.value);
          if (registryPath.value) fd.append("registry_path", registryPath.value);
          fd.append("auto_enroll_unknown", enrollIntoRegistry.checked ? "true" : "false");
          // Dual-enroll redundant in compare mode (we already run both backends).
          if (dualEnroll.checked && !dual) fd.append("dual_enroll", "true");
          if (cmpId) fd.append("comparison_id", cmpId);
          res = await api.uploadJob(fd);
        } else {
          res = await api.submitJob({
            mode: mode.value,
            backend: be,
            audio_path: audioPath.value || null,
            audio_url: audioUrl.value || null,
            languages: languages.value ? languages.value.split(",").map(s => s.trim()).filter(Boolean) : [],
            registry_path: registryPath.value || null,
            auto_enroll_unknown: enrollIntoRegistry.checked,
            dual_enroll: dualEnroll.checked && !dual,
            comparison_id: cmpId,
          });
        }
        submitted.push(res.job_id);
      }
      toast(root, "ok",
        dual
          ? `Comparison submitted: ${submitted.join(" + ")}`
          : `Job ${submitted[0]} submitted`,
      );
      upload.value = "";
      onSubmitted();
    } catch (e) {
      toast(root, "err", e.message);
    } finally {
      submit.disabled = false;
    }
  });

  function randomCompareId() {
    return "cmp_" + Math.random().toString(36).slice(2, 10);
  }

  const card = el("section", { class: "card glow jobs-form" },
    el("div", { class: "card-head" },
      el("span", { class: "card-title" }, "Submit job"),
      el("span", { class: "card-sub" }, "fast / batch via Azure Speech"),
    ),
    el("div", { class: "jobs-form-grid" },
      el("label", { class: "field" }, withHint(el("span", {}, "Mode"), HINTS.mode), mode),
      el("label", { class: "field" }, withHint(el("span", {}, "Backend"), HINTS.backend), backend),
      el("label", { class: "field jobs-form-wide" },
        withHint(el("span", {}, "Upload audio"), HINTS.upload),
        upload,
      ),
      el("label", { class: "field jobs-form-wide" },
        withHint(el("span", {}, "or audio_path"), HINTS.audioPath),
        audioPath,
      ),
      el("label", { class: "field jobs-form-wide" },
        withHint(el("span", {}, "or audio_url (SAS, batch only)"), HINTS.audioUrl),
        audioUrl,
      ),
      el("label", { class: "field" },
        withHint(el("span", {}, "Languages"), HINTS.languages),
        languages,
      ),
      el("label", { class: "field" },
        withHint(el("span", {}, "Registry path"), HINTS.registryPath),
        registryPath,
      ),
      el("label", { class: "field jobs-form-wide jobs-compare-toggle" },
        enrollIntoRegistry,
        withHint(
          el("span", {}, " Enroll new speakers into registry"),
          HINTS.enroll,
        ),
        el("span", { class: "field-sub" },
          "默认勾选。关掉后只做声纹匹配 ", el("strong", {}, "不"),
          " 把没识别出的新声音入库 —— 适合一次性实验。",
        ),
      ),
      el("label", { class: "field jobs-form-wide jobs-compare-toggle" },
        dualEnroll,
        withHint(
          el("span", {}, " Also enroll the ", el("strong", {}, "other"), " backend (cross-backend matching)"),
          HINTS.dual,
        ),
        el("span", { class: "field-sub" },
          "勾上后,主 backend 跑完会再用另一个 backend 在同一段音频做一次 diarize,把它的向量也挂到同一个 speaker_id 下。",
          " 以后任意 backend 都能识别同一个人。Compare 模式自动跳过。",
        ),
      ),
      el("label", { class: "field jobs-form-wide jobs-compare-toggle" },
        compareBoth,
        withHint(
          el("span", {}, " Run on ", el("strong", {}, "both"), " backends and compare side-by-side"),
          "勾上后会用同一份音频提交两次(speechbrain + pyannote),完成后在列表里点 Compare 看对比。两份输出共享 STT(文本一致),只比较声纹分离结果。",
        ),
        el("span", { class: "field-sub" },
          "→ 同一份音频会跑 ", el("strong", {}, "两个独立 Job"), "(",
          el("code", {}, "speechbrain"), " + ", el("code", {}, "pyannote"),
          ",共用同一 ", el("code", {}, "comparison_id"),
          "),完成后下面表格出现 “Compare” 按钮。Azure 计费两次。",
        ),
      ),
    ),
    el("div", { class: "jobs-form-actions" }, submit),
  );
  attachProximityGlow(card);
  return card;
}

function buildListing(root) {
  const tbody = el("tbody");
  const card = el("section", { class: "card glow jobs-listing" },
    el("div", { class: "card-head" },
      el("span", { class: "card-title" }, "Jobs"),
      el("span", { class: "card-sub" }, "most recent first"),
    ),
    el("table", { class: "table" },
      el("thead", {},
        el("tr", {},
          el("th", {}, "ID"),
          el("th", {}, "Mode"),
          el("th", {}, "Status"),
          el("th", {}, "Submitted"),
          el("th", {}, "Result"),
        ),
      ),
      tbody,
    ),
  );
  attachProximityGlow(card);

  async function refresh() {
    let jobs;
    try { jobs = await api.listJobs(); }
    catch (e) {
      tbody.innerHTML = `<tr><td colspan="5"><div class="toast err">${escapeHtml(e.message)}</div></td></tr>`;
      return;
    }
    if (!jobs.length) {
      tbody.innerHTML = `<tr><td colspan="5"><div class="empty">尚未提交任何作业。</div></td></tr>`;
      return;
    }
    // Group jobs by comparison_id; ungrouped jobs stay alone. Render in
    // reverse-chronological order (api already returns recent-first).
    const groups = collectComparisons(jobs);
    tbody.innerHTML = "";
    for (const g of groups) {
      for (let i = 0; i < g.jobs.length; i++) {
        const j = g.jobs[i];
        const isFirst = i === 0;
        const isLast = i === g.jobs.length - 1;
        const klass = g.cmpId ? `cmp-row${isFirst ? " cmp-first" : ""}${isLast ? " cmp-last" : ""}` : "";
        tbody.append(
          el("tr", { class: klass, "data-cmp": g.cmpId || "" },
            el("td", { class: "mono" }, j.job_id,
              g.cmpId && isFirst
                ? el("span", { class: "cmp-tag", title: `comparison ${g.cmpId}` }, "compare ×" + g.jobs.length)
                : null,
            ),
            el("td", {},
              j.mode,
              el("span", { class: "cmp-backend" }, " · " + (j.request?.backend || "?")),
            ),
            el("td", {}, statusBadge(j.status)),
            el("td", {}, fmtDateTime(j.submitted_at)),
            el("td", {}, jobActions(root, j, g)),
          ),
        );
      }
    }
  }

  return { node: card, refresh };
}

function statusBadge(status) {
  const klass = ({
    queued:  "muted",
    running: "warn",
    done:    "ok",
    failed:  "err",
  })[status] || "muted";
  return el("span", { class: `badge ${klass}` }, status);
}

function jobActions(root, job, group) {
  const buttons = [];
  if (job.status === "done" && job.result) {
    buttons.push(el("button", {
      class: "btn-ghost",
      onclick: () => showResultDrawer(root, job),
    }, "View"));
    buttons.push(el("button", {
      class: "btn-ghost",
      title: "Download result JSON",
      onclick: () => downloadJSON(job.result, `job_${job.job_id}.json`),
    }, "↓"));
  } else if (job.status === "failed") {
    buttons.push(el("span", { class: "live-conf", title: job.error || "" }, (job.error || "").slice(0, 60)));
  } else {
    buttons.push(el("span", { class: "spinner" }));
  }
  // Show "Compare" only on the first row of a comparison group, and only
  // when every member has finished. Both jobs share STT so the diff is
  // meaningful only after both end.
  if (group?.cmpId && group.jobs[0]?.job_id === job.job_id) {
    const allDone = group.jobs.every(j => j.status === "done" && j.result);
    if (allDone) {
      buttons.push(el("button", {
        class: "btn-primary cmp-btn",
        onclick: () => showComparisonDrawer(root, group),
      }, "Compare"));
    } else {
      buttons.push(el("span", { class: "badge muted" },
        `compare in ${group.jobs.filter(j => j.status !== "done").length}…`));
    }
  }
  return el("div", { class: "row-actions" }, ...buttons);
}

// Group jobs by comparison_id while preserving the api's recent-first order.
// Each group renders contiguously: the comparison's most-recent job dictates
// position, but inside the group jobs are ordered oldest-first so backends
// read left-to-right in submission order.
function collectComparisons(jobs) {
  const seen = new Set();
  const groups = [];
  for (const j of jobs) {
    const cmp = j.comparison_id || null;
    if (!cmp) {
      groups.push({ cmpId: null, jobs: [j] });
      continue;
    }
    if (seen.has(cmp)) continue;
    seen.add(cmp);
    const members = jobs.filter(x => x.comparison_id === cmp);
    members.sort((a, b) => a.submitted_at - b.submitted_at);
    groups.push({ cmpId: cmp, jobs: members });
  }
  return groups;
}

// ---- result drawer ----

function showResultDrawer(root, job) {
  const existing = document.getElementById("jobDrawer");
  if (existing) existing.remove();

  const result = job.result || {};
  const utterances = result.utterances || [];
  const speakers = collectSpeakers(utterances);

  const summary = el("div", { class: "result-summary" },
    metric("Speakers", String(speakers.size)),
    metric("Utterances", String(utterances.length)),
    metric("Duration", result.duration != null ? fmtTime(result.duration) : "—"),
    metric("Language", result.language || "—"),
    metric("Backend", result.voiceprint_backend || "—"),
  );

  const speakerLegend = el("div", { class: "result-legend" },
    ...[...speakers.entries()].map(([label, count]) => {
      const c = speakerColor(label);
      return el("span", { class: "result-legend-item", style: { borderColor: c, color: c } },
        el("span", { class: "result-legend-dot", style: { background: c } }),
        label,
        el("small", { class: "result-legend-count" }, ` × ${count}`),
      );
    }),
  );

  const transcript = renderTranscript(utterances);

  const downloadBtn = el("button", { class: "btn-primary" }, "↓ Download JSON");
  downloadBtn.addEventListener("click", () => downloadJSON(result, `${job.job_id}.json`));

  const copyBtn = el("button", { class: "btn-ghost" }, "Copy");
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      copyBtn.textContent = "Copied ✓";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
    } catch {
      copyBtn.textContent = "Copy failed";
    }
  });

  const rawJsonToggle = el("summary", { class: "result-raw-toggle" }, "Raw JSON");
  const rawPre = el("pre", { class: "json-dump" }, JSON.stringify(result, null, 2));
  const rawDetails = el("details", { class: "result-raw" }, rawJsonToggle, rawPre);

  const drawer = el("div", { class: "drawer", id: "jobDrawer" },
    el("div", { class: "drawer-card result-drawer-card" },
      el("div", { class: "drawer-head" },
        el("h3", {}, `Job ${job.job_id}`),
        el("div", { class: "result-actions" }, copyBtn, downloadBtn,
          el("button", { class: "icon-btn", onclick: () => drawer.remove() }, "✕"),
        ),
      ),
      summary,
      speakers.size > 0 ? speakerLegend : null,
      transcript,
      rawDetails,
    ),
  );
  drawer.addEventListener("click", (e) => { if (e.target === drawer) drawer.remove(); });
  document.body.append(drawer);
}

function collectSpeakers(utterances) {
  // Map<label, count>, preserving first-seen order.
  const m = new Map();
  for (const u of utterances) {
    const k = u.speaker || "—";
    m.set(k, (m.get(k) || 0) + 1);
  }
  return m;
}

function renderTranscript(utterances) {
  if (!utterances.length) {
    return el("div", { class: "empty" }, "(no utterances)");
  }
  const list = el("div", { class: "result-transcript" });
  let prevSpeaker = null;
  for (const u of utterances) {
    const speaker = u.speaker || "—";
    const c = speakerColor(speaker);
    const sameAsPrev = speaker === prevSpeaker;
    list.append(
      el("div", {
        class: "result-utterance" + (sameAsPrev ? " is-continuation" : ""),
        style: { borderLeftColor: c },
      },
        el("div", { class: "result-utterance-head" },
          sameAsPrev
            ? el("span", { class: "result-speaker-spacer" })
            : el("span", { class: "result-speaker-chip", style: { color: c, borderColor: c } }, speaker),
          el("span", { class: "result-time" }, `${fmtTime(u.start)} – ${fmtTime(u.end)}`),
          u.speaker_confidence != null
            ? el("span", { class: "result-conf" }, fmtFloat(u.speaker_confidence, 2))
            : null,
        ),
        el("div", { class: "result-text" }, u.text || ""),
      ),
    );
    prevSpeaker = speaker;
  }
  return list;
}

function metric(label, value) {
  return el("div", { class: "metric" },
    el("span", { class: "metric-label" }, label),
    el("span", { class: "metric-value" }, value),
  );
}

function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: filename, style: { display: "none" } });
  document.body.append(a);
  a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 0);
}

// ---- comparison drawer ----

function showComparisonDrawer(root, group) {
  const existing = document.getElementById("jobDrawer");
  if (existing) existing.remove();

  const [a, b] = group.jobs; // oldest first; if more than 2 we still take first two
  const ar = a.result || {};
  const br = b.result || {};
  const aBackend = a.request?.backend || "?";
  const bBackend = b.request?.backend || "?";

  const aUtter = ar.utterances || [];
  const bUtter = br.utterances || [];

  const aSpeakers = collectSpeakers(aUtter);
  const bSpeakers = collectSpeakers(bUtter);

  const wallA = (a.finished_at && a.started_at) ? (a.finished_at - a.started_at) : null;
  const wallB = (b.finished_at && b.started_at) ? (b.finished_at - b.started_at) : null;
  // ⭐ on wallclock winner only when the gap is meaningful (>5%) and both
  // sides finished. A near-tie shouldn't crown a winner — that's noise.
  let wallWinner = null;
  if (wallA != null && wallB != null) {
    const gap = Math.abs(wallA - wallB) / Math.max(wallA, wallB);
    if (gap >= 0.05) wallWinner = wallA < wallB ? "a" : "b";
  }

  // Speaker-label agreement: zip by index. Both jobs share STT so utterances
  // line up 1:1 when text matches; misaligned texts count as a mismatch.
  // We canonicalize per-side speaker labels to a stable index (first occurrence
  // → 0, second → 1, …) so "Speaker_A" on side A and "Speaker_B" on side B
  // both map to 0 when they appear first; agreement is then index equality.
  const agreement = computeAgreement(aUtter, bUtter);

  const summary = el("div", { class: "cmp-summary" },
    el("div", { class: "cmp-col cmp-col-a" },
      el("span", { class: "cmp-col-title" }, aBackend),
      cmpMetric("Utterances", String(aUtter.length)),
      cmpMetric("Speakers", String(aSpeakers.size)),
      cmpMetric("Wallclock", wallA != null ? wallA.toFixed(1) + "s" : "—",
                "job 真实墙钟时间(finished − started)。⭐ = 显著更快(差距 ≥5%)。",
                wallWinner === "a"),
    ),
    el("div", { class: "cmp-col cmp-col-mid" },
      el("span", { class: "cmp-col-title" }, "agreement"),
      cmpMetric("Speaker label", `${(agreement.rate * 100).toFixed(1)}%`,
                "两 backend 在每条 utterance 上分到“相同 speaker 序号”的比例"),
      cmpMetric("Aligned", `${agreement.aligned}/${agreement.total}`,
                "STT 文本能逐句对齐的句数 / 总句数"),
      cmpMetric("Audio", ar.duration != null ? fmtTime(ar.duration) : "—",
                "音频时长(供算 wallclock 对比基准)"),
    ),
    el("div", { class: "cmp-col cmp-col-b" },
      el("span", { class: "cmp-col-title" }, bBackend),
      cmpMetric("Utterances", String(bUtter.length)),
      cmpMetric("Speakers", String(bSpeakers.size)),
      cmpMetric("Wallclock", wallB != null ? wallB.toFixed(1) + "s" : "—",
                "job 真实墙钟时间(finished − started)。⭐ = 显著更快(差距 ≥5%)。",
                wallWinner === "b"),
    ),
  );

  const transcript = renderComparisonTranscript(aUtter, bUtter, aBackend, bBackend);

  const downloadBtn = el("button", { class: "btn-primary" }, "↓ Download both JSON");
  downloadBtn.addEventListener("click", () => {
    downloadJSON({ comparison_id: group.cmpId, [aBackend]: ar, [bBackend]: br }, `compare_${group.cmpId}.json`);
  });

  const drawer = el("div", { class: "drawer", id: "jobDrawer" },
    el("div", { class: "drawer-card cmp-drawer-card" },
      el("div", { class: "drawer-head" },
        el("h3", {}, `Compare · ${group.cmpId}`),
        el("div", { class: "result-actions" }, downloadBtn,
          el("button", { class: "icon-btn", onclick: () => drawer.remove() }, "✕"),
        ),
      ),
      summary,
      transcript,
    ),
  );
  drawer.addEventListener("click", (e) => { if (e.target === drawer) drawer.remove(); });
  document.body.append(drawer);
}

function cmpMetric(label, value, hint, isWinner = false) {
  const labelNode = hint
    ? withHint(el("span", { class: "metric-label" }, label), hint)
    : el("span", { class: "metric-label" }, label);
  const valueNode = isWinner
    ? el("span", { class: "metric-value cmp-winner", title: "Significantly faster" },
        el("span", { class: "cmp-star" }, "⭐"), " ", value)
    : el("span", { class: "metric-value" }, value);
  return el("div", { class: "cmp-metric" + (isWinner ? " cmp-metric-winner" : "") },
    labelNode,
    valueNode,
  );
}

function canonicalSpeakerIndex(utterances) {
  // Map each utterance's speaker label to a small int representing its
  // appearance order on its own side. Different backends may name speakers
  // differently (Speaker_A vs Speaker_B), but if both sides put the same
  // index on the same utterance position, that's effective agreement.
  const order = new Map();
  return utterances.map(u => {
    const k = u.speaker || "";
    if (!order.has(k)) order.set(k, order.size);
    return order.get(k);
  });
}

function computeAgreement(aUtter, bUtter) {
  const aIdx = canonicalSpeakerIndex(aUtter);
  const bIdx = canonicalSpeakerIndex(bUtter);
  const total = Math.max(aUtter.length, bUtter.length);
  const aligned = Math.min(aUtter.length, bUtter.length);
  let match = 0;
  for (let i = 0; i < aligned; i++) {
    if (aIdx[i] === bIdx[i]) match++;
  }
  return { match, aligned, total, rate: aligned ? match / aligned : 0 };
}

function renderComparisonTranscript(aUtter, bUtter, aBackend, bBackend) {
  const n = Math.max(aUtter.length, bUtter.length);
  if (!n) return el("div", { class: "empty" }, "(both backends produced no utterances)");
  const list = el("div", { class: "cmp-transcript" });
  list.append(el("div", { class: "cmp-transcript-head" },
    el("span", {}, aBackend),
    el("span", {}, "text"),
    el("span", {}, bBackend),
  ));
  for (let i = 0; i < n; i++) {
    const a = aUtter[i];
    const b = bUtter[i];
    const aSp = a?.speaker || "—";
    const bSp = b?.speaker || "—";
    const aColor = speakerColor(aSp);
    const bColor = speakerColor(bSp);
    const text = a?.text || b?.text || "—";
    const time = a ? `${fmtTime(a.start)}–${fmtTime(a.end)}` : "";
    const matches = aSp === bSp; // strict label match (visual signal only)
    list.append(
      el("div", { class: "cmp-row" + (matches ? "" : " cmp-row-disagree") },
        el("span", {
          class: "result-speaker-chip",
          style: { color: aColor, borderColor: aColor },
        }, aSp),
        el("div", { class: "cmp-row-text" },
          el("span", { class: "cmp-row-time" }, time),
          el("span", {}, text),
        ),
        el("span", {
          class: "result-speaker-chip",
          style: { color: bColor, borderColor: bColor },
        }, bSp),
      ),
    );
  }
  return list;
}
