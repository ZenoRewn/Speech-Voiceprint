// Live transcription page. Mirrors the legacy `web/index.html` behavior
// but rebuilt against the new design system + the api's `/ws/events`.
//
// Phase B additions: a "Start session" card lets the browser drive the
// realtime pipeline directly — either by uploading a 16k mono WAV (server
// pushes through the same Azure realtime path) or by capturing the
// microphone and streaming PCM over /ws/ingest. This is what makes the
// dashboard usable in AKS / pure-cloud deployments where there's no shell
// to start a `pipeline.streaming` process.

import { api, wsUrlForEvents, wsUrlForIngest } from "../api.js";
import { attachProximityGlow, el, escapeHtml, fmtFloat, pageHint, speakerColor, toast } from "../util.js";

export async function renderLive(root) {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const initialSession = params.get("session") || "default";

  root.append(pageHint("live", "Live transcript — 实时字幕滚屏", [
    el("p", {}, "页面通过 WebSocket 订阅服务端的某个 ", el("strong", {}, "session"), ",收到 STT + 声纹融合后的事件就追加到下面。"),
    el("p", {}, "推事件有 ", el("strong", {}, "三种方式"), ":(1) 浏览器上传 wav,在下面 ", el("strong", {}, "Start session"), " 卡片里选 “File”;(2) 浏览器麦克风,选 “Microphone”;(3) CLI 旁路:"),
    el("p", {}, el("code", {}, ".venv/bin/python -m pipeline.streaming --mic --session demo --api-target http://localhost:8080")),
    el("p", {}, "切换 session:右上角下拉选已有 session,或在右侧输入新名字点 Switch。点 speaker 标签可改名(命令通过 WS 发到服务端,会 rewrite 历史)。"),
  ]));

  const sessionPicker = el("select", { class: "field-select", id: "livePicker" });
  const customLabel = el("input", {
    type: "text", id: "liveCustom", placeholder: "session id…", class: "live-custom",
  });
  customLabel.value = initialSession;

  const transcript = el("div", { class: "live-transcript", id: "transcript" });
  const empty = el("div", { class: "empty" },
    el("div", { class: "empty-icon" }, "●"),
    el("div", {}, "尚无事件 — 用下方 Start session 卡片(File / Microphone)在浏览器里直接起一路,或用 CLI 旁路推事件。"),
  );

  const liveCard = el("section", { class: "card glow live-card" },
    el("div", { class: "card-head" },
      el("div", { class: "live-head-left" },
        el("span", { class: "card-title" }, "Live transcript"),
        el("span", { class: "badge purple" },
          el("span", { class: "blink-dot" }), "session ", el("strong", { id: "liveSessionLabel" }, initialSession),
        ),
      ),
      el("div", { class: "live-head-right" },
        el("label", { class: "live-pick" },
          el("span", {}, "session"),
          sessionPicker,
        ),
        customLabel,
        el("button", { class: "btn-ghost", id: "liveSwitch" }, "Switch"),
      ),
    ),
    transcript,
    empty,
  );
  attachProximityGlow(liveCard);
  root.append(liveCard);

  // Start session card — drives the streaming pipeline from the browser.
  const ingestCard = renderIngestCard(root, initialSession);
  root.append(ingestCard);

  // ---- Wire session picker ----
  await refreshSessions();
  document.getElementById("liveSwitch").addEventListener("click", () => {
    const next = customLabel.value.trim() || "default";
    location.hash = `#/live?session=${encodeURIComponent(next)}`;
  });
  sessionPicker.addEventListener("change", () => {
    if (sessionPicker.value === "__custom__") {
      customLabel.focus();
      return;
    }
    location.hash = `#/live?session=${encodeURIComponent(sessionPicker.value)}`;
  });

  async function refreshSessions() {
    try {
      const list = await api.listSessions();
      sessionPicker.innerHTML = "";
      const ids = list.map(s => s.session);
      if (!ids.includes(initialSession)) ids.unshift(initialSession);
      for (const id of ids) {
        const opt = el("option", { value: id }, id);
        if (id === initialSession) opt.selected = true;
        sessionPicker.append(opt);
      }
      sessionPicker.append(el("option", { value: "__custom__" }, "(custom…)"));
    } catch {
      sessionPicker.innerHTML = `<option>${escapeHtml(initialSession)}</option>`;
    }
  }

  // ---- WebSocket subscription ----
  let ws;
  let reconnectTimer = null;
  let teardown = false;
  function connect() {
    ws = new WebSocket(wsUrlForEvents(initialSession));
    ws.addEventListener("message", (e) => {
      try { ingest(JSON.parse(e.data)); } catch (err) { console.warn("bad event", err); }
    });
    ws.addEventListener("close", () => {
      if (teardown) return;
      reconnectTimer = setTimeout(connect, 2000);
    });
    ws.addEventListener("error", () => ws.close());
  }
  connect();

  // ---- Render utterances ----
  // Each `final`/`tentative` shares an id; `revised` updates the same DOM node.
  const seenById = new Map();

  function ingest(rec) {
    if (!rec || !rec.text) {
      // stream_end markers have no text; surface them in the toast bar.
      if (rec && rec.event === "stream_end") {
        toast(root, rec.ok ? "ok" : "err",
              rec.ok ? "Stream finished" : `Stream failed: ${rec.error || "unknown"}`);
      }
      return;
    }
    if (rec.session && rec.session !== initialSession) return;
    empty.remove();

    const id = `${rec.start.toFixed(3)}-${rec.end.toFixed(3)}-${rec.azure_speaker || ""}`;
    let row = seenById.get(id);
    if (!row) {
      row = el("div", { class: "live-row", "data-event": rec.event || "final" });
      transcript.append(row);
      seenById.set(id, row);
      requestAnimationFrame(() => row.scrollIntoView({ block: "end", behavior: "smooth" }));
    }
    row.dataset.event = rec.event || "final";
    row.innerHTML = "";

    const start = rec.start.toFixed(2).padStart(6, " ");
    const end = rec.end.toFixed(2).padStart(6, " ");
    const conf = fmtFloat(rec.speaker_confidence, 2);

    row.append(
      el("span", { class: "live-time" }, `${start}–${end}`),
      el("span", {
        class: "live-chip",
        style: { color: speakerColor(rec.speaker), borderColor: speakerColor(rec.speaker) },
        title: "click to rename",
        onclick: () => promptRename(rec.speaker),
      }, rec.speaker || "—"),
      el("span", { class: "live-conf" }, `(${conf})`),
      el("span", { class: "live-text" }, rec.text),
      el("span", { class: `live-event live-event-${rec.event || "final"}` }, rec.event || "final"),
    );
  }

  function promptRename(label) {
    if (!label) return;
    const next = prompt(`重命名 ${label} 为:`);
    if (!next) return;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "rename", old: label, new: next.trim() }));
    }
  }

  return () => {
    teardown = true;
    if (ws) try { ws.close(); } catch {}
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (ingestCard._teardown) ingestCard._teardown();
  };
}


// ---------------------------------------------------------------------------
// Start-session card: tabs for File and Microphone ingest. Both target the
// same session as the page (so the live transcript above streams as it
// runs). Mic capture pulls from `getUserMedia` and pipes through an
// AudioWorklet that resamples to 16k int16 → /ws/ingest.

function renderIngestCard(root, session) {
  const fileTab = el("button", { class: "ingest-tab active", "data-tab": "file" }, "Upload file");
  const micTab = el("button", { class: "ingest-tab", "data-tab": "mic" }, "Microphone");

  const fileBody = renderFileTab(root, session);
  const micBody = renderMicTab(root, session);
  micBody.style.display = "none";

  const card = el("section", { class: "card glow ingest-card" },
    el("div", { class: "card-head" },
      el("span", { class: "card-title" }, "Start session"),
      el("span", { class: "card-sub" }, `→ ${session}`),
    ),
    el("p", { class: "ingest-hint" },
      "在浏览器里直接起一路 streaming(Realtime STT + 声纹)。事件会推到本页 session,",
      el("strong", {}, "无需"), " 在服务器上单独开 ",
      el("code", {}, "pipeline.streaming"),
      "。",
    ),
    el("div", { class: "ingest-tabs" }, fileTab, micTab),
    el("div", { class: "ingest-body" }, fileBody, micBody),
  );
  attachProximityGlow(card);

  fileTab.addEventListener("click", () => {
    fileTab.classList.add("active");
    micTab.classList.remove("active");
    fileBody.style.display = "block";
    micBody.style.display = "none";
  });
  micTab.addEventListener("click", () => {
    micTab.classList.add("active");
    fileTab.classList.remove("active");
    fileBody.style.display = "none";
    micBody.style.display = "block";
  });

  card._teardown = () => {
    if (micBody._teardown) micBody._teardown();
  };
  return card;
}


function renderFileTab(root, session) {
  const fileInput = el("input", { type: "file", accept: ".wav,audio/wav" });
  const langInput = el("input", { type: "text", value: "en-US", placeholder: "en-US" });
  const pacing = el("input", { type: "checkbox", checked: true });
  const submit = el("button", { class: "btn-primary" }, "Start streaming");
  const status = el("div", { class: "ingest-status" });

  submit.addEventListener("click", async () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) {
      toast(root, "warn", "请选择一个 16k mono WAV 文件");
      return;
    }
    submit.disabled = true;
    status.textContent = "Uploading…";
    const form = new FormData();
    form.append("upload", f, f.name);
    form.append("language", langInput.value || "en-US");
    form.append("realtime_pacing", pacing.checked ? "true" : "false");
    try {
      const reply = await api.streamFile(session, form);
      status.textContent = `Streaming ${reply.audio} as session “${reply.session}”…`;
      toast(root, "ok", `Streaming started (${reply.audio})`);
      pollStatus(root, session, status, submit);
    } catch (e) {
      toast(root, "err", `Upload failed: ${e.message}`);
      status.textContent = "";
      submit.disabled = false;
    }
  });

  return el("div", { class: "ingest-file" },
    el("div", { class: "field" },
      el("span", {}, "WAV file (16k mono int16)"),
      fileInput,
    ),
    el("div", { class: "ingest-row" },
      el("label", { class: "field" },
        el("span", {}, "Language"),
        langInput,
      ),
      el("label", { class: "ingest-pacing" },
        pacing, el("span", {}, "Real-time pacing"),
      ),
    ),
    el("div", { class: "ingest-actions" }, submit, status),
  );
}


async function pollStatus(root, session, statusEl, submitBtn) {
  // Light poll to detect when the stream finishes.
  for (let i = 0; i < 1200; i++) {
    try {
      const s = await api.streamStatus(session);
      if (!s.running) {
        statusEl.textContent = "Stream finished.";
        if (submitBtn) submitBtn.disabled = false;
        return;
      }
    } catch {}
    await new Promise(r => setTimeout(r, 1000));
  }
  if (submitBtn) submitBtn.disabled = false;
}


function renderMicTab(root, session) {
  const langInput = el("input", { type: "text", value: "en-US", placeholder: "en-US" });
  const startBtn = el("button", { class: "btn-primary" }, "Start microphone");
  const stopBtn = el("button", { class: "btn-ghost", disabled: true }, "Stop");
  const status = el("div", { class: "ingest-status" });
  const meter = el("div", { class: "ingest-meter" }, el("div", { class: "ingest-meter-fill" }));

  let audioCtx = null;
  let mediaStream = null;
  let workletNode = null;
  let socket = null;

  async function teardownMic() {
    try { if (workletNode) workletNode.disconnect(); } catch {}
    try { if (audioCtx) await audioCtx.close(); } catch {}
    try { if (mediaStream) mediaStream.getTracks().forEach(t => t.stop()); } catch {}
    try { if (socket && socket.readyState === WebSocket.OPEN) socket.send("stop"); } catch {}
    try { if (socket) socket.close(); } catch {}
    audioCtx = null;
    mediaStream = null;
    workletNode = null;
    socket = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    status.textContent = "";
    meter.querySelector(".ingest-meter-fill").style.width = "0%";
  }

  startBtn.addEventListener("click", async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast(root, "err", "Browser doesn't expose getUserMedia");
      return;
    }
    if (!window.AudioWorkletNode) {
      toast(root, "err", "AudioWorklet not supported in this browser");
      return;
    }
    if (location.protocol !== "https:" && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
      toast(root, "warn", "getUserMedia 仅在 https / localhost 可用");
    }

    startBtn.disabled = true;
    status.textContent = "Requesting microphone…";

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        video: false,
      });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      await audioCtx.audioWorklet.addModule("/static/js/pcm-worklet.js");

      // Open WS first so we don't drop the first chunks.
      socket = new WebSocket(wsUrlForIngest(session, langInput.value || "en-US"));
      socket.binaryType = "arraybuffer";
      let ready = false;
      socket.addEventListener("message", (e) => {
        try {
          const m = JSON.parse(e.data);
          if (m.type === "ingest_ready") {
            ready = true;
            status.textContent = "Microphone live → " + session;
            stopBtn.disabled = false;
          } else if (m.type === "error") {
            toast(root, "err", m.error || "ingest error");
            teardownMic();
          }
        } catch {}
      });
      socket.addEventListener("close", () => {
        if (status.textContent.startsWith("Microphone live")) {
          status.textContent = "WS closed.";
        }
        teardownMic();
      });
      socket.addEventListener("error", () => {
        toast(root, "err", "WebSocket error");
      });

      // Wait until the WS is open before piping audio.
      await waitForOpen(socket);

      const source = audioCtx.createMediaStreamSource(mediaStream);
      workletNode = new AudioWorkletNode(audioCtx, "pcm-downsampler");
      workletNode.port.onmessage = (e) => {
        if (!ready) return;
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(e.data);
          // crude audio-level meter: peak of int16
          const view = new Int16Array(e.data);
          let peak = 0;
          for (let i = 0; i < view.length; i++) {
            const a = Math.abs(view[i]);
            if (a > peak) peak = a;
          }
          const pct = Math.min(100, (peak / 0x7FFF) * 100);
          meter.querySelector(".ingest-meter-fill").style.width = pct.toFixed(1) + "%";
        }
      };
      source.connect(workletNode);
      // We don't connect the worklet to the destination — we don't want to
      // play back the mic into the speakers (feedback).
    } catch (e) {
      toast(root, "err", `Mic init failed: ${e.message}`);
      teardownMic();
    }
  });

  stopBtn.addEventListener("click", teardownMic);

  function waitForOpen(ws) {
    return new Promise((resolve, reject) => {
      if (ws.readyState === WebSocket.OPEN) return resolve();
      ws.addEventListener("open", () => resolve(), { once: true });
      ws.addEventListener("error", () => reject(new Error("ws error")), { once: true });
      setTimeout(() => reject(new Error("ws open timeout")), 5000);
    });
  }

  const node = el("div", { class: "ingest-mic" },
    el("div", { class: "ingest-row" },
      el("label", { class: "field" },
        el("span", {}, "Language"),
        langInput,
      ),
    ),
    el("div", { class: "ingest-actions" }, startBtn, stopBtn, status),
    el("div", { class: "ingest-meter-wrap" },
      el("span", {}, "level"),
      meter,
    ),
    el("p", { class: "ingest-hint-small" },
      "需要 https 或 localhost。第一次点 “Start microphone” 浏览器会要权限。停止后会自动 close WS。",
    ),
  );
  node._teardown = () => { teardownMic(); };
  return node;
}
