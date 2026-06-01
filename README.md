# Speech_Voiceprint

Azure Speech STT 与开源声纹模型(pyannote.audio / SpeechBrain)拼装,在匿名 `Guest-N` 之上输出可识别的 Speaker 标签。

## 快速开始

```bash
# 安装(选其一 backend)
pip install -e ".[pyannote,dev]"
# 或
pip install -e ".[speechbrain,dev]"

# 配置
cp config.example.yaml config.yaml
export AZURE_SPEECH_KEY=...
export AZURE_SPEECH_REGION=eastasia
export HF_TOKEN=...   # 仅 pyannote 需要

# 端到端跑通
python -m pipeline.orchestrator \
  --mode fast \
  --audio sample.wav \
  --backend pyannote \
  --out result.json
```

## 模块

| 路径 | 职责 |
|------|------|
| `stt/` | Azure Speech 适配:realtime / fast / batch |
| `voiceprint/` | 声纹后端:pyannote / speechbrain |
| `merger/` | STT 时间戳与声纹段对齐 |
| `registry/` | 可选,SQLite / MySQL 跨会话声纹匹配 |
| `pipeline/` | 编排器与 CLI |
| `benchmarks/` | 双 backend 对比评测 |

## 输出

```json
{
  "utterances": [
    {
      "text": "你好,我们今天讨论的是...",
      "start": 1.20, "end": 4.85,
      "speaker": "Speaker_A", "speaker_confidence": 0.92,
      "words": [
        {"text": "你好", "start": 1.20, "end": 1.62, "speaker": "Speaker_A"}
      ]
    }
  ]
}
```

## 三种 STT 模式

```bash
# Fast Transcription(<5min,<500MB,推荐默认)
python -m pipeline.orchestrator --mode fast --audio sample.wav --backend pyannote ...

# Batch v3.2(长音频,接受 SAS URL)
python -m pipeline.orchestrator --mode batch \
  --audio-url "https://acct.blob.core.windows.net/...sample.wav?<SAS>" \
  --audio sample.wav      # 同一份音频本地副本,供声纹提取
  --backend pyannote --language en-US --out result.json

# 实时流式(WAV / 麦克风)
python -m pipeline.streaming --audio sample.wav --language en-US \
  --registry ~/.speaker_registry.db [--ws-port 8765]

python -m pipeline.streaming --mic --language zh-CN \
  --mic-device 1 --registry ~/.speaker_registry.db --ws-port 8765
python -m pipeline.streaming --list-devices    # 列出可用麦克风
```

## 实时 Web 查看器(单进程模式)

```bash
python -m pipeline.streaming --audio sample.wav --language en-US \
  --registry ~/.speaker_registry.db --ws-port 8765 --session katiesteve
# 访问 http://localhost:8765/?session=katiesteve  → 实时滚动字幕,speaker 自动着色
# WS 事件源:ws://localhost:8765/events?session=katiesteve
# 列出活跃会话:GET http://localhost:8765/sessions
```

事件格式:`{start, end, speaker, speaker_confidence, text, azure_speaker, session, event}`,`event` ∈ `{tentative, final, revised}`。

`--session` 是会话标签(默认 `default`)。同一进程内的多个 viewer 通过 URL 上的 `?session=` 隔离;页面顶部 dropdown 列出当前服务器上所有会话,切换时直接刷新。

## 管理台 / Web Dashboard(REST API + UI)

新的 FastAPI 入口把 live 查看器 + Speaker registry CRUD + 离线作业提交 + 健康面板拼到一起,部署在浏览器单一站点。

```bash
# 一键启动 / 停止(推荐)—— 后台跑、等 /api/ready、PID + 日志写到 .run/
./start.sh                # 启动
./start.sh status         # 查看运行状态
./start.sh logs -f        # 跟随日志
./start.sh stop           # SIGTERM 优雅停机
./start.sh restart
# 端口 / host / 注册表路径可用环境变量覆盖:SV_PORT、SV_HOST、SV_REGISTRY_PATH
# 鉴权:在 .env 里设 SV_API_TOKEN=<bearer> 即可全路由生效

# 或直接调命令(开发期前台调试)
.venv/bin/python -m pipeline.api --port 8080 --registry ~/.speaker_registry.db

# 浏览 http://localhost:8080/  → sidebar 导航:Live / Sessions / Registry / Jobs / Health / Maintenance

# 让 streaming 进程把事件转发给中央 api(而非自己起 ws-port)
.venv/bin/python -m pipeline.streaming --mic --session katiesteve \
  --api-target http://localhost:8080
```

REST 端点(全部走 Bearer 当 `SV_API_TOKEN` 已设;探针除外):
- `GET /livez` —— liveness,无鉴权(K8s / Docker probe 用)
- `GET /readyz` / `GET /api/ready` —— readiness,SIGTERM 后翻 503;**无需 token**(K8s readiness 用)
- `GET /api/health` —— UI/运维健康详情;当 `SV_API_TOKEN` 已设时需要 Bearer
- `GET /api/schema` —— Pydantic JSON Schema
- `GET /api/sessions`、`GET /api/sessions/{id}/history`、`POST /api/sessions/{id}/events`、`DELETE /api/sessions/{id}`
- `GET|PATCH|DELETE /api/registry/speakers[/{id}]`、`DELETE /api/registry/speakers?scope=all|unnamed`(批量)
- `POST /api/jobs/transcribe`(JSON)/ `POST /api/jobs/transcribe-upload`(multipart)
- `GET /api/jobs[/{id}]`、`GET /api/jobs/{id}/download`、`DELETE /api/jobs/{id}`
- `GET /api/maintenance/usage`、`POST /api/maintenance/cleanup`(scope:uploads/stream/outputs/jobs/sessions,带 dry_run)
- `WebSocket /ws/events?session=<id>` —— 浏览器订阅 + 命令(rename/forget)
- `WebSocket /ws/ingest?session=<id>` —— 浏览器麦克风 PCM 推流

设计语言深色优先,glassmorphic 卡片 + Inter / JetBrains Mono;暗 / 亮主题切换持久化。

## 跨 backend 复用 Speaker(双注册)

speechbrain 的 192-dim ECAPA 和 pyannote 的 512-dim x-vector 是两套不可比对的向量空间——同一个 speaker_id 下要让两个 backend 都能识别,需要把两套 embedding 都写进库。Job 表单的 ☑ "Also enroll the **other** backend" 就是干这件事:

- 主 backend 跑完 STT + 声纹分离 + registry 匹配后,自动用另一个 backend 在同一段音频再 diarize 一次
- 用时间重叠把 secondary 段落映射回 primary 已经决定好的 `speaker_id`
- secondary 的均值 embedding 写进 registry,model 字段标 `pyannote-512` / `speechbrain-192`

之后任意 backend 跑,这个人都能被识别成同一身份。Compare 模式下自动跳过(本来就跑了两次)。pyannote 路径需要 `HF_TOKEN`;缺失时 dual-enroll 优雅降级,主流程不受影响。

> **未完成的 B 方案**:跨模型映射学习。离线训一个小 MLP 把 192→512(或反向),库里只存一套向量,匹配时按需映射。优点是不用每次跑双 diarize;代价是要标注样本 + 训模型。先记在这里,有需要再做。

## 数据目录布局

所有 IO 集中在一个根目录(`SV_DATA_DIR`,默认 `<repo>/data/`):

| 子目录 | 用途 | 可清理 |
|--------|------|--------|
| `resources/` | 用户上传的原始音频(只读) | 手动 |
| `uploads/` | `POST /api/jobs/transcribe-upload` 暂存 | 是,Maintenance 页 |
| `stream/` | `POST /api/sessions/{id}/stream-file` 暂存 | 是 |
| `outputs/` | Job 完成的 result JSON,Jobs 页 ↓ 下载源 | 是 |
| `registry/` | `speakers.db` 声纹库 SQLite;或通过 `SV_REGISTRY_PATH=mysql+pymysql://...` 使用 MySQL | 仅通过 Registry 页 |

每个子目录可单独 override:`SV_RESOURCES_DIR / SV_UPLOADS_DIR / SV_STREAM_DIR / SV_OUTPUTS_DIR / SV_REGISTRY_DIR`。AKS 场景常把 `data` 和 `registry` 挂到不同 PVC。

## Python SDK

```python
from pipeline import transcribe_file
result = transcribe_file("katiesteve.wav", mode="fast", backend="speechbrain",
                         registry_path="~/.speaker_registry.db")
print(result.utterances[0].speaker, result.utterances[0].text)
print(result.model_dump_json(exclude_none=True))   # 与 CLI 输出等价
```

## 部署

详见 [`deploy/README.md`](deploy/README.md):Mac 本地、Azure VM、AKS,以及跨端共享 SQLite 声纹库(litestream → Azure Blob)。

```bash
cd deploy

# 本地 / VM:跑管理台 api
docker compose up -d api
# 一次性离线作业
docker compose run --rm orchestrator --mode fast \
  --audio /data/resources/sample.wav --backend pyannote \
  --out /data/outputs/out.json
```

**AKS** 一键(模板已就位,先填 `deploy/k8s/secret.yaml` 三个值或使用 sealed-secrets):

```bash
kubectl apply -k deploy/k8s/
```

manifests 包含 Deployment(replicas=1,startup/liveness/readiness 三探针,SIGTERM 优雅停机 60s)+ Service + Ingress + 三个 PVC(data / registry / models)+ Secret + ConfigMap。**当前是 single-replica**:SessionHub 是进程内 dict,水平扩展需先迁到 Redis/Postgres;registry 可先切到 MySQL。

## 测试

```bash
.venv/bin/pytest tests/         # 126 用例:aligner / registry / registry_factory / dual_enroll / azure / config / mic / ws_server / retry / sdk / api / vad / paths
.venv/bin/python -m benchmarks.run_eval --audio ... --backend pyannote --backend speechbrain
bash deploy/smoke_litestream.sh # docker daemon 必需:litestream WAL round-trip
```

最新 backend 对比报告:[`benchmarks/report.md`](benchmarks/report.md)。
