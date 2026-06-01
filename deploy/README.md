# Speech_Voiceprint 部署指南

本文档覆盖四种部署形态:Mac 本地开发、Azure VM 生产、跨端共享声纹库、AKS / Kubernetes。

---

## 1. Mac 本地(开发与试跑)

### 1.1 一键启动管理台 api(最常用)

```bash
cp .env.example .env  # 填 AZURE_SPEECH_KEY / AZURE_SPEECH_REGION,可选 HF_TOKEN / SV_API_TOKEN
./start.sh            # 后台启 pipeline.api,等 /api/ready 200,PID + 日志在 .run/
./start.sh status     # 看运行状态
./start.sh logs -f    # 跟随日志
./start.sh stop       # SIGTERM 优雅停机
# 浏览 http://127.0.0.1:8080/  → Live / Sessions / Registry / Jobs / Health / Maintenance
```

### 1.2 直接用 venv 跑 CLI(批处理 / 实时调试)

```bash
uv venv --python 3.13 .venv
.venv/bin/pip install -e ".[pyannote,speechbrain,dev]"

# 离线 Fast Transcription
.venv/bin/python -m pipeline.orchestrator \
  --mode fast --audio sample.wav --backend pyannote \
  --language en-US --language zh-CN --device cpu --out result.json

# 实时流式
.venv/bin/python -m pipeline.streaming \
  --audio sample.wav --language en-US \
  --registry ~/.speaker_registry.db
```

Apple Silicon 想用 GPU(MPS):`--device mps`。pyannote/SpeechBrain 都会自动用 Metal 后端。

`--registry` 接受裸路径或 URI(`sqlite:///abs/path.db`),工厂在 `registry.open_store`;未来加 MySQL/Postgres 后端走同一入口。

### 1.3 用 Docker(隔离依赖)

```bash
cd deploy
docker compose up -d api                            # 主服务 = pipeline.api(8080)
docker compose run --rm orchestrator \
  --mode fast --audio /data/resources/katiesteve.wav \
  --backend pyannote --device cpu --out /data/outputs/result.json
```

模型权重保存到 `sv-models` volume,跨多次 `run` 复用。**第一次启动会从 HF/Hub 下载约 1GB**(pyannote 250MB + SpeechBrain 70MB + 一些段模型)。

把权重直接打到镜像里:
```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg BAKE_MODELS=1 \
  -t speech-voiceprint:latest -f deploy/Dockerfile .
```

---

## 2. Azure VM(生产)

### 2.1 推荐机型

| 用途 | VM 大小 | 备注 |
|---|---|---|
| Fast 后处理 | D4s_v5 (4 vCPU / 16GB) | 已足够 RTF 0.6 (pyannote) / 0.02 (speechbrain) |
| 长 batch + 实时多并发 | D8s_v5 (8 vCPU / 32GB) | 同时跑 4-8 路实时 |
| GPU 加速(可选) | NC4as_T4_v3 | pyannote on GPU,RTF 提升 5-10× |

### 2.2 准备

```bash
# 在 VM 上
az login --identity                                 # 推荐用 Managed Identity
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER && newgrp docker

git clone <your-repo> speech-voiceprint && cd speech-voiceprint
mkdir -p deploy/out deploy/bench-out
```

### 2.3 secrets 注入

把 key 放到 `deploy/.env`(被 docker-compose 自动读取):

```bash
cat > deploy/.env <<EOF
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=southeastasia
HF_TOKEN=hf_...
EOF
chmod 600 deploy/.env
```

更安全:用 Azure Key Vault + Managed Identity 注入:
```bash
export AZURE_SPEECH_KEY=$(az keyvault secret show --vault-name <kv> --name speech-key --query value -o tsv)
```

### 2.4 启动

```bash
cd deploy
docker compose build --build-arg BAKE_MODELS=1 api
docker compose up -d api                              # 主服务,8080
curl -fsS http://127.0.0.1:8080/readyz                # {"ready":true}
docker compose run --rm orchestrator --help          # 一次性 CLI
```

### 2.5 反向代理 / WS

api 同时承载 REST + WS(`/ws/events`、`/ws/ingest`)。Caddy 例:
```caddyfile
streaming.example.com {
  reverse_proxy api:8080
}
```

---

## 3. 跨端共享声纹库(litestream)

让 Mac 给 Katie/Steve 命名后,Azure VM 立即生效;反之亦然。

### 3.1 创建 Azure Blob 容器

```bash
RG=speech-voiceprint-rg
SA=svvoiceprintdb
az storage account create -g $RG -n $SA --sku Standard_LRS
az storage container create --account-name $SA --name speech-voiceprint
```

### 3.2 配置 litestream

在 `deploy/.env` 里追加:
```bash
LITESTREAM_AZURE_ACCOUNT=svvoiceprintdb
LITESTREAM_AZURE_ACCOUNT_KEY=...
```

litestream 服务在 compose 里挂在 `registry-sync` profile 下,默认不随 `docker compose up` 一起启动:
```bash
docker compose --profile registry-sync up -d litestream
```

litestream 每 10s 把 `sv-registry/speakers.db` 的 WAL 同步到 Azure Blob。**新 VM 第一次启动用 `litestream restore` 拉回库**:
```bash
docker compose run --rm --entrypoint /usr/local/bin/litestream litestream \
  restore -config /etc/litestream.yml /registry/speakers.db
```

### 3.2.1 端到端 smoke(本地)

`deploy/smoke_litestream.sh` 用本地 `file` driver 跑 WAL → snapshot → restore 完整回环,在不连 Azure 的情况下验证 litestream 容器和你的 SQLite 一起工作:

```bash
bash deploy/smoke_litestream.sh
# 期望末行:OK: round-trip restored Katie
```

> 注:litestream 的 `abs` 驱动 URL 硬编码 `*.blob.core.windows.net`,无法指向 Azurite,所以本脚本只覆盖 WAL 复制语义;真实 Azure ABS 仍走 §4 健康检查清单做手测。

### 3.3 替代方案

| 方案 | 优点 | 缺点 |
|---|---|---|
| **litestream(推荐)** | SQLite 不用换;实时同步;Azure 原生 | 新增 sidecar 容器 |
| Azure Files | 简单 mount 即可 | NFS 上 SQLite 锁有性能问题 |
| MySQL | 真正多写者并发;AKS 原生适配 Azure Database for MySQL | 需要安装 `.[mysql]` 并管理连接串 |
| Postgres | 真正多写者并发 | Schema 迁移、运维成本 |
| 手工 rsync | 零依赖 | 只能定时,有冲突风险 |

如果只读多写少(常见情况),litestream 完全够用。

### 3.4 MySQL registry

如果希望 registry 跨 Pod/节点共享,可把 speaker/voiceprint 元数据放进 MySQL:

```bash
pip install -e ".[mysql]"
export SV_REGISTRY_PATH='mysql+pymysql://user:password@mysql-host:3306/speech_voiceprint?charset=utf8mb4'
```

MySQL backend 会自动创建 `speakers` 与 `voiceprints` 两张表。上传音频、stream 暂存和 job output JSON 仍走 `SV_DATA_DIR` / PVC;这些大对象后续更适合接 Azure Blob,不建议写进 MySQL。

---

## 4. 健康检查清单

```bash
# 1. api 就绪
curl -fsS http://127.0.0.1:8080/livez                  # {"ok":true}
curl -fsS http://127.0.0.1:8080/readyz                 # {"ready":true}
curl -fsS -H "Authorization: Bearer $SV_API_TOKEN" \
  http://127.0.0.1:8080/api/health                     # 含 registry_path / workers

# 2. 跑 katiesteve 看 7 句全到(orchestrator CLI 模式)
docker compose run --rm orchestrator \
  --mode fast --audio /data/resources/katiesteve.wav --backend pyannote \
  --language en-US --device cpu --out /data/outputs/health.json

# 3. registry 持久化:跑两次,第二次出 sp_xxxx
docker compose run --rm orchestrator \
  --mode fast --audio /data/resources/katiesteve.wav --backend pyannote \
  --language en-US --device cpu \
  --registry /data/registry/speakers.db --out /data/outputs/run1.json
docker compose run --rm orchestrator \
  --mode fast --audio /data/resources/katiesteve.wav --backend pyannote \
  --language en-US --device cpu \
  --registry /data/registry/speakers.db --out /data/outputs/run2.json

# 第二次的 utterance.speaker 应该是 sp_xxxxxxxx 而非 Speaker_A/B
```

---

## 5. AKS / Kubernetes 部署

### 5.1 文件清单

`deploy/k8s/`:

| 文件 | 作用 |
|---|---|
| `namespace.yaml` | `speech-voiceprint` namespace |
| `configmap.yaml` | 非敏感配置(region、log 设置、`SV_DATA_DIR`、`SV_SHUTDOWN_TIMEOUT`) |
| `secret.yaml` | **模板** —— Azure key、HF token、`SV_API_TOKEN`(替换或用 sealed-secrets) |
| `pvc.yaml` | 三个 PVC:`sv-data`(20Gi)/ `sv-registry`(2Gi)/ `sv-models`(5Gi) |
| `deployment.yaml` | replicas=1,startup/liveness/readiness 三探针,`Recreate` 滚动策略 |
| `service.yaml` | ClusterIP:80 → 8080 |
| `ingress.yaml` | NGINX 模板,WS 友好的 timeout |
| `kustomization.yaml` | 入口,`kubectl apply -k` 用 |

### 5.2 部署流程

```bash
# 1. 推镜像到 ACR(假设你的 ACR 名 myacr)
docker build -t myacr.azurecr.io/speech-voiceprint:v0.1.0 -f deploy/Dockerfile .
docker push myacr.azurecr.io/speech-voiceprint:v0.1.0

# 2. 改镜像 tag
cd deploy/k8s
kustomize edit set image ghcr.io/example/speech-voiceprint=myacr.azurecr.io/speech-voiceprint:v0.1.0

# 3. 创建 secret(三选一)
# a. 命令行(适合实验/Demo):
kubectl create namespace speech-voiceprint
kubectl create secret generic sv-secrets -n speech-voiceprint \
  --from-literal=AZURE_SPEECH_KEY="$AZURE_SPEECH_KEY" \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  --from-literal=SV_API_TOKEN="$(openssl rand -hex 32)"
# b. sealed-secrets:用 kubeseal 加密 secret.yaml 后提交 git
# c. Azure Key Vault CSI driver:把 secretObjects 改写到 secret.yaml

# 4. 部署
kubectl apply -k deploy/k8s/

# 5. 验证
kubectl -n speech-voiceprint get pods
kubectl -n speech-voiceprint port-forward svc/sv-api 8080:80
curl -fsS http://127.0.0.1:8080/api/ready    # {"ready":true}

# 6. 浏览器:配置 ingress.yaml 的 host,或用 port-forward 临时访问
```

### 5.3 关键设计点

- **`replicas: 1` 且 `Recreate` 策略**:JobStore 与 SessionHub 是进程内 dict,跨副本不共享;PVC 默认 RWO 也不允许两个 pod 同时挂。要水平扩展先把这两块迁出(Redis/Postgres)。
- **三探针的分工**:
  - `startupProbe`(180s 容错窗)—— 走 `/livez`,进程启动即可
  - `livenessProbe` 走 `/livez` —— 进程存活,无鉴权
  - `readinessProbe` 走 `/readyz` —— SIGTERM 后翻 503,让 ingress 主动停发新流量,无鉴权
- **优雅停机**:Pod `terminationGracePeriodSeconds: 90` ≥ `SV_SHUTDOWN_TIMEOUT` 60s + 30s 余量。lifespan 收到 SIGTERM 翻 ready=False,然后等 in-flight Job 完成再退出。
- **Bearer token + Ingress**:`SV_API_TOKEN` 是 **defense-in-depth**。即使前面挂了 Azure AD / API Management / oauth2-proxy 做用户认证,也建议保留:Ingress 转发到内部网络时附加 `Authorization: Bearer $SV_API_TOKEN`,这样直接 hit Pod IP 也访问不到 API。
- **WebSocket**:`/ws/events`、`/ws/ingest` 都是长连接。NGINX ingress 已配置 `proxy-read-timeout: 3600`;AGIC 用户改用 `appgw-ingress.kubernetes.io/request-timeout` 等同义注解。
- **资源建议**:0.5 CPU / 1.5Gi 起步,4Gi 上限够 pyannote + 2 个并发 Job。GPU 节点把 device 改 `cuda` 并加 `nodeSelector` / `tolerations`(模板里没默认放 GPU)。
- **PVC**:`sv-data` 用 RWO 即可。如果 future 分离 api 与 worker,把 `sv-data` 升 RWX(`azurefile-csi`)便于共享 outputs。

### 5.4 SV_API_TOKEN 何时该用

| 场景 | 是否设 | 备注 |
|---|---|---|
| 本地开发,只在 localhost 跑 | 否 | 留空,所有路由免认证,体验最丝滑 |
| 直接对外暴露(LoadBalancer / NodePort / Tunnel) | **必须** | 否则任何人都能 CRUD registry / 提交作业 |
| AKS + Ingress 但 **没有** 用户级认证 | **必须** | 同上,公开就要保护 |
| AKS + Ingress + Azure AD / APIM / oauth2-proxy | 推荐保留 | defense-in-depth;让 Ingress 注入这个 token,即使 Pod 被旁路也无法访问 API |

token 推荐用 `openssl rand -hex 32` 生成,配合 sealed-secrets / Key Vault 持久化。**不要** 在前端表单里让用户输入 —— 只做内部认证 token,前端要鉴权请走 Ingress 那一层。

---

## 6. 故障排查

| 现象 | 原因 / 解决 |
|---|---|
| `huggingface_hub.errors.GatedRepoError 401` | 浏览器登录 HF,接受 pyannote/speaker-diarization-community-1 与 pyannote/embedding 的用户协议;然后重试 |
| `OSError: cannot load library libsndfile` | base 镜像缺 `libsndfile1`;Dockerfile 已装,确认是 host pip 安装的话 `apt install libsndfile1` |
| Azure 401 | `AZURE_SPEECH_KEY/REGION` 不匹配;Fast 用 `cognitive.microsoft.com`,Batch 用 `cognitiveservices.azure.com`,host 在代码里已对应 |
| MP3 在 pyannote 报 sample mismatch | 已通过 `librosa.load(sr=16000)` 预处理修复 |
| Registry 第二次仍出 `sp_xxxx` 而非姓名 | 没 rename;`sqlite3 speakers.db "UPDATE speakers SET display_name='Katie' WHERE id='sp_xxxx'"` |
