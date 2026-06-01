# Voiceprint Backend Comparison — M1 Report (post-optimization)

## 数据集
| audio | 时长 | 语言 | 真实说话人 | 备注 |
|---|---:|---|---:|---|
| katiesteve.wav | 29.5s | en-US | 2 (Katie/Steve) | 双人对话,干净录音 |
| resources/1.mp3 | 210.7s | zh-CN | 1 | 中文单人讲解(夹杂英文短语) |
| resources/2.mp3 | 810.0s | en-US | 3 | 三人英文播客,有背景音/笑声 |

## 总表(Mac CPU)

| audio | duration(s) | backend | RTF | speakers | segments | utterances | mixed | mixed_ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| katiesteve | 29.5 | pyannote | 1.13 | **2** ✓ | 11 | 8 | 0 | 0.000 |
| katiesteve | 29.5 | speechbrain | 0.03 | **2** ✓ | 6 | 8 | 0 | 0.000 |
| 1 | 210.7 | pyannote | 0.58 | **1** ✓ | 9 | 8 | 0 | 0.000 |
| 1 | 210.7 | speechbrain | 0.02 | **1** ✓ | 1 | 8 | 0 | 0.000 |
| 2 | 810.0 | pyannote | 0.60 | **3** ✓ | 534 | 332 | 22 | 0.066 |
| 2 | 810.0 | speechbrain | 0.02 | **3** ✓ | 146 | 332 | 20 | **0.060** |

两个 backend 在三个数据集上 speaker_count 全对,SpeechBrain mixed_ratio 略低且**速度 30×**。

## 优化清单(本轮)

### 1. SpeechBrain 聚类:AHC(固定阈值)→ spectral + auto-k
旧实现在 13.5min 三人音频上切出 109 簇(阈值 0.6 在长片段同人 embedding 离散度变大时彻底失效)。新实现:

1. **Monologue gate**:median pairwise cosine 距离 < 0.5 直接判 1 人
2. **Outlier 剥离**:5-NN 平均距离的 90 百分位剔除噪声窗口
3. **Spectral + silhouette**:k=2..max_k 跑 spectral,过滤 min_size < 5% 的退化划分,选最大 cosine-silhouette
4. **Min silhouette gate**:全部 k 的最佳 silhouette < 0.10 → fallback k=1

位置:`voiceprint/speechbrain_provider.py::{_auto_k, _filter_outliers, _spectral}`

### 2. M3 streaming 接 Registry
`pipeline/streaming.py` 加 `--registry --match-threshold --unknown-threshold` 实现 Katie/Steve 跨会话识别。第二次跑同一音频直接出真实姓名。

### 3. 冷启动 pending → revised
置信 < 0.5 的 utterance 先发 `tentative`,流式结束后再聚类 → emit `revised`(只在 speaker 变化或置信提升 ≥ 0.1 时)。

### 4. `--no-pacing` 收尾
`AzureRealtimeTranscriber.end_input(timeout=60)` 等 `session_stopped` 后才 close,不再丢尾部事件(原本 7 句只到 3 句)。

### 5. 在线 cluster threshold 0.6 → 0.7
防止 < 1s 短窗(笑声/感叹词)被误切为新 speaker。"Absolutely." 从 Speaker_C 假阳改回 Katie。

## 推荐默认配置

| 场景 | 推荐 backend | 备注 |
|---|---|---|
| 离线批处理(任意长度) | pyannote 或 speechbrain | 三测试集 accuracy 持平,speechbrain 30× 快 |
| Fast Transcription 后处理 | **speechbrain** | RTF 0.02 |
| 长会议/播客(高质量需求) | **pyannote** | community-1 有内置 embedding,边界更精细 |
| 实时流式 | **speechbrain**(`streaming.py` 默认) | pyannote 在线推理需另写适配器 |

## 复现命令

```bash
# 离线对比
.venv/bin/python -m benchmarks.run_eval \
  --audio katiesteve.wav --audio resources/1.mp3 --audio resources/2.mp3 \
  --backend pyannote --backend speechbrain \
  --language en-US --language zh-CN --device cpu

# 实时流式 + Registry 命名持久化
.venv/bin/python -m pipeline.streaming \
  --audio katiesteve.wav --language en-US --language zh-CN \
  --registry ~/.speaker_registry.db
```

JSON 详细结果:`benchmarks/out/{audio_name}__{backend}.json`
