# 会议/课堂录音转写纪要系统 — 部署与操作手册

> 本手册面向**初次部署**的使用者，逐步说明从零到跑通的全过程，以及日常使用中的进阶操作与问题排查。

---

## 目录

1. [系统要求](#1-系统要求)
2. [环境准备](#2-环境准备)
3. [获取 API Key](#3-获取-api-key)
4. [配置文件说明](#4-配置文件说明)
5. [首次运行](#5-首次运行)
6. [读懂输出结果](#6-读懂输出结果)
7. [崩溃续跑](#7-崩溃续跑)
8. [关键词词典定制](#8-关键词词典定制)
9. [提示词定制](#9-提示词定制)
10. [切换大模型](#10-切换大模型)
11. [进阶配置](#11-进阶配置)
12. [故障排查](#12-故障排查)
13. [常见问题](#13-常见问题)

---

## 1. 系统要求

### 硬件

| 场景 | CPU | 内存 | GPU（可选） | 磁盘 |
|------|-----|------|------------|------|
| 纯 CPU（慢） | 4 核+ | 8 GB+ | — | 10 GB+ |
| GPU 加速（推荐） | 4 核+ | 16 GB+ | 4 GB+ 显存（CUDA） | 10 GB+ |
| Apple Silicon | M1/M2/M3 | 16 GB+ | — (MPS 自动) | 10 GB+ |

> **说明**：磁盘空间主要用于 FunASR 自动下载的模型文件（约 3–5 GB）。

### 软件

- **Python**：3.9 — 3.11（推荐 3.10）
- **操作系统**：Linux / macOS / Windows（WSL2 均可）
- **CUDA**（可选）：12.x，需与 PyTorch 版本匹配

### 网络

- 首次运行需要**联网**，FunASR 会自动从 ModelScope 下载模型（约 3–5 GB）。
- 调用大模型 API 时需要能访问对应服务（Moonshot / Anthropic 等）。
- 后续运行可完全离线（模型缓存在 `~/.cache/modelscope/`）。

---

## 2. 环境准备

### 2.1 创建虚拟环境（强烈推荐）

```bash
# 用 conda
conda create -n meeting_asr python=3.10 -y
conda activate meeting_asr

# 或者用 venv
python3 -m venv meeting_asr
source meeting_asr/bin/activate   # Linux / macOS
# meeting_asr\Scripts\activate    # Windows
```

### 2.2 安装 PyTorch

根据你的硬件选择对应命令（在 https://pytorch.org/get-started/locally/ 生成精确命令）：

```bash
# GPU（CUDA 12.1 示例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon（macOS）
pip install torch torchvision torchaudio
```

### 2.3 安装 FunASR

```bash
pip install funasr
```

> FunASR 会一并安装 `scipy`、`numpy`、`soundfile` 等依赖，无需单独安装。

### 2.4 安装本流水线依赖

```bash
# 进入流水线目录
cd examples/industrial_data_pretraining/meeting_transcription

# 核心依赖：openai SDK（兼容 Kimi K2）+ PyYAML
pip install openai pyyaml

# 如需使用 Claude（可选）
pip install anthropic
```

### 2.5 验证安装

```bash
python -c "from funasr import AutoModel; print('FunASR OK')"
python -c "import openai, yaml; print('openai + yaml OK')"
```

两行均打印 OK 即可继续。

---

## 3. 获取 API Key

### Kimi K2（默认，推荐）

1. 访问 [https://platform.moonshot.cn/](https://platform.moonshot.cn/) 注册/登录
2. 进入「控制台」→「API Keys」→「新建 API Key」
3. 复制生成的 Key（格式：`sk-...`）
4. 设置环境变量：

```bash
# Linux / macOS — 临时（当前终端有效）
export MOONSHOT_API_KEY=sk-你的key

# 永久写入（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export MOONSHOT_API_KEY=sk-你的key' >> ~/.bashrc
source ~/.bashrc
```

```cmd
# Windows CMD
set MOONSHOT_API_KEY=sk-你的key

# Windows PowerShell
$env:MOONSHOT_API_KEY="sk-你的key"
```

### Claude（可选）

1. 访问 [https://console.anthropic.com/](https://console.anthropic.com/) 注册/登录
2. 「API Keys」→「Create Key」
3. 设置 `ANTHROPIC_API_KEY`（方法同上）

---

## 4. 配置文件说明

```bash
# 将示例配置复制为你的配置文件
cp config.example.yaml config.yaml
```

打开 `config.yaml`，按需修改。下面是各参数的详细说明：

```yaml
# ── ASR 模型 ──────────────────────────────────────────────────────────────────
asr:
  model: "paraformer-zh"       # 中文 ASR 模型（首次运行自动下载）
  vad_model: "fsmn-vad"        # 语音活动检测（过滤静音）
  punc_model: "ct-punc"        # 标点恢复
  spk_model: "cam++"           # 说话人分离（声纹聚类）
  language: "auto"             # 语言：auto/zh/en/ja 等（auto 自动识别）
  use_itn: true                # 数字/日期文字转阿拉伯数字/日期格式
  merge_vad: true              # 合并相邻短语音段（减少碎句）
  merge_length_s: 15           # 合并后最长段落（秒）
  batch_size_s: 300            # FunASR 内部批处理大小（秒），显存不足时调小

# ── 分窗与说话人 ──────────────────────────────────────────────────────────────
pipeline:
  window_s: null               # 每窗时长（秒）；null=自动：GPU 600s / CPU 180s
                               # 调大可提升说话人一致性，但占用更多内存
  overlap_s: 5                 # 窗口重叠（秒），用于保证边界句子不被截断
  merge_thr: 0.78              # 说话人合并阈值（余弦相似度）
                               # 越高 → 更保守（倾向于更多说话人）
                               # 越低 → 更激进（倾向于更少说话人）
  preset_spk_num: null         # 强制指定说话人数；null = 自动推断

# ── 设备 ──────────────────────────────────────────────────────────────────────
device: null                   # null=自动检测（cuda:0 → mps → cpu）
                               # 也可手动指定："cuda:0"/"cuda:1"/"cpu"

# ── 目录 ──────────────────────────────────────────────────────────────────────
work_dir: "./work_dir"         # 缓存目录：各窗 JSON 文件 + LLM 缓存
output:
  dir: "./output"              # 最终输出目录

# ── 热词（可选） ──────────────────────────────────────────────────────────────
# hotword: "FunASR 阿里 会议"  # 用空格分隔，提升特定词汇识别率

# ── 关键词筛查 ────────────────────────────────────────────────────────────────
keywords:
  dict_path: null              # 关键词词典路径；null = 跳过关键词筛查
  # dict_path: "./keywords.example.json"  # 取消注释以启用

# ── 大模型纪要 ────────────────────────────────────────────────────────────────
llm:
  enabled: false               # 是否生成 LLM 结构化纪要（需要 API Key）
  provider: "kimi"             # 提供方：kimi（默认）/ claude / openai_compat
  model: "kimi-k2"             # 模型 ID（Kimi K2 的推荐值）
  max_tokens: 8192             # LLM 输出最大 token 数
  system_prompt_path: null     # 系统提示词文件路径（null=使用内置默认值）
  user_prompt_path: null       # 用户提示词模板文件路径

log_level: "INFO"              # 日志级别：DEBUG / INFO / WARNING / ERROR
```

---

## 5. 首次运行

### 5.1 最小运行（仅转写，不调 LLM）

```bash
cd examples/industrial_data_pretraining/meeting_transcription

python run.py --audio /path/to/your/meeting.wav
```

**第一次运行时**，FunASR 会自动从 ModelScope 下载四个模型（约 3–5 GB），请耐心等待，看到如下日志即在下载：

```
Downloading model paraformer-zh ...
Downloading model fsmn-vad ...
Downloading model ct-punc ...
Downloading model cam++ ...
```

下载完成后流水线开始处理。典型日志如下：

```
10:01:00 [INFO] Audio loaded: 3620.5s @ 16000Hz from meeting.wav
10:01:00 [INFO] Created new manifest: 7 windows planned
10:01:00 [INFO] Loading FunASR model paraformer-zh on cuda:0 …
10:01:05 [INFO] Processing window 1/7 (0–605s) …
10:06:22 [INFO] Window 0 done: 48 sentences, 3 speakers
10:06:22 [INFO] Processing window 2/7 (600–1205s) …
...
10:30:11 [INFO] Global re-clustering: 21 (window,spk) pairs → 3 global speakers
10:30:11 [INFO] Assembly done: 312 sentences
10:30:11 [INFO] Transcript saved to work_dir/a3f2b1c4d5e6f7a8/
```

### 5.2 带 LLM 纪要的完整运行

```bash
# 1. 先编辑 config.yaml，设置 llm.enabled: true
# 2. 确认已设置 MOONSHOT_API_KEY
echo $MOONSHOT_API_KEY   # 应打印你的 Key

# 3. 运行（使用你的配置 + 关键词词典）
python run.py \
  --audio /path/to/meeting.wav \
  --config config.yaml
```

### 5.3 常用命令行参数速查

```bash
# 指定输出目录
python run.py --audio meeting.wav --out-dir ./my_output

# 指定缓存目录（多个项目隔离）
python run.py --audio meeting.wav --work-dir ./proj_A_cache

# 强制指定说话人数（已知有 3 人）
python run.py --audio meeting.wav --spk-num 3

# 跳过 LLM 步骤（即使配置文件里 enabled: true）
python run.py --audio meeting.wav --no-llm

# 详细调试日志
python run.py --audio meeting.wav --log-level DEBUG
```

---

## 6. 读懂输出结果

运行完成后，`output/` 目录下会生成以下文件：

```
output/
├── transcript.json       ← 完整转写（结构化）
├── transcript.txt        ← 人读版转写稿
├── keyword_hits.json     ← 关键词命中记录（如配置了词典）
├── minutes.json          ← 结构化纪要（需 LLM）
└── minutes.md            ← 可读纪要草稿（需 LLM）
```

### transcript.txt — 转写稿示例

```
[说话人0] [0.3s–12.8s] 好，今天主要讨论三个议题，第一个是Q3的产品路线图。
[说话人1] [13.2s–28.5s] 我这边准备了一个方案，核心是把交付周期从8周缩短到5周。
[说话人0] [29.1s–35.6s] 这个方案有什么风险？
[说话人2] [36.0s–52.3s] 主要风险是测试资源不够，需要再招2名测试工程师。
```

> **说话人编号**（0, 1, 2）是机器自动聚类的结果，不代表真实姓名。你可以在 `minutes.md` 的"参与者"部分手动备注真实姓名。

### minutes.md — 纪要草稿示例

文件顶部有如下免责提示：

```
> AI 生成草稿，需人工复核。内容由大模型生成，可能存在遗漏或错误。
```

随后是结构化的会议纪要，包含：摘要、议题、决议、行动项、风险点、关键词亮点。

### transcript.json — 结构化数据

每条句子包含：

```json
{
  "text": "这个方案有什么风险？",
  "start": 29100,        // 绝对开始时间（毫秒）
  "end": 35600,          // 绝对结束时间（毫秒）
  "spk": 0,              // 全局说话人编号
  "window_id": 0,        // 来源窗口编号（调试用）
  "local_spk": 2         // 窗口内局部说话人编号（调试用）
}
```

---

## 7. 崩溃续跑

这是本流水线的核心设计之一。当长录音（数小时）中途因 OOM、网络断开、手动 Ctrl-C 等原因中断时，**再次执行同样的命令即可从断点继续**，已完成的窗口不会重新计算。

```bash
# 第一次运行，处理到第 4 个窗口时崩溃
python run.py --audio long_meeting.wav --config config.yaml
# [Ctrl+C] 或 OOM 崩溃

# 再次运行同一命令 → 自动续跑
python run.py --audio long_meeting.wav --config config.yaml
```

续跑时的日志提示：

```
10:31:00 [INFO] Loaded existing manifest (12 windows)
10:31:00 [INFO] Reconcile: adopted orphan files / reset stale running windows
10:31:00 [INFO] All windows already done — skipping ASR   ← 全部完成时直接跳过
# 或：
10:31:00 [INFO] Processing window 5/12 (600–720s) …      ← 从第 5 窗口继续
```

**工作原理**：`work_dir/<音频哈希>/manifest.json` 记录每个窗口的状态（`pending` / `running` / `done` / `failed`）。每个窗口写入磁盘后才标记为 `done`（原子操作）。重启时 `running` 状态会被重置为 `pending` 重新处理。

**注意**：如果你修改了 `window_s`、`overlap_s` 或 ASR 模型，会生成新的哈希，视为全新的任务（旧缓存不会被误用）。

---

## 8. 关键词词典定制

关键词功能用于标记转写中出现的重要词汇，并将命中记录提供给 LLM 作为"重点线索"。

### 启用关键词词典

在 `config.yaml` 中：

```yaml
keywords:
  dict_path: "./my_keywords.json"
```

### 词典格式

参考 `keywords.example.json`，以分类 → 关键词列表的形式组织：

```json
{
  "决议": ["决定", "通过", "批准", "同意", "确认"],
  "行动项": ["负责", "跟进", "落实", "截止", "提交"],
  "风险": ["风险", "延期", "问题", "阻碍", "超预算"],
  "人员": ["张三", "李四", "产品经理", "技术负责人"],
  "项目": ["Q3", "一期", "二期", "上线", "发布"]
}
```

### 命中结果示例（`keyword_hits.json`）

```json
[
  {
    "category": "决议",
    "keyword": "决定",
    "spk": 0,
    "start_ms": 45200,
    "end_ms": 58300,
    "text": "我们决定把上线时间推迟到下个月底。"
  }
]
```

### 热词加速识别（提升 ASR 准确率）

如果你的会议中有专有名词（人名、产品名、技术术语）经常被识别错误，可以在 `config.yaml` 中配置热词：

```yaml
hotword: "FunASR 张三 Q3路线图 阿里云 OpenAPI"
```

> 热词用空格分隔，直接嵌入到 ASR 识别过程中，提升命中率。

---

## 9. 提示词定制

默认提示词适合通用会议场景。如果你的场景有特殊要求（课堂笔记、医疗问诊、法律庭审等），可以自定义提示词文件。

### 创建自定义提示词

```bash
cp prompts/minutes_system.txt prompts/my_system.txt
cp prompts/minutes_user.txt prompts/my_user.txt
```

### 系统提示词（`my_system.txt`）

定义大模型的角色和输出要求。示例（课堂场景）：

```
你是一位专业的课堂助教，擅长整理讲课要点。你的任务是将课堂录音转写整理成结构化笔记。
- 识别老师讲授的核心知识点（说话人0通常为老师）
- 归纳学生提问及老师回答
- 标注重要公式、定义、例题
- 输出严格遵循要求的 JSON 格式
```

### 用户提示词模板（`my_user.txt`）

模板中必须保留 `{transcript}` 和 `{keyword_hits}` 两个占位符：

```
请根据以下课堂录音转写，生成结构化课堂笔记（JSON格式）。

## 课堂转写
{transcript}

## 关键词/知识点命中
{keyword_hits}

请按 JSON schema 输出笔记，topics 字段代表知识模块，每个模块包含讲解要点和示例。
```

### 在配置文件中指定路径

```yaml
llm:
  system_prompt_path: "./prompts/my_system.txt"
  user_prompt_path: "./prompts/my_user.txt"
```

> **提示词缓存机制**：修改提示词后，LLM 缓存会自动失效（缓存 Key 包含系统提示词哈希），下次运行会重新调用 LLM。

---

## 10. 切换大模型

### 10.1 Kimi K2（默认）

```yaml
llm:
  enabled: true
  provider: "kimi"
  model: "kimi-k2"          # 也可指定 "kimi-k2-0711" 等具体版本
  max_tokens: 8192
```

```bash
export MOONSHOT_API_KEY=sk-...
```

### 10.2 Claude

```yaml
llm:
  enabled: true
  provider: "claude"
  model: "claude-opus-4-8"
  max_tokens: 16000
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 10.3 阿里云百炼（DashScope / Qwen）

```yaml
llm:
  enabled: true
  provider: "openai_compat"
  model: "qwen-max"
  max_tokens: 8192
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

```bash
export LLM_API_KEY=sk-...   # DashScope API Key
```

### 10.4 本地 Ollama

```yaml
llm:
  enabled: true
  provider: "openai_compat"
  model: "qwen2.5:14b"       # 你本地拉取的模型名
  max_tokens: 4096
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"           # Ollama 不校验 key，填任意值即可
```

---

## 11. 进阶配置

### 11.1 多 GPU 场景

```yaml
device: "cuda:1"   # 指定使用第二块 GPU
```

如需并行处理多个音频，建议分别指定 `work_dir` 隔离：

```bash
python run.py --audio meeting_A.wav --work-dir ./cache_A --out-dir ./out_A &
CUDA_VISIBLE_DEVICES=1 python run.py --audio meeting_B.wav --work-dir ./cache_B --out-dir ./out_B &
```

### 11.2 超长录音（>3 小时）的内存调优

```yaml
pipeline:
  window_s: 300      # GPU 显存 4–8 GB 时建议调到 300s
  batch_size_s: 60   # FunASR 内部批处理，进一步减少峰值显存
asr:
  batch_size_s: 60
```

纯 CPU 场景：

```yaml
pipeline:
  window_s: 120      # CPU 内存 8 GB 时建议 120–180s
  batch_size_s: 30
```

### 11.3 已知说话人数

如果你事先知道会议有 N 个人，指定后可以显著提升说话人聚类质量：

```yaml
pipeline:
  preset_spk_num: 3   # 强制 3 个说话人
```

或命令行：

```bash
python run.py --audio meeting.wav --spk-num 3
```

### 11.4 多语言会议

```yaml
asr:
  language: "auto"   # 自动识别（中英混合、纯英文均可）
  use_itn: true
```

如果会议语言固定，明确指定可提升准确率：

- 中文：`language: "zh"`
- 英文：`language: "en"`
- 日文：`language: "ja"`

---

## 12. 故障排查

### 问题 A：模型下载失败 / 超时

**现象**：
```
ConnectionError: HTTPSConnectionPool ... timed out
```

**解决**：
```bash
# 方法 1：设置 ModelScope 镜像（国内推荐）
export MODELSCOPE_CACHE=~/.cache/modelscope
pip install modelscope

# 方法 2：手动设置代理
export HTTPS_PROXY=http://your-proxy:port

# 方法 3：手动下载模型并放入 modelscope 缓存目录
```

---

### 问题 B：CUDA out of memory

**现象**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**解决**：
```yaml
# 在 config.yaml 中减小窗口和批次
pipeline:
  window_s: 120
asr:
  batch_size_s: 30
```

如果显存极其有限（< 4 GB）：

```yaml
device: "cpu"   # 强制使用 CPU（速度慢但稳定）
```

---

### 问题 C：所有说话人被合并成"说话人0"

**现象**：`transcript.txt` 里所有句子都是说话人0。

**原因**：`merge_thr` 过低，或说话人声纹本身很相似，或录音质量差。

**解决**：
```yaml
pipeline:
  merge_thr: 0.85    # 提高阈值，减少合并（从 0.78 调到 0.85–0.92）
  # 或者直接指定说话人数
  preset_spk_num: 3
```

---

### 问题 D：句子数量为 0，没有转写结果

**现象**：`Assembly done: 0 sentences`。

**排查步骤**：
1. 检查音频格式是否支持：`file meeting.wav`（应为 PCM WAV / MP3 / FLAC / M4A 均可）
2. 检查音频是否有内容：用任意播放器确认不是空文件
3. 开启 DEBUG 日志查看详情：`python run.py --audio meeting.wav --log-level DEBUG`
4. 查看 `work_dir/<key>/window_0.json`，确认 `sentence_info` 字段是否为空

---

### 问题 E：LLM 返回 JSON 解析失败

**现象**：`minutes.json` 里有 `"_parse_error"` 字段，或 `title: "解析失败"`。

**原因**：大模型输出了非 JSON 格式的内容（如加了额外说明文字）。

**解决**：
1. 检查 `work_dir/llm_*.json` 里是否有缓存的原始响应
2. 删除对应缓存文件，重新运行
3. 在系统提示词中更明确地强调"只输出 JSON，不要其他文字"
4. 换用支持更好的模型（如 `kimi-k2` 通常比小模型稳定）

---

### 问题 F：Kimi API 报错 401 / 403

**现象**：
```
openai.AuthenticationError: 401 Unauthorized
```

**解决**：
```bash
# 确认 Key 已设置
echo $MOONSHOT_API_KEY

# 确认 Key 格式正确（以 sk- 开头）
# 在 Moonshot 控制台确认 Key 状态（未过期、有余额）
```

---

### 问题 G：`manifest.json` 损坏后无法续跑

**现象**：启动时报 `json.JSONDecodeError`。

**解决**：
```bash
# 查看损坏的 manifest
cat work_dir/<key>/manifest.json

# 如果损坏，删除 manifest 让流水线重建（窗口 JSON 文件会被重新认领）
rm work_dir/<key>/manifest.json

# 再次运行，_reconcile() 会自动认领已有的 window_*.json
python run.py --audio meeting.wav
```

---

## 13. 常见问题

**Q：支持哪些音频格式？**

A：支持 WAV、MP3、FLAC、M4A、OGG、MP4（音轨）等常见格式，FunASR 内部会自动重采样到 16kHz 单声道。建议使用 WAV 或 MP3，其他格式如遇问题可先用 `ffmpeg` 转换：

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

---

**Q：一段 1 小时的录音大概需要多长时间？**

A：参考时间（不含模型下载）：

| 硬件 | 1 小时音频 | 2 小时音频 |
|------|-----------|-----------|
| RTX 3080 (10GB) | ~5–8 分钟 | ~10–15 分钟 |
| Apple M2 (MPS) | ~15–20 分钟 | ~30 分钟 |
| 8 核 CPU | ~40–60 分钟 | ~80–120 分钟 |

LLM 纪要生成额外需要 10–60 秒（取决于转写长度和 API 响应速度），且有磁盘缓存（第二次运行秒出）。

---

**Q：说话人分离有多准确？**

A：在录音质量良好（无严重回声/混响）、说话人交替清晰的场景下，CAM++ 声纹聚类效果较好。影响准确率的主要因素：

- **录音质量**：信噪比低、回声重会导致识别错误
- **说话人相似度**：声音非常相似的两人可能被合并
- **窗口大小**：`window_s` 越大，跨窗口一致性越好
- **说话人数**：已知说话人数时建议设 `preset_spk_num`

---

**Q：纪要草稿里说话人0/1/2 对应谁？**

A：机器无法自动推断真实姓名。你可以：

1. 通过 `transcript.txt` 中的时间戳定位每位说话人的语段，对照实际录音确认
2. 在 `minutes.md` 的"参与者"部分手动标注（如"说话人0 = 张三（主持人）"）
3. 如果在会议开始时每人都做了自我介绍，可在提示词中指导 LLM 推断角色

---

**Q：如何处理中英文混合的会议？**

A：设置 `language: "auto"`（默认），Paraformer 会自动识别中英文混合的语段。如识别率不满意，也可尝试 `language: "zh"` + 热词列表补充关键英文词汇：

```yaml
hotword: "API SDK Dashboard Q3 Review"
```

---

**Q：能否对已有的 `transcript.json` 重新生成纪要（不重跑 ASR）？**

A：可以。删除 LLM 缓存文件即可：

```bash
rm work_dir/<audio_key>/llm_*.json  # 删除 LLM 结果缓存

# 然后带 --no-llm=false 的配置重跑（ASR 缓存保留，只重跑 LLM）
python run.py --audio meeting.wav --config config.yaml
```

或者直接修改提示词文件（提示词变化会自动使缓存失效）。

---

**Q：如何批量处理多个录音文件？**

A：写一个简单的 shell 脚本：

```bash
#!/bin/bash
for audio in ./recordings/*.wav; do
    echo "Processing: $audio"
    python run.py \
        --audio "$audio" \
        --config config.yaml \
        --out-dir "./output/$(basename "$audio" .wav)" \
        --work-dir "./cache/$(basename "$audio" .wav)"
done
```

---

*本手册对应版本：pipeline v1.0 — 如遇问题欢迎在 [FunASR GitHub Issues](https://github.com/alibaba-damo-academy/FunASR/issues) 反馈。*
