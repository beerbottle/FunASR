# 会议/课堂录音 → 结构化纪要

VAD → 本地 ASR (FunASR/Paraformer) → 说话人分离 (CAM++) → 关键词粗筛 → **大模型结构化纪要**

面向日常工作/学习场景，输出供人复核的草稿。关键词词典与提示词可按业务自定义。

## 特性

- **本地 ASR + 声纹分离**：FunASR `paraformer-zh` + `fsmn-vad` + `ct-punc` + `cam++`，无需上传音频
- **全局说话人一致性**：跨窗口 scipy 层次聚类，绕开 ClusterBackend 的 <20 行缺陷
- **崩溃可续跑**：每窗原子落盘，重启跳过已完成窗口
- **可插拔 LLM**：**Kimi K2 (`kimi-k2`) 为默认**，Claude / 任意 OpenAI-compatible 接口均可切换
- **关键词留痕**：外置词典，命中记录含说话人/时间戳

## 安装

```bash
# FunASR（如未安装）
pip install funasr

# 额外依赖（openai SDK 兼容 Kimi K2）
pip install openai pyyaml
# scipy 已是 FunASR 的依赖，无需单独安装

# 如需使用 Claude：
pip install anthropic
```

## 快速开始

```bash
# 1. 仅转写（不调 LLM）
python run.py --audio meeting.wav

# 2. 带 LLM 纪要（Kimi K2，默认）
export MOONSHOT_API_KEY=sk-...        # 在 https://platform.moonshot.cn/ 获取
python run.py --audio meeting.wav --config config.example.yaml

# 编辑 config.example.yaml 中 llm.enabled: true 后运行
```

## 输出文件

```
output/
  transcript.json      # 结构化句子列表（含说话人/时间/文字）
  transcript.txt       # 人读版转写稿
  keyword_hits.json    # 关键词命中记录
  minutes.json         # 结构化纪要（需 LLM）
  minutes.md           # 可读纪要草稿（需 LLM）

work_dir/<audio_key>/
  manifest.json        # 分窗状态（续跑用）
  window_0.json        # 各窗 ASR 结果 + 说话人质心
  window_1.json
  ...
  transcript.json/.txt # 同 output/ 的副本
```

## 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml（至少设 llm.enabled: true）
python run.py --audio meeting.wav --config config.yaml
```

主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `asr.model` | `paraformer-zh` | ASR 模型 |
| `pipeline.window_s` | 自动(GPU:600s/CPU:180s) | 分窗大小 |
| `pipeline.merge_thr` | `0.78` | 说话人聚类相似度阈值 |
| `pipeline.preset_spk_num` | `null` | 指定说话人数(null=自动) |
| `llm.enabled` | `false` | 是否生成 LLM 纪要 |
| `llm.provider` | `kimi` | LLM 提供方 |
| `llm.model` | `kimi-k2` | LLM 模型 ID |
| `keywords.dict_path` | `null` | 关键词词典路径 |

## LLM 切换

### Kimi K2（默认）

```yaml
llm:
  enabled: true
  provider: "kimi"
  model: "kimi-k2"          # 也可用 "kimi-k2-0711" 等具体版本
```

```bash
export MOONSHOT_API_KEY=sk-...
```

### Claude

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

### OpenAI-compatible（DashScope / vLLM / Ollama）

```yaml
llm:
  enabled: true
  provider: "openai_compat"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-max"
```

```bash
export LLM_API_KEY=sk-...
```

## 关键词词典自定义

参考 `keywords.example.json`，按业务分类添加关键词：

```json
{
  "决议": ["决定", "通过", "批准"],
  "行动项": ["负责", "跟进", "截止"],
  "风险": ["风险", "延期", "问题"]
}
```

## 提示词自定义

编辑 `prompts/minutes_system.txt`（系统提示）和 `prompts/minutes_user.txt`（用户提示模板，含 `{transcript}` 和 `{keyword_hits}` 占位符），然后在 config 中指定路径：

```yaml
llm:
  system_prompt_path: "./prompts/minutes_system.txt"
  user_prompt_path: "./prompts/minutes_user.txt"
```

改提示词后 LLM 缓存自动失效，重新生成纪要。

## 续跑说明

中途中断后再次运行同一命令，日志会显示跳过已完成窗口：

```
11:22:33 [INFO] Loaded existing manifest (12 windows)
11:22:33 [INFO] Processing window 5/12 (600–720s) …  ← 从断点继续
```

## 说话人标签说明

- 说话人编号为整数 (0, 1, 2, ...)，与 `transcript.txt` / `minutes.md` 中的"说话人N"对应
- 跨窗口聚类基于 CAM++ 质心的余弦相似度；同一人在不同窗口会被归为同一编号
- 调参建议：`merge_thr` 越高合并越保守（更多说话人），越低合并越积极（更少说话人）
