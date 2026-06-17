#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Meeting / Classroom Recording → Structured Minutes

Usage:
    python run.py --audio meeting.wav [--config config.yaml] [--out-dir ./output]

The pipeline:
  1. Windowed ASR + Speaker diarization (FunASR / paraformer-zh + cam++)
  2. Crash-resumable disk cache per window
  3. Global speaker re-clustering (scipy)
  4. Keyword screening (optional external dict)
  5. LLM structured minutes (Claude by default, pluggable)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
    )


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_config() -> dict:
    return {
        "asr": {
            "model": "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
            "spk_model": "cam++",
            "language": "auto",
            "use_itn": True,
            "merge_vad": True,
            "merge_length_s": 15,
            "batch_size_s": 300,
        },
        "pipeline": {
            "window_s": None,
            "overlap_s": 5,
            "merge_thr": 0.78,
            "preset_spk_num": None,
        },
        "device": None,
        "work_dir": "./work_dir",
        "keywords": {"dict_path": None},
        "llm": {
            "enabled": False,
            "provider": "claude",
            "model": "claude-opus-4-8",
            "max_tokens": 16000,
            "system_prompt_path": None,
            "user_prompt_path": None,
        },
        "output": {"dir": "./output"},
        "log_level": "INFO",
    }


def _merge_configs(defaults: dict, overrides: dict) -> dict:
    result = {}
    for k, v in defaults.items():
        if k in overrides:
            if isinstance(v, dict) and isinstance(overrides[k], dict):
                result[k] = _merge_configs(v, overrides[k])
            else:
                result[k] = overrides[k]
        else:
            result[k] = v
    for k, v in overrides.items():
        if k not in result:
            result[k] = v
    return result


def _load_prompt(path: Optional[str], default_text: str) -> str:  # noqa: F821
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return default_text


# Default prompt texts (used when no external file provided)
_DEFAULT_SYSTEM_PROMPT = """你是一位专业的会议/课堂记录员。你的任务是将提供的转写文本整理成结构化纪要。
- 保持客观，忠实于原文
- 识别并列出主要议题、决议、行动项和风险点
- 关键词命中内容作为重点线索
- 输出严格遵循要求的 JSON 格式
- 对于无法确认的信息，在相应字段注明"待确认"
"""

_DEFAULT_USER_PROMPT = """请根据以下会议/课堂转写文本，生成结构化纪要（JSON格式）。

## 转写文本
{transcript}

## 关键词命中
{keyword_hits}

请按照要求的 JSON schema 输出结构化纪要。"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meeting/Classroom Recording → Structured Minutes (FunASR + LLM)"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--out-dir", default=None, help="Output directory override")
    parser.add_argument("--work-dir", default=None, help="Work/cache directory override")
    parser.add_argument(
        "--spk-num", type=int, default=None, help="Force speaker count"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Skip LLM minutes generation"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING)"
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)
    logger = logging.getLogger("run")

    # Build config
    cfg = _default_config()
    if args.config:
        cfg = _merge_configs(cfg, _load_config(args.config))
    if args.out_dir:
        cfg["output"]["dir"] = args.out_dir
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    if args.spk_num:
        cfg["pipeline"]["preset_spk_num"] = args.spk_num
    if args.no_llm:
        cfg["llm"]["enabled"] = False

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------- Step 1: ASR + Diarization ---------------
    from pipeline import MeetingTranscriber

    transcriber = MeetingTranscriber(
        model_tag=cfg["asr"]["model"],
        vad_model=cfg["asr"]["vad_model"],
        punc_model=cfg["asr"]["punc_model"],
        spk_model=cfg["asr"]["spk_model"],
        device=cfg.get("device"),
        work_dir=cfg["work_dir"],
        window_s=cfg["pipeline"].get("window_s"),
        overlap_s=cfg["pipeline"].get("overlap_s", 5),
        batch_size_s=cfg["asr"].get("batch_size_s", 300),
        merge_thr=cfg["pipeline"].get("merge_thr", 0.78),
        preset_spk_num=cfg["pipeline"].get("preset_spk_num"),
        hotword=cfg.get("hotword", ""),
        language=cfg["asr"].get("language", "auto"),
        use_itn=cfg["asr"].get("use_itn", True),
        merge_vad=cfg["asr"].get("merge_vad", True),
        merge_length_s=cfg["asr"].get("merge_length_s", 15),
    )

    logger.info("=== Step 1: Transcription + Diarization ===")
    result = transcriber.transcribe(args.audio)
    sentences = result["sentences"]
    transcript_text = result["transcript_text"]
    logger.info(
        "Transcription done: %d sentences, %d speakers",
        len(sentences),
        result["num_speakers"],
    )

    # Copy transcript outputs to out_dir
    _copy_if_exists(Path(result["work_subdir"]) / "transcript.json", out_dir / "transcript.json")
    _copy_if_exists(Path(result["work_subdir"]) / "transcript.txt", out_dir / "transcript.txt")
    logger.info("Transcript saved to %s", out_dir)

    # --------------- Step 2: Keyword Screening ---------------
    logger.info("=== Step 2: Keyword Screening ===")
    from keywords import load_keyword_dict, scan_sentences

    kw_cfg = cfg.get("keywords", {})
    kw_dict_path = kw_cfg.get("dict_path")
    keyword_dict = load_keyword_dict(kw_dict_path) if kw_dict_path else {}
    keyword_hits = scan_sentences(sentences, keyword_dict)

    kw_out = out_dir / "keyword_hits.json"
    with open(kw_out, "w", encoding="utf-8") as f:
        json.dump(keyword_hits, f, ensure_ascii=False, indent=2)
    logger.info("Keyword hits: %d — saved to %s", len(keyword_hits), kw_out)

    # --------------- Step 3: LLM Minutes ---------------
    if cfg["llm"].get("enabled", False):
        logger.info("=== Step 3: LLM Structured Minutes ===")

        from llm_summarizer import build_llm_client, format_minutes_markdown

        llm_cfg = cfg["llm"]
        llm_client = build_llm_client(llm_cfg, cache_dir=cfg["work_dir"])

        system_prompt = _load_prompt(
            llm_cfg.get("system_prompt_path"), _DEFAULT_SYSTEM_PROMPT
        )
        user_prompt_template = _load_prompt(
            llm_cfg.get("user_prompt_path"), _DEFAULT_USER_PROMPT
        )

        minutes = llm_client.summarize(
            transcript_text=transcript_text,
            keyword_hits=keyword_hits,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )

        minutes_json_path = out_dir / "minutes.json"
        with open(minutes_json_path, "w", encoding="utf-8") as f:
            json.dump(minutes, f, ensure_ascii=False, indent=2)

        minutes_md = format_minutes_markdown(minutes, audio_path=args.audio)
        minutes_md_path = out_dir / "minutes.md"
        minutes_md_path.write_text(minutes_md, encoding="utf-8")

        logger.info("Minutes saved to %s and %s", minutes_json_path, minutes_md_path)
    else:
        logger.info("LLM minutes disabled (use --config with llm.enabled=true or omit --no-llm)")

    # --------------- Summary ---------------
    print(f"\n{'='*60}")
    print(f"Audio:       {args.audio}")
    print(f"Speakers:    {result['num_speakers']}")
    print(f"Sentences:   {len(sentences)}")
    print(f"Output dir:  {out_dir.resolve()}")
    print(f"Cache dir:   {result['work_subdir']}")
    if cfg["llm"].get("enabled"):
        print(f"Minutes:     {out_dir / 'minutes.md'}")
    print(f"{'='*60}\n")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        import shutil
        shutil.copy2(src, dst)
