#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Pluggable LLM client for generating structured meeting minutes.

Default: KimiClient — Moonshot AI Kimi K2 via OpenAI-compatible API.
Also available: ClaudeClient (provider: "claude"), OpenAICompatClient (provider: "openai_compat").
"""

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schema for structured output
# ---------------------------------------------------------------------------
MINUTES_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "会议/课堂主题"},
        "date": {"type": "string", "description": "推断或留空"},
        "attendees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "spk": {"type": "integer"},
                    "name_or_role": {"type": "string"},
                },
                "required": ["spk"],
            },
            "description": "说话人列表(编号/推断角色)",
        },
        "summary": {"type": "string", "description": "50-150字总结"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "detail": {"type": "string"},
                    "related_spks": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["topic", "detail"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "明确决议列表",
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner_spk": {"type": "integer"},
                    "task": {"type": "string"},
                    "due": {"type": "string"},
                },
                "required": ["task"],
            },
        },
        "risks_or_open_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "keyword_highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "keyword": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["category", "keyword", "context"],
            },
        },
    },
    "required": ["title", "summary", "topics"],
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class LLMClient(ABC):
    @abstractmethod
    def summarize(
        self,
        transcript_text: str,
        keyword_hits: List[dict],
        system_prompt: str,
        user_prompt_template: str,
    ) -> Dict[str, Any]:
        """Return structured minutes dict matching MINUTES_SCHEMA."""
        ...


# ---------------------------------------------------------------------------
# Claude implementation
# ---------------------------------------------------------------------------
class ClaudeClient(LLMClient):
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 16000,
        cache_dir: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _cache_key(self, transcript: str, system_prompt: str) -> str:
        raw = f"{transcript}\x00{system_prompt}\x00{self.model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:20]

    def _load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"llm_{key}.json"
        if p.exists():
            logger.info("LLM cache hit: %s", key)
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, result: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.cache_dir / f"llm_{key}.json"
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def summarize(
        self,
        transcript_text: str,
        keyword_hits: List[dict],
        system_prompt: str,
        user_prompt_template: str,
    ) -> Dict[str, Any]:
        from keywords import format_hits_for_prompt

        keyword_section = format_hits_for_prompt(keyword_hits)
        user_msg = user_prompt_template.format(
            transcript=transcript_text,
            keyword_hits=keyword_section,
        )

        cache_key = self._cache_key(transcript_text, system_prompt)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # Use streaming for long transcripts to avoid timeout
        use_stream = len(transcript_text) > 8000

        system_block = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        create_kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system_block,
            messages=[{"role": "user", "content": user_msg}],
            betas=["output-128k-2025-02-19"],
        )

        logger.info("Calling Claude %s (stream=%s) for structured minutes …", self.model, use_stream)

        if use_stream:
            with self._client.messages.stream(**create_kwargs) as stream:
                response = stream.get_final_message()
        else:
            response = self._client.messages.create(**create_kwargs)

        # Extract text from response
        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text += block.text

        # Parse JSON
        result = _parse_json_from_response(result_text)
        self._save_cache(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# Kimi K2 (Moonshot AI) — DEFAULT
# ---------------------------------------------------------------------------
class KimiClient(LLMClient):
    """
    Kimi K2 from Moonshot AI via OpenAI-compatible API.
    Requires: pip install openai
    API key:  MOONSHOT_API_KEY environment variable
    """

    KIMI_BASE_URL = "https://api.moonshot.cn/v1"

    def __init__(
        self,
        model: str = "kimi-k2",
        max_tokens: int = 8192,
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not key:
            raise ValueError(
                "Kimi K2 requires MOONSHOT_API_KEY environment variable "
                "(get one at https://platform.moonshot.cn/)"
            )
        self._client = OpenAI(base_url=self.KIMI_BASE_URL, api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_key(self, transcript: str, system_prompt: str) -> str:
        raw = f"{transcript}\x00{system_prompt}\x00{self.model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:20]

    def _load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"llm_{key}.json"
        if p.exists():
            logger.info("LLM cache hit: %s", key)
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, result: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.cache_dir / f"llm_{key}.json"
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def summarize(
        self,
        transcript_text: str,
        keyword_hits: List[dict],
        system_prompt: str,
        user_prompt_template: str,
    ) -> Dict[str, Any]:
        from keywords import format_hits_for_prompt

        keyword_section = format_hits_for_prompt(keyword_hits)
        user_msg = user_prompt_template.format(
            transcript=transcript_text,
            keyword_hits=keyword_section,
        )

        cache_key = self._cache_key(transcript_text, system_prompt)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        use_stream = len(transcript_text) > 8000
        logger.info(
            "Calling Kimi %s (stream=%s) for structured minutes …", self.model, use_stream
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        if use_stream:
            result_text = ""
            with self._client.chat.completions.create(stream=True, **kwargs) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        result_text += delta
        else:
            resp = self._client.chat.completions.create(**kwargs)
            result_text = resp.choices[0].message.content or ""

        result = _parse_json_from_response(result_text)
        self._save_cache(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# OpenAI-compatible (DashScope, local vLLM, Ollama, etc.)
# ---------------------------------------------------------------------------
class OpenAICompatClient(LLMClient):
    """
    Generic OpenAI-compatible client for DashScope, vLLM, Ollama, etc.
    Requires: pip install openai
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
        cache_dir: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        key = api_key or os.environ.get("LLM_API_KEY", "none")
        self._client = OpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_key(self, transcript: str, system_prompt: str) -> str:
        raw = f"{transcript}\x00{system_prompt}\x00{self.model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:20]

    def _load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"llm_{key}.json"
        if p.exists():
            logger.info("LLM cache hit: %s", key)
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, result: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.cache_dir / f"llm_{key}.json"
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def summarize(
        self,
        transcript_text: str,
        keyword_hits: List[dict],
        system_prompt: str,
        user_prompt_template: str,
    ) -> Dict[str, Any]:
        from keywords import format_hits_for_prompt

        keyword_section = format_hits_for_prompt(keyword_hits)
        user_msg = user_prompt_template.format(
            transcript=transcript_text,
            keyword_hits=keyword_section,
        )

        cache_key = self._cache_key(transcript_text, system_prompt)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        use_stream = len(transcript_text) > 8000
        logger.info(
            "Calling OpenAI-compat %s (stream=%s) for structured minutes …",
            self.model,
            use_stream,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.3,
        )
        # response_format is widely supported but not universal; try with it
        try:
            kwargs_json = dict(kwargs, response_format={"type": "json_object"})
            if use_stream:
                result_text = ""
                with self._client.chat.completions.create(stream=True, **kwargs_json) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            result_text += delta
            else:
                resp = self._client.chat.completions.create(**kwargs_json)
                result_text = resp.choices[0].message.content or ""
        except Exception:
            # Fallback: omit response_format for providers that don't support it
            if use_stream:
                result_text = ""
                with self._client.chat.completions.create(stream=True, **kwargs) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            result_text += delta
            else:
                resp = self._client.chat.completions.create(**kwargs)
                result_text = resp.choices[0].message.content or ""

        result = _parse_json_from_response(result_text)
        self._save_cache(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_llm_client(config: dict, cache_dir: Optional[str] = None) -> LLMClient:
    """Instantiate LLM client from config dict."""
    provider = config.get("provider", "kimi").lower()
    if provider == "kimi":
        return KimiClient(
            model=config.get("model", "kimi-k2"),
            max_tokens=config.get("max_tokens", 8192),
            cache_dir=cache_dir,
            api_key=config.get("api_key") or os.environ.get("MOONSHOT_API_KEY"),
        )
    elif provider == "claude":
        return ClaudeClient(
            model=config.get("model", "claude-opus-4-8"),
            max_tokens=config.get("max_tokens", 16000),
            cache_dir=cache_dir,
        )
    elif provider in ("openai", "openai_compat", "dashscope", "vllm", "ollama"):
        return OpenAICompatClient(
            base_url=config["base_url"],
            model=config["model"],
            api_key=config.get("api_key") or os.environ.get("LLM_API_KEY"),
            max_tokens=config.get("max_tokens", 8192),
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r} — expected 'kimi', 'claude', or 'openai_compat'")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_minutes_markdown(minutes: dict, audio_path: str = "") -> str:
    """Render structured minutes dict as a reviewable Markdown draft."""
    lines = [
        "> **AI 生成草稿，需人工复核。** 内容由大模型生成，可能存在遗漏或错误。",
        "",
        f"# {minutes.get('title', '会议纪要')}",
    ]
    if minutes.get("date"):
        lines.append(f"\n**日期**: {minutes['date']}")
    if audio_path:
        lines.append(f"\n**音频来源**: `{audio_path}`")
    lines.append("")

    if minutes.get("attendees"):
        lines.append("## 参与者")
        for a in minutes["attendees"]:
            role = a.get("name_or_role", "")
            lines.append(f"- 说话人 {a['spk']}" + (f" ({role})" if role else ""))
        lines.append("")

    lines.append("## 摘要")
    lines.append(minutes.get("summary", ""))
    lines.append("")

    if minutes.get("topics"):
        lines.append("## 议题")
        for t in minutes["topics"]:
            spks = t.get("related_spks", [])
            spk_str = f" _(说话人 {', '.join(str(s) for s in spks)})_" if spks else ""
            lines.append(f"### {t['topic']}{spk_str}")
            lines.append(t.get("detail", ""))
            lines.append("")

    if minutes.get("decisions"):
        lines.append("## 决议")
        for d in minutes["decisions"]:
            lines.append(f"- {d}")
        lines.append("")

    if minutes.get("action_items"):
        lines.append("## 行动项")
        for ai in minutes["action_items"]:
            owner = f"说话人{ai['owner_spk']} " if ai.get("owner_spk") is not None else ""
            due = f" (截止: {ai['due']})" if ai.get("due") else ""
            lines.append(f"- {owner}{ai['task']}{due}")
        lines.append("")

    if minutes.get("risks_or_open_questions"):
        lines.append("## 风险 / 待讨论")
        for r in minutes["risks_or_open_questions"]:
            lines.append(f"- {r}")
        lines.append("")

    if minutes.get("keyword_highlights"):
        lines.append("## 关键词亮点")
        for kh in minutes["keyword_highlights"]:
            lines.append(f"- **[{kh['category']}] {kh['keyword']}**: {kh['context']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json_from_response(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response text (may be wrapped in code block)."""
    text = text.strip()
    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from LLM response: %s\nRaw: %.500s", e, text)
        # Return partial structure to avoid crashing
        return {"title": "解析失败", "summary": text[:500], "topics": [], "_parse_error": str(e)}
