#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Keyword screening: load a categorized keyword dictionary and scan transcript sentences.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_keyword_dict(path: str) -> Dict[str, List[str]]:
    """
    Load keyword dictionary from JSON file.

    Expected format:
        {
            "决议": ["决定", "通过", "批准"],
            "行动项": ["负责", "跟进", "落实"],
            ...
        }
    Also accepts list-style aliases as values (first entry is canonical keyword).
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Keyword dict not found: %s — skipping keyword screening", path)
        return {}

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Keyword dict must be a JSON object, got {type(data)}")

    # Normalize: values can be list of strings or list of lists (aliases)
    normalized: Dict[str, List[str]] = {}
    for category, keywords in data.items():
        if not isinstance(keywords, list):
            raise ValueError(f"Category '{category}' must map to a list")
        flat = []
        for kw in keywords:
            if isinstance(kw, str):
                flat.append(kw)
            elif isinstance(kw, list):
                flat.extend(kw)
        normalized[category] = flat

    total = sum(len(v) for v in normalized.values())
    logger.info("Loaded %d keywords in %d categories from %s", total, len(normalized), path)
    return normalized


def scan_sentences(
    sentences: List[dict],
    keyword_dict: Dict[str, List[str]],
    hotword_boost_path: Optional[str] = None,
) -> List[dict]:
    """
    Scan a list of sentence dicts for keyword matches.

    Args:
        sentences: list of assembled sentence dicts (with global spk, abs times, text).
        keyword_dict: {category: [keyword, ...]} from load_keyword_dict.
        hotword_boost_path: optional path to a plain-text hotword list for FunASR
                            (not used here, reserved for pipeline caller).

    Returns:
        list of hit dicts: {category, keyword, spk, start_ms, end_ms, text}
    """
    if not keyword_dict:
        return []

    hits = []
    for sent in sentences:
        text = sent.get("text", "")
        if not text:
            continue
        for category, keywords in keyword_dict.items():
            for kw in keywords:
                if kw and kw in text:
                    hits.append(
                        {
                            "category": category,
                            "keyword": kw,
                            "spk": sent.get("spk", -1),
                            "start_ms": sent.get("start", 0),
                            "end_ms": sent.get("end", 0),
                            "text": text,
                        }
                    )
    logger.info("Keyword scan: %d hits across %d sentences", len(hits), len(sentences))
    return hits


def build_hotword_string(keyword_dict: Dict[str, List[str]], sep: str = " ") -> str:
    """Flatten all keywords into a space-separated string for FunASR hotword argument."""
    all_kw = []
    for kws in keyword_dict.values():
        all_kw.extend(kws)
    return sep.join(all_kw)


def format_hits_for_prompt(hits: List[dict], max_hits: int = 50) -> str:
    """Format keyword hits as a concise text block for the LLM prompt."""
    if not hits:
        return "(无关键词命中)"
    lines = []
    for h in hits[:max_hits]:
        spk = h.get("spk", "?")
        start_s = h.get("start_ms", 0) / 1000
        lines.append(
            f"[{h['category']}] 「{h['keyword']}」 说话人{spk} "
            f"@{start_s:.1f}s: {h['text'][:60]}"
        )
    if len(hits) > max_hits:
        lines.append(f"... 及 {len(hits) - max_hits} 条更多命中")
    return "\n".join(lines)
