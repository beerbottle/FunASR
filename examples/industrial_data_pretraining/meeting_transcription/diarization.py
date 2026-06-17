#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Global speaker re-clustering across windows using scipy hierarchical clustering.

FunASR's ClusterBackend collapses all speakers to label 0 when given fewer than
20 embeddings (cluster_backend.py:224-225). Since cross-window centroids for a
whole meeting are typically < 20, we use scipy instead, which has no such limit.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


def recluster_speakers(
    window_centroids: List[Tuple[int, int, np.ndarray]],
    merge_thr: float = 0.78,
    preset_spk_num: Optional[int] = None,
    min_spk_duration_s: float = 3.0,
    window_sentence_info: Optional[Dict[int, List[dict]]] = None,
) -> Dict[Tuple[int, int], int]:
    """
    Map (window_id, local_spk) → global_spk using scipy average-linkage clustering.

    Args:
        window_centroids: list of (window_id, local_spk, centroid_vec[192]) tuples.
        merge_thr: cosine similarity threshold; speakers with similarity ≥ merge_thr
                   are merged. Mirrors ClusterBackend.merge_by_cos semantics.
        preset_spk_num: if given, force exactly this many speakers (maxclust criterion).
        min_spk_duration_s: filter centroids with < this many seconds of total owned
                             sentences before clustering; these are re-assigned via
                             nearest centroid after clustering.
        window_sentence_info: {window_id: sentence_info_list} used only when
                              min_spk_duration_s > 0 to compute owned durations.

    Returns:
        remap dict: {(window_id, local_spk): global_spk_int}
    """
    if not window_centroids:
        return {}

    # Build index arrays
    meta = [(wid, lspk) for wid, lspk, _ in window_centroids]
    C = np.stack([c for _, _, c in window_centroids], axis=0).astype(np.float64)

    if C.shape[0] == 1:
        return {meta[0]: 0}

    # L2-normalize
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    Cn = C / norms

    # Identify short-duration centroids to handle separately
    keep_mask = np.ones(len(meta), dtype=bool)
    if min_spk_duration_s > 0 and window_sentence_info:
        durations = _compute_centroid_durations(meta, window_sentence_info)
        keep_mask = np.array([durations.get(m, 0) >= min_spk_duration_s * 1000 for m in meta])
        if keep_mask.sum() < 1:
            keep_mask[:] = True  # all short: cluster everything

    main_idx = np.where(keep_mask)[0]
    short_idx = np.where(~keep_mask)[0]

    remap: Dict[Tuple[int, int], int] = {}

    if main_idx.shape[0] == 1:
        remap[meta[main_idx[0]]] = 0
        base_labels = np.array([0])
        main_Cn = Cn[main_idx]
    else:
        main_Cn = Cn[main_idx]
        dist_vec = pdist(main_Cn, metric="cosine")
        dist_vec = np.clip(dist_vec, 0.0, 2.0)
        Z = linkage(dist_vec, method="average")

        if preset_spk_num is not None:
            raw_labels = fcluster(Z, t=preset_spk_num, criterion="maxclust")
        else:
            t = 1.0 - merge_thr  # cosine distance threshold
            raw_labels = fcluster(Z, t=t, criterion="distance")

        # Re-index from 0
        unique, inv = np.unique(raw_labels, return_inverse=True)
        base_labels = inv  # 0-based

        for i, idx in enumerate(main_idx):
            remap[meta[idx]] = int(base_labels[i])

    # Assign short-duration centroids to nearest main centroid by cosine
    if short_idx.shape[0] > 0:
        for idx in short_idx:
            sims = main_Cn @ Cn[idx]
            nearest = int(np.argmax(sims))
            remap[meta[idx]] = int(base_labels[nearest])

    logger.info(
        "Global re-clustering: %d (window,spk) pairs → %d global speakers",
        len(meta),
        len(set(remap.values())),
    )
    return remap


def _compute_centroid_durations(
    meta: List[Tuple[int, int]],
    window_sentence_info: Dict[int, List[dict]],
) -> Dict[Tuple[int, int], float]:
    """Sum sentence durations (ms) owned by each (window_id, local_spk)."""
    durations: Dict[Tuple[int, int], float] = {}
    for wid, lspk in meta:
        key = (wid, lspk)
        durations[key] = 0.0
    for wid, sentences in window_sentence_info.items():
        for sent in sentences:
            key = (wid, int(sent["spk"]))
            if key in durations:
                durations[key] = durations.get(key, 0.0) + (
                    sent.get("end", 0) - sent.get("start", 0)
                )
    return durations


def remap_sentences(
    sentences: List[dict],
    window_id: int,
    remap: Dict[Tuple[int, int], int],
    window_start_ms: int,
    core_start_ms: int,
    core_end_ms: int,
) -> List[dict]:
    """
    Convert per-window sentence_info to absolute times and apply global speaker labels.
    Emits only sentences whose midpoint falls inside [core_start_ms, core_end_ms).

    Args:
        sentences: sentence_info list from one window (times are window-relative ms).
        window_id: index of this window.
        remap: output of recluster_speakers.
        window_start_ms: absolute start of window in ms (add to all times).
        core_start_ms: absolute start of owned zone.
        core_end_ms: absolute end of owned zone.

    Returns:
        list of sentence dicts with absolute times and global spk labels.
    """
    out = []
    for sent in sentences:
        abs_start = int(sent["start"]) + window_start_ms
        abs_end = int(sent["end"]) + window_start_ms
        midpoint = (abs_start + abs_end) / 2.0
        if not (core_start_ms <= midpoint < core_end_ms):
            continue

        local_spk = int(sent["spk"])
        global_spk = remap.get((window_id, local_spk), local_spk)

        new_sent = dict(sent)
        new_sent["start"] = abs_start
        new_sent["end"] = abs_end
        new_sent["spk"] = global_spk
        new_sent["local_spk"] = local_spk
        new_sent["window_id"] = window_id

        # Shift inner word-level timestamps if present
        if "timestamp" in new_sent and new_sent["timestamp"]:
            new_sent["timestamp"] = [
                [t[0] + window_start_ms, t[1] + window_start_ms]
                for t in new_sent["timestamp"]
            ]

        out.append(new_sent)
    return out


def merge_adjacent_same_speaker(
    sentences: List[dict], gap_ms: int = 500
) -> List[dict]:
    """Merge consecutive sentences with the same global speaker if gap < gap_ms."""
    if not sentences:
        return sentences
    merged = [dict(sentences[0])]
    for sent in sentences[1:]:
        prev = merged[-1]
        if (
            sent["spk"] == prev["spk"]
            and sent["start"] - prev["end"] < gap_ms
        ):
            prev["end"] = sent["end"]
            prev["text"] = prev["text"] + sent["text"]
            if "timestamp" in prev and "timestamp" in sent and prev["timestamp"] and sent["timestamp"]:
                prev["timestamp"] = prev["timestamp"] + sent["timestamp"]
        else:
            merged.append(dict(sent))
    return merged
