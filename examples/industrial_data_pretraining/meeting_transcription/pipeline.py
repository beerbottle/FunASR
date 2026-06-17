#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
MeetingTranscriber: windowed, crash-resumable ASR + speaker diarization pipeline.

Key design choices:
- Audio split into overlap-windows; each window processed by FunASR AutoModel
  (VAD → ASR → punctuation → CAM++ diarization) in one generate() call.
- Per-window results atomically cached to disk; crash-safe via temp-then-replace.
- Startup reconcile: orphan window files adopted, running→pending reset.
- Global speaker re-clustering via scipy (bypasses ClusterBackend's <20-row bug).
- Overlap dedup: sentence emitted only when its midpoint falls in window's owned core.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diarization import merge_adjacent_same_speaker, recluster_speakers, remap_sentences

logger = logging.getLogger(__name__)

# Default window sizes (seconds)
_DEFAULT_WINDOW_GPU = 600
_DEFAULT_WINDOW_CPU = 180
_DEFAULT_OVERLAP = 5


class MeetingTranscriber:
    """
    End-to-end windowed transcription pipeline.

    Args:
        model_tag: FunASR ASR model name (e.g. "paraformer-zh").
        vad_model: VAD model name (e.g. "fsmn-vad").
        punc_model: Punctuation model name (e.g. "ct-punc").
        spk_model: Speaker model name (e.g. "cam++").
        device: "cuda:0" / "mps" / "cpu" (auto-detected if None).
        work_dir: root directory for cache files.
        window_s: window size in seconds (auto-selects by device if None).
        overlap_s: overlap in seconds at each boundary.
        batch_size_s: FunASR batch_size_s parameter.
        merge_thr: cosine similarity threshold for global speaker merging.
        preset_spk_num: force this many speakers (skips auto detection).
        hotword: space-separated hotwords for FunASR.
        language: ASR language code passed to generate().
        use_itn: whether to apply inverse text normalization.
        merge_vad: merge short VAD segments.
        merge_length_s: max merged VAD length.
    """

    def __init__(
        self,
        model_tag: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        spk_model: str = "cam++",
        device: Optional[str] = None,
        work_dir: str = "./work_dir",
        window_s: Optional[int] = None,
        overlap_s: int = _DEFAULT_OVERLAP,
        batch_size_s: int = 300,
        merge_thr: float = 0.78,
        preset_spk_num: Optional[int] = None,
        hotword: str = "",
        language: str = "auto",
        use_itn: bool = True,
        merge_vad: bool = True,
        merge_length_s: int = 15,
    ):
        self.model_tag = model_tag
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.work_dir = Path(work_dir)
        self.overlap_s = overlap_s
        self.batch_size_s = batch_size_s
        self.merge_thr = merge_thr
        self.preset_spk_num = preset_spk_num
        self.hotword = hotword
        self.language = language
        self.use_itn = use_itn
        self.merge_vad = merge_vad
        self.merge_length_s = merge_length_s

        self.device = device or _auto_device()
        self.window_s = window_s or (
            _DEFAULT_WINDOW_GPU if "cuda" in self.device or self.device == "mps"
            else _DEFAULT_WINDOW_CPU
        )

        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Full pipeline: load audio → windowed ASR → global recluster → assemble.

        Returns dict with keys:
            sentences: list of sentence dicts (abs times, global spk, text)
            transcript_text: flat text string
            num_speakers: int
            audio_key: cache key used
            work_subdir: path to cache directory for this audio
        """
        audio_path = str(Path(audio_path).resolve())
        audio, sr = self._load_audio(audio_path)
        duration_s = len(audio) / sr
        logger.info("Audio loaded: %.1fs @ %dHz from %s", duration_s, sr, audio_path)

        audio_key = self._compute_audio_key(audio_path, duration_s)
        work_subdir = self.work_dir / audio_key
        work_subdir.mkdir(parents=True, exist_ok=True)

        manifest = self._load_or_init_manifest(
            work_subdir, audio_path, audio_key, duration_s
        )
        self._reconcile(manifest, work_subdir)
        self._save_manifest(manifest, work_subdir)

        self._process_pending_windows(manifest, work_subdir, audio, sr)

        sentences = self._assemble(manifest, work_subdir)
        transcript_text = _build_transcript_text(sentences)

        # Save assembled transcript
        self._save_transcript(sentences, transcript_text, work_subdir)

        num_spk = len({s["spk"] for s in sentences}) if sentences else 0
        return {
            "sentences": sentences,
            "transcript_text": transcript_text,
            "num_speakers": num_spk,
            "audio_key": audio_key,
            "work_subdir": str(work_subdir),
        }

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        from funasr.utils.load_utils import load_audio_text_image_video

        audio = load_audio_text_image_video(audio_path, fs=16000, audio_fs=16000)
        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio, 16000

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def _compute_audio_key(self, audio_path: str, duration_s: float) -> str:
        stat = os.stat(audio_path)
        raw = (
            f"{audio_path}|{stat.st_size}|{stat.st_mtime:.3f}"
            f"|{duration_s:.3f}|{self.window_s}|{self.overlap_s}"
            f"|{self.model_tag}|{self.vad_model}|{self.spk_model}"
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _manifest_path(self, work_subdir: Path) -> Path:
        return work_subdir / "manifest.json"

    def _load_or_init_manifest(
        self,
        work_subdir: Path,
        audio_path: str,
        audio_key: str,
        duration_s: float,
    ) -> dict:
        mp = self._manifest_path(work_subdir)
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                manifest = json.load(f)
            logger.info("Loaded existing manifest (%d windows)", len(manifest.get("windows", [])))
            return manifest

        # Build window plan
        windows = _plan_windows(duration_s, self.window_s, self.overlap_s)
        manifest = {
            "audio_path": audio_path,
            "audio_key": audio_key,
            "duration_s": duration_s,
            "window_s": self.window_s,
            "overlap_s": self.overlap_s,
            "model_tag": self.model_tag,
            "windows": windows,
            "global": {"status": "stale"},
        }
        logger.info("Created new manifest: %d windows planned", len(windows))
        return manifest

    def _save_manifest(self, manifest: dict, work_subdir: Path) -> None:
        mp = self._manifest_path(work_subdir)
        tmp = mp.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp, mp)

    def _reconcile(self, manifest: dict, work_subdir: Path) -> None:
        """Adopt orphan window files; reset running→pending."""
        changed = False
        for w in manifest.get("windows", []):
            wpath = work_subdir / f"window_{w['id']}.json"
            if w["status"] == "running":
                w["status"] = "pending"
                changed = True
            elif w["status"] == "pending" and wpath.exists():
                w["status"] = "done"
                changed = True
        if changed:
            logger.info("Reconcile: adopted orphan files / reset stale running windows")

    def _mark_window_stale(self, manifest: dict) -> None:
        manifest["global"]["status"] = "stale"

    # ------------------------------------------------------------------
    # FunASR model
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from funasr import AutoModel

            logger.info(
                "Loading FunASR model %s on %s …", self.model_tag, self.device
            )
            self._model = AutoModel(
                model=self.model_tag,
                vad_model=self.vad_model,
                punc_model=self.punc_model,
                spk_model=self.spk_model,
                device=self.device,
            )
        return self._model

    # ------------------------------------------------------------------
    # Per-window processing
    # ------------------------------------------------------------------

    def _process_pending_windows(
        self,
        manifest: dict,
        work_subdir: Path,
        audio: np.ndarray,
        sr: int,
    ) -> None:
        windows = manifest["windows"]
        pending = [w for w in windows if w["status"] == "pending"]
        if not pending:
            logger.info("All windows already done — skipping ASR")
            return

        model = self._get_model()
        total = len(windows)

        for w in pending:
            i = w["id"]
            logger.info("Processing window %d/%d (%.0f–%.0fs) …", i + 1, total, w["start_s"], w["end_s"])

            start_sample = int(w["start_s"] * sr)
            end_sample = int(w["end_s"] * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) == 0:
                self._save_window(work_subdir, i, {"sentence_info": [], "centroids": []})
                w["status"] = "done"
                self._mark_window_stale(manifest)
                self._save_manifest(manifest, work_subdir)
                continue

            w["status"] = "running"
            self._save_manifest(manifest, work_subdir)

            try:
                result = model.generate(
                    input=chunk,
                    cache={},
                    language=self.language,
                    use_itn=self.use_itn,
                    batch_size_s=self.batch_size_s,
                    merge_vad=self.merge_vad,
                    merge_length_s=self.merge_length_s,
                    hotword=self.hotword if self.hotword else "",
                    return_spk_res=True,
                    return_spk_center=True,
                )

                sentence_info = result[0].get("sentence_info", []) if result else []
                centroids_raw = result[0].get("spk_embedding_center", None) if result else None

                if centroids_raw is not None and hasattr(centroids_raw, "tolist"):
                    centroids = centroids_raw.tolist()
                elif centroids_raw is not None:
                    centroids = list(centroids_raw)
                else:
                    centroids = _derive_centroids_from_sentences(sentence_info)

                window_data = {
                    "sentence_info": sentence_info,
                    "centroids": centroids,
                }
                self._save_window(work_subdir, i, window_data)
                w["status"] = "done"
                self._mark_window_stale(manifest)
                self._save_manifest(manifest, work_subdir)
                logger.info(
                    "Window %d done: %d sentences, %d speakers",
                    i,
                    len(sentence_info),
                    len(centroids),
                )

            except Exception as exc:
                logger.error("Window %d failed: %s", i, exc, exc_info=True)
                w["status"] = "failed"
                self._save_manifest(manifest, work_subdir)
                raise

    def _save_window(self, work_subdir: Path, window_id: int, data: dict) -> None:
        wpath = work_subdir / f"window_{window_id}.json"
        tmp = wpath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, wpath)

    def _load_window(self, work_subdir: Path, window_id: int) -> dict:
        wpath = work_subdir / f"window_{window_id}.json"
        with open(wpath, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, manifest: dict, work_subdir: Path) -> List[dict]:
        windows = manifest["windows"]
        total_dur_ms = int(manifest["duration_s"] * 1000)

        # Collect all centroids for global recluster
        all_centroids: List[Tuple[int, int, np.ndarray]] = []
        window_sentence_info: Dict[int, List[dict]] = {}

        for w in windows:
            if w["status"] != "done":
                continue
            wdata = self._load_window(work_subdir, w["id"])
            sinfo = wdata.get("sentence_info", [])
            centroids = wdata.get("centroids", [])
            window_sentence_info[w["id"]] = sinfo
            for local_spk, c in enumerate(centroids):
                all_centroids.append((w["id"], local_spk, np.asarray(c, dtype=np.float64)))

        if not all_centroids:
            logger.warning("No speaker centroids found — returning empty transcript")
            return []

        remap = recluster_speakers(
            all_centroids,
            merge_thr=self.merge_thr,
            preset_spk_num=self.preset_spk_num,
            min_spk_duration_s=3.0,
            window_sentence_info=window_sentence_info,
        )

        # Emit sentences with global speaker labels
        all_sentences: List[dict] = []
        for w in windows:
            if w["status"] != "done":
                continue
            start_ms = int(w["start_s"] * 1000)
            end_ms = int(w["end_s"] * 1000)
            overlap_ms = int(self.overlap_s * 1000)

            core_start_ms = start_ms + (overlap_ms if w["id"] > 0 else 0)
            core_end_ms = end_ms  # last window owns its full tail

            sinfo = window_sentence_info.get(w["id"], [])
            emitted = remap_sentences(
                sinfo,
                window_id=w["id"],
                remap=remap,
                window_start_ms=start_ms,
                core_start_ms=core_start_ms,
                core_end_ms=min(core_end_ms, total_dur_ms),
            )
            all_sentences.extend(emitted)

        all_sentences.sort(key=lambda s: s["start"])
        all_sentences = merge_adjacent_same_speaker(all_sentences, gap_ms=500)
        logger.info("Assembly done: %d sentences", len(all_sentences))
        return all_sentences

    def _save_transcript(
        self, sentences: List[dict], transcript_text: str, work_subdir: Path
    ) -> None:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        # JSON
        tj = work_subdir / "transcript.json"
        tmp = tj.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        os.replace(tmp, tj)

        # Human-readable TXT
        tt = work_subdir / "transcript.txt"
        tmp = tt.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for sent in sentences:
                text = rich_transcription_postprocess(sent.get("text", ""))
                start_s = sent["start"] / 1000
                end_s = sent["end"] / 1000
                f.write(
                    f"[说话人{sent['spk']}] [{start_s:.1f}s–{end_s:.1f}s] {text}\n"
                )
        os.replace(tmp, tt)
        logger.info("Transcript saved to %s", work_subdir)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _plan_windows(
    duration_s: float, window_s: int, overlap_s: int
) -> List[Dict[str, Any]]:
    """Return list of window dicts {id, start_s, end_s, status}."""
    windows = []
    i = 0
    start = 0.0
    while start < duration_s:
        end = min(start + window_s + overlap_s, duration_s)
        # First window starts at 0 without prefix overlap offset
        w_start = max(0.0, start - (overlap_s if i > 0 else 0))
        windows.append(
            {
                "id": i,
                "start_s": w_start,
                "end_s": end,
                "status": "pending",
            }
        )
        start += window_s
        i += 1
    return windows


def _derive_centroids_from_sentences(sentence_info: List[dict]) -> List[List[float]]:
    """Fallback: build dummy centroids (zero vectors) when model doesn't return them."""
    if not sentence_info:
        return []
    spks = {int(s["spk"]) for s in sentence_info}
    # Return 192-dim zeros per speaker; global recluster will handle them as separate
    return [[0.0] * 192 for _ in range(max(spks) + 1)]


def _build_transcript_text(sentences: List[dict]) -> str:
    """Build flat transcript string with speaker labels."""
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    lines = []
    for sent in sentences:
        text = rich_transcription_postprocess(sent.get("text", ""))
        lines.append(f"说话人{sent['spk']}: {text}")
    return "\n".join(lines)
