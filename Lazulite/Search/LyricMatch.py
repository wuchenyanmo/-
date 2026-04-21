from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from opencc import OpenCC

from Lazulite.Align import LyricAligner
from Lazulite.Lyric import LyricLineStamp
from Lazulite.Transcribe import WhisperChunkResult, WhisperTrackResult

_OPENCC_T2S = OpenCC("t2s")


@dataclass(slots=True)
class LyricContentMatchResult:
    usable_chunk_count: int
    matched_chunk_count: int
    unique_matched_line_count: int
    avg_best_similarity: float
    confidence_weighted_similarity: float
    unique_matched_line_ratio: float
    content_match_score: float

    def to_dict(self) -> dict:
        return {
            "usable_chunk_count": self.usable_chunk_count,
            "matched_chunk_count": self.matched_chunk_count,
            "unique_matched_line_count": self.unique_matched_line_count,
            "avg_best_similarity": self.avg_best_similarity,
            "confidence_weighted_similarity": self.confidence_weighted_similarity,
            "unique_matched_line_ratio": self.unique_matched_line_ratio,
            "content_match_score": self.content_match_score,
        }


class LyricContentMatcher:
    def __init__(
        self,
        min_chunk_confidence: float = 0.42,
        max_hallucination_risk: float = 0.55,
        max_self_repeat_score: float = 0.60,
        min_similarity: float = 0.28,
    ):
        self.min_chunk_confidence = min_chunk_confidence
        self.max_hallucination_risk = max_hallucination_risk
        self.max_self_repeat_score = max_self_repeat_score
        self.min_similarity = min_similarity
        self.aligner = LyricAligner()

    @staticmethod
    def _to_simplified_text(text: str) -> str:
        normalized = LyricLineStamp.normalize_lyric_text(text)
        if not normalized:
            return ""
        return LyricLineStamp.normalize_lyric_text(_OPENCC_T2S.convert(normalized))

    def _text_similarity(self, lyric_text: str, chunk_text: str) -> float:
        direct_score = self.aligner.text_similarity(lyric_text, chunk_text)
        simplified_lyric = self._to_simplified_text(lyric_text)
        simplified_chunk = self._to_simplified_text(chunk_text)
        simplified_score = self.aligner.text_similarity(simplified_lyric, simplified_chunk)
        return max(float(direct_score), float(simplified_score))

    def _is_usable_chunk(self, chunk: WhisperChunkResult) -> bool:
        if not chunk.text:
            return False
        if float(chunk.avg_confidence or 0.0) < self.min_chunk_confidence:
            return False
        if float(chunk.hallucination_risk or 0.0) > self.max_hallucination_risk:
            return False
        if float(chunk.self_repeat_score or 0.0) > self.max_self_repeat_score:
            return False
        return True

    def score(self, lyric: LyricLineStamp, transcription: WhisperTrackResult) -> LyricContentMatchResult:
        lyric_lines = [line for line in lyric.lyric_lines if line.normalized_text]
        usable_chunks = [chunk for chunk in transcription.chunks if self._is_usable_chunk(chunk)]
        if not lyric_lines or not usable_chunks:
            return LyricContentMatchResult(
                usable_chunk_count=len(usable_chunks),
                matched_chunk_count=0,
                unique_matched_line_count=0,
                avg_best_similarity=0.0,
                confidence_weighted_similarity=0.0,
                unique_matched_line_ratio=0.0,
                content_match_score=0.0,
            )

        matched_scores: list[float] = []
        weighted_scores: list[float] = []
        matched_line_indices: set[int] = set()

        for chunk in usable_chunks:
            best_line_index = -1
            best_similarity = 0.0
            for line_index, line in enumerate(lyric_lines):
                similarity = self._text_similarity(line.normalized_text, chunk.text)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_line_index = line_index

            if best_similarity < self.min_similarity:
                continue

            matched_scores.append(float(best_similarity))
            weighted_scores.append(float(best_similarity) * float(chunk.avg_confidence or 0.0))
            if best_line_index >= 0:
                matched_line_indices.add(best_line_index)

        matched_chunk_count = len(matched_scores)
        avg_best_similarity = float(np.mean(matched_scores)) if matched_scores else 0.0
        confidence_weighted_similarity = (
            float(sum(weighted_scores) / matched_chunk_count)
            if matched_chunk_count > 0 else 0.0
        )
        unique_matched_line_ratio = len(matched_line_indices) / max(min(len(usable_chunks), len(lyric_lines)), 1)
        content_match_score = float(np.clip(
            0.70 * confidence_weighted_similarity + 0.30 * unique_matched_line_ratio,
            0.0,
            1.0,
        ))

        return LyricContentMatchResult(
            usable_chunk_count=len(usable_chunks),
            matched_chunk_count=matched_chunk_count,
            unique_matched_line_count=len(matched_line_indices),
            avg_best_similarity=avg_best_similarity,
            confidence_weighted_similarity=confidence_weighted_similarity,
            unique_matched_line_ratio=unique_matched_line_ratio,
            content_match_score=content_match_score,
        )
