from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from app.services.ingestion import Chunk


@dataclass
class BuiltPrompt:
    prompt: str
    citations: List[Dict[str, str]]


def _estimate_tokens(text: str) -> int:
    # Crude token estimate ~ 4 chars per token
    return max(len(text) // 4, 1)


class PromptBuilder:
    """Simple prompt builder that composes user question with retrieved chunks.
    Includes basic metadata to aid traceability.
    """

    def __init__(self, *, system_preamble: Optional[str] = None, language: Optional[str] = None) -> None:
        self.system_preamble = system_preamble or (
            "You are a smart farming assistant. Provide actionable, safe, and concise advice."
        )
        self.language = language or "auto"

    def build(
        self,
        question: str,
        chunks: List[Chunk],
        *,
        max_context_chars: int = 2000,
        max_context_tokens: Optional[int] = None,
        external_signals: Optional[Dict[str, Any]] = None,
    ) -> BuiltPrompt:
        context_parts: List[str] = []
        citations: List[Dict[str, str]] = []
        used_chars = 0
        used_tokens = 0
        for ch in chunks:
            part = ch.text.strip()
            if not part:
                continue
            if used_chars + len(part) > max_context_chars:
                break
            # respect token budget if provided
            t = _estimate_tokens(part)
            if max_context_tokens is not None and used_tokens + t > max_context_tokens:
                # Try to truncate the part to fit remaining tokens
                remaining = max(0, max_context_tokens - used_tokens)
                approx_chars = remaining * 4
                if approx_chars <= 0:
                    break
                part = part[:approx_chars].rstrip()
                t = _estimate_tokens(part)
            context_parts.append(part)
            used_chars += len(part)
            used_tokens += t
            citations.append(
                {
                    "doc_id": ch.doc_id,
                    "chunk_id": ch.id,
                    "source_url": ch.metadata.get("source_url", ""),
                    "region": ch.metadata.get("region", ""),
                    "crop": ch.metadata.get("crop", ""),
                    "chunk_index": ch.metadata.get("chunk_index", ""),
                }
            )
        ctx = "\n\n".join(context_parts)

        signals_text = ""
        if external_signals:
            # Compact serialization of external signals
            lines: List[str] = []
            for k, v in external_signals.items():
                if isinstance(v, (dict, list)):
                    s = str(v)
                else:
                    s = f"{v}"
                # Truncate very long lines
                if len(s) > 400:
                    s = s[:400] + "…"
                lines.append(f"- {k}: {s}")
            signals_text = "\n".join(lines)

        prompt = (
            f"[lang={self.language}]\n"
            f"System: {self.system_preamble}\n\n"
            f"Context:\n{ctx}\n\n"
            + (f"External Signals:\n{signals_text}\n\n" if signals_text else "")
            +
            f"User Question: {question}\n"
            f"Answer in the specified language, cite sources by doc_id and chunk_index."
        )
        return BuiltPrompt(prompt=prompt, citations=citations)
