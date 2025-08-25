from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, str]


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: Dict[str, str]


def _hash_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Simple paragraph/sentence aware chunking.
    - Splits by double newlines; if a block is larger than max_chars, do greedy slicing with overlap.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks: List[str] = []
    for b in blocks:
        if len(b) <= max_chars:
            chunks.append(b)
        else:
            start = 0
            while start < len(b):
                end = min(start + max_chars, len(b))
                chunks.append(b[start:end])
                if end == len(b):
                    break
                start = end - overlap
    return chunks


def enrich_metadata(
    base_meta: Optional[Dict[str, str]] = None,
    *,
    region: Optional[str] = None,
    crop: Optional[str] = None,
    authority: Optional[str] = None,
    source_url: Optional[str] = None,
    effective_date: Optional[datetime] = None,
) -> Dict[str, str]:
    meta = dict(base_meta or {})
    if region:
        meta["region"] = region.lower()
    if crop:
        meta["crop"] = crop.lower()
    if authority:
        meta["authority"] = authority
    if source_url:
        meta["source_url"] = source_url
    meta["ingested_at"] = (effective_date or datetime.now(UTC)).isoformat()
    return meta


class UpsertStore:
    """In-memory upsert interface (stub) to simulate vector/db persistence."""

    def __init__(self) -> None:
        self.docs: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}

    def upsert_document(self, text: str, metadata: Dict[str, str]) -> Document:
        doc_id = _hash_id(text, metadata.get("source_url", ""))
        doc = Document(id=doc_id, text=text, metadata=metadata)
        self.docs[doc_id] = doc
        return doc

    def upsert_chunks(self, doc: Document, parts: Iterable[str]) -> List[Chunk]:
        out: List[Chunk] = []
        for idx, part in enumerate(parts):
            chunk_id = _hash_id(doc.id, str(idx), part[:32])
            meta = dict(doc.metadata)
            meta["chunk_index"] = str(idx)
            ch = Chunk(id=chunk_id, doc_id=doc.id, text=part, metadata=meta)
            self.chunks[chunk_id] = ch
            out.append(ch)
        return out


def ingest_text(
    store: UpsertStore,
    text: str,
    *,
    region: Optional[str] = None,
    crop: Optional[str] = None,
    authority: Optional[str] = None,
    source_url: Optional[str] = None,
    effective_date: Optional[datetime] = None,
    max_chars: int = 800,
    overlap: int = 100,
) -> Tuple[Document, List[Chunk]]:
    meta = enrich_metadata(
        region=region,
        crop=crop,
        authority=authority,
        source_url=source_url,
        effective_date=effective_date,
    )
    doc = store.upsert_document(text=text, metadata=meta)
    parts = chunk_text(text, max_chars=max_chars, overlap=overlap)
    chs = store.upsert_chunks(doc, parts)
    return doc, chs
