from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, UTC


@dataclass
class TemplateVersion:
    version: int
    content: str
    created_at: str


class TemplateRegistry:
    """In-memory versioned template store for prompt/policy templates."""

    def __init__(self) -> None:
        self._store: Dict[str, List[TemplateVersion]] = {}

    def set(self, name: str, content: str) -> TemplateVersion:
        if not content or not isinstance(content, str):
            raise ValueError("content must be a non-empty string")
        versions = self._store.setdefault(name, [])
        ver = (versions[-1].version + 1) if versions else 1
        tv = TemplateVersion(version=ver, content=content, created_at=datetime.now(UTC).isoformat())
        versions.append(tv)
        return tv

    def current(self, name: str) -> Optional[TemplateVersion]:
        versions = self._store.get(name, [])
        return versions[-1] if versions else None

    def list_versions(self, name: str) -> List[TemplateVersion]:
        return list(self._store.get(name, []))

    def rollback(self, name: str, version: int) -> TemplateVersion:
        versions = self._store.get(name, [])
        if not versions:
            raise KeyError("template not found")
        target = None
        for tv in versions:
            if tv.version == version:
                target = tv
                break
        if target is None:
            raise ValueError("version not found")
        # Append a new head identical to target to mark rollback as a new version
        new_ver = versions[-1].version + 1
        new_tv = TemplateVersion(version=new_ver, content=target.content, created_at=datetime.now(UTC).isoformat())
        versions.append(new_tv)
        return new_tv
