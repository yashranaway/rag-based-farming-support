from __future__ import annotations

from typing import List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class Location(BaseModel):
    gps: Optional[Tuple[float, float]] = Field(default=None, description="(lat, lon)")
    pincode: Optional[str] = None
    district: Optional[str] = None


class UserPreferences(BaseModel):
    language: Optional[str] = None
    verbosity: Optional[str] = Field(default="basic", description="basic|detailed")

    @field_validator("verbosity")
    @classmethod
    def validate_verbosity(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"basic", "detailed"}
        if v not in allowed:
            raise ValueError(f"verbosity must be one of {allowed}")
        return v


class Citation(BaseModel):
    title: str
    url: Optional[str] = None
    timestamp: Optional[datetime] = None


class Diagnostics(BaseModel):
    latency_ms: Optional[int] = None
    tokens_prompt: Optional[int] = None
    tokens_output: Optional[int] = None
    retrieval_k: Optional[int] = None


class QueryRequest(BaseModel):
    text: str
    locale: Optional[str] = None
    location: Optional[Location] = None
    preferences: Optional[UserPreferences] = None


class AnswerResponse(BaseModel):
    answer: str
    language: str
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    diagnostics: Optional[Diagnostics] = None


# Admin models
class ReindexRequest(BaseModel):
    text: str
    region: Optional[str] = None
    crop: Optional[str] = None
    source_url: Optional[str] = None
    max_chars: Optional[int] = 800
    overlap: Optional[int] = 100


class TemplateSetRequest(BaseModel):
    content: str


class TemplateRollbackRequest(BaseModel):
    version: int
