# Design Document: Smart Farming Advice AI Agent

## Overview

This document describes the system design for the Smart Farming Advice AI Agent that delivers real-time, localized agricultural guidance to small-scale farmers using a Retrieval-Augmented Generation (RAG) pipeline powered by IBM Granite models. The system ingests trusted sources (weather, soil, crops, pest advisories, mandi prices, government updates), stores them in a vector store, and answers user queries in local languages with grounded, citeable outputs. The solution targets IBM Cloud Lite services for cost-constrained, scalable deployment, with secure secret management and observability.

Key goals derived from requirements:
- Multilingual farmer Q&A with grounded, citeable answers.
- Weather/soil-aware, market- and policy-informed recommendations.
- IBM Granite for LLM/vision; IBM Cloud Lite for hosting, storage, and secrets.
- Safety, privacy, and observability built-in.

References:
- IBM Granite models: https://www.ibm.com/granite/docs/models/granite/
- LangChain: https://python.langchain.com/
- Docling: https://docling-project.github.io/docling/
- Milvus: https://milvus.io/ and LangChain integration https://python.langchain.com/docs/integrations/vectorstores/milvus/
- eNAM (mandi prices): https://enam.gov.in/
- India Meteorological Department (IMD): https://mausam.imd.gov.in/

## Architecture

```mermaid
flowchart TD
  subgraph Client
    U[Farmer (text/voice, local language)]
  end

  subgraph API[API Layer]
    G[Gateway / HTTPS]
    A[App Service / Code Engine]
  end

  subgraph RAG[RAG Orchestrator]
    QP[Query Processor\nLang detect, locale]
    RET[Retriever\nVector DB + Filters]
    RNK[Re-Ranker (optional)]
    PT[Prompt Builder\nTokenizerChatPromptTemplate]
    LLM[IBM Granite LLM\n(Replicate or watsonx.ai)]
  end

  subgraph DataPlane[Knowledge & External Sources]
    DOCS[Doc Ingestion Pipeline\nDocling, ETL]
    VS[Vector Store\nMilvus/OpenSearch]
    MET[Metadata Store\n(COS/DB)]
    WAPI[Weather API]
    MKT[Mandi Prices API]
    SOIL[Soil dataset/API]
    GOV[Govt advisories\n(PDF/HTML feeds)]
  end

  subgraph Platform[IBM Cloud Lite]
    CE[Code Engine (API, workers)]
    COS[Cloud Object Storage]
    SM[Secrets Manager]
    LOG[Logging/Monitoring]
  end

  U -- query --> G --> A --> QP --> RET --> RNK --> PT --> LLM --> A --> G --> U
  DOCS --> VS
  DOCS --> MET
  RET --> VS
  QP --> WAPI
  QP --> MKT
  QP --> SOIL
  QP --> GOV
  A --> SM
  A --> LOG
  VS --> COS
```

Deployment options (Lite-friendly):
- Compute: IBM Code Engine (Lite) for API and background ingestion workers.
- Storage: IBM Cloud Object Storage (Lite) for raw docs, chunked JSON, artifacts.
- Secrets: IBM Secrets Manager (Lite) for API tokens (Replicate, IMD, eNAM, etc.).
- Vector DB: 
  - Option A: Milvus (embedded/standalone) running on Code Engine, with COS persistence.
  - Option B: OpenSearch/Elasticsearch with kNN plugin (if available) as vector store.
- LLM access:
  - Preferred: IBM watsonx.ai Granite endpoints (if accessible under trial/Lite).
  - Alternative: Replicate `ibm-granite` models (requires credits; cost controls required).

## Components and Interfaces

1) API Layer (FastAPI recommended)
- Endpoints:
  - `POST /v1/query` → body: `QueryRequest` → returns `AnswerResponse`
  - `GET /v1/sources` → list indexed sources and freshness
  - `POST /v1/admin/reindex` (protected) → trigger ingestion for a source
  - `GET /v1/healthz` / `GET /v1/readyz`
- Responsibilities: auth (if needed), rate limiting, request validation, streaming responses.

2) Query Processor
- Functions:
  - Language detection (fasttext/langid) and locale normalization.
  - Location resolution (GPS, pincode, district) → region filters.
  - External signals fetch: weather, prices, soil defaults.
  - Query classification (intent: weather, crop advice, pest, market, schemes).

3) Retrieval & Ranking
- Vector retriever (Milvus/OpenSearch): top-k by similarity, filters by region/crop/date.
- Optional re-ranker (cross-encoder or Granite-in-the-loop scoring) under budget.
- Freshness and authority weighting.

4) Prompt Builder
- Uses Granite chat templates (`TokenizerChatPromptTemplate`) to assemble:
  - System instructions: safety, tone, language code.
  - Retrieved snippets (truncated to context budget, with citations metadata).
  - External signals (weather snapshot, price table) as structured bullets.
  - User question.

5) LLM Adapter (Granite)
- Providers:
  - watsonx.ai Granite or Replicate `ibm-granite/granite-3.3-8b-instruct`.
- Streaming token output with cost/latency caps (max_tokens, temperature, stop).

6) Ingestion Pipeline
- Docling-based PDF/HTML parsing → text, tables, images.
- Image-to-text (Granite Vision) to caption charts/figures.
- Chunking (token-aware), metadata enrichment (region, date, source authority).
- Embedding (Granite embeddings on Hugging Face) → upsert vector store.

7) Data Connectors
- Weather: IMD or other regional API.
- Market prices: eNAM/state APIs.
- Soil: static datasets + optional local APIs.
- Government updates: RSS/HTML/PDF crawlers with date extraction.

8) Observability & Safety
- Structured logging, request/trace IDs, slow-span capture.
- Guardrails: prompt policies, unsafe content filter, pesticide safety blocks.

## Data Models

- `QueryRequest`
  - `text: str`
  - `locale: Optional[str]` (optional; auto-detect if missing)
  - `location: Optional[Location]` (gps/pincode/district)
  - `preferences: Optional[UserPreferences]`

- `AnswerResponse`
  - `answer: str`
  - `language: str`
  - `citations: List[Citation]` (title, url, timestamp)
  - `warnings: List[str]`
  - `diagnostics: Optional[Diagnostics]` (latency, tokens, selected sources)

- `Document`
  - `id: str`, `title: str`, `source: str (url/type)`
  - `region_tags: List[str]`, `crop_tags: List[str]`, `date: datetime`
  - `content: str`, `modality: enum(text, table_md, image_caption)`

- `Chunk`
  - `doc_id: str`, `chunk_id: str`, `text: str`, `tokens: int`
  - `embedding: vector`, `metadata: dict` (region, crop, date, authority)

- `RetrievalResult`
  - `chunk: Chunk`, `score: float`, `rank: int`

- `Location`
  - `gps: Optional[lat,lon]`, `pincode: Optional[str]`, `district: Optional[str]`

- `UserPreferences`
  - `language: Optional[str]`, `verbosity: enum(basic, detailed)`

- `Citation`
  - `title: str`, `url: Optional[str]`, `timestamp: Optional[datetime]`

- `Diagnostics`
  - `latency_ms: int`, `tokens_prompt: int`, `tokens_output: int`, `retrieval_k: int`

## Error Handling

- Input validation errors → 400 with message; suggest minimal info to proceed (e.g., crop and district).
- Upstream API timeouts → return partial results with disclaimer; include which sources failed.
- Retrieval empty/low-similarity → ask clarifying question; expose “insufficient context” note.
- LLM provider errors (quota/credits) → fallback to cached answers (if policy allows) or suggest retry.
- Safety violations → block unsafe advice and provide safe alternatives + source links.
- All errors logged with correlation IDs; PII minimized and anonymized.

## Testing Strategy

- Unit tests
  - Language detection & locale mapping.
  - Chunking, embeddings, and metadata tagging.
  - Retriever filters (region/crop/date) and freshness weighting.
  - Prompt assembly with token budgeting.

- Integration tests
  - Ingestion → vector store upsert → retrieval pipeline.
  - External connectors (weather, mandi) using mocked APIs.
  - Granite LLM adapter using stubbed providers and contract tests.

- RAG Evaluation
  - Curated QA set per region/crop.
  - Metrics: groundedness (source overlap), relevance, citation accuracy, refusal correctness.
  - Regression gates on PR.

- Safety Tests
  - Red-team prompts for unsafe pesticide instructions.
  - Ensure warnings and safe alternatives appear; no unsafe dosages without labels.

- Performance & Cost Tests
  - Token budgeting adherence under large contexts.
  - Streaming first-token latency target; rate-limit behavior.

## Research Findings and Design Choices

- IBM Granite: High-quality instruction models and vision support align with multimodal needs; prompt templates available via community utils and LangChain wrappers.
- Docling: Reliable PDF/HTML parsing for government advisories; supports tables/pictures and image extraction.
- Vector DB: Milvus offers strong vector ops; on Lite, deployment via Code Engine is feasible; OpenSearch kNN is a pragmatic alternative if managed service preferred.
- IBM Cloud Lite: Code Engine + COS + Secrets Manager provide a low-cost, secure baseline with autoscaling.
- External sources: IMD and eNAM commonly used in India; availability and rate limits vary—implement caching and fallbacks.

## Open Questions (to finalize in implementation planning)

- Exact list of supported languages and voice I/O scope.
- Preferred vector store option (Milvus on Code Engine vs OpenSearch).
- Granite provider path (watsonx.ai vs Replicate), cost envelope, and quotas.
- Specific government feeds/URLs per target states; PDF vs HTML mix.
- Target SLOs (P50/P95 latencies, monthly budget caps).
