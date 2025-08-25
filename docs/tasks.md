# Implementation Plan

- [x] 1. Set up project structure and core services — testing complete
  - Create FastAPI service skeleton with endpoints: `/v1/query`, `/v1/sources`, `/v1/admin/reindex`, `/v1/healthz`, `/v1/readyz`
  - Wire streaming responses and request validation schemas
  - _Requirements: 11.1, 11.2, 11.3, 9.2_

- [x] 1.1 Create API request/response models — testing complete
  - Implement `QueryRequest`, `AnswerResponse`, `Citation`, `Diagnostics`, `Location`, `UserPreferences` in `api/models.py`
  - Unit tests for schema validation and serialization
  - _Requirements: 1.1–1.5, 2.1–2.5, 6.1–6.3, 7.1–7.3, 8.1–8.3, 10.1–10.3_

- [x] 1.2 Implement health/readiness probes — testing complete
  - `/v1/healthz` basic liveness; `/v1/readyz` checks vector store and model connectivity (mocked)
  - Tests for HTTP status and JSON payload
  - _Requirements: 11.1_

- [x] 2. Language detection and localization — testing complete
  - Implement language detection utility (e.g., fasttext/langid wrapper) with confidence threshold
  - Add locale normalization and auto-reply-in-same-language policy
  - Unit tests for detection thresholds and fallbacks
  - _Requirements: 1.1–1.3, 8.2_

- [x] 3. Location resolution utilities — testing complete
  - Implement GPS/pincode/district normalization and region tagging
  - Fallback default region when missing with user prompt requirements
  - Unit tests for parsing and edge cases
  - _Requirements: 2.5, 3.1, 4.1_

- [x] 4. Data connectors (mock-first) — testing complete
  - Define interfaces and mock clients for Weather (IMD), Mandi (eNAM), Soil defaults, Govt advisories
  - Implement caching layer and timeouts in each connector
  - Integration tests with mocked responses, including timeout/failure paths
  - _Requirements: 3.1–3.3, 6.1–6.3, 7.1–7.3, 11.1, 11.2_

- [x] 5. Ingestion pipeline — testing complete
  - Implement Docling-based parser for PDF/HTML → text/tables/images
  - Implement Granite Vision captioning for images (behind feature flag)
  - Chunking (token-aware), metadata enrichment (region/crop/date/authority)
  - Unit and integration tests for parsing, chunking, and metadata tagging
  - _Requirements: 2.1–2.5, 7.2, 11.1, 12.2, 13.2_

- [x] 6. Embeddings and Vector Store — testing complete
  - Integrate Granite embeddings via HuggingFace; build vector upsert functions
  - Implement Milvus/OpenSearch adapter behind interface; choose provider by env var
  - Tests: upsert, similarity search, filtered retrieval by region/crop/date
  - _Requirements: 2.1–2.5, 9.2, 12.2_

- [x] 7. Retriever and ranking — testing complete
  - Implement retriever interface with top-k, filters, freshness weighting
  - Optional re-ranker stub (feature flag)
  - Unit tests for ranking/freshness behavior
  - _Requirements: 2.1–2.5, 12.2_

- [x] 8. Prompt builder — testing complete
  - Implement `TokenizerChatPromptTemplate`-based builder with:
    - System policies (safety, tone, language)
    - Retrieved snippets with citation metadata
    - External signals (weather snapshot, price list) compacted to fit budget
  - Token budgeting utilities (truncate/rank chunks to cap context window)
  - Unit tests for template assembly and budgeting
  - _Requirements: 1.3–1.5, 2.1–2.4, 3.2, 12.2, 10.3_

- [x] 9. Granite LLM adapter — testing complete
  - Implement provider abstraction for watsonx.ai Granite and Replicate Granite
  - Streaming output support; configurable temperature/max_tokens/stop
  - Contract tests (stubbed providers); error handling for quota/credit failures
  - _Status: Adapters completed with streaming, parameters (temperature/max_tokens/stop), and simulated error modes; comprehensive tests added_
  - _Requirements: 9.1, 11.2, 12.1, 12.3_

- [x] 10. Query orchestration — testing complete
  - Implement end-to-end flow: classify intent → fetch external signals → retrieve → prompt → stream answer
  - Safety intercepts for unsafe content (pesticide usage warnings)
  - Integration tests for streaming and safety
  - _Status: Full orchestration implemented with intent classification, external signals (weather/mandi), streaming endpoint, and safety intercept; tests passing_
  - _Requirements: 1.4–1.5, 2.1–2.5, 3.1–3.3, 4.1–4.3, 5.1–5.3, 6.1–6.3, 7.1–7.3, 10.3, 11.1–11.3, 12.1–12.3_

- [ ] 11. Admin endpoints
  - `POST /v1/admin/reindex` to trigger ingestion for a source; validate configs
  - Versioned prompt/policy templates and rollback support
  - Tests for admin flows and validation errors
  - _Requirements: 13.1–13.3_

- [ ] 12. Observability and logging (partial)
  - Add structured logs with correlation IDs; redact PII
  - Capture slow spans and retrieval/generation diagnostics in responses
  - Tests for log fields and diagnostics shape
  - _Status: trace id + latency headers added for /v1/query; structured logging/tests pending_
  - _Requirements: 10.1, 11.2, 11.3_

- [ ] 13. Evaluation harness
  - Implement RAG evaluation suite for groundedness, relevance, citation accuracy
  - Seed curated QA set per region/crop; run on CI
  - Tests asserting minimum thresholds and regression prevention
  - _Requirements: 14.2_

- [ ] 14. Safety tests
  - Red-team prompts for unsafe pesticide guidance; expect warnings and safe alternatives
  - Add CI checks for safety policy changes
  - _Requirements: 5.2, 10.3, 14.3_

- [ ] 15. Performance/cost controls
  - First-token streaming target checks; rate limit middleware
  - Token budgeting across retrieval and generation; enforce caps
  - Tests for throttling and budgeting behavior
  - _Requirements: 12.1–12.3_

- [ ] 16. IBM Cloud Lite readiness
  - Configuration for Code Engine deployment; COS storage paths; Secrets Manager integration
  - Provide adapters for Milvus-on-Code-Engine vs OpenSearch
  - Smoke tests for Lite environment (using mocks/no external billing)
  - _Requirements: 9.2, 9.3_
