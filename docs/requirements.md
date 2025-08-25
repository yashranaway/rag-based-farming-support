# Requirements

## Environment Configuration

Set the following environment variables to control runtime behavior. Defaults are safe for local development.

- FEATURE_ORCHESTRATOR
  - Values: 0|1 (default: 0)
  - When 1, `/v1/query` uses the orchestrated RAG path.

- GRANITE_API_KEY
  - When set, uses `GraniteAdapter`; otherwise uses `FakeAdapter`.

- FEATURE_VISION
  - Values: 0|1 (default: 0)
  - When 1, parsing enables stub image capture in `parse_html()`.

- RETRIEVAL_PROVIDER
  - Values: `keyword` (default) | `embedding`
  - Selects `InMemoryRetriever` or embeddings-based retriever.

- VECTOR_PROVIDER
  - Values: `memory` (default) | `opensearch`
  - `memory` uses in-process store. `opensearch` requires an injected client in app wiring.

- RETRIEVAL_FRESHNESS
  - Values: 0|1 (default: 0)
  - When 1, applies exponential freshness weighting to retrieval scores.

- RETRIEVAL_RERANKER
  - Values: 0|1 (default: 0)
  - When 1, applies a simple reranker that boosts authoritative sources.

# Requirements Document

## Introduction

This document defines the initial functional and non-functional requirements for the Smart Farming Advice AI Agent. The agent provides localized, real-time agricultural guidance to small-scale farmers using a Retrieval-Augmented Generation (RAG) pipeline powered by IBM Granite models. The system ingests and retrieves trusted data sources (weather, soil, crops, pest control, mandi prices, and government updates), reasons over them using IBM Granite LLMs, and returns answers in the farmer’s local language. The solution must leverage IBM Cloud Lite services for scalable, secure, and cost-conscious deployment.

## Requirements

### Requirement 1: Farmer Q&A in Local Languages

**User Story:** As a small-scale farmer, I want to ask questions in my local language and receive accurate, easy-to-understand answers, so that I can make timely and informed farming decisions.

#### Acceptance Criteria
1. WHEN a user submits a question in any supported local language THEN the system SHALL detect the language and process the query without requiring manual language selection.
2. IF the system cannot confidently detect the language (below threshold) THEN the system SHALL ask the user to confirm or select a language.
3. WHEN generating an answer THEN the system SHALL respond in the same language as the user’s input unless the user explicitly requests another language.
4. WHEN a response is presented THEN the system SHALL include concise, step-by-step guidance suitable for low-literacy users where applicable.
5. IF the model detects potentially harmful or unsafe instructions THEN the system SHALL warn the user and suggest a safe alternative.

### Requirement 2: RAG over Trusted Agricultural Sources

**User Story:** As a farmer, I want the agent to use trusted local data sources, so that recommendations are accurate, up-to-date, and relevant to my location and crop.

#### Acceptance Criteria
1. WHEN answering a question THEN the system SHALL retrieve supporting passages from indexed sources (weather, soil, crop advisories, pest bulletins, mandi prices, government updates) and ground the final answer on them.
2. WHEN retrieval returns insufficient relevant context (below similarity threshold or empty) THEN the system SHALL inform the user and request clarifying details or propose alternative phrasing.
3. WHEN presenting an answer THEN the system SHALL cite the top supporting sources (title and timestamp, and link when available).
4. IF a source has stale data (past freshness window) THEN the system SHALL either prioritize fresher sources or indicate staleness in the response.
5. WHEN the user provides a location (GPS/pincode/district) THEN the system SHALL prioritize retrieval from region-specific sources.

### Requirement 3: Weather-Aware Recommendations

**User Story:** As a farmer, I want recommendations that factor in current and forecasted weather, so that I can plan sowing, irrigation, and pest control effectively.

#### Acceptance Criteria
1. WHEN a user asks about crop planning or operations THEN the system SHALL retrieve current conditions and short-term forecasts for the user’s location.
2. IF severe weather is forecast (e.g., heavy rain, heatwave, frost) THEN the system SHALL include precautionary guidance in the answer.
3. WHEN forecasts are unavailable for the location THEN the system SHALL disclose unavailability and provide general best-practice guidance.

### Requirement 4: Soil and Crop Suitability Guidance

**User Story:** As a farmer, I want crop suggestions based on soil conditions and season, so that I can improve yield and reduce risk.

#### Acceptance Criteria
1. WHEN the user provides soil parameters (e.g., pH, moisture, texture) or region defaults are available THEN the system SHALL recommend suitable crops and varieties for the current season.
2. IF soil parameter values are missing or inconsistent THEN the system SHALL request the minimal additional inputs required (e.g., pH range).
3. WHEN giving recommendations THEN the system SHALL include expected sowing windows and basic input requirements (seed rate, spacing).

### Requirement 5: Pest and Disease Advisories

**User Story:** As a farmer, I want early warnings and treatment guidance for pests and diseases, so that I can prevent damage and reduce costs.

#### Acceptance Criteria
1. WHEN pest alerts exist for the user’s region and crop THEN the system SHALL include detection signs, thresholds, and IPM-first treatment steps (cultural/biological/chemical with safety).
2. IF pesticide use is advised THEN the system SHALL include dilution/usage instructions, pre-harvest interval, and safety precautions.
3. WHEN relevant advisories are out-of-date THEN the system SHALL flag their date and recommend contacting local extension officers if uncertainty remains.

### Requirement 6: Market/Mandi Price Updates

**User Story:** As a farmer, I want current mandi prices for my crop, so that I can decide where and when to sell.

#### Acceptance Criteria
1. WHEN the user requests prices for a crop and region THEN the system SHALL retrieve and present current or most recent mandi prices with market names and timestamps.
2. IF multiple markets are nearby THEN the system SHALL list top N markets with highest prices and distance (if available).
3. IF prices are unavailable or delayed THEN the system SHALL disclose the last known date and suggest the nearest markets with data.

### Requirement 7: Government and Extension Updates

**User Story:** As a farmer, I want relevant government schemes and extension advisories, so that I can access benefits and follow best practices.

#### Acceptance Criteria
1. WHEN a user asks about schemes or assistance THEN the system SHALL retrieve region-appropriate schemes with eligibility and application steps.
2. WHEN an advisory is cited THEN the system SHALL present the authoritative source and date.
3. IF conflicting advisories exist THEN the system SHALL prefer the most recent and authoritative source and note the conflict.

### Requirement 8: Multilingual UX and Accessibility

**User Story:** As a farmer, I want a simple interface that supports voice/text and offline-friendly behavior, so that I can use the agent in the field.

#### Acceptance Criteria
1. WHEN bandwidth is low THEN the system SHALL degrade gracefully (reduced images, shorter responses, minimal citations) while preserving correctness.
2. IF the user opts for voice input/output THEN the system SHALL support speech-to-text and text-to-speech for supported languages.
3. WHEN presenting complex instructions THEN the system SHALL include concise bullet lists and optional “more details” expansion.

### Requirement 9: IBM Granite and IBM Cloud Lite Compliance

**User Story:** As a platform owner, I want the system to use IBM Granite models and IBM Cloud Lite services, so that it is enterprise-grade, secure, and scalable within cost limits.

#### Acceptance Criteria
1. WHEN running LLM inference THEN the system SHALL use IBM Granite models for generation and vision where applicable.
2. WHEN deploying core services (vector DB, API, storage, monitoring) THEN the system SHALL use IBM Cloud Lite-compatible services, staying within free/low-cost tiers where possible.
3. WHEN handling secrets (API keys, tokens) THEN the system SHALL store them securely using IBM Cloud secret management or environment variables, never in code.

### Requirement 10: Data Privacy, Security, and Safety

**User Story:** As a platform owner, I want data to be protected and usage auditable, so that we comply with privacy laws and build user trust.

#### Acceptance Criteria
1. WHEN handling user inputs and locations THEN the system SHALL minimize data retention and anonymize logs.
2. WHEN storing documents and vectors THEN the system SHALL encrypt data at rest and in transit.
3. WHEN generating outputs THEN the system SHALL include guardrails (content filters and safety checks) to prevent harmful or illegal guidance.

### Requirement 11: Observability and Reliability

**User Story:** As an operator, I want monitoring and graceful error handling, so that issues are detected and resolved quickly.

#### Acceptance Criteria
1. WHEN any upstream data source fails or times out THEN the system SHALL return a partial-but-useful response with a clear note about missing data.
2. WHEN retrieval or generation fails THEN the system SHALL log structured error details and present a user-friendly fallback message.
3. WHEN latencies exceed thresholds THEN the system SHALL log slow spans and surface minimal viable responses first (streaming), followed by details.

### Requirement 12: Performance and Cost Efficiency

**User Story:** As a platform owner, I want fast responses at low cost, so that the solution remains viable at scale.

#### Acceptance Criteria
1. WHEN answering typical queries THEN the system SHALL return a first token within an acceptable latency target (e.g., <2s on average) using streaming.
2. WHEN retrieving documents THEN the system SHALL cap total tokens (context window budget) using rankers and chunking policies.
3. WHEN usage exceeds rate or budget limits THEN the system SHALL throttle requests and present a polite notice to the user.

### Requirement 13: Administrative Controls

**User Story:** As an administrator, I want to manage data sources, schemas, and policies, so that the system remains accurate and compliant.

#### Acceptance Criteria
1. WHEN an admin updates a data source configuration THEN the system SHALL validate connectivity and schema before applying changes.
2. WHEN new document types are added THEN the system SHALL index them via an ingestion pipeline with validation and monitoring hooks.
3. WHEN prompt or policy templates are updated THEN the system SHALL version them and support rollback.

### Requirement 14: Testing and Quality Gates

**User Story:** As an engineer, I want automated tests and evaluation harnesses, so that changes don’t regress answer quality or safety.

#### Acceptance Criteria
1. WHEN code is committed THEN the system SHALL run unit and integration tests for ingestion, retrieval, ranking, and generation.
2. WHEN prompts are changed THEN the system SHALL run RAG evaluation (groundedness, relevancy) on a curated set of questions.
3. WHEN safety filters are changed THEN the system SHALL test red-team prompts and block unsafe completions.
