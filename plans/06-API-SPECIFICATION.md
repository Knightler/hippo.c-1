# API Specification: Hippo.c Memory System

## Overview

This document provides the complete API specification for the Hippo.c memory system.

**Base URL**: `http://localhost:8000` (development) or `https://api.your-domain.com` (production)

**Content-Type**: `application/json`

**Authentication**: API Key (header: `X-API-Key`) - *Not yet implemented*

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/encode` | Encode text into memory |
| POST | `/api/v1/recall` | Recall memories by query |
| POST | `/api/v1/recall/llm` | Recall formatted for LLM |
| GET | `/api/v1/facts` | List all facts |
| GET | `/api/v1/facts/{id}` | Get a specific fact |
| DELETE | `/api/v1/facts/{id}` | Delete a fact |
| GET | `/api/v1/labels` | List all labels |
| GET | `/api/v1/labels/{id}` | Get a specific label |

---

## Health Check

### `GET /health`

Check system health status.

#### Response

```json
{
    "status": "healthy",
    "database": true,
    "version": "0.3.0"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | System is healthy |
| 503 | System is degraded |

---

## Encode API

### `POST /api/v1/encode`

Process text and extract facts into memory.

#### Request Body

```json
{
    "text": "I really love Italian food, especially pasta.",
    "role": "user"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| text | string | Yes | - | Text to process |
| role | string | No | "user" | Message role: "user", "ai", "system" |

#### Response

```json
{
    "success": true,
    "updates": [
        {
            "action": "create",
            "entity": "fact",
            "entity_id": "550e8400-e29b-41d4-a716-446655440000",
            "confidence": 0.6,
            "reason": "new_fact"
        }
    ],
    "latency_ms": 145
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether encoding succeeded |
| updates | array | List of memory updates |
| latency_ms | integer | Processing time in milliseconds |

#### Update Object

| Field | Type | Description |
|-------|------|-------------|
| action | string | "create", "update", or "skip" |
| entity | string | "fact" or "label" |
| entity_id | string | UUID of created/updated entity |
| confidence | number | Confidence score (0.0-1.0) |
| reason | string | Reason for action |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid request |
| 500 | Internal error |

---

## Recall API

### `POST /api/v1/recall`

Query memories using natural language.

#### Request Body

```json
{
    "query": "What food does the user like?",
    "context": "User is asking for dinner recommendations",
    "limit": 10,
    "min_confidence": 0.3,
    "categories": ["preference", "food"]
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Yes | - | Natural language query |
| context | string | No | "" | Conversation context |
| limit | integer | No | 10 | Max results (1-50) |
| min_confidence | number | No | 0.3 | Min confidence (0.0-1.0) |
| categories | array | No | null | Filter by categories |

#### Response

```json
{
    "success": true,
    "query_id": "550e8400-e29b-41d4-a716-446655440001",
    "latency_ms": 45,
    "results": [
        {
            "fact_id": "550e8400-e29b-41d4-a716-446655440000",
            "content": "loves Italian food",
            "category": "preference",
            "label": "food_preferences",
            "confidence": 0.85,
            "relevance_score": 0.92,
            "last_seen": "2024-01-15T10:30:00Z",
            "evidence_count": 5,
            "explanation": "high semantic similarity; matched: food"
        }
    ],
    "summary": "User preferences: loves Italian food; enjoys pasta",
    "total_matches": 3
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether recall succeeded |
| query_id | string | Unique query identifier |
| latency_ms | integer | Query time in milliseconds |
| results | array | Matched facts |
| summary | string | Natural language summary |
| total_matches | integer | Total results returned |

#### Result Object

| Field | Type | Description |
|-------|------|-------------|
| fact_id | string | UUID of the fact |
| content | string | Fact content |
| category | string | Fact category |
| label | string | Associated label name |
| confidence | number | Fact confidence (0.0-1.0) |
| relevance_score | number | Query relevance (0.0-1.0) |
| last_seen | string | ISO 8601 timestamp |
| evidence_count | integer | Times fact reinforced |
| explanation | string | Why result matched |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid query |
| 500 | Internal error |

---

### `POST /api/v1/recall/llm`

Get recall results formatted for LLM context injection.

#### Request Body

```json
{
    "query": "food preferences and dietary restrictions",
    "context": "User asking for dinner recommendations",
    "max_tokens": 500
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Yes | - | What to recall |
| context | string | No | "" | Conversation context |
| max_tokens | integer | No | 500 | Max tokens (100-2000) |

#### Response

```json
{
    "context": "## User Memory Context\nSummary: User preferences: loves Italian food; enjoys pasta\n\nSpecific facts:\n- [preference] loves Italian food (high confidence)\n- [preference] enjoys pasta dishes (high confidence)"
}
```

| Field | Type | Description |
|-------|------|-------------|
| context | string | Formatted context for LLM |

#### Usage Example

```python
# Get memory context
response = requests.post(
    "http://localhost:8000/api/v1/recall/llm",
    json={"query": "user preferences", "max_tokens": 300}
)
memory_context = response.json()["context"]

# Inject into LLM prompt
system_prompt = f"""You are a helpful assistant.

{memory_context}

Use the above context to personalize your responses."""

# Send to your LLM...
```

---

## Facts API

### `GET /api/v1/facts`

List all facts with pagination. *(Future endpoint)*

#### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| limit | integer | 20 | Max results per page |
| offset | integer | 0 | Pagination offset |
| category | string | - | Filter by category |
| min_confidence | number | 0 | Min confidence filter |

#### Response

```json
{
    "facts": [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "content": "loves Italian food",
            "category": "preference",
            "label_id": "550e8400-e29b-41d4-a716-446655440010",
            "label_name": "food_preferences",
            "confidence": 0.85,
            "evidence_count": 5,
            "created_at": "2024-01-10T08:00:00Z",
            "updated_at": "2024-01-15T10:30:00Z",
            "last_seen_at": "2024-01-15T10:30:00Z"
        }
    ],
    "total": 42,
    "limit": 20,
    "offset": 0
}
```

---

### `GET /api/v1/facts/{id}`

Get a specific fact by ID. *(Future endpoint)*

#### Response

```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "loves Italian food",
    "category": "preference",
    "label_id": "550e8400-e29b-41d4-a716-446655440010",
    "label_name": "food_preferences",
    "confidence": 0.85,
    "source_role": "user",
    "evidence_count": 5,
    "created_at": "2024-01-10T08:00:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "last_seen_at": "2024-01-15T10:30:00Z",
    "metadata": {}
}
```

---

### `DELETE /api/v1/facts/{id}`

Soft delete a fact. *(Future endpoint)*

#### Response

```json
{
    "success": true,
    "message": "Fact deleted"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Fact not found |

---

## Labels API

### `GET /api/v1/labels`

List all labels. *(Future endpoint)*

#### Response

```json
{
    "labels": [
        {
            "id": "550e8400-e29b-41d4-a716-446655440010",
            "name": "food_preferences",
            "kind": "topic",
            "category": "preference",
            "usage_count": 15,
            "fact_count": 8,
            "created_at": "2024-01-10T08:00:00Z",
            "updated_at": "2024-01-15T10:30:00Z"
        }
    ],
    "total": 12
}
```

---

### `GET /api/v1/labels/{id}`

Get a specific label with its facts. *(Future endpoint)*

#### Response

```json
{
    "id": "550e8400-e29b-41d4-a716-446655440010",
    "name": "food_preferences",
    "kind": "topic",
    "category": "preference",
    "usage_count": 15,
    "created_at": "2024-01-10T08:00:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "facts": [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "content": "loves Italian food",
            "confidence": 0.85
        }
    ]
}
```

---

## Error Responses

All error responses follow this format:

```json
{
    "detail": {
        "code": "ERROR_CODE",
        "message": "Human-readable error message",
        "details": {}
    }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_QUERY | 400 | Query is empty or malformed |
| INVALID_REQUEST | 400 | Request body is invalid |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMITED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Internal server error |
| DATABASE_ERROR | 503 | Database unavailable |

---

## Categories

The system uses these built-in categories:

| Category | Description | Examples |
|----------|-------------|----------|
| preference | Likes, dislikes, favorites | "loves coffee", "prefers dark mode" |
| relationship | People connections | "friend named John", "married to Sarah" |
| location | Places and geography | "lives in NYC", "from California" |
| goal | Aspirations and plans | "wants to learn Python", "planning to travel" |
| emotion | Feelings and moods | "feels happy", "anxious about work" |
| habit | Regular behaviors | "exercises daily", "wakes up early" |
| opinion | Beliefs and views | "thinks AI is useful", "believes in X" |
| fact | General information | "born in 1990", "has a cat" |

---

## Rate Limits

*(Future implementation)*

| Tier | Requests/minute | Requests/day |
|------|-----------------|--------------|
| Free | 20 | 1,000 |
| Basic | 60 | 10,000 |
| Pro | 300 | 100,000 |
| Enterprise | Custom | Custom |

Rate limit headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1704067200
```

---

## SDK Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

class HippoClient:
    def __init__(self, base_url: str = BASE_URL, api_key: str = None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def encode(self, text: str, role: str = "user") -> dict:
        response = requests.post(
            f"{self.base_url}/api/v1/encode",
            json={"text": text, "role": role},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
    
    def recall(
        self,
        query: str,
        context: str = "",
        limit: int = 10,
        categories: list[str] = None,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/api/v1/recall",
            json={
                "query": query,
                "context": context,
                "limit": limit,
                "categories": categories,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
    
    def recall_for_llm(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 500,
    ) -> str:
        response = requests.post(
            f"{self.base_url}/api/v1/recall/llm",
            json={
                "query": query,
                "context": context,
                "max_tokens": max_tokens,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()["context"]


# Usage
client = HippoClient()

# Encode new information
client.encode("I love pizza and Italian food")

# Recall memories
results = client.recall("What food does the user like?")
print(results["summary"])

# Get context for LLM
context = client.recall_for_llm("user preferences")
print(context)
```

### JavaScript/TypeScript

```typescript
interface RecallResult {
    fact_id: string;
    content: string;
    category: string;
    confidence: number;
    relevance_score: number;
}

interface RecallResponse {
    success: boolean;
    results: RecallResult[];
    summary: string;
}

class HippoClient {
    private baseUrl: string;
    private apiKey?: string;

    constructor(baseUrl: string = "http://localhost:8000", apiKey?: string) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
    }

    private getHeaders(): HeadersInit {
        const headers: HeadersInit = {
            "Content-Type": "application/json",
        };
        if (this.apiKey) {
            headers["X-API-Key"] = this.apiKey;
        }
        return headers;
    }

    async encode(text: string, role: string = "user"): Promise<any> {
        const response = await fetch(`${this.baseUrl}/api/v1/encode`, {
            method: "POST",
            headers: this.getHeaders(),
            body: JSON.stringify({ text, role }),
        });
        return response.json();
    }

    async recall(
        query: string,
        options: { context?: string; limit?: number; categories?: string[] } = {}
    ): Promise<RecallResponse> {
        const response = await fetch(`${this.baseUrl}/api/v1/recall`, {
            method: "POST",
            headers: this.getHeaders(),
            body: JSON.stringify({
                query,
                context: options.context || "",
                limit: options.limit || 10,
                categories: options.categories,
            }),
        });
        return response.json();
    }

    async recallForLLM(
        query: string,
        context: string = "",
        maxTokens: number = 500
    ): Promise<string> {
        const response = await fetch(`${this.baseUrl}/api/v1/recall/llm`, {
            method: "POST",
            headers: this.getHeaders(),
            body: JSON.stringify({ query, context, max_tokens: maxTokens }),
        });
        const data = await response.json();
        return data.context;
    }
}

// Usage
const client = new HippoClient();

await client.encode("I love hiking and outdoor activities");

const results = await client.recall("What does the user like to do?");
console.log(results.summary);

const context = await client.recallForLLM("user hobbies");
console.log(context);
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Encode text
curl -X POST http://localhost:8000/api/v1/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "I love hiking in the mountains", "role": "user"}'

# Recall memories
curl -X POST http://localhost:8000/api/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "outdoor activities", "limit": 5}'

# Recall for LLM
curl -X POST http://localhost:8000/api/v1/recall/llm \
  -H "Content-Type: application/json" \
  -d '{"query": "hobbies and interests", "max_tokens": 300}'
```

---

## OpenAPI Specification

The full OpenAPI spec is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Changelog

### v0.3.0 (Current)
- Added recall system
- Added LLM context formatting
- Connection pooling
- Quality embeddings

### v0.2.2
- Initial encode system
- Pattern extraction
- Basic memory storage

---

## Support

For issues and questions:
- GitHub Issues: [repository URL]
- Documentation: [docs URL]
