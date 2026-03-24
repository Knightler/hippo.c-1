# Security Guide: Hippo.c Memory System

## Overview

This document provides comprehensive security guidance for the Hippo.c memory system, covering authentication, data protection, and privacy compliance.

---

## Security Model

### Threat Model

| Threat | Risk Level | Mitigation |
|--------|------------|------------|
| Unauthorized API access | High | API key authentication |
| SQL injection | Medium | Parameterized queries (already done) |
| Data exfiltration | High | Encryption, access control |
| Cross-user data access | Critical | User isolation |
| API key leakage | High | Secret rotation, scoping |
| Denial of service | Medium | Rate limiting |
| PII exposure | High | Data minimization, encryption |

### Security Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Network Security                                        │
│ - HTTPS/TLS encryption                                          │
│ - Firewall rules                                                │
│ - DDoS protection                                               │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Authentication & Authorization                          │
│ - API key validation                                            │
│ - User isolation                                                │
│ - Role-based access                                             │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Input Validation                                        │
│ - Request validation                                            │
│ - Content sanitization                                          │
│ - Size limits                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Data Protection                                         │
│ - Encryption at rest                                            │
│ - Encryption in transit                                         │
│ - Secure key storage                                            │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: Audit & Monitoring                                      │
│ - Access logging                                                │
│ - Anomaly detection                                             │
│ - Alerting                                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Authentication

### API Key Authentication

#### Implementation

```python
# api/auth/middleware.py

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable

from fastapi import HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
import logging

logger = logging.getLogger(__name__)

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyManager:
    """Manage API keys securely."""
    
    def __init__(self):
        # In production, load from database/secrets manager
        self._keys: dict[str, dict] = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from secure storage."""
        # Master key from environment
        master_key = os.getenv("HIPPO_MASTER_API_KEY")
        if master_key:
            self._keys[self._hash_key(master_key)] = {
                "user_id": "master",
                "scopes": ["read", "write", "admin"],
                "expires_at": None,
            }
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for secure comparison."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def validate_key(self, key: str) -> dict | None:
        """Validate an API key and return associated user info."""
        if not key:
            return None
        
        hashed = self._hash_key(key)
        key_info = self._keys.get(hashed)
        
        if not key_info:
            return None
        
        # Check expiration
        if key_info.get("expires_at"):
            if datetime.utcnow() > key_info["expires_at"]:
                return None
        
        return key_info
    
    def generate_key(self, user_id: str, scopes: list[str], expires_days: int = 30) -> str:
        """Generate a new API key."""
        key = f"hippo_{secrets.token_urlsafe(32)}"
        
        self._keys[self._hash_key(key)] = {
            "user_id": user_id,
            "scopes": scopes,
            "expires_at": datetime.utcnow() + timedelta(days=expires_days),
            "created_at": datetime.utcnow(),
        }
        
        return key
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        hashed = self._hash_key(key)
        if hashed in self._keys:
            del self._keys[hashed]
            return True
        return False


# Global key manager
key_manager = APIKeyManager()


async def verify_api_key(
    request: Request,
    api_key: str = Depends(api_key_header),
) -> dict:
    """Verify API key and return user context."""
    if not api_key:
        logger.warning(f"Missing API key from {request.client.host}")
        raise HTTPException(
            status_code=401,
            detail={"code": "MISSING_API_KEY", "message": "API key required"},
        )
    
    key_info = key_manager.validate_key(api_key)
    
    if not key_info:
        logger.warning(f"Invalid API key from {request.client.host}")
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_API_KEY", "message": "Invalid or expired API key"},
        )
    
    # Log access
    logger.info(f"API access: user={key_info['user_id']} endpoint={request.url.path}")
    
    return key_info


def require_scope(scope: str) -> Callable:
    """Decorator to require specific scope."""
    async def scope_checker(key_info: dict = Depends(verify_api_key)) -> dict:
        if scope not in key_info.get("scopes", []):
            raise HTTPException(
                status_code=403,
                detail={"code": "INSUFFICIENT_SCOPE", "message": f"Requires '{scope}' scope"},
            )
        return key_info
    return scope_checker
```

#### Usage in Endpoints

```python
# api/main.py

from api.auth.middleware import verify_api_key, require_scope

@app.post("/api/v1/recall")
async def recall_endpoint(
    request: RecallRequest,
    auth: dict = Depends(verify_api_key),  # Requires valid API key
):
    user_id = auth["user_id"]
    # ... use user_id to filter results
    

@app.post("/api/v1/admin/keys")
async def create_api_key(
    request: CreateKeyRequest,
    auth: dict = Depends(require_scope("admin")),  # Requires admin scope
):
    # ... create new API key
```

### Environment Variables

```bash
# .env (NEVER commit this file)
SUPABASE_DATABASE_URL=postgresql://...
HIPPO_MASTER_API_KEY=hippo_your_secret_key_here
DEEPSEEK_API_KEY=sk-...
```

```python
# .gitignore
.env
.env.local
.env.*.local
```

---

## User Isolation

### Multi-User Schema

```sql
-- Migration: Add user isolation

-- Add user_id to all tables
ALTER TABLE labels ADD COLUMN user_id uuid NOT NULL;
ALTER TABLE facts ADD COLUMN user_id uuid NOT NULL;
ALTER TABLE prompts ADD COLUMN user_id uuid NOT NULL;
ALTER TABLE patterns ADD COLUMN user_id uuid;  -- Global patterns have NULL

-- Create indexes for user filtering
CREATE INDEX idx_labels_user ON labels(user_id);
CREATE INDEX idx_facts_user ON facts(user_id);
CREATE INDEX idx_prompts_user ON prompts(user_id);

-- Row Level Security (for Supabase/PostgreSQL)
ALTER TABLE labels ENABLE ROW LEVEL SECURITY;
ALTER TABLE facts ENABLE ROW LEVEL SECURITY;
ALTER TABLE prompts ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY labels_user_isolation ON labels
    FOR ALL
    USING (user_id = current_setting('app.user_id')::uuid);

CREATE POLICY facts_user_isolation ON facts
    FOR ALL
    USING (user_id = current_setting('app.user_id')::uuid);

CREATE POLICY prompts_user_isolation ON prompts
    FOR ALL
    USING (user_id = current_setting('app.user_id')::uuid);
```

### Application-Level Isolation

```python
# memory/client.py - User-scoped client

class UserMemoryClient(MemoryClient):
    """Memory client scoped to a specific user."""
    
    def __init__(self, user_id: str, dsn_env: str = "SUPABASE_DATABASE_URL"):
        super().__init__(dsn_env)
        self.user_id = user_id
    
    @contextmanager
    def _connect(self) -> Generator[psycopg.Connection, None, None]:
        """Get connection with user context."""
        with super()._connect() as conn:
            # Set user context for RLS
            with conn.cursor() as cur:
                cur.execute(
                    "SET LOCAL app.user_id = %s",
                    (self.user_id,)
                )
            yield conn
    
    def search_facts(self, embedding: list[float], **kwargs) -> list[tuple[Fact, float]]:
        """Search facts for current user only."""
        # RLS automatically filters, but we can also explicitly filter
        return super().search_facts(embedding, **kwargs)
```

---

## Input Validation

### Request Validation

```python
# api/validation.py

from pydantic import BaseModel, Field, validator
import re

MAX_TEXT_LENGTH = 10000
MAX_QUERY_LENGTH = 1000


class EncodeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    role: str = Field("user", regex="^(user|ai|system)$")
    
    @validator("text")
    def sanitize_text(cls, v):
        # Remove control characters
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        # Limit consecutive whitespace
        v = re.sub(r"\s{10,}", " " * 10, v)
        return v.strip()


class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    context: str = Field("", max_length=MAX_TEXT_LENGTH)
    limit: int = Field(10, ge=1, le=50)
    min_confidence: float = Field(0.3, ge=0, le=1)
    categories: list[str] | None = None
    
    @validator("categories")
    def validate_categories(cls, v):
        if v is None:
            return v
        
        ALLOWED_CATEGORIES = {
            "preference", "relationship", "location", "goal",
            "emotion", "habit", "opinion", "fact"
        }
        
        invalid = set(v) - ALLOWED_CATEGORIES
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}")
        
        return v
```

### Content Sanitization

```python
# utils/sanitize.py

import re
import html


def sanitize_content(text: str) -> str:
    """Sanitize user content before storage."""
    # Escape HTML entities
    text = html.escape(text)
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    
    # Limit length
    if len(text) > 10000:
        text = text[:10000]
    
    return text.strip()


def sanitize_for_embedding(text: str) -> str:
    """Prepare text for embedding (less strict)."""
    # Just basic cleanup
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

---

## Data Protection

### Encryption at Rest

Supabase and most managed databases encrypt data at rest. For self-hosted:

```sql
-- PostgreSQL with encryption
-- Use pgcrypto for column-level encryption

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt sensitive content
-- Note: This impacts search performance
UPDATE facts
SET content = pgp_sym_encrypt(content, current_setting('app.encryption_key'))
WHERE category = 'sensitive';

-- Decrypt when reading
SELECT pgp_sym_decrypt(content::bytea, current_setting('app.encryption_key')) as content
FROM facts;
```

**Better approach**: Use application-level encryption for sensitive fields:

```python
# utils/crypto.py

from cryptography.fernet import Fernet
import os


class DataEncryptor:
    def __init__(self):
        key = os.getenv("HIPPO_ENCRYPTION_KEY")
        if not key:
            raise ValueError("HIPPO_ENCRYPTION_KEY not set")
        self.fernet = Fernet(key.encode())
    
    def encrypt(self, data: str) -> str:
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        return self.fernet.decrypt(encrypted.encode()).decode()


# Generate a key (do once, store securely)
# from cryptography.fernet import Fernet
# print(Fernet.generate_key().decode())
```

### Secrets Management

**Environment Variables** (minimum):
```bash
# Use .env for development
# Use secrets manager for production

# Never commit secrets!
echo ".env" >> .gitignore
```

**Production Options**:
- AWS Secrets Manager
- HashiCorp Vault
- Google Secret Manager
- Azure Key Vault

```python
# utils/secrets.py

import os


def get_secret(name: str) -> str:
    """Get secret from appropriate source."""
    # Check environment first
    value = os.getenv(name)
    if value:
        return value
    
    # Check secrets manager (example: AWS)
    if os.getenv("USE_AWS_SECRETS"):
        import boto3
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=name)
        return response["SecretString"]
    
    raise ValueError(f"Secret {name} not found")
```

---

## Rate Limiting

### Implementation

```python
# api/auth/rate_limit.py

import time
from collections import defaultdict
from fastapi import HTTPException, Request


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._limits = {
            "default": (60, 60),    # 60 requests per 60 seconds
            "encode": (30, 60),     # 30 encodes per minute
            "recall": (100, 60),    # 100 recalls per minute
        }
    
    def check(self, key: str, endpoint: str = "default") -> tuple[bool, dict]:
        """Check if request is allowed."""
        limit, window = self._limits.get(endpoint, self._limits["default"])
        
        now = time.time()
        window_start = now - window
        
        # Clean old requests
        self._requests[key] = [
            t for t in self._requests[key] if t > window_start
        ]
        
        # Check limit
        if len(self._requests[key]) >= limit:
            retry_after = int(self._requests[key][0] - window_start) + 1
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset": int(window_start + window),
                "retry_after": retry_after,
            }
        
        # Record request
        self._requests[key].append(now)
        
        return True, {
            "limit": limit,
            "remaining": limit - len(self._requests[key]),
            "reset": int(now + window),
        }


rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """Rate limit middleware."""
    # Get client identifier
    client_key = request.headers.get("X-API-Key", request.client.host)
    
    # Determine endpoint type
    endpoint = "default"
    if "/encode" in request.url.path:
        endpoint = "encode"
    elif "/recall" in request.url.path:
        endpoint = "recall"
    
    # Check rate limit
    allowed, info = rate_limiter.check(client_key, endpoint)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "RATE_LIMITED",
                "message": "Too many requests",
                "retry_after": info["retry_after"],
            },
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(info["retry_after"]),
            },
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(info["reset"])
    
    return response
```

---

## Audit Logging

### Access Logs

```python
# api/audit/logger.py

import logging
import json
from datetime import datetime
from typing import Any

# Configure audit logger
audit_logger = logging.getLogger("hippo.audit")
audit_logger.setLevel(logging.INFO)

# File handler for audit logs
handler = logging.FileHandler("/var/log/hippo/audit.log")
handler.setFormatter(logging.Formatter("%(message)s"))
audit_logger.addHandler(handler)


def log_access(
    user_id: str,
    action: str,
    resource: str,
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
    success: bool = True,
    ip_address: str | None = None,
):
    """Log an access event."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "resource_id": resource_id,
        "success": success,
        "ip_address": ip_address,
        "details": details or {},
    }
    
    audit_logger.info(json.dumps(entry))


# Usage in endpoints
@app.post("/api/v1/recall")
async def recall_endpoint(
    request: Request,
    body: RecallRequest,
    auth: dict = Depends(verify_api_key),
):
    log_access(
        user_id=auth["user_id"],
        action="recall",
        resource="facts",
        ip_address=request.client.host,
        details={"query": body.query[:100]},  # Truncate for logging
    )
    
    # ... rest of endpoint
```

### Audit Trail Schema

```sql
-- Audit log table
CREATE TABLE audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp timestamptz NOT NULL DEFAULT now(),
    user_id uuid NOT NULL,
    action text NOT NULL,
    resource text NOT NULL,
    resource_id uuid,
    success boolean NOT NULL DEFAULT true,
    ip_address inet,
    user_agent text,
    details jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- Index for querying
CREATE INDEX idx_audit_user ON audit_log(user_id, timestamp DESC);
CREATE INDEX idx_audit_resource ON audit_log(resource, resource_id, timestamp DESC);
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp DESC);
```

---

## Privacy & GDPR

### Data Export

```python
# api/privacy.py

from datetime import datetime
import json

from memory import MemoryClient


def export_user_data(user_id: str) -> dict:
    """Export all data for a user (GDPR Article 20)."""
    client = MemoryClient()
    
    # Get all user data
    with client._connect() as conn:
        with conn.cursor() as cur:
            # Export facts
            cur.execute(
                "SELECT * FROM facts WHERE user_id = %s",
                (user_id,)
            )
            facts = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
            
            # Export labels
            cur.execute(
                "SELECT * FROM labels WHERE user_id = %s",
                (user_id,)
            )
            labels = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
            
            # Export prompts
            cur.execute(
                "SELECT * FROM prompts WHERE user_id = %s",
                (user_id,)
            )
            prompts = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
    
    return {
        "export_date": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "facts": facts,
        "labels": labels,
        "prompts": prompts,
    }


def delete_user_data(user_id: str) -> dict:
    """Delete all data for a user (GDPR Article 17)."""
    client = MemoryClient()
    
    deleted = {"facts": 0, "labels": 0, "prompts": 0}
    
    with client._connect() as conn:
        with conn.cursor() as cur:
            # Delete facts
            cur.execute("DELETE FROM facts WHERE user_id = %s", (user_id,))
            deleted["facts"] = cur.rowcount
            
            # Delete labels
            cur.execute("DELETE FROM labels WHERE user_id = %s", (user_id,))
            deleted["labels"] = cur.rowcount
            
            # Delete prompts
            cur.execute("DELETE FROM prompts WHERE user_id = %s", (user_id,))
            deleted["prompts"] = cur.rowcount
            
            conn.commit()
    
    return deleted
```

### Privacy API Endpoints

```python
# api/routes/privacy.py

@app.get("/api/v1/privacy/export")
async def export_data(auth: dict = Depends(verify_api_key)):
    """Export all user data (GDPR)."""
    data = export_user_data(auth["user_id"])
    return data


@app.delete("/api/v1/privacy/delete")
async def delete_data(auth: dict = Depends(verify_api_key)):
    """Delete all user data (GDPR)."""
    result = delete_user_data(auth["user_id"])
    return {"success": True, "deleted": result}


@app.get("/api/v1/privacy/policy")
async def privacy_policy():
    """Return privacy policy information."""
    return {
        "data_collected": [
            "User messages for memory extraction",
            "Extracted facts and preferences",
            "Usage patterns and access logs",
        ],
        "data_retention": "Data is retained until user deletion request",
        "data_sharing": "Data is not shared with third parties",
        "contact": "privacy@your-domain.com",
    }
```

---

## Security Checklist

### Development
- [ ] No secrets in code
- [ ] .env in .gitignore
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] Dependency vulnerability scanning

### Pre-Production
- [ ] API authentication implemented
- [ ] Rate limiting enabled
- [ ] Audit logging enabled
- [ ] HTTPS/TLS configured
- [ ] CORS properly configured
- [ ] Security headers set

### Production
- [ ] User isolation verified
- [ ] Encryption at rest enabled
- [ ] Secrets in secure storage
- [ ] Monitoring and alerting
- [ ] Incident response plan
- [ ] Penetration testing

### Compliance
- [ ] Privacy policy published
- [ ] Data export functionality
- [ ] Data deletion functionality
- [ ] Consent management
- [ ] Data retention policy

---

## Security Headers

```python
# api/main.py

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


app.add_middleware(SecurityHeadersMiddleware)
```

---

## Incident Response

### Response Plan

1. **Detection**: Alert triggers from monitoring
2. **Containment**: Isolate affected systems
3. **Investigation**: Determine scope and cause
4. **Remediation**: Fix vulnerability
5. **Recovery**: Restore normal operation
6. **Post-mortem**: Document and improve

### Emergency Actions

```bash
# Revoke all API keys
curl -X POST http://localhost:8000/api/v1/admin/revoke-all-keys \
  -H "X-API-Key: $ADMIN_KEY"

# Disable authentication (emergency read-only)
export HIPPO_EMERGENCY_MODE=true

# Enable verbose logging
export HIPPO_LOG_LEVEL=DEBUG
```

---

This completes the security guide. Implement these measures progressively, starting with authentication and input validation.
