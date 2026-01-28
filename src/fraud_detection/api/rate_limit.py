"""Rate limiting middleware for the Fraud Detection API."""

import asyncio
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Requests per window
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    
    # Burst allowance (token bucket)
    burst_size: int = 10
    
    # Headers to use for client identification
    client_header: str = "X-API-Key"
    fallback_header: str = "X-Forwarded-For"
    
    # Endpoints exempt from rate limiting
    exempt_paths: set = field(default_factory=lambda: {"/health", "/metrics", "/"})
    
    # Enable Redis backend (for distributed deployments)
    use_redis: bool = False
    redis_url: str = ""


@dataclass
class RateLimitState:
    """Track rate limit state for a client."""
    
    tokens: float = 10.0
    last_update: float = field(default_factory=time.time)
    minute_requests: int = 0
    minute_start: float = field(default_factory=time.time)
    hour_requests: int = 0
    hour_start: float = field(default_factory=time.time)


class InMemoryRateLimiter:
    """In-memory rate limiter using token bucket algorithm."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.clients: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> tuple[bool, Dict[str, str]]:
        """
        Check if request is allowed for the client.
        
        Returns:
            Tuple of (is_allowed, headers_dict)
        """
        async with self._lock:
            state = self.clients[client_id]
            now = time.time()
            
            # Refill tokens based on time passed
            time_passed = now - state.last_update
            tokens_to_add = time_passed * (self.config.requests_per_minute / 60)
            state.tokens = min(self.config.burst_size, state.tokens + tokens_to_add)
            state.last_update = now
            
            # Reset minute counter if minute has passed
            if now - state.minute_start >= 60:
                state.minute_requests = 0
                state.minute_start = now
            
            # Reset hour counter if hour has passed
            if now - state.hour_start >= 3600:
                state.hour_requests = 0
                state.hour_start = now
            
            # Calculate remaining limits
            minute_remaining = max(0, self.config.requests_per_minute - state.minute_requests)
            hour_remaining = max(0, self.config.requests_per_hour - state.hour_requests)
            
            # Common headers
            headers = {
                "X-RateLimit-Limit": str(self.config.requests_per_minute),
                "X-RateLimit-Remaining": str(minute_remaining),
                "X-RateLimit-Reset": str(int(state.minute_start + 60)),
            }
            
            # Check limits
            if state.tokens < 1:
                retry_after = int((1 - state.tokens) * (60 / self.config.requests_per_minute))
                headers["Retry-After"] = str(max(1, retry_after))
                return False, headers
            
            if state.minute_requests >= self.config.requests_per_minute:
                headers["Retry-After"] = str(int(60 - (now - state.minute_start)))
                return False, headers
            
            if state.hour_requests >= self.config.requests_per_hour:
                headers["Retry-After"] = str(int(3600 - (now - state.hour_start)))
                return False, headers
            
            # Allow request
            state.tokens -= 1
            state.minute_requests += 1
            state.hour_requests += 1
            
            headers["X-RateLimit-Remaining"] = str(minute_remaining - 1)
            
            return True, headers
    
    async def get_client_stats(self, client_id: str) -> Dict:
        """Get current stats for a client."""
        state = self.clients.get(client_id)
        if not state:
            return {}
        
        return {
            "tokens": state.tokens,
            "minute_requests": state.minute_requests,
            "hour_requests": state.hour_requests,
        }


class RedisRateLimiter:
    """Redis-backed rate limiter for distributed deployments."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._redis = None
        self._initialized = False
    
    async def _get_redis(self):
        """Lazy initialization of Redis connection."""
        if not self._initialized:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(
                    self.config.redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
                )
                await self._redis.ping()
                self._initialized = True
                logger.info("Redis rate limiter connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to in-memory")
                self._redis = None
        return self._redis
    
    async def is_allowed(self, client_id: str) -> tuple[bool, Dict[str, str]]:
        """Check if request is allowed using Redis."""
        redis = await self._get_redis()
        
        if not redis:
            # Fallback: allow all if Redis is unavailable
            return True, {"X-RateLimit-Limit": "unlimited"}
        
        now = int(time.time())
        minute_key = f"ratelimit:{client_id}:minute:{now // 60}"
        hour_key = f"ratelimit:{client_id}:hour:{now // 3600}"
        
        try:
            pipe = redis.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            results = await pipe.execute()
            
            minute_count = results[0]
            hour_count = results[2]
            
            minute_remaining = max(0, self.config.requests_per_minute - minute_count)
            
            headers = {
                "X-RateLimit-Limit": str(self.config.requests_per_minute),
                "X-RateLimit-Remaining": str(minute_remaining),
                "X-RateLimit-Reset": str((now // 60 + 1) * 60),
            }
            
            if minute_count > self.config.requests_per_minute:
                headers["Retry-After"] = str(60 - (now % 60))
                return False, headers
            
            if hour_count > self.config.requests_per_hour:
                headers["Retry-After"] = str(3600 - (now % 3600))
                return False, headers
            
            return True, headers
            
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            return True, {}
    
    async def get_client_stats(self, client_id: str) -> Dict:
        """Get current stats for a client from Redis."""
        redis = await self._get_redis()
        if not redis:
            return {}
        
        now = int(time.time())
        minute_key = f"ratelimit:{client_id}:minute:{now // 60}"
        hour_key = f"ratelimit:{client_id}:hour:{now // 3600}"
        
        try:
            minute_count = await redis.get(minute_key) or 0
            hour_count = await redis.get(hour_key) or 0
            
            return {
                "minute_requests": int(minute_count),
                "hour_requests": int(hour_count),
            }
        except Exception:
            return {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        app,
        config: Optional[RateLimitConfig] = None,
    ):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        
        # Choose limiter based on config
        if self.config.use_redis:
            self.limiter = RedisRateLimiter(self.config)
        else:
            self.limiter = InMemoryRateLimiter(self.config)
        
        logger.info(
            f"Rate limiting enabled: {self.config.requests_per_minute}/min, "
            f"{self.config.requests_per_hour}/hour, "
            f"backend={'redis' if self.config.use_redis else 'memory'}"
        )
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try API key first
        api_key = request.headers.get(self.config.client_header)
        if api_key:
            return f"key:{api_key[:16]}"  # Use prefix of key
        
        # Fall back to IP address
        forwarded = request.headers.get(self.config.fallback_header)
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        if request.client:
            return f"ip:{request.client.host}"
        
        return "unknown"
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and apply rate limiting."""
        # Skip rate limiting for exempt paths
        if request.url.path in self.config.exempt_paths:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        allowed, headers = await self.limiter.is_allowed(client_id)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_id}")
            response = Response(
                content='{"detail": "Rate limit exceeded. Try again later."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
            )
            for key, value in headers.items():
                response.headers[key] = value
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response


def create_rate_limiter(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    use_redis: bool = False,
    redis_url: str = "",
) -> RateLimitMiddleware:
    """
    Create a rate limiting middleware instance.
    
    Args:
        requests_per_minute: Max requests per minute per client
        requests_per_hour: Max requests per hour per client
        use_redis: Use Redis backend for distributed deployments
        redis_url: Redis connection URL
        
    Returns:
        Configured RateLimitMiddleware
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        use_redis=use_redis,
        redis_url=redis_url,
    )
    return lambda app: RateLimitMiddleware(app, config)
