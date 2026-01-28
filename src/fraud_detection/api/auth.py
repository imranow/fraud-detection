"""API Key Authentication middleware for the Fraud Detection API."""

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# API Key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load API keys from environment
# Format: comma-separated list of valid keys
_API_KEYS: Optional[set] = None


def get_api_keys() -> set:
    """Get the set of valid API keys from environment."""
    global _API_KEYS
    
    if _API_KEYS is None:
        keys_str = os.getenv("API_KEYS", "")
        if keys_str:
            _API_KEYS = set(key.strip() for key in keys_str.split(",") if key.strip())
        else:
            # Generate a default key for development if none provided
            default_key = os.getenv("API_KEY", "")
            if default_key:
                _API_KEYS = {default_key}
            else:
                logger.warning(
                    "No API keys configured. Set API_KEYS or API_KEY environment variable. "
                    "API authentication is currently DISABLED."
                )
                _API_KEYS = set()
    
    return _API_KEYS


def is_auth_enabled() -> bool:
    """Check if API authentication is enabled."""
    return len(get_api_keys()) > 0


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """
    Verify the API key from the request header.
    
    If authentication is disabled (no keys configured), allows all requests.
    If enabled, validates the X-API-Key header.
    
    Returns:
        The validated API key or None if auth is disabled.
        
    Raises:
        HTTPException: If authentication fails.
    """
    # If no keys are configured, auth is disabled
    if not is_auth_enabled():
        return None
    
    # Key is required when auth is enabled
    if api_key is None:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Validate the key using constant-time comparison
    valid_keys = get_api_keys()
    if not any(secrets.compare_digest(api_key, valid_key) for valid_key in valid_keys):
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    logger.debug(f"API key validated: {api_key[:8]}...")
    return api_key


async def optional_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """
    Optional API key verification for public endpoints.
    
    Validates the key if provided, but doesn't require it.
    Useful for endpoints that have different behavior for authenticated users.
    
    Returns:
        The validated API key or None if not provided/invalid.
    """
    if api_key is None:
        return None
    
    if not is_auth_enabled():
        return api_key
    
    valid_keys = get_api_keys()
    if any(secrets.compare_digest(api_key, valid_key) for valid_key in valid_keys):
        return api_key
    
    return None


def generate_api_key(prefix: str = "fd") -> str:
    """
    Generate a new secure API key.
    
    Args:
        prefix: Prefix for the key (default: "fd" for fraud-detection)
        
    Returns:
        A new API key in format: {prefix}_{random_32_chars}
    """
    random_part = secrets.token_urlsafe(24)  # 32 chars after base64
    return f"{prefix}_{random_part}"


# CLI utility for key generation
if __name__ == "__main__":
    print("Generated API Key:")
    print(generate_api_key())
