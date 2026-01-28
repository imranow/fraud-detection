"""Simple feature store for real-time feature serving."""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a feature set."""
    
    name: str
    version: str
    created_at: datetime
    updated_at: datetime
    schema: Dict[str, str]  # feature_name -> type
    ttl_seconds: int = 3600
    description: str = ""


@dataclass
class FeatureValue:
    """A cached feature value with metadata."""
    
    entity_id: str
    features: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    version: str = "1.0"
    
    def is_fresh(self, ttl_seconds: int = 3600) -> bool:
        """Check if the feature value is still fresh."""
        return (time.time() - self.timestamp) < ttl_seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "features": self.features,
            "timestamp": self.timestamp,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureValue":
        """Create from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            features=data["features"],
            timestamp=data.get("timestamp", time.time()),
            version=data.get("version", "1.0"),
        )


class InMemoryFeatureStore:
    """In-memory feature store for development and testing."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, FeatureValue] = {}
        self._metadata: Dict[str, FeatureMetadata] = {}
    
    def _make_key(self, feature_set: str, entity_id: str) -> str:
        """Create a cache key."""
        return f"{feature_set}:{entity_id}"
    
    def get(
        self,
        feature_set: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get features for an entity.
        
        Args:
            feature_set: Name of the feature set
            entity_id: Entity identifier (e.g., transaction ID, user ID)
            feature_names: Optional list of specific features to retrieve
            
        Returns:
            Dictionary of features or None if not found/stale
        """
        key = self._make_key(feature_set, entity_id)
        value = self._cache.get(key)
        
        if value is None:
            return None
        
        if not value.is_fresh(self.ttl_seconds):
            del self._cache[key]
            return None
        
        features = value.features
        
        if feature_names:
            features = {k: v for k, v in features.items() if k in feature_names}
        
        return features
    
    def set(
        self,
        feature_set: str,
        entity_id: str,
        features: Dict[str, Any],
        version: str = "1.0",
    ) -> None:
        """
        Store features for an entity.
        
        Args:
            feature_set: Name of the feature set
            entity_id: Entity identifier
            features: Dictionary of feature values
            version: Feature version
        """
        key = self._make_key(feature_set, entity_id)
        self._cache[key] = FeatureValue(
            entity_id=entity_id,
            features=features,
            version=version,
        )
    
    def get_batch(
        self,
        feature_set: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get features for multiple entities."""
        return {
            entity_id: self.get(feature_set, entity_id, feature_names)
            for entity_id in entity_ids
        }
    
    def set_batch(
        self,
        feature_set: str,
        entities: Dict[str, Dict[str, Any]],
        version: str = "1.0",
    ) -> None:
        """Store features for multiple entities."""
        for entity_id, features in entities.items():
            self.set(feature_set, entity_id, features, version)
    
    def delete(self, feature_set: str, entity_id: str) -> bool:
        """Delete features for an entity."""
        key = self._make_key(feature_set, entity_id)
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self, feature_set: Optional[str] = None) -> int:
        """Clear all or a specific feature set."""
        if feature_set is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        
        count = 0
        keys_to_delete = [k for k in self._cache if k.startswith(f"{feature_set}:")]
        for key in keys_to_delete:
            del self._cache[key]
            count += 1
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        fresh = sum(1 for v in self._cache.values() if v.is_fresh(self.ttl_seconds))
        return {
            "total_entries": len(self._cache),
            "fresh_entries": fresh,
            "stale_entries": len(self._cache) - fresh,
            "ttl_seconds": self.ttl_seconds,
        }


class RedisFeatureStore:
    """Redis-backed feature store for production use."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
        key_prefix: str = "features",
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self._redis = None
        self._connected = False
    
    def _get_redis(self):
        """Lazy initialization of Redis connection."""
        if not self._connected:
            try:
                import redis
                self._redis = redis.from_url(self.redis_url)
                self._redis.ping()
                self._connected = True
                logger.info("Redis feature store connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._redis = None
        return self._redis
    
    def _make_key(self, feature_set: str, entity_id: str) -> str:
        """Create a Redis key."""
        return f"{self.key_prefix}:{feature_set}:{entity_id}"
    
    def get(
        self,
        feature_set: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get features for an entity."""
        redis = self._get_redis()
        if not redis:
            return None
        
        try:
            key = self._make_key(feature_set, entity_id)
            data = redis.get(key)
            
            if not data:
                return None
            
            value = FeatureValue.from_dict(json.loads(data))
            features = value.features
            
            if feature_names:
                features = {k: v for k, v in features.items() if k in feature_names}
            
            return features
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(
        self,
        feature_set: str,
        entity_id: str,
        features: Dict[str, Any],
        version: str = "1.0",
    ) -> None:
        """Store features for an entity."""
        redis = self._get_redis()
        if not redis:
            return
        
        try:
            key = self._make_key(feature_set, entity_id)
            value = FeatureValue(entity_id=entity_id, features=features, version=version)
            redis.setex(key, self.ttl_seconds, json.dumps(value.to_dict()))
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def get_batch(
        self,
        feature_set: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get features for multiple entities using Redis pipeline."""
        redis = self._get_redis()
        if not redis:
            return {eid: None for eid in entity_ids}
        
        try:
            pipe = redis.pipeline()
            keys = [self._make_key(feature_set, eid) for eid in entity_ids]
            
            for key in keys:
                pipe.get(key)
            
            results = pipe.execute()
            
            output = {}
            for entity_id, data in zip(entity_ids, results):
                if data:
                    value = FeatureValue.from_dict(json.loads(data))
                    features = value.features
                    if feature_names:
                        features = {k: v for k, v in features.items() if k in feature_names}
                    output[entity_id] = features
                else:
                    output[entity_id] = None
            
            return output
            
        except Exception as e:
            logger.error(f"Redis batch get error: {e}")
            return {eid: None for eid in entity_ids}
    
    def set_batch(
        self,
        feature_set: str,
        entities: Dict[str, Dict[str, Any]],
        version: str = "1.0",
    ) -> None:
        """Store features for multiple entities using Redis pipeline."""
        redis = self._get_redis()
        if not redis:
            return
        
        try:
            pipe = redis.pipeline()
            
            for entity_id, features in entities.items():
                key = self._make_key(feature_set, entity_id)
                value = FeatureValue(entity_id=entity_id, features=features, version=version)
                pipe.setex(key, self.ttl_seconds, json.dumps(value.to_dict()))
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Redis batch set error: {e}")
    
    def delete(self, feature_set: str, entity_id: str) -> bool:
        """Delete features for an entity."""
        redis = self._get_redis()
        if not redis:
            return False
        
        try:
            key = self._make_key(feature_set, entity_id)
            return bool(redis.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        redis = self._get_redis()
        if not redis:
            return {"connected": False}
        
        try:
            info = redis.info("memory")
            keys = redis.keys(f"{self.key_prefix}:*")
            
            return {
                "connected": True,
                "total_keys": len(keys),
                "used_memory": info.get("used_memory_human", "unknown"),
                "ttl_seconds": self.ttl_seconds,
            }
        except Exception as e:
            return {"connected": True, "error": str(e)}


class FeatureStore:
    """
    Feature store with automatic backend selection.
    
    Uses Redis if available, falls back to in-memory storage.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
        prefer_redis: bool = True,
    ):
        self.ttl_seconds = ttl_seconds
        
        # Try Redis first if preferred
        if prefer_redis and redis_url:
            self._store = RedisFeatureStore(redis_url, ttl_seconds)
            if self._store._get_redis():
                logger.info("Feature store using Redis backend")
                return
        
        # Fall back to in-memory
        self._store = InMemoryFeatureStore(ttl_seconds)
        logger.info("Feature store using in-memory backend")
    
    def get(
        self,
        feature_set: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        return self._store.get(feature_set, entity_id, feature_names)
    
    def set(
        self,
        feature_set: str,
        entity_id: str,
        features: Dict[str, Any],
        version: str = "1.0",
    ) -> None:
        self._store.set(feature_set, entity_id, features, version)
    
    def get_batch(
        self,
        feature_set: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        return self._store.get_batch(feature_set, entity_ids, feature_names)
    
    def set_batch(
        self,
        feature_set: str,
        entities: Dict[str, Dict[str, Any]],
        version: str = "1.0",
    ) -> None:
        self._store.set_batch(feature_set, entities, version)
    
    def delete(self, feature_set: str, entity_id: str) -> bool:
        return self._store.delete(feature_set, entity_id)
    
    def stats(self) -> Dict[str, Any]:
        return self._store.stats()


# Global feature store instance
_store: Optional[FeatureStore] = None


def get_feature_store(
    redis_url: Optional[str] = None,
    ttl_seconds: int = 3600,
) -> FeatureStore:
    """Get or create the global feature store instance."""
    global _store
    if _store is None:
        _store = FeatureStore(
            redis_url=redis_url or os.getenv("REDIS_URL"),
            ttl_seconds=ttl_seconds,
        )
    return _store
