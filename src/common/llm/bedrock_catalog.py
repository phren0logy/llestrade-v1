"""
Utilities for discovering Anthropic Claude models available via AWS Bedrock.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BedrockModel:
    """Lightweight description of an Anthropic Claude model exposed by Bedrock."""

    model_id: str
    name: str
    provider_name: str = "Anthropic"
    region: Optional[str] = None


_DEFAULT_MODELS: List[BedrockModel] = [
    BedrockModel(
        model_id="anthropic.claude-sonnet-4-5-v1",
        name="Claude Sonnet 4.5",
    ),
    BedrockModel(
        model_id="anthropic.claude-opus-4-6-v1",
        name="Claude Opus 4.6",
    ),
]

_CLAUDE_MODEL_VERSION_RE = re.compile(
    r"claude-(?:[a-z]+-)?(?P<major>\d+)(?:-(?P<minor>\d+))?",
    re.IGNORECASE,
)

_catalog_lock = Lock()
_cached_models: List[BedrockModel] = list(_DEFAULT_MODELS)
_cache_key: tuple[Optional[str], Optional[str]] = (None, None)
_cache_expiry: float = 0.0
_CACHE_TTL_SECONDS = 900.0  # 15 minutes
DEFAULT_BEDROCK_MODELS: List[BedrockModel] = list(_DEFAULT_MODELS)


def _create_session(profile: Optional[str]):
    """Create a boto3 session using the optional profile name."""
    try:
        import boto3  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "boto3 is required for AWS Bedrock support. Install anthropic[bedrock]."
        ) from exc

    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile
    return boto3.Session(**session_kwargs)


def _fetch_models_from_bedrock(
    *,
    region: Optional[str],
    profile: Optional[str],
) -> Iterable[BedrockModel]:
    """Retrieve Anthropic Claude models from AWS Bedrock."""
    try:
        session = _create_session(profile)
        client_kwargs = {}
        if region:
            client_kwargs["region_name"] = region

        bedrock_client = session.client("bedrock", **client_kwargs)
        discovered_region = bedrock_client.meta.region_name

        models: List[BedrockModel] = []
        next_token: Optional[str] = None

        while True:
            params = {"byProvider": "Anthropic"}
            if next_token:
                params["nextToken"] = next_token

            response = bedrock_client.list_foundation_models(**params)
            summaries = response.get("modelSummaries") or []
            for summary in summaries:
                model_id = summary.get("modelId")
                model_name = summary.get("modelName") or model_id
                if not model_id:
                    continue
                if not _supports_minimum_version(model_id):
                    continue
                models.append(
                    BedrockModel(
                        model_id=model_id,
                        name=model_name,
                        provider_name=summary.get("providerName", "Anthropic"),
                        region=discovered_region,
                    )
                )

            next_token = response.get("nextToken")
            if not next_token:
                break

        if models:
            return models

    except Exception as exc:
        logger.debug("Bedrock model discovery failed: %s", exc)

    return list(DEFAULT_BEDROCK_MODELS)


def _supports_minimum_version(model_id: str) -> bool:
    """Allow only Claude models at version 4.1 or newer."""
    match = _CLAUDE_MODEL_VERSION_RE.search(model_id or "")
    if not match:
        return False

    major = int(match.group("major"))
    minor_text = match.group("minor")
    minor = int(minor_text) if minor_text else 0
    return major > 4 or (major == 4 and minor >= 1)


def list_bedrock_models(
    *,
    region: Optional[str],
    profile: Optional[str],
    force_refresh: bool = False,
) -> List[BedrockModel]:
    """
    Return a cached list of Anthropic Claude models exposed via Bedrock.

    Args:
        region: Optional AWS region override.
        profile: Optional named profile from the AWS config/credentials files.
        force_refresh: If True, bypass the existing cache.
    """
    global _cached_models, _cache_expiry, _cache_key

    key = (profile, region)
    now = time.time()

    with _catalog_lock:
        if (
            not force_refresh
            and key == _cache_key
            and now < _cache_expiry
            and _cached_models
        ):
            return list(_cached_models)

        models = list(
            _fetch_models_from_bedrock(region=region, profile=profile)
        )

        _cached_models = models
        _cache_key = key
        _cache_expiry = now + _CACHE_TTL_SECONDS
        return list(models)


def preferred_bedrock_model(region: Optional[str], profile: Optional[str]) -> str:
    """Return the highest-priority model ID based on availability."""
    models = list_bedrock_models(region=region, profile=profile)
    if not models:
        return DEFAULT_BEDROCK_MODELS[0].model_id
    # Prefer the first entry, which is the newest known model.
    return models[0].model_id


__all__ = [
    "BedrockModel",
    "DEFAULT_BEDROCK_MODELS",
    "list_bedrock_models",
    "preferred_bedrock_model",
]
