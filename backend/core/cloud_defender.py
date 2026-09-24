"""
Cloud Defender client interface for Aegis EHR.

EdgeGuard escalates only PHI-scrubbed ambiguous prompts here.
Attack category assignment happens in the cloud (not on the edge
binary classifier).

This module provides a typed client stub suitable for dependency
injection and local development. Wire a real HTTP/API backend later.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

VALID_ATTACK_CLASSES = frozenset(
    {
        "DIRECT_INJECTION",
        "INDIRECT_INJECTION",
        "PERSONA_JAILBREAK",
        "TOOL_EXFILTRATION",
        "UNKNOWN",
    }
)

VALID_THREAT_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


@runtime_checkable
class CloudDefenderProtocol(Protocol):
    """Minimal protocol EdgeGuard depends on."""

    def analyze(self, scrubbed_prompt: str) -> dict[str, Any]:
        """
        Analyze a PHI-scrubbed prompt.

        Expected return keys:
            is_attack: bool
            attack_class: str
            threat_level: str
            extracted_intent: str
            hot_patch_signature: str | None
            confidence_score: float
        """
        ...


class CloudDefenderError(Exception):
    """Raised when cloud analysis fails or times out."""


class CloudDefenderUnavailable(CloudDefenderError):
    """Raised when the cloud defender cannot be reached."""


class CloudDefender:
    """
    Cloud forensics / hot-patch client.

    By default this client is a fail-closed stub: calling analyze()
    raises CloudDefenderUnavailable unless ``handler`` is provided
    or a concrete HTTP backend is configured later.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 8.0,
        handler: Any | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._handler = handler

    def analyze(self, scrubbed_prompt: str) -> dict[str, Any]:
        if not isinstance(scrubbed_prompt, str) or not scrubbed_prompt.strip():
            raise ValueError("scrubbed_prompt must be a non-empty string.")

        # Never log the prompt body (even scrubbed) at info level.
        logger.info(
            "cloud_defender.analyze",
            extra={"prompt_chars": len(scrubbed_prompt)},
        )

        if self._handler is not None:
            result = self._handler(scrubbed_prompt)
            return normalize_cloud_result(result)

        if not self.endpoint:
            raise CloudDefenderUnavailable(
                "CloudDefender endpoint is not configured."
            )

        # Placeholder for a future HTTP implementation.
        raise CloudDefenderUnavailable(
            f"CloudDefender HTTP backend is not implemented "
            f"(endpoint={self.endpoint!r})."
        )


def normalize_cloud_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a cloud defender response."""
    if not isinstance(result, dict):
        raise CloudDefenderError("Cloud defender returned a non-dict result.")

    required = (
        "is_attack",
        "attack_class",
        "threat_level",
        "extracted_intent",
        "confidence_score",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise CloudDefenderError(
            f"Cloud defender response missing keys: {missing}"
        )

    attack_class = str(result["attack_class"]).upper()
    if attack_class not in VALID_ATTACK_CLASSES:
        attack_class = "UNKNOWN"

    threat_level = str(result["threat_level"]).upper()
    if threat_level not in VALID_THREAT_LEVELS:
        raise CloudDefenderError(
            f"Invalid threat_level from cloud: {result['threat_level']!r}"
        )

    confidence = float(result["confidence_score"])
    if not 0.0 <= confidence <= 1.0:
        raise CloudDefenderError(
            f"confidence_score out of range: {confidence}"
        )

    signature = result.get("hot_patch_signature")
    if signature is not None:
        signature = str(signature).strip() or None

    return {
        "is_attack": bool(result["is_attack"]),
        "attack_class": attack_class,
        "threat_level": threat_level,
        "extracted_intent": str(result["extracted_intent"]),
        "hot_patch_signature": signature,
        "confidence_score": confidence,
    }
