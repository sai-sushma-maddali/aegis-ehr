"""
Test-facing Attacker Agent entrypoint.

Re-exports ``AttackerAgent`` from ``backend.core.attacker`` with the default
seed path pinned to ``tests/attack_prompts.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.attacker import (  # noqa: E402
    DEFAULT_SEED_PATH,
    MUTATION_SYSTEM_PROMPT,
    VALID_ATTACK_CLASSES,
    AttackerAgent,
    AttackClass,
)

SEED_PATH = Path(__file__).resolve().parent / "attack_prompts.json"


def build_attacker(
    mode: str = "benchmark",
    model_provider: str = "groq",
    model_name: str = "llama-3.1-8b-instant",
    **kwargs,
) -> AttackerAgent:
    """Factory that defaults seed_path to tests/attack_prompts.json."""
    kwargs.setdefault("seed_path", SEED_PATH)
    return AttackerAgent(
        mode=mode,
        model_provider=model_provider,
        model_name=model_name,
        **kwargs,
    )


__all__ = [
    "AttackerAgent",
    "AttackClass",
    "VALID_ATTACK_CLASSES",
    "MUTATION_SYSTEM_PROMPT",
    "DEFAULT_SEED_PATH",
    "SEED_PATH",
    "build_attacker",
]
