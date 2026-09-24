"""Aegis EHR backend core modules."""

__all__ = [
    "ClinicalAssistant",
    "ClinicalAgent",
    "EdgeGuard",
    "CloudDefender",
    "AttackerAgent",
    "EHRIngestion",
]


def __getattr__(name: str):
    if name in {"ClinicalAssistant", "ClinicalAgent"}:
        from .clinical_agent import ClinicalAgent, ClinicalAssistant

        return ClinicalAgent if name == "ClinicalAgent" else ClinicalAssistant
    if name == "EdgeGuard":
        from .edge_guard import EdgeGuard

        return EdgeGuard
    if name == "CloudDefender":
        from .cloud_defender import CloudDefender

        return CloudDefender
    if name == "AttackerAgent":
        from .attacker import AttackerAgent

        return AttackerAgent
    if name == "EHRIngestion":
        from .ingestion import EHRIngestion

        return EHRIngestion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
