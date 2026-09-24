"""Aegis EHR backend core modules."""

__all__ = ["ClinicalAssistant", "EdgeGuard", "CloudDefender"]


def __getattr__(name: str):
    if name == "ClinicalAssistant":
        from .clinical_agent import ClinicalAssistant

        return ClinicalAssistant
    if name == "EdgeGuard":
        from .edge_guard import EdgeGuard

        return EdgeGuard
    if name == "CloudDefender":
        from .cloud_defender import CloudDefender

        return CloudDefender
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
