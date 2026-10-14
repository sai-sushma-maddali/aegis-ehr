"""Load deployment settings from ``aegis.config.json`` at the repo root.

Resolution order for each value:
1. Environment variable ``AEGIS_*`` (when set)
2. ``aegis.config.json`` (or path in ``AEGIS_CONFIG``)
3. Portable built-in defaults (safe for laptop / missing GPU assets)

Absolute paths are used as-is. Relative paths are resolved against the
repository root (parent of ``backend/``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("aegis.config")

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_CONFIG_NAME = "aegis.config.json"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _dig(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _resolve_path(raw: str | None, *, project_dir: Path) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (project_dir / path).resolve()
    return path


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for Nano, Fury, or a local laptop."""

    project_dir: Path
    config_path: Path | None

    # Platform labels (UI / telemetry)
    platform_display_name: str = "HP ZGX"
    gpu_fallback_name: str = "NVIDIA GPU"
    architecture_blurb: str = (
        "On-device GPU (CPU/GPU load from nvidia-smi when available)"
    )
    memory_label: str = "System memory"
    rated_tops_fp4: float | None = None

    # Paths (None / missing → optional feature skipped)
    guard_merged_model: Path | None = None
    qwen_base_model: Path | None = None
    mistral_model_dir: Path | None = None
    zrt_metrics_socket: Path | None = None
    chroma_dir: Path = field(
        default_factory=lambda: BACKEND_DIR / "data" / "chroma_aegis_db"
    )
    raw_data_dir: Path = field(
        default_factory=lambda: BACKEND_DIR / "data" / "raw"
    )
    ehr_records_file: Path = field(
        default_factory=lambda: BACKEND_DIR / "data" / "ehr_records.txt"
    )
    seed_threats_file: Path = field(
        default_factory=lambda: BACKEND_DIR / "data" / "seed_threats.json"
    )

    # Services
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_host: str = "0.0.0.0"
    ui_port: int = 5173
    clinical_llm_base_url: str = "http://127.0.0.1:8080/v1"
    clinical_llm_model: str = "hf:mistralai/Mistral-7B-Instruct-v0.3@main"
    weight_cache_host: str = "127.0.0.1"
    weight_cache_port: int = 8091

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    peft_adapter_id: str = "SushmaMaddali/qwen-medical-prompt-guard-lora"
    cloud_defender_model: str = "gemma-4-31b-it"
    cloud_defender_fallback: str = "gemma-4-26b-a4b-it"
    ehr_repo_id: str = "serag-ai/Synthetic-EHR-Llama"

    # Thresholds
    vector_similarity: float = 0.82
    allow_threshold: float = 0.15
    block_threshold: float = 0.85

    # Flags
    offline_mode: bool = False
    skip_clinical: bool = False
    weight_cache: bool = True
    reset_on_boot: bool = False

    # CORS
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    cors_origin_regex: str = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    @property
    def weight_cache_url(self) -> str:
        return f"http://{self.weight_cache_host}:{self.weight_cache_port}"

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    def platform_public(self) -> dict[str, Any]:
        return {
            "display_name": self.platform_display_name,
            "gpu_fallback_name": self.gpu_fallback_name,
            "architecture_blurb": self.architecture_blurb,
            "memory_label": self.memory_label,
            "rated_tops_fp4": self.rated_tops_fp4,
        }


def _config_candidates(project_dir: Path) -> list[Path]:
    env_path = os.getenv("AEGIS_CONFIG", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(project_dir / DEFAULT_CONFIG_NAME)
    candidates.append(project_dir / "aegis.config.example.json")
    return candidates


def _load_raw(project_dir: Path) -> tuple[dict[str, Any], Path | None]:
    for path in _config_candidates(project_dir):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("config.load_failed path=%s err=%s", path, exc)
            continue
        if isinstance(data, dict):
            return data, path.resolve()
    return {}, None


def load_settings(project_dir: Path | None = None) -> Settings:
    root = (project_dir or PROJECT_DIR).resolve()
    raw, config_path = _load_raw(root)

    chroma = _resolve_path(
        _env_str(
            "AEGIS_CHROMA_PATH",
            str(_dig(raw, "paths", "chroma_dir", default="backend/data/chroma_aegis_db")),
        ),
        project_dir=root,
    ) or (BACKEND_DIR / "data" / "chroma_aegis_db")

    raw_dir = _resolve_path(
        str(_dig(raw, "paths", "raw_data_dir", default="backend/data/raw")),
        project_dir=root,
    ) or (BACKEND_DIR / "data" / "raw")

    ehr_file = _resolve_path(
        str(
            _dig(
                raw,
                "paths",
                "ehr_records_file",
                default="backend/data/ehr_records.txt",
            )
        ),
        project_dir=root,
    ) or (BACKEND_DIR / "data" / "ehr_records.txt")

    seed_threats = _resolve_path(
        _env_str(
            "AEGIS_SEED_THREATS_FILE",
            str(
                _dig(
                    raw,
                    "paths",
                    "seed_threats_file",
                    default="backend/data/seed_threats.json",
                )
            ),
        ),
        project_dir=root,
    ) or (BACKEND_DIR / "data" / "seed_threats.json")

    cors_list = _dig(raw, "cors", "origins", default=None)
    if isinstance(cors_list, list) and cors_list:
        cors_origins = tuple(str(item).strip() for item in cors_list if str(item).strip())
    else:
        cors_origins = (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        )
    cors_env = os.getenv("AEGIS_CORS_ORIGINS")
    if cors_env:
        cors_origins = tuple(
            part.strip() for part in cors_env.split(",") if part.strip()
        )

    rated = _dig(raw, "platform", "rated_tops_fp4", default=None)
    if rated is not None:
        try:
            rated = float(rated)
        except (TypeError, ValueError):
            rated = None

    wc_host = str(
        _dig(raw, "services", "weight_cache_host", default="127.0.0.1")
    )
    wc_port = int(
        _dig(raw, "services", "weight_cache_port", default=8091)
    )
    wc_url_default = f"http://{wc_host}:{wc_port}"

    settings = Settings(
        project_dir=root,
        config_path=config_path,
        platform_display_name=str(
            _dig(raw, "platform", "display_name", default="HP ZGX")
        ),
        gpu_fallback_name=str(
            _dig(raw, "platform", "gpu_fallback_name", default="NVIDIA GPU")
        ),
        architecture_blurb=str(
            _dig(
                raw,
                "platform",
                "architecture_blurb",
                default=(
                    "On-device GPU (CPU/GPU load from nvidia-smi when available)"
                ),
            )
        ),
        memory_label=str(
            _dig(raw, "platform", "memory_label", default="System memory")
        ),
        rated_tops_fp4=rated,
        guard_merged_model=_resolve_path(
            _env_str(
                "AEGIS_GUARD_MODEL_PATH",
                str(_dig(raw, "paths", "guard_merged_model", default="") or ""),
            ),
            project_dir=root,
        ),
        qwen_base_model=_resolve_path(
            _env_str(
                "AEGIS_QWEN_BASE_MODEL",
                str(_dig(raw, "paths", "qwen_base_model", default="") or ""),
            ),
            project_dir=root,
        ),
        mistral_model_dir=_resolve_path(
            _env_str(
                "AEGIS_MISTRAL_MODEL_DIR",
                str(_dig(raw, "paths", "mistral_model_dir", default="") or ""),
            ),
            project_dir=root,
        ),
        zrt_metrics_socket=_resolve_path(
            _env_str(
                "AEGIS_ZRT_METRICS_SOCKET",
                str(_dig(raw, "paths", "zrt_metrics_socket", default="") or ""),
            ),
            project_dir=root,
        ),
        chroma_dir=chroma,
        raw_data_dir=raw_dir,
        ehr_records_file=ehr_file,
        seed_threats_file=seed_threats,
        api_host=_env_str(
            "AEGIS_HOST",
            str(_dig(raw, "services", "api_host", default="127.0.0.1")),
        ),
        api_port=_env_int(
            "AEGIS_PORT",
            int(_dig(raw, "services", "api_port", default=8000)),
        ),
        ui_host=str(_dig(raw, "services", "ui_host", default="0.0.0.0")),
        ui_port=int(_dig(raw, "services", "ui_port", default=5173)),
        clinical_llm_base_url=_env_str(
            "AEGIS_CLINICAL_LLM_BASE_URL",
            str(
                _dig(
                    raw,
                    "services",
                    "clinical_llm_base_url",
                    default="http://127.0.0.1:8080/v1",
                )
            ),
        ).rstrip("/"),
        clinical_llm_model=_env_str(
            "AEGIS_CLINICAL_LLM_MODEL",
            str(
                _dig(
                    raw,
                    "services",
                    "clinical_llm_model",
                    default="hf:mistralai/Mistral-7B-Instruct-v0.3@main",
                )
            ),
        ),
        weight_cache_host=_env_str("AEGIS_WEIGHT_CACHE_HOST", wc_host),
        weight_cache_port=_env_int("AEGIS_WEIGHT_CACHE_PORT", wc_port),
        embedding_model=_env_str(
            "AEGIS_EMBEDDING_MODEL",
            str(
                _dig(
                    raw,
                    "models",
                    "embedding",
                    default="sentence-transformers/all-MiniLM-L6-v2",
                )
            ),
        ),
        peft_adapter_id=str(
            _dig(
                raw,
                "models",
                "peft_adapter_id",
                default="SushmaMaddali/qwen-medical-prompt-guard-lora",
            )
        ),
        cloud_defender_model=_env_str(
            "AEGIS_CLOUD_DEFENDER_MODEL",
            str(_dig(raw, "models", "cloud_defender", default="gemma-4-31b-it")),
        ),
        cloud_defender_fallback=str(
            _dig(
                raw,
                "models",
                "cloud_defender_fallback",
                default="gemma-4-26b-a4b-it",
            )
        ),
        ehr_repo_id=str(
            _dig(
                raw,
                "models",
                "ehr_repo_id",
                default="serag-ai/Synthetic-EHR-Llama",
            )
        ),
        vector_similarity=_env_float(
            "AEGIS_VECTOR_SIMILARITY",
            float(_dig(raw, "thresholds", "vector_similarity", default=0.82)),
        ),
        allow_threshold=_env_float(
            "AEGIS_ALLOW_THRESHOLD",
            float(_dig(raw, "thresholds", "allow", default=0.15)),
        ),
        block_threshold=_env_float(
            "AEGIS_BLOCK_THRESHOLD",
            float(_dig(raw, "thresholds", "block", default=0.85)),
        ),
        offline_mode=_env_flag(
            "AEGIS_OFFLINE_MODE",
            bool(_dig(raw, "flags", "offline_mode", default=False)),
        ),
        skip_clinical=_env_flag(
            "AEGIS_SKIP_CLINICAL",
            bool(_dig(raw, "flags", "skip_clinical", default=False)),
        ),
        weight_cache=_env_flag(
            "AEGIS_WEIGHT_CACHE",
            bool(_dig(raw, "flags", "weight_cache", default=True)),
        ),
        reset_on_boot=_env_flag(
            "AEGIS_RESET_ON_BOOT",
            bool(_dig(raw, "flags", "reset_on_boot", default=False)),
        ),
        cors_origins=cors_origins,
        cors_origin_regex=_env_str(
            "AEGIS_CORS_ORIGIN_REGEX",
            str(
                _dig(
                    raw,
                    "cors",
                    "origin_regex",
                    default=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
                )
            ),
        ),
    )

    # Allow full URL override for weight cache.
    url_override = os.getenv("AEGIS_WEIGHT_CACHE_URL", "").strip()
    if url_override:
        # Host/port already set; callers prefer weight_cache_url property —
        # stash via env is handled in weight_cache.py reading settings URL.
        pass

    logger.info(
        "config.loaded path=%s offline=%s skip_clinical=%s guard=%s",
        config_path or "(defaults)",
        settings.offline_mode,
        settings.skip_clinical,
        settings.guard_merged_model or "(none)",
    )
    # Silence unused if no override — keep API for callers that check env.
    _ = wc_url_default
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
