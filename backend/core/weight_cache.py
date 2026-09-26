"""
Resident cache for the models the API would otherwise reload on every start.

The Qwen guard and the MiniLM embedder stay in this process, on GPU.
Restarting uvicorn reconnects to the cache instead of loading the weights again.
The process keeps running after the API exits. Stop it with the pid in
backend/data/weight_cache.pid when you want the GPU memory back.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("aegis.weight_cache")

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
PID_PATH = DATA_DIR / "weight_cache.pid"
LOG_PATH = DATA_DIR / "weight_cache.log"
LOCK_PATH = DATA_DIR / "weight_cache.lock"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PORT = 8091

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "ready": False,
    "embedder": False,
    "guard": False,
    "model_path": "",
    "error": None,
}
_EMBEDDER: Any = None
_TOKENIZER: Any = None
_GUARD: Any = None


def cache_port() -> int:
    raw = os.getenv("AEGIS_WEIGHT_CACHE_PORT", str(DEFAULT_PORT))
    return int(raw)


def cache_url() -> str:
    return os.getenv(
        "AEGIS_WEIGHT_CACHE_URL",
        f"http://127.0.0.1:{cache_port()}",
    ).rstrip("/")


def _read_pid() -> int | None:
    try:
        return int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def fetch_health(timeout: float = 0.4) -> dict[str, Any] | None:
    try:
        with urlopen(f"{cache_url()}/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        f"{cache_url()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise RuntimeError(f"Weight cache returned an invalid {path} response.")
    if body.get("error"):
        raise RuntimeError(str(body["error"]))
    return body


def classify_text(text: str, max_length: int = 512) -> dict[str, Any]:
    return _post(
        "/classify",
        {"text": text, "max_length": max_length},
        timeout=60,
    )


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    body = _post(
        "/embed",
        {"texts": texts, "batch_size": batch_size},
        timeout=180,
    )
    vectors = body.get("vectors")
    if not isinstance(vectors, list):
        raise RuntimeError("Weight cache returned no embeddings.")
    return vectors


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


class RemoteEmbedder:
    """SentenceTransformer-shaped client. Weights stay in the cache process."""

    def __init__(self) -> None:
        self._tokenizer: Any | None = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    EMBEDDING_MODEL_NAME,
                    local_files_only=True,
                )
            except Exception:  # noqa: BLE001
                logger.warning("weight_cache.tokenizer_fallback")
                self._tokenizer = _WhitespaceTokenizer()
        return self._tokenizer

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **_kwargs: Any,
    ) -> Any:
        del normalize_embeddings, show_progress_bar
        single = isinstance(sentences, str)
        texts = [sentences] if single else [str(item) for item in sentences]
        import numpy as np

        if not texts:
            array = np.zeros((0, 384), dtype=np.float32)
        else:
            array = np.asarray(
                embed_texts(texts, batch_size=batch_size),
                dtype=np.float32,
            )
        if single and len(array):
            array = array[0]
        if not convert_to_numpy:
            return array.tolist()
        return array


def _spawn(model_path: str) -> None:
    import fcntl

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if fetch_health() is not None or _pid_alive(_read_pid()):
            return
        log_file = LOG_PATH.open("a")
        env = os.environ.copy()
        env["AEGIS_WEIGHT_CACHE_CHILD"] = "1"
        env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "core.weight_cache",
                "--serve",
                "--model",
                model_path,
            ],
            cwd=str(BACKEND_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        PID_PATH.write_text(str(process.pid))


def ensure_weight_cache(model_path: str, timeout: float = 240) -> None:
    """Connect to the resident cache, starting it once if it is not up."""
    wanted = str(Path(model_path))
    health = fetch_health()
    if health and health.get("ready") and health.get("model_path") == wanted:
        logger.info("weight_cache.reused url=%s", cache_url())
        return
    if health and health.get("error"):
        raise RuntimeError(str(health["error"]))
    if health is None and not _pid_alive(_read_pid()):
        logger.info("weight_cache.starting url=%s", cache_url())
        _spawn(wanted)

    deadline = time.time() + timeout
    while time.time() < deadline:
        health = fetch_health(timeout=1)
        if health and health.get("ready") and health.get("model_path") == wanted:
            logger.info(
                "weight_cache.ready url=%s pid=%s",
                cache_url(),
                health.get("pid"),
            )
            return
        if health and health.get("error"):
            raise RuntimeError(str(health["error"]))
        time.sleep(0.5)
    raise TimeoutError(
        "The weight cache did not finish loading. "
        f"See {LOG_PATH}."
    )


def cache_is_alive() -> bool:
    return fetch_health() is not None or _pid_alive(_read_pid())


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_embedder() -> None:
    global _EMBEDDER
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    logger.info("weight_cache.loading_embedder")
    try:
        _EMBEDDER = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=_device(),
            local_files_only=True,
        )
    except TypeError:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device())
    _STATE["embedder"] = True


def _load_guard(model_path: str) -> None:
    global _TOKENIZER, _GUARD
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    merged = Path(model_path)
    if not merged.exists():
        raise FileNotFoundError(f"Guard weights not found at {merged}")

    dtype = (
        torch.bfloat16
        if _device() == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    logger.info("weight_cache.loading_guard model=%s", merged)
    tokenizer = AutoTokenizer.from_pretrained(
        str(merged),
        trust_remote_code=True,
        local_files_only=True,
    )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(merged),
            num_labels=2,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(merged),
            num_labels=2,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(_device())
    model.eval()
    _TOKENIZER = tokenizer
    _GUARD = model
    _STATE["guard"] = True
    _STATE["model_path"] = str(merged)


def _classify(text: str, max_length: int) -> dict[str, float | str]:
    import torch

    if _GUARD is None or _TOKENIZER is None:
        raise RuntimeError("Guard weights are not loaded in the cache.")
    with _LOCK:
        encoded = _TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        encoded = {key: value.to(_device()) for key, value in encoded.items()}
        with torch.no_grad():
            logits = _GUARD(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1).squeeze(0)
    safe = float(probabilities[0].item())
    attack = float(probabilities[1].item())
    label = "PROMPT_INJECTION" if attack >= safe else "SAFE"
    return {
        "safe_probability": safe,
        "attack_probability": attack,
        "predicted_label": label,
    }


def _embed(texts: list[str], batch_size: int) -> list[list[float]]:
    if _EMBEDDER is None:
        raise RuntimeError("Embedding weights are not loaded in the cache.")
    with _LOCK:
        vectors = _EMBEDDER.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return vectors.tolist()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON object required.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(
            200,
            {
                "ready": _STATE["ready"],
                "embedder": _STATE["embedder"],
                "guard": _STATE["guard"],
                "model_path": _STATE["model_path"],
                "error": _STATE["error"],
                "pid": os.getpid(),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            payload = self._read_json()
            if path == "/classify":
                text = str(payload.get("text") or "")
                result = _classify(text, int(payload.get("max_length") or 512))
                self._send(200, result)
                return
            if path == "/embed":
                texts = payload.get("texts") or []
                if isinstance(texts, str):
                    texts = [texts]
                vectors = _embed(
                    [str(item) for item in texts],
                    int(payload.get("batch_size") or 32),
                )
                self._send(200, {"vectors": vectors})
                return
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("weight_cache.request_failed")
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("weight_cache.http %s", fmt % args)


def _load_all(model_path: str) -> None:
    try:
        _load_embedder()
        _load_guard(model_path)
        _STATE["ready"] = True
        logger.info("weight_cache.models_resident pid=%s", os.getpid())
    except Exception as exc:  # noqa: BLE001
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("weight_cache.load_failed")
        time.sleep(1.5)
        os._exit(1)


def serve(model_path: str) -> None:
    logging.basicConfig(
        level=os.getenv("AEGIS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))
    server = ThreadingHTTPServer(("127.0.0.1", cache_port()), _Handler)
    threading.Thread(
        target=_load_all,
        args=(model_path,),
        name="weight-cache-load",
        daemon=True,
    ).start()
    logger.info("weight_cache.listening url=%s pid=%s", cache_url(), os.getpid())
    try:
        server.serve_forever()
    finally:
        if _read_pid() == os.getpid():
            PID_PATH.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aegis resident model cache")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if not args.serve:
        parser.error("--serve is required")
    serve(args.model)


if __name__ == "__main__":
    main()
