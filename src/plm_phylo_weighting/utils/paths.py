from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def result_subdir(work_dir: str | Path, name: str) -> Path:
    return ensure_dir(Path(work_dir) / name)


def embedding_dir(work_dir: str | Path) -> Path:
    return result_subdir(work_dir, "embeddings")


def model_cache_dir(work_dir: str | Path, model_key: str) -> Path:
    return ensure_dir(embedding_dir(work_dir) / model_key)
