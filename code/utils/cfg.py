"""Config loader for PQ-CANchor.

Reads `code/config.yaml` (or whichever path `MAKALE2_CONFIG` env var points to)
and resolves simple `${var}` interpolation. The only interpolation source is
top-level scalar strings under `datasets`, so a `makale1_root` key becomes
`${makale1_root}` available everywhere downstream.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _interpolate(node: Any, vars_: dict[str, str]) -> Any:
    if isinstance(node, str):
        out = node
        for k, v in vars_.items():
            out = out.replace("${" + k + "}", v)
        return out
    if isinstance(node, dict):
        return {k: _interpolate(v, vars_) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(item, vars_) for item in node]
    return node


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("MAKALE2_CONFIG") or DEFAULT_CONFIG)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ds = cfg.get("datasets", {}) or {}
    vars_: dict[str, str] = {k: v for k, v in ds.items() if isinstance(v, str)}
    return _interpolate(cfg, vars_)


def project_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["project"]["paper_root"])


def dataset_path(name: str, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    entry = cfg["datasets"][name]
    p = Path(entry["path"])
    if not p.is_absolute():
        p = project_root(cfg) / p
    return p
