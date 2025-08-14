"""Utility helpers for loading and saving persistent agent state.

This simple JSON-based storage layer keeps the demo free of external
infrastructure requirements. When the environment variable
``POLY_AGENT_DATA_DIR`` is set, state files are written there, otherwise the
local ``data/`` directory relative to the project root is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic.json import pydantic_encoder

from .models import OrderModel, StateModel

_DATA_DIR_ENV = "POLY_AGENT_DATA_DIR"
_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "data"

STATE_FILE = "state.json"
ORDERS_FILE = "orders.json"


def _data_dir() -> Path:
    root = os.getenv(_DATA_DIR_ENV)
    if root:
        return Path(root)
    return _DEFAULT_DIR


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(obj: Any, fp: Path) -> None:
    fp.write_text(json.dumps(obj, default=pydantic_encoder, indent=2))


def _json_load(fp: Path) -> Any | None:
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def load_state() -> StateModel | None:
    """Read the latest ``StateModel`` from disk."""
    fp = _data_dir() / STATE_FILE
    payload = _json_load(fp)
    if payload is None:
        return None
    return StateModel.model_validate(payload)


def save_state(state: StateModel) -> None:
    """Persist the given ``StateModel`` to disk."""
    target = _data_dir()
    _ensure_dir(target)
    _json_dump(state.model_dump(), target / STATE_FILE)


def append_order(order: OrderModel) -> None:
    """Append an ``OrderModel`` entry to ``orders.json`` lazily creating the file."""
    target = _data_dir()
    _ensure_dir(target)
    orders_fp = target / ORDERS_FILE
    orders = _json_load(orders_fp) or []
    orders.append(order.model_dump())
    _json_dump(orders, orders_fp)
