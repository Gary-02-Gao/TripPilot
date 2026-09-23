"""Runtime configuration shared by the Agent, tools and local services.

The project intentionally keeps configuration dependency-free.  Values are read
from the process environment and, when present, from ``.env`` in the project
root.  Existing environment variables always win.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Optional[Path] = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def require_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}。请复制 .env.example 为 .env 后填写，"
            "或使用离线 Demo 模式。"
        )
    return value


load_dotenv()
