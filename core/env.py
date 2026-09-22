"""Minimal .env loading, stdlib only.

The service is configured entirely through environment variables, and every one of them is
documented in `.env.example`. Nothing was reading a `.env` file, so editing it had no
effect at all - a trap worth removing rather than working around.

Deliberately tiny, and deliberately not a dependency:

  * `KEY=VALUE`, one per line; blank lines and `#` comments ignored;
  * surrounding single or double quotes are stripped;
  * a leading `export ` is tolerated, so a file can be sourced by a shell as well;
  * **a real environment variable always wins.** An operator exporting a value for one run
    must not have it silently overridden by a checked-in file.

This runs before the modules that read configuration at import time, so it has to be
called early - see the top of `api/server.py`.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_env(text: str) -> dict[str, str]:
    """Parse .env text into a mapping. Pure, so it can be tested without touching the fs."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def load_env(path: str | os.PathLike[str] | None = None, override: bool = False) -> list[str]:
    """Load `.env` into ``os.environ``. Returns the names it set.

    ``override=False`` (the default) keeps any variable that is already present in the
    process environment.
    """
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return []

    applied: list[str] = []
    for key, value in parse_env(text).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
