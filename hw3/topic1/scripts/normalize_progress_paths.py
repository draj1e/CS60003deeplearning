#!/usr/bin/env python3
"""Normalize progress.json artifact paths to repository-relative paths."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRESS_JSON = ROOT / "outputs" / "progress.json"


def rel(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return path
    return path


def main() -> None:
    data = json.loads(PROGRESS_JSON.read_text())
    if "workspace" in data:
        data["workspace"] = "."
    for stage in data.get("stages", {}).values():
        stage["artifacts"] = [rel(a) for a in stage.get("artifacts", [])]
    PROGRESS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
