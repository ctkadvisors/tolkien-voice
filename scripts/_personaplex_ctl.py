#!/usr/bin/env python3
"""Start or stop the PersonaPlex runtime directly, bypassing the FastAPI
controller/session layer. Used by the offline native A/B evaluation, which
talks to PersonaPlex's raw websocket protocol directly rather than through
a browser session.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tolkien_voice.runtime import PersonaPlexRuntime, RuntimeStartError
from tolkien_voice.settings import load_runtime_settings


async def start(config_path: Path) -> int:
    settings = load_runtime_settings(config_path)
    runtime = PersonaPlexRuntime(settings)
    try:
        await runtime.start()
    except RuntimeStartError as error:
        print(f"start failed: {error}\n{error.diagnostics}", file=sys.stderr)
        return 1
    print("ready")
    return 0


async def stop(config_path: Path) -> int:
    settings = load_runtime_settings(config_path)
    runtime = PersonaPlexRuntime(settings)
    runtime._started = True
    await runtime.stop()
    print("stopped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "stop"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "start":
        return asyncio.run(start(args.config))
    return asyncio.run(stop(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
