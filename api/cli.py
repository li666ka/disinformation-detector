"""CLI utilities. Запуск: `python -m api.cli <command>`.

Commands:
  check-ml-server   — health-probe Colab ML server, друкує JSON + exit code.
"""
from __future__ import annotations

import argparse
import json
import sys

from api.ml_client import check_status


def _cmd_check_ml_server(args: argparse.Namespace) -> int:
    status = check_status(force=True)  # CLI завжди свіжий probe
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m api.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_health = sub.add_parser(
        "check-ml-server",
        help="Health-probe Colab ML server (друкує JSON, exit 0=ok, 1=offline)",
    )
    p_health.set_defaults(func=_cmd_check_ml_server)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
