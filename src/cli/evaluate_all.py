from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cli.evaluate import evaluate_checkpoint
from src.utils.config import JsonConfigLoader
from src.utils.logging import enable_file_logging, logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base eval config (shared)")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to search for checkpoints (.pt) via rglob",
    )
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help="Glob pattern for checkpoints (default: *.pt)",
    )
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_eval(args.config)
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    enable_file_logging(root / "evaluate_all.log", mode="w")

    checkpoints = sorted(root.rglob(args.pattern))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found under {root} with {args.pattern}"
        )

    logger.info(f"Found {len(checkpoints)} checkpoints under {root}")

    for ckpt_path in checkpoints:
        metrics = evaluate_checkpoint(cfg, ckpt_path)
        out_name = f"eval_metrics__{ckpt_path.stem}.json"
        out_path = ckpt_path.parent / out_name
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
