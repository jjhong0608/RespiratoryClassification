from __future__ import annotations

import argparse
import json

from transformers import ASTConfig

from src.utils.logging import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_or_path",
        help="Hugging Face AST model id or local config/checkpoint directory",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON to stdout.",
    )
    args = parser.parse_args()

    if not args.name_or_path:
        raise SystemExit("Error: provide --name_or_path.")

    cfg = ASTConfig.from_pretrained(args.name_or_path, cache_dir=args.cache_dir)
    payload = {
        "name_or_path": args.name_or_path,
        "num_mel_bins": int(cfg.num_mel_bins),
        "max_length": int(cfg.max_length),
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "intermediate_size": int(cfg.intermediate_size),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    logger.info("name_or_path=%s", args.name_or_path)
    logger.info(
        "ast_config: num_mel_bins=%d, max_length=%d, hidden_size=%d, "
        "num_hidden_layers=%d, num_attention_heads=%d, intermediate_size=%d",
        payload["num_mel_bins"],
        payload["max_length"],
        payload["hidden_size"],
        payload["num_hidden_layers"],
        payload["num_attention_heads"],
        payload["intermediate_size"],
    )
    logger.info("Suggested config snippet:")
    logger.info(
        json.dumps(
            {
                "data": {
                    "features": {
                        "ast_fbank": {
                            "num_mel_bins": payload["num_mel_bins"],
                            "max_length": payload["max_length"],
                        }
                    }
                },
                "model": {
                    "encoder": {
                        "pretrained_name_or_path": args.name_or_path,
                    }
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
