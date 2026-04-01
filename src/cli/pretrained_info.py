from __future__ import annotations

import argparse
import json

from src.pretrained.whisper import OpenAIWhisperCheckpointLoader
from src.utils.logging import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_or_path",
        help="OpenAI Whisper model name (e.g. tiny/base/...) or local .pt checkpoint path",
    )
    parser.add_argument(
        "--download_root",
        default=None,
        help="Cache directory for official model downloads (default: ~/.cache/whisper)",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="Print the supported official model names and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON to stdout.",
    )
    args = parser.parse_args()

    loader = OpenAIWhisperCheckpointLoader()
    if args.list_models:
        for name in loader.available_models():
            logger.info(name)
        return

    if not args.name_or_path:
        raise SystemExit("Error: provide --name_or_path (or use --list_models).")

    resolved = loader.resolve_checkpoint_path(args.name_or_path, args.download_root)
    dims = loader.inspect_encoder_dims(
        args.name_or_path, download_root=args.download_root
    )

    if args.json:
        print(
            json.dumps(
                {
                    "name_or_path": args.name_or_path,
                    "resolved_path": str(resolved),
                    "encoder_dims": {
                        "n_mels": dims.n_mels,
                        "n_audio_ctx": dims.n_audio_ctx,
                        "n_audio_state": dims.n_audio_state,
                        "n_audio_head": dims.n_audio_head,
                        "n_audio_layer": dims.n_audio_layer,
                    },
                },
                indent=2,
            )
        )
        return

    logger.info(f"name_or_path={args.name_or_path}")
    logger.info(f"resolved_path={resolved}")
    logger.info(
        f"encoder_dims: n_mels={dims.n_mels}, n_audio_ctx={dims.n_audio_ctx}, "
        f"n_audio_state={dims.n_audio_state}, n_audio_head={dims.n_audio_head}, "
        f"n_audio_layer={dims.n_audio_layer}"
    )
    logger.info("Suggested config snippet (encoder dims):")
    logger.info(
        json.dumps(
            {
                "model": {
                    "n_mels": dims.n_mels,
                    "n_audio_ctx": dims.n_audio_ctx,
                    "n_audio_state": dims.n_audio_state,
                    "n_audio_head": dims.n_audio_head,
                    "n_audio_layer": dims.n_audio_layer,
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
