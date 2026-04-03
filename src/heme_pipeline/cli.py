from __future__ import annotations

import sys

from heme_pipeline.runner import build_arg_parser, run_pipeline


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(config_path=args.config, only=args.only)
    sys.exit(0)


if __name__ == "__main__":
    main()
