"""Command-line entry point for nano-vLLM."""

from __future__ import annotations

import argparse

from nano_vllm import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by the package entry point."""
    parser = argparse.ArgumentParser(
        prog="nano-vllm",
        description="A small, vLLM-inspired inference engine for Qwen3 models.",
    )
    parser.add_argument("--version", action="version", version=f"nano-vllm {__version__}")
    return parser


def main() -> None:
    """Run the command-line interface.

    Generation options will be added with the offline Qwen3 API in M5.
    """
    build_parser().parse_args()


if __name__ == "__main__":
    main()
