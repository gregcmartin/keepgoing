#!/usr/bin/env python3
"""
Start the mlx_lm model server with TurboQuant KV cache compression.

TurboQuant (PolarQuant) achieves 3-5x KV cache memory compression
with minimal quality loss, enabling much longer context windows
on the same hardware.

Usage:
    python scripts/start_server.py [--bits 4] [--port 8000] [--model MODEL]

Without TurboQuant installed, falls back to the standard server.
"""
import argparse
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Start mlx_lm server with TurboQuant KV cache")
    parser.add_argument(
        "--model",
        default="nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx",
        help="Model name or path",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to generate")
    parser.add_argument(
        "--bits",
        type=float,
        default=4,
        choices=[2, 3, 3.5, 4],
        help="KV cache quantization bits (lower = more compression, slight quality loss)",
    )
    parser.add_argument(
        "--no-turboquant",
        action="store_true",
        help="Disable TurboQuant and use standard KV cache",
    )
    args = parser.parse_args()

    turboquant_available = False
    if not args.no_turboquant:
        try:
            from mlx_turboquant.integration import patch_sdpa
            patch_sdpa(bits=args.bits)
            turboquant_available = True
            print(f"[keepgoing] TurboQuant enabled: {args.bits}-bit KV cache quantization")
            print(f"[keepgoing] Expected ~{4.6 if args.bits >= 3 else 4.0}x KV cache compression")
        except ImportError:
            print("[keepgoing] TurboQuant not installed, using standard KV cache")
            print("[keepgoing] Install with: pip install mlx-turboquant")
        except Exception as e:
            print(f"[keepgoing] TurboQuant patch failed ({e}), falling back to standard KV cache")

    # Start the server via mlx_lm.server
    # We import and run it directly so the monkey-patch stays active
    try:
        from mlx_lm.server import main as server_main
        sys.argv = [
            "mlx_lm.server",
            "--model", args.model,
            "--port", str(args.port),
            "--max-tokens", str(args.max_tokens),
        ]
        print(f"[keepgoing] Starting model server on port {args.port}")
        print(f"[keepgoing] Model: {args.model}")
        server_main()
    except ImportError:
        print("[keepgoing] mlx-lm not installed. Install with: pip install mlx-lm")
        sys.exit(1)


if __name__ == "__main__":
    main()
