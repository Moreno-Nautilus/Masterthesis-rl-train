#!/usr/bin/env python3
"""Check CUDA memory for silent byte corruption.

Run with no training or other CUDA jobs active. The process exits nonzero when any
written byte reads back incorrectly.
"""

import argparse
import gc
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-gib", type=int, nargs="+", default=[1, 4, 8, 12, 18])
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    patterns = (0x00, 0xFF, 0xAA, 0x55)
    chunk_bytes = args.chunk_mib * 1024**2
    any_bad = False
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for size_gib in args.sizes_gib:
        size_bytes = size_gib * 1024**3
        free_bytes, _ = torch.cuda.mem_get_info()
        if size_bytes + chunk_bytes > free_bytes:
            print(f"{size_gib:>2} GiB: SKIP (only {free_bytes / 1024**3:.1f} GiB free)")
            continue

        buffer = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        counts = []
        started = time.monotonic()
        for pattern in patterns:
            buffer.fill_(pattern)
            torch.cuda.synchronize()
            bad_bytes = 0
            for start in range(0, size_bytes, chunk_bytes):
                stop = min(start + chunk_bytes, size_bytes)
                bad_bytes += int((buffer[start:stop] != pattern).sum())
            counts.append(bad_bytes)
            any_bad |= bad_bytes > 0

        status = "FAIL" if any(counts) else "PASS"
        print(f"{size_gib:>2} GiB: {status} bad bytes={counts} ({time.monotonic() - started:.2f}s)")
        del buffer
        gc.collect()
        torch.cuda.empty_cache()

    if any_bad:
        print("CUDA memory corruption detected", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
