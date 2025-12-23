#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import resource
import sys
import time
from typing import Any, Dict, Optional


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def read_peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def create_fts_index(
    dataset: Any,
    column: str,
    replace: bool,
    index_name: Optional[str],
    tokenizer: Optional[str],
    with_position: bool,
) -> None:
    kwargs: Dict[str, Any] = {
        "index_type": "INVERTED",
        "replace": replace,
        "with_position": with_position,
    }
    if index_name:
        kwargs["name"] = index_name
    if tokenizer:
        kwargs["base_tokenizer"] = tokenizer

    dataset.create_scalar_index(column, **kwargs)


def main() -> None:
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - only used when lance missing
        raise SystemExit(
            "lance is required for this script. Install with 'pip install pylance'."
        ) from exc

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_uri", help="lance dataset URI")
    parser.add_argument("--column", default="doc")
    parser.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to replace an existing FTS index",
    )
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--with-position", action="store_true", default=False)
    args = parser.parse_args()

    dataset = lance.dataset(args.dataset_uri)

    peak_before = read_peak_rss_bytes()
    start = time.perf_counter()
    create_fts_index(
        dataset,
        column=args.column,
        replace=args.replace,
        index_name=args.index_name,
        tokenizer=args.tokenizer,
        with_position=args.with_position,
    )
    duration = time.perf_counter() - start
    peak_after = read_peak_rss_bytes()
    peak_delta = max(peak_after - peak_before, 0)

    result = {
        "dataset_uri": args.dataset_uri,
        "column": args.column,
        "replace": args.replace,
        "index_name": args.index_name,
        "tokenizer": args.tokenizer,
        "with_position": args.with_position,
        "duration_s": duration,
        "peak_rss_bytes": peak_after,
        "peak_rss_human": human_bytes(peak_after),
        "peak_rss_delta_bytes": peak_delta,
        "peak_rss_delta_human": human_bytes(peak_delta),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
