#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect
import json
import time
from typing import Any, Dict, Optional

try:
    import lancedb
except ImportError as exc:  # pragma: no cover - only used when lancedb missing
    raise SystemExit(
        "lancedb is required for this script. Install with 'pip install lancedb'."
    ) from exc


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def create_fts_index(
    table: Any,
    column: str,
    replace: bool,
    index_name: Optional[str],
    tokenizer: Optional[str],
    with_position: bool,
) -> None:
    if hasattr(table, "create_fts_index"):
        fn = table.create_fts_index
        kwargs: Dict[str, Any] = {"replace": replace}
        if index_name:
            for key in ("index_name", "name"):
                if key in inspect.signature(fn).parameters:
                    kwargs[key] = index_name
                    break
        if tokenizer:
            for key in ("tokenizer_name", "tokenizer"):
                if key in inspect.signature(fn).parameters:
                    kwargs[key] = tokenizer
                    break
        if with_position:
            for key in ("with_position", "with_positions"):
                if key in inspect.signature(fn).parameters:
                    kwargs[key] = True
                    break
        kwargs = _filter_kwargs(fn, kwargs)
        fn(column, **kwargs)
        return

    if hasattr(table, "create_index"):
        fn = table.create_index
        kwargs = {"index_type": "FTS", "replace": replace}
        if index_name:
            kwargs["index_name"] = index_name
        kwargs = _filter_kwargs(fn, kwargs)
        fn(column, **kwargs)
        return

    raise RuntimeError("table does not support FTS index creation")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("db_uri", help="lancedb database URI")
    parser.add_argument("table", help="table name")
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

    db = lancedb.connect(args.db_uri)
    if hasattr(db, "open_table"):
        table = db.open_table(args.table)
    else:
        table = db.table(args.table)

    start = time.perf_counter()
    create_fts_index(
        table,
        column=args.column,
        replace=args.replace,
        index_name=args.index_name,
        tokenizer=args.tokenizer,
        with_position=args.with_position,
    )
    duration = time.perf_counter() - start

    result = {
        "table": args.table,
        "column": args.column,
        "replace": args.replace,
        "index_name": args.index_name,
        "tokenizer": args.tokenizer,
        "with_position": args.with_position,
        "duration_s": duration,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
