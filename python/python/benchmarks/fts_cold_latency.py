# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from typing import Iterable, Sequence

DEFAULT_QUERY_SIZES = (5, 10, 20, 50, 100)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_text(text: str, min_len: int) -> list[str]:
    tokens = (token.lower() for token in TOKEN_RE.findall(text))
    return [token for token in tokens if len(token) >= min_len]


def extract_word_pool(values: Iterable[object], min_len: int) -> set[str]:
    word_pool: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            items = value
        else:
            items = [value]
        for item in items:
            if not isinstance(item, str):
                continue
            word_pool.update(tokenize_text(item, min_len))
    return word_pool


def build_queries(
    word_pool: set[str],
    sizes: Sequence[int],
    warm_word: str | None,
    rng: random.Random,
) -> tuple[str, list[list[str]]]:
    if not word_pool:
        raise ValueError("Word pool is empty. Increase sample size or adjust filters.")
    for size in sizes:
        if size <= 0:
            raise ValueError("Query sizes must be positive.")

    pool = list(word_pool)
    if warm_word is None:
        warm_word = rng.choice(pool)
        pool.remove(warm_word)
    else:
        warm_word = warm_word.lower()
        if warm_word not in word_pool:
            raise ValueError(
                f"Warm word '{warm_word}' not found in sampled words. "
                "Increase --sample-rows or choose a different word."
            )
        pool.remove(warm_word)

    rng.shuffle(pool)
    total_needed = sum(sizes)
    if len(pool) < total_needed:
        raise ValueError(
            "Not enough unique words to build queries. "
            f"Need {total_needed}, but only have {len(pool)} after excluding warm word."
        )

    queries: list[list[str]] = []
    offset = 0
    for size in sizes:
        queries.append(pool[offset : offset + size])
        offset += size
    return warm_word, queries


def parse_query_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            sizes.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid query size: {part}") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("Query sizes must not be empty.")
    return sizes


def run_fts_query(
    dataset,
    query: str,
    columns: list[str],
    limit: int | None,
    with_row_id: bool,
    include_score: bool,
) -> tuple[float, int]:
    start = time.perf_counter()
    builder = dataset.scanner().full_text_search(query, columns=columns)
    if limit is not None:
        builder = builder.limit(limit)
    builder = builder.with_row_id(with_row_id)
    if not include_score:
        builder = builder.disable_scoring_autoprojection(True)
    builder = builder.columns([])
    table = builder.to_table()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, table.num_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure cold FTS latency for an InvertedIndex."
    )
    parser.add_argument("dataset", help="Path to the Lance dataset")
    parser.add_argument(
        "--column",
        default="text",
        help="Text column used for sampling and FTS queries (default: text)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
        help="Number of rows to sample for building the word pool (default: 2000)",
    )
    parser.add_argument(
        "--min-word-len",
        type=int,
        default=3,
        help="Minimum word length to keep (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and query generation (default: 42)",
    )
    parser.add_argument(
        "--warm-word",
        default=None,
        help="Warm word used to load the index (default: auto-selected)",
    )
    parser.add_argument(
        "--query-sizes",
        type=parse_query_sizes,
        default=list(DEFAULT_QUERY_SIZES),
        help="Comma-separated query sizes (default: 5,10,20,50,100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Limit rows returned for each query (default: 1)",
    )
    parser.add_argument(
        "--print-queries",
        action="store_true",
        help="Print the query strings used for benchmarking",
    )
    parser.add_argument(
        "--include-score",
        action="store_true",
        help="Keep the score column in results (default: disabled)",
    )
    parser.add_argument(
        "--no-row-id",
        action="store_true",
        help="Do not include _rowid in results (default: include)",
    )

    args = parser.parse_args(argv)

    if args.sample_rows <= 0:
        print("Sample rows must be positive.", file=sys.stderr)
        return 2
    if args.min_word_len <= 0:
        print("Minimum word length must be positive.", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 0:
        print("Limit must be non-negative.", file=sys.stderr)
        return 2

    try:
        import lance
    except ImportError as exc:
        print("Failed to import lance. Build the python package first.", file=sys.stderr)
        raise exc

    random.seed(args.seed)
    rng = random.Random(args.seed)

    dataset = lance.dataset(args.dataset)
    if args.column not in dataset.schema.names:
        print(
            f"Column '{args.column}' not found in dataset schema: "
            f"{', '.join(dataset.schema.names)}",
            file=sys.stderr,
        )
        return 2

    total_rows = dataset.count_rows()
    if total_rows == 0:
        print("Dataset is empty.", file=sys.stderr)
        return 2

    sample_rows = min(args.sample_rows, total_rows)
    table = dataset.sample(sample_rows, columns=[args.column])
    values = table[args.column].to_pylist()
    word_pool = extract_word_pool(values, args.min_word_len)

    required_words = 1 + sum(args.query_sizes)
    if len(word_pool) < required_words:
        print(
            "Not enough unique words sampled to satisfy query sizes. "
            f"Need {required_words}, but only found {len(word_pool)}. "
            "Increase --sample-rows or lower --min-word-len.",
            file=sys.stderr,
        )
        return 2

    warm_word, query_words = build_queries(
        word_pool, args.query_sizes, args.warm_word, rng
    )

    used_words = {warm_word}
    for words in query_words:
        overlap = used_words.intersection(words)
        if overlap:
            raise RuntimeError(f"Query words overlap: {sorted(overlap)}")
        used_words.update(words)

    print(f"Dataset: {args.dataset}")
    print(f"Text column: {args.column}")
    print(f"Warm word: {warm_word}")
    print(f"Query sizes: {', '.join(str(size) for size in args.query_sizes)}")
    print(f"Sample rows: {sample_rows}")
    print(f"Min word length: {args.min_word_len}")
    print(f"Limit: {args.limit}")
    print(f"Seed: {args.seed}")
    print()

    warm_latency_ms, warm_rows = run_fts_query(
        dataset,
        warm_word,
        [args.column],
        args.limit,
        with_row_id=not args.no_row_id,
        include_score=args.include_score,
    )
    print(f"Warm query latency: {warm_latency_ms:.3f} ms (rows={warm_rows})")
    print()

    results: list[tuple[int, float, int, str]] = []
    for size, words in zip(args.query_sizes, query_words):
        query = " ".join(words)
        latency_ms, rows = run_fts_query(
            dataset,
            query,
            [args.column],
            args.limit,
            with_row_id=not args.no_row_id,
            include_score=args.include_score,
        )
        results.append((size, latency_ms, rows, query))

    print("Query results:")
    print("size\tlatency_ms\trows")
    for size, latency_ms, rows, _ in results:
        print(f"{size}\t{latency_ms:.3f}\t{rows}")

    if args.print_queries:
        print()
        print("Queries:")
        for size, _, _, query in results:
            print(f"{size} words: {query}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
