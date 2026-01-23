# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import importlib.util
import random
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "fts_cold_latency.py"
_SPEC = importlib.util.spec_from_file_location("fts_cold_latency", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load fts_cold_latency module for tests.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_queries = _MODULE.build_queries
extract_word_pool = _MODULE.extract_word_pool
tokenize_text = _MODULE.tokenize_text


def test_tokenize_text_min_len():
    tokens = tokenize_text("Hello, World! It's Lance.", min_len=3)
    assert tokens == ["hello", "world", "lance"]


def test_extract_word_pool_handles_lists_and_nulls():
    values = ["Hello world", None, ["Mixed CASE", None], 123]
    pool = extract_word_pool(values, min_len=3)
    assert pool == {"hello", "world", "mixed", "case"}


def test_build_queries_unique_and_sizes():
    pool = {f"word{i}" for i in range(300)}
    rng = random.Random(0)
    warm_word, queries = build_queries(pool, [5, 10, 20, 50, 100], "word0", rng)

    assert warm_word == "word0"
    assert [len(words) for words in queries] == [5, 10, 20, 50, 100]

    used = {warm_word}
    for words in queries:
        assert used.isdisjoint(words)
        used.update(words)


def test_build_queries_raises_for_insufficient_words():
    pool = {f"word{i}" for i in range(10)}
    rng = random.Random(1)
    with pytest.raises(ValueError):
        build_queries(pool, [5, 10], None, rng)


def test_build_queries_requires_warm_word_in_pool():
    pool = {f"word{i}" for i in range(50)}
    rng = random.Random(2)
    with pytest.raises(ValueError):
        build_queries(pool, [5], "missing", rng)
