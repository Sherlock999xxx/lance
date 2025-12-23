#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

BENCH_PATH = Path(__file__).with_name("bench_fts_index.py")

spec = importlib.util.spec_from_file_location("fts_bench", BENCH_PATH)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError("Failed to load bench_fts_index module")

fts_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fts_bench)


class DummyDataset:
    def __init__(self):
        self.called = None

    def create_scalar_index(self, column, **kwargs):
        self.called = {"column": column, **kwargs}


def test_create_fts_index_passes_expected_kwargs():
    dummy = DummyDataset()

    fts_bench.create_fts_index(
        dummy,
        column="doc",
        replace=False,
        index_name="fts_idx",
        tokenizer="whitespace",
        with_position=True,
    )

    assert dummy.called is not None
    assert dummy.called["column"] == "doc"
    assert dummy.called["index_type"] == "INVERTED"
    assert dummy.called["replace"] is False
    assert dummy.called["with_position"] is True
    assert dummy.called["name"] == "fts_idx"
    assert dummy.called["base_tokenizer"] == "whitespace"
