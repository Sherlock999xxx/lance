## Index selection (quick)

Use this file when the user asks "which index should I use" or "how do I tune it".

Always confirm:

- The query pattern (filter-only, vector-only, hybrid)
- Data scale (rows, vector dimension)
- Update pattern (append vs frequent updates/deletes)
- Correctness needs (must return top-k within a filtered subset vs best-effort)

## Decision table

| Workload | Recommended starting point | Notes |
| --- | --- | --- |
| Filter-only scans (`scanner(filter=...)`) | Create a scalar index on the filtered column | Choose scalar index type based on predicate shape and cardinality |
| Vector search only (`nearest=...`) on large data | Build a vector index | Start with `IVF_PQ` if you need compression; tune `nprobes` / `refine_factor` |
| Vector search + selective filter | Scalar index for filter + vector index for search | Use `prefilter=True` when you need true top-k among filtered rows |
| Vector search + non-selective filter | Vector index only (or scalar index optional) | Consider `prefilter=False` for speed; accept fewer than k results |
| Text search | Create a text-oriented scalar index | Use `full_text_query=...` when available; verify the supported index type in the current Lance version |

## Vector index types (user-facing summary)

Vector index names typically follow a pattern like `{clustering}_{sub_index}_{quantization}`.

Common combinations:

- `IVF_PQ`: IVF clustering + PQ compression
- `IVF_HNSW_SQ`: IVF clustering + HNSW + SQ
- `IVF_SQ`: IVF clustering + SQ
- `IVF_RQ`: IVF clustering + RQ
- `IVF_FLAT`: IVF clustering + no quantization (exact vectors within clusters)

If you are unsure which types are supported in the user's environment, recommend starting with `IVF_PQ` and fall back to "try and see" (the API will error on unsupported types).

## Vector index creation defaults

Start with:

- `index_type="IVF_PQ"`
- `num_partitions`: 64 to 1024 (higher for larger datasets)
- `num_sub_vectors`: choose a value that divides the vector dimension

Practical warning (performance):

- Avoid misalignment: `(dimension / num_sub_vectors) % 8 == 0` is a common sweet spot for faster index creation.

## Vector search tuning defaults

Tune recall vs latency with:

- `nprobes`: how many IVF partitions to search
- `refine_factor`: how many candidates to re-rank to improve accuracy

When a user reports "too slow" or "bad recall", ask for:

- Current `nprobes`, `refine_factor`, and index type
- Whether the query is using `prefilter`

## Scalar index selection (starting guidance)

Choose scalar index type based on the filter expression:

- Equality filters on high-cardinality columns: start with `BTREE`
- Equality / IN-list filters on low-cardinality columns: start with `BITMAP`
- Text search: start with `FTS` (or other text index types supported by the version)
- Range filters: start with range-friendly options (for example `ZONEMAP` when appropriate)

If you cannot confidently map the filter to an index type, recommend `BTREE` as a safe baseline and confirm via a small benchmark on representative queries.
