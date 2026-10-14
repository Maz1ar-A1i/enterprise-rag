# Postgresql - 02 Indexes Queries Performance

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **postgresql**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## POSTGRESQL-001 | Why indexes exist

An index can let PostgreSQL locate qualifying rows faster than scanning an entire table, but indexes add write and storage overhead. More indexes are not automatically better. Every index should have a workload reason.

## POSTGRESQL-002 | B-tree indexes

B-tree is the general-purpose index type used for many equality, range and ordering queries. It is often the default starting point for ordinary scalar columns, subject to actual query patterns.

## POSTGRESQL-003 | GIN, GiST and BRIN

PostgreSQL supports several index types with different data-structure strengths. GIN is useful for certain composite and containment-style workloads; GiST supports generalized search structures; BRIN summarizes physical ranges and can be very small for naturally ordered large tables. Choose based on operator and data characteristics.

## POSTGRESQL-004 | Composite indexes

A multi-column index can support queries that constrain or order by multiple columns. Column order matters. Test real query plans because an index that seems logically appropriate may still be unused or dominated by another access path.

## POSTGRESQL-005 | Partial indexes

A partial index covers only rows satisfying a predicate. This can make an index smaller and more selective when the application repeatedly queries a subset such as active records. The query predicate must be compatible enough for the planner to use the index.

## POSTGRESQL-006 | Expression indexes

An index can be built on an expression, such as a normalized form of a value. This is useful when queries repeatedly apply the same transformation. The expression must match planner expectations and the workload should justify the extra write cost.

## POSTGRESQL-007 | EXPLAIN and EXPLAIN ANALYZE

`EXPLAIN` shows the planned execution strategy. `EXPLAIN ANALYZE` actually executes the query and reports observed timing and row counts. Because it executes the statement, use caution with writes and production data. Compare estimated versus actual cardinalities to diagnose planner misestimates.

## POSTGRESQL-008 | Statistics and ANALYZE

The planner relies on table statistics to estimate row counts and choose plans. `ANALYZE` updates statistics. Stale or insufficient statistics can produce poor plans even when indexes exist.

## POSTGRESQL-009 | Slow query diagnosis

Start with the actual query, parameters and execution plan. Identify scans, joins, sorts, I/O and row-estimate errors. Do not add an index before understanding whether the bottleneck is CPU, I/O, locking, network transfer or application misuse.

## POSTGRESQL-010 | RAG metadata indexing

For the capstone, PostgreSQL can store document metadata, source identifiers, version tags, ingestion status, evaluation records and experiment metadata. Index fields used for filtering, but avoid placing large document bodies into indexes merely because they exist in the table.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://www.postgresql.org/docs/current/indexes.html
- https://www.postgresql.org/docs/current/mvcc-intro.html
- https://www.postgresql.org/docs/current/backup.html
- https://www.postgresql.org/docs/current/sql-createindex.html

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
