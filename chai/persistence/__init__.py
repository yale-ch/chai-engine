"""Persistence: the CHAI-TEA schema, its rows, and writing them to files and databases.

What the ``chai.storage`` components write, and how -- with no components here:

* ``schema`` -- the CHAI-TEA tables (``TABLES``, the DDL, ``column_types``/``column_kinds``)
* ``records`` -- ``Project`` and ``WorkflowRun``, the PROJECT and WORKFLOW_RUN rows as objects
* ``provenance`` -- where a result came from: its input, that input's md5, where in it the result is
* ``rows`` -- a result as a RESULT row (``result_row``), or a file storage's record (``build_record``)
* ``jsonl`` -- JSON-Lines files, the run records kept beside them, and ``token_usage_summary``
* ``parquet`` -- ``ParquetRecordWriter``, ``jsonl_to_parquet`` and reading a Parquet file back
* ``sql`` -- ``write_rows``, the upsert writer for both databases
* ``sqlite`` / ``postgres`` -- each database: its tables, writing rows, loading files, corrections
* ``bundle`` -- moving rows between PostgreSQL databases as a bundle of Parquet files
* ``legacy`` -- reshaping files stored before the storages wrote RESULT rows
* ``vectors`` -- ``VectorStore``, embeddings in SQLite

The public functions are importable from here as well as from their modules.
"""

from .bundle import (  # noqa: F401
    BUNDLE_MANIFEST,
    BUNDLE_ORDER,
    BUNDLE_VERSION,
    PERMISSION_ORDER,
    bundle_order,
    parquet_bundle_to_postgres,
    postgres_to_parquet,
    read_bundle_manifest,
)
from .jsonl import add_run_records, read_jsonl, read_run_records, run_records_file, token_usage_summary  # noqa: F401
from .legacy import jsonl_to_result_rows, record_to_result_row  # noqa: F401
from .parquet import ParquetRecordWriter, jsonl_to_parquet, parquet_columns, parquet_run_records  # noqa: F401
from .postgres import (  # noqa: F401
    ensure_postgres_database,
    ensure_postgres_schema,
    load_run_parquet,
    parquet_to_postgres,
    postgres_params,
    save_postgres_correction,
    store_postgres_rows,
)
from .provenance import (  # noqa: F401
    content_md5,
    correction_target,
    input_locator,
    input_provenance,
    recorded_source,
    result_md5,
    result_sequence,
    source_value,
)
from .records import Project, WorkflowRun, as_datetime, as_uuid, now  # noqa: F401
from .rows import (  # noqa: F401
    COMPUTED_FIELDS,
    RESULT_COLUMNS,
    build_record,
    correction_row,
    is_result_row,
    recorded_cost,
    result_row,
    result_to_json,
)
from .schema import TABLE_ORDER, TABLES, column_kinds, column_types, table_names  # noqa: F401
from .sql import write_rows  # noqa: F401
from .sqlite import ensure_database, get_result, list_processors, list_results, save_correction, store_rows  # noqa: F401
from .vectors import VectorStore  # noqa: F401
