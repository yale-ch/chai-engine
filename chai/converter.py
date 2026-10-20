"""Converters: components that turn a file into a file of another format.

A converter takes a file (a ``FileItemResult``, or a result whose value is a path) and writes the same
data out in a different format, returning a ``FileItemResult`` for the file it wrote -- so the next
step, typically an ``Uploader``, can pick the new file up. ``JsonLinesToParquet`` packs the
``.jsonl`` file a ``JsonLinesStorage`` streamed a run into as one Parquet file for a bulk load.
"""

import logging
import os
from datetime import datetime, timezone

from .core import Component, Result
from .result import FileItemResult
from .storage import jsonl_to_parquet

logger = logging.getLogger("chai")


def result_file_path(result):
    """The local path *result* stands for: a ``FileItemResult``'s file, or a path given as the value.

    Never reads a ``FileItemResult``'s content. Raises ``ValueError`` when the result names no file.
    """
    if isinstance(result, FileItemResult):
        path = result.file_name
    elif isinstance(result, Result):
        path = result.value
    else:
        path = result
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path:
        raise ValueError(f"Expected a file, or a result whose value is a file path; got {result!r}")
    return path


class Converter(Component):
    """Take a file and write it out in another format, returning a ``FileItemResult`` for the new file.

    Subclasses implement ``convert(source, target)``, which writes *target* and returns a dict of
    information about it (recorded in the output's metadata), or ``None`` when there was nothing to
    write -- in which case the converter produces no result and its ``next_steps`` do not run.

    The file is written under a temporary name and renamed into place once complete, so whatever
    picks it up never sees a half-written file.

    Settings:
        - file: path of the file to write; ``{dir}`` (the source's directory), ``{stem}`` (its name
          without the extension) and ``{name}`` (its name) are filled in from the source file
          (default '{dir}/{stem}' plus the converter's ``EXTENSION``)
        - directory: shorthand for writing '{stem}' plus the extension into another directory; ignored
          when ``file`` is set
    """

    EXTENSION = ""

    def target_path(self, source):
        """Where to write the converted form of *source*."""
        template = self.settings.get("file")
        if not template:
            directory = self.settings.get("directory") or "{dir}"
            template = os.path.join(directory, "{stem}" + self.EXTENSION)
        name = os.path.basename(source)
        return os.path.abspath(
            template.format(
                dir=os.path.dirname(os.path.abspath(source)),
                stem=os.path.splitext(name)[0],
                name=name,
            )
        )

    def convert(self, source, target):
        raise NotImplementedError(f"{self.__class__.__name__} must implement convert()")

    def _process(self, input: Result) -> Result:
        source = result_file_path(input)
        if not os.path.exists(source):
            raise ValueError(f"{self} cannot convert {source}: no such file")
        target = self.target_path(source)
        partial = f"{target}.partial"
        try:
            info = self.convert(source, partial)
        except BaseException:
            if os.path.exists(partial):
                os.remove(partial)
            raise
        if info is None:
            if os.path.exists(partial):
                os.remove(partial)
            logger.warning(f"{self} had nothing to convert in {source}; wrote no file")
            return None
        os.replace(partial, target)
        info = dict(info, path=target)
        return FileItemResult(target, input=input, processor=self, metadata={"type": "DATA", **info})


class JsonLinesToParquet(Converter):
    """Convert a JSON-Lines file (as written by ``JsonLinesStorage``) into one Parquet file.

    The workflow form of ``storage.jsonl_to_parquet``: reads and writes in batches, so neither file is
    ever fully in memory. Each line becomes a row; nested dicts and lists, and columns of mixed types,
    become JSON-encoded text columns (see ``ParquetRecordWriter``). The output's metadata records the
    ``rows`` written and the ``columns`` with their types, and the Parquet file's own metadata records
    the source file, workflow and component it was made by.

    A file with no lines writes no Parquet file (there is nothing to infer the columns from) unless
    ``schema`` declares them, in which case it writes an empty file with those columns.

    Settings (as well as ``file``/``directory``, see ``Converter``):
        - schema: ``{column: type}`` declaring column types and their order (string, json, int, float,
          bool, timestamp); columns not listed are inferred from the first batch
        - batch_size: lines per Parquet row group (default 50000); column types are inferred from the
          first batch
        - compression: Parquet codec, e.g. 'snappy' (default), 'zstd', 'gzip', 'none'
        - metadata: ``{key: value}`` added to the Parquet file's metadata -- a batch name, a target
          table -- for the system that loads it
    """

    EXTENSION = ".parquet"

    def convert(self, source, target):
        metadata = {
            "chai_source": os.path.abspath(source),
            "chai_workflow_id": self.workflow.id if self.workflow else "",
            "chai_component_id": self.id,
            "chai_converted_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata.update(self.settings.get("metadata") or {})
        schema = self.settings.get("schema")
        info = jsonl_to_parquet(
            source,
            target,
            schema=schema,
            batch_size=self.settings.get("batch_size", 50000),
            compression=self.settings.get("compression", "snappy"),
            metadata=metadata,
        )
        if not info["rows"] and not schema:
            return None
        return {"rows": info["rows"], "columns": info["columns"]}
