import os
import sqlite3

import ujson as json

from .core import Component, Result


class Storage(Component):
    """Take the input and store it somewhere"""

    def build_json(self, input: Result):
        return input.to_json(recurse=False)

    def _process(self, input: Result) -> Result:
        # Do persistence here
        return input


class FileSystemStorage(Storage):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "directory" not in self.settings:
            self.settings["directory"] = "results"
        dn = self.settings["directory"]
        if not os.path.exists(dn):
            os.makedirs(dn, exist_ok=True)

    def _process(self, input: Result) -> Result:
        """Write the result to the filesystem according to the processor id"""
        if input.processor is not None:
            # Store in directory per processor
            base = os.path.join(self.settings["directory"], input.processor.id)
        else:
            base = os.path.join(self.settings["directory"], "base")
        if not os.path.exists(base):
            os.makedirs(base, exist_ok=True)
        # Now make a pair-tree
        pair = os.path.join(base, input.id[0:2], input.id[2:4])
        if not os.path.exists(pair):
            os.makedirs(pair, exist_ok=True)
        fn = os.path.join(pair, f"{input.id}.json")
        if os.path.exists(fn):
            # make a new version
            vn = 1  # FIXME: Make this the count of files with this name
            fn = f"{fn}.{vn}"
        js = self.build_json(input)
        with open(fn, "w") as fh:
            json.dump(js, fh)

        return input


class PostgresStorage(Storage):
    pass


class SqliteStorage(Storage):
    """Store results in a SQLite database"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "database" not in self.settings:
            self.settings["database"] = "results.db"
        self.conn = None

    def _get_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.settings["database"])
            self._init_db()
        return self.conn

    def _init_db(self):
        """Initialize the database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                processor_id TEXT,
                workflow_id TEXT,
                value_json TEXT,
                metadata_json TEXT,
                extra_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS derivatives (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                component_id TEXT,
                result_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_processor ON results(processor_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_workflow ON results(workflow_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_derivatives_source ON derivatives(source_id)
        """)
        conn.commit()

    def _process(self, input: Result) -> Result:
        """Store the result in SQLite"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build JSON representations
        value_json = json.dumps(input.value) if input.value else None
        metadata_json = json.dumps(input.metadata) if input.metadata else None
        extra_json = json.dumps(input.extra) if input.extra else None

        # Insert result
        cursor.execute(
            """
            INSERT OR REPLACE INTO results
            (id, processor_id, workflow_id, value_json, metadata_json, extra_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                input.id,
                input.processor.id if input.processor else None,
                input.workflow.id if input.workflow else None,
                value_json,
                metadata_json,
                extra_json,
            ),
        )

        # Store derivatives
        for component, results in input.derivative_results.items():
            for result in results:
                result_json = json.dumps(result.to_json())
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO derivatives
                    (id, source_id, component_id, result_json)
                    VALUES (?, ?, ?, ?)
                """,
                    (result.id, input.id, component.id, result_json),
                )

        conn.commit()
        return input

    def __del__(self):
        """Close connection on cleanup"""
        if self.conn is not None:
            self.conn.close()
