"""Move CHAI-TEA results and annotations out of a local PostgreSQL instance and into a central one.

The command line for ``chai.storage.postgres_to_parquet`` and ``parquet_bundle_to_postgres``: export
writes one Parquet file per table, plus a ``manifest.json``, into a directory; load puts that
directory into the central database in one transaction. The rows the exported results and
annotations reference travel with them, so nothing dangles (see ``postgres_to_parquet``).

    # on the local instance
    python chai_tea_parquet.py export ./upload --database chai_tea
    python chai_tea_parquet.py export ./upload --database chai_tea --since 2026-09-01

    # against the central database
    python chai_tea_parquet.py load ./upload --host master.example.edu --database chai_tea

Needs ``pyarrow`` (as the other Parquet paths do) and ``psycopg``.
"""

import argparse
import logging
import sys

from chai.storage import BUNDLE_MANIFEST, parquet_bundle_to_postgres, postgres_to_parquet


def _connection_args(parser):
    """The connection options both subcommands take."""
    parser.add_argument("--dsn", default=None, help="connection string; overrides the options below")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--database", default=None, help="default: chai")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)


def _connection(args):
    names = ("dsn", "host", "port", "database", "user", "password")
    return {name: getattr(args, name) for name in names if getattr(args, name)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="write the local results and annotations to a bundle")
    export.add_argument("directory", help="where to write the Parquet files and manifest.json")
    export.add_argument("--since", default=None, help="only results and annotations from this time on")
    export.add_argument("--project", default=None, help="limit to one project, by name or id")
    export.add_argument("--workflow", default=None, help="limit to one workflow, by name or id")
    export.add_argument("--include-permissions", action="store_true", help="also export roles and permissions")
    export.add_argument("--batch-size", type=int, default=50000)
    export.add_argument("--compression", default="snappy")
    _connection_args(export)

    load = sub.add_parser("load", help="load a bundle into the central database")
    load.add_argument("directory", help="the bundle to load")
    load.add_argument(
        "--on-conflict",
        choices=("skip", "update"),
        default="skip",
        help="rows the central database already has: leave them (default) or overwrite them",
    )
    load.add_argument("--no-create", action="store_true", help="fail instead of creating missing tables there")
    load.add_argument("--batch-size", type=int, default=10000)
    _connection_args(load)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "export":
        manifest = postgres_to_parquet(
            args.directory,
            _connection(args),
            since=args.since,
            project=args.project,
            workflow=args.workflow,
            include_permissions=args.include_permissions,
            batch_size=args.batch_size,
            compression=args.compression,
        )
        for entry in manifest["tables"]:
            print(f"  {entry['rows']:>9,} {entry['table']} -> {entry['file']}")
        print(f"Wrote {args.directory}/{BUNDLE_MANIFEST}")
    else:
        loaded = parquet_bundle_to_postgres(
            args.directory,
            _connection(args),
            on_conflict=args.on_conflict,
            create=not args.no_create,
            batch_size=args.batch_size,
        )
        for counts in loaded.values():
            print(f"  {counts['written']:>9,} written, {counts['skipped']:>9,} already there  {counts['table']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
