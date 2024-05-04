#!/usr/bin/env python3

import lmdb
import argparse
import sys
from pathlib import Path

def merge_dbs(src_db_paths: list[Path], target_db_path: Path) -> None:
    """
    Merges source dbs into target dbs.
    :param src_db_paths (list[Path]): source databases to merge together
    :param target_db_path (Path): target database to merge sources into
    """
    for db_path in src_db_paths:
        if not db_path.exists():
            raise ValueError(f'Database {str(db_path)} does not exist!')

    target_db = lmdb.open(str(target_db_path), create=True)
    for db_path in src_db_paths:
        print(f'Merging database {str(db_path)}')
        source_db = lmdb.open(str(db_path))
        source_tx = source_db.begin()
        source_cursor = source_tx.cursor()
        for i, (key, value) in enumerate(
            source_cursor.iternext(keys=True, values=True)
        ):
            target_tx = target_db.begin(write=True)
            try:
                target_tx.put(key=key, value=value)
                target_tx.commit()
            except lmdb.MapFullError:
                target_tx.abort()
                target_db.set_mapsize(target_db.info()['map_size'] * 2)
                target_tx = target_db.begin(write=True)
                target_tx.put(key=key, value=value)
                target_tx.commit()
            if i % 1000 == 0:
                print(f'Merged records: {i}')
        source_db.close()
    target_db.close()

def parse_args():
    parser = argparse.ArgumentParser('Merges lmdb databases together.')
    parser.add_argument(
        '-s', '--source-databases',
        help='Source databases to merge.',
        nargs='+',
        type=Path,
        required=True
    )
    parser.add_argument(
        '-t', '--target-database',
        help='Target database to merge source databases to.',
        type=Path,
        required=True
    )
    return parser.parse_args()

def main():
    args = parse_args()
    merge_dbs(
        src_db_paths=args.source_databases,
        target_db_path=args.target_database
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
