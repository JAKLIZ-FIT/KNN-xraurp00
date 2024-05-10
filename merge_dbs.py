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

def separate_dbs(
    source_path: Path,
    target_path: Path,
    ids_path: Path,
    split_labels: bool = True
) -> None:
    """
    Extracts part of the database to separate db.
    :param source_path (pathlib.Path): source database path
    :param target_path (pathlib.Path): target database path
    :param ids_path (pathlib.Path): path to ids of data to extract
    :param split_labels (bool): if lines in id files contains format
        (id 0 label), this option splits the line and uses only id
    """
    if not source_path.exists():
        raise ValueError(f"File {str(source_path)} does not exist!")
    if not ids_path.exists():
        raise ValueError(f"File {str(ids_path)} does not exist!")
    
    id_file = open(str(ids_path), 'r')
    target_db = lmdb.open(str(target_path), create=True)
    source_db = lmdb.open(str(source_path))
    source_tx = source_db.begin()
    
    for i in id_file:
        i = i.strip()
        if split_labels:
            i = i.split(' 0 ')[0]
        i = i.encode()
        target_tx = target_db.begin(write=True)
        data = source_tx.get(i)
        try:
            target_tx.put(key=i, value=data)
            target_tx.commit()
        except lmdb.MapFullError:
            target_tx.abort()
            target_db.set_mapsize(target_db.info()['map_size'] * 2)
            target_tx = target_db.begin(write=True)
            target_tx.put(key=i, value=data)
            target_tx.commit()

    source_db.close()
    target_db.close()
    id_file.close()

def parse_args():
    parser = argparse.ArgumentParser(
        'Merges lmdb databases together, '
        'or extracts part of db to separate one.'
    )
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
    parser.add_argument(
        '-i', '--ids-file',
        help='File with ids if database separation should be used. '
             'Only one source db must be used with this option.',
        type=Path,
        default=None
    )
    parser.add_argument(
        '-l', '--label-file',
        help='Tels if ids-file contains labels or not. (default=False)',
        default=False,
        action='store_true'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.ids_file:
        merge_dbs(
            src_db_paths=args.source_databases,
            target_db_path=args.target_database
        )
    else:
        separate_dbs(
            source_path=args.source_databases[0],
            target_path=args.target_database,
            ids_path=args.ids_file,
            split_labels=args.label_file
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
