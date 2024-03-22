#!/usr/bin/env python3

import lmdb
import os
import argparse
import sys

def extract_ds(source_path: str, target_path: str, n: int = 0) -> None:
    """
    Extracts dataset from source path leading to lmdb file to target
    path leading to output folder.
    :param source_path (str): path leading to source lmdb file
    :param target_path (str): path leading to output folder
    :param n (int): number of records to extract (0 == all)
    """
    env = lmdb.open(source_path)
    with env.begin() as transaction:
        cursor = transaction.cursor()
        for i, (key, value) in enumerate(cursor.iternext(keys=True, values=True)):
            if n != 0 and i >= n:
                break
            #print(key.decode('utf-8'))
            output = os.path.join(target_path, key.decode('utf-8'))
            with open(output, 'wb') as output_file:
                output_file.write(value)

def parse_args():
    parser = argparse.ArgumentParser('Extract lmdb to folder.')
    parser.add_argument(
        '-s', '--source',
        help='Source lmdb to extract.',
        required=True
    )
    parser.add_argument(
        '-t', '--target',
        help='Target folder to extract database to.',
        required=True
    )
    parser.add_argument(
        '-n', '--number-of-records',
        help='Number of records to extract. (0 == all) (default == 0)',
        type=int,
        default=0
    )
    return parser.parse_args()

def main():
    args = parse_args()

    extract_ds(
        source_path=args.source,
        target_path=args.target,
        n=args.number_of_records
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

