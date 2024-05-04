#!/usr/bin/env python3

from PIL import Image
from io import BytesIO
import lmdb
import sys
import argparse
from pathlib import Path
import traceback

def check_db(path: Path, labels: Path) -> None:
    """
    Tries to load each image from db using PIL.
    :param path (pathlib.Path): path to database
    """
    if not path.exists():
        raise ValueError(f'Path {str(path)} does not exist!')
    if not labels.exists():
        raise ValueError(f'Path {str(labels)} does not exist!')
    
    label_file = open(str(labels), 'r')

    db = lmdb.open(str(path))
    tx = db.begin()
    cursor = tx.cursor()
    '''
    for key, value in cursor.iternext(keys=True, values=True):
        try:
            image = Image.open(BytesIO(value)).convert('RGB')
        except Exception as e:
            print(f'Error when opening image {key}!')
            print(traceback.format_exc())
    '''
    for line in label_file:
        key, label = line.strip().split(' 0 ')
        try:
            image = Image.open(BytesIO(tx.get(key.encode()))).convert('RGB')
        except Exception as e:
            print(f'Error when opening image {key}!')
            print(traceback.format_exc())
    
    db.close()
    label_file.close()

def parse_args():
    parser = argparse.ArgumentParser('Checks database of images for broken images.')
    parser.add_argument(
        '-d', '--database',
        help='Database to check.',
        required=True,
        type=Path
    )
    parser.add_argument(
        '-l', '--labels',
        help='Labels for db.',
        required=True,
        type=Path
    )
    return parser.parse_args()

def main():
    args = parse_args()
    check_db(args.database, args.labels)
    return 0

if __name__ == '__main__':
    sys.exit(main())
