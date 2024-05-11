#!/usr/bin/env python3

import random
import os
import argparse
import sys

def mix_lines(filename: str):
    """
    Mixes lines in file.
    :param filename (str): name of file to mix
    """
    file = open(filename, 'r')
    lines = []
    for line in file:
        lines.append(line)
    file.close()
    random.shuffle(lines)
    file = open(filename, 'w')
    for line in lines:
        file.write(line)
    file.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Mixes lines in a file.')
    parser.add_argument(
        '-f', '--filenames',
        help='Filenames to mix.',
        nargs='+',
        default=[]
    )
    parser.add_argument(
        '-d', '--directory',
        help='Directory with files to mix.'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    for filename in args.filenames:
        mix_lines(filename)
    if args.directory:
        for filename in os.listdir(args.directory):
            mix_lines(os.path.join(args.directory, filename))
    return 0

if __name__ == '__main__':
    sys.exit(main())
