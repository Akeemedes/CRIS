"""Plot local inverse accuracy and global convergence for the Bratu problem."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training/bratu'))
import plot_bratu_details as details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['local','global'],required=True)
    args=parser.parse_args()
    if args.only=='local':details.local(details.selection())
    else:details.global_figures()


if __name__=='__main__':main()
