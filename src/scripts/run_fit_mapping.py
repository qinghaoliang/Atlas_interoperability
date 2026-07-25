"""
Entry point for fitting atlas-to-atlas mapping matrices.

This wrapper calls the atlas transformation module used in the manuscript.
It requires paired node-wise timeseries data parcellated with both source and
target atlases. Controlled-access datasets are not included in this repository.
"""

import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, help="Source atlas name")
    parser.add_argument("-t", "--target", required=True, help="Target atlas name")
    parser.add_argument("-p", type=int, default=50, help="Spectral embedding dimension")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(project_root, "src", "atlas_transform", "map_nodetime.py")

    subprocess.run(
        [
            sys.executable,
            script,
            "-s",
            args.source,
            "-t",
            args.target,
            "-p",
            str(args.p),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
