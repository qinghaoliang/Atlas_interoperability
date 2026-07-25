"""
Entry point for applying learned atlas-to-atlas mappings to SC and FC.

The mapped outputs are saved under data/mapped/ and can be used by downstream
FLECHA and MECHA analyses.
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
    script = os.path.join(project_root, "src", "atlas_transform", "map_connectome.py")

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
