#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def pack(source_dir: Path, lang: str, dest_dir: Path) -> Path:
    if not source_dir.is_dir():
        print(f"Error: source directory '{source_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    package_name = source_dir.resolve().name
    deployment_dir = dest_dir / package_name

    # Clean previous deployment if present
    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)

    # Copy source to deployment location
    shutil.copytree(source_dir, deployment_dir)
    print(f"Copied '{source_dir}' -> '{deployment_dir}'")

    if lang == "python":
        requirements = deployment_dir / "requirements.txt"
        if not requirements.is_file():
            print(f"Warning: no requirements.txt found in '{source_dir}', skipping dependency install", file=sys.stderr)
            return deployment_dir

        packages_dir = deployment_dir / ".python_packages"
        packages_dir.mkdir(exist_ok=True)

        print(f"Installing Python dependencies into '{packages_dir}'")
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--target", str(packages_dir),
                "-r", str(requirements),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"pip install failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(result.stdout)

    return deployment_dir


def main():
    parser = argparse.ArgumentParser(description="Pack a code directory for container deployment")
    parser.add_argument("source", type=Path, help="Source code directory to pack")
    parser.add_argument("type", choices=["python", "c++"], help="Language type (python or c++)")
    parser.add_argument("--dest", type=Path, default=Path(tempfile.gettempdir()),
                        help="Destination base directory (default: /tmp)")
    args = parser.parse_args()

    deployment_dir = pack(args.source, args.type, args.dest)
    print(f"Deployment ready at: {deployment_dir}")


if __name__ == "__main__":
    main()
