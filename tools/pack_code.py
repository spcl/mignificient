#!/usr/bin/env python3
"""Pack a function: <dest>/<name>/{code, env/ (python), package.json}.

Python packages carry a full conda env with *dynamically linked* conda PyTorch;
pip torch wheels statically link cudart and break gpuless LD_PRELOAD interception.

The env is copied from conda's package cache unless the cache is on the same filesystem
as --dest; set CONDA_PKGS_DIRS=<dir on dest filesystem> to get hard links.
Speeds up the build and saves space for bare-metal packages.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIMES = {
    "torch-1.12.1-cu116": [
        "python=3.9",
        "pytorch=1.12.1",
        "torchvision",
        "cudatoolkit=11.6",
    ]
}
CHANNELS = ["pytorch", "conda-forge"]  # cudatoolkit 11.6 lives on conda-forge
TORCH_LIB = "lib/python3.9/site-packages/torch/lib/libc10_cuda.so"


def find_conda(explicit=None):
    cands = (
        [explicit]
        if explicit
        else [shutil.which(t) for t in ("micromamba", "mamba", "conda")]
        + [os.environ.get("CONDA_EXE"), "/opt/miniconda3/bin/conda"]
    )
    for c in cands:
        if c and os.access(c, os.X_OK):
            return c
    raise SystemExit("Error: no micromamba/mamba/conda found (use --conda-exe)")


def has_dynamic_cudart(lib: Path) -> bool:
    """True if libc10_cuda.so has a NEEDED libcudart.so (i.e. not a static pip wheel)."""
    if not shutil.which("readelf"):
        raise SystemExit("Error: readelf not found (install binutils)")
    r = subprocess.run(["readelf", "-d", str(lib)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"Error: readelf -d {lib} failed: {r.stderr.strip()}")
    return re.search(r"\(NEEDED\).*\[libcudart\.so", r.stdout) is not None


def check_reserved(src: Path):
    for reserved in ("env", "package.json"):
        if (src / reserved).exists():
            raise SystemExit(f"Error: '{reserved}' is reserved in the package; remove it from '{src}'")


def copy_code(src: Path, pkg: Path):
    src = src.resolve()
    skip = shutil.ignore_patterns(".git", "__pycache__", "*.pyc")

    def ignore(
        d, names
    ):  # also skip the package and its ancestors when dest lies inside src
        return set(skip(d, names)) | {
            n for n in names if pkg.is_relative_to((Path(d) / n).resolve())
        }

    shutil.copytree(src, pkg, symlinks=True, ignore=ignore, dirs_exist_ok=True)


def build(src, lang, pkg, runtime, so, conda_exe):
    if lang == "python":
        # conda wants to create the prefix itself, so env goes first, code copied over after
        subprocess.run(
            [
                find_conda(conda_exe),
                "create",
                "-y",
                "-p",
                str(pkg / "env"),
                *RUNTIMES[runtime],
                *[x for c in CHANNELS for x in ("-c", c)],
            ],
            check=True,
        )
        req = src / "requirements.txt"
        if req.is_file():
            subprocess.run(
                [str(pkg / "env/bin/pip"), "install", "-r", str(req)], check=True
            )
        lib = pkg / "env" / TORCH_LIB
        if not lib.is_file():
            raise SystemExit(
                f"Error: {lib} not found: torch is missing or has an unexpected layout"
            )
        if not has_dynamic_cudart(lib):
            raise SystemExit(
                f"Error: {lib} has statically linked cudart (a pip torch wheel replaced "
                "conda's PyTorch); gpuless interception would fail. Remove the pip torch "
                "from requirements.txt."
            )
    copy_code(src, pkg)
    meta = {
        "language": lang,
        "runtime": runtime if lang == "python" else None,
        "name": pkg.name,
    }
    if lang == "cpp":
        meta["function-file"] = so
    (pkg / "package.json").write_text(json.dumps(meta, indent=2) + "\n")


def pack(
    src: Path,
    lang: str,
    dest: Path,
    runtime: str,
    name=None,
    so=None,
    conda_exe=None,
    force=False,
) -> Path:
    if not src.is_dir():
        raise SystemExit(f"Error: source directory '{src}' does not exist")
    name = src.resolve().name if name is None else name
    if lang == "cpp" and not (so and (src / so).is_file()):
        raise SystemExit(
            f"Error: function .so '{so}' not found in '{src}' (use --so FILE)"
        )
    if lang == "python" and runtime not in RUNTIMES:
        raise SystemExit(
            f"Error: unknown runtime '{runtime}', known: {', '.join(RUNTIMES)}"
        )

    if name in ("", ".", "..") or name != Path(name).name:
        raise SystemExit(f"Error: invalid package name '{name}'")
    check_reserved(src)
    pkg = (dest / name).resolve()
    assert pkg.parent == dest.resolve(), pkg  # never rmtree outside dest
    if pkg.exists():
        if not force:
            raise SystemExit(f"Error: '{pkg}' exists (use --force to replace it)")
        shutil.rmtree(pkg)
    pkg.parent.mkdir(parents=True, exist_ok=True)
    try:
        build(src, lang, pkg, runtime, so, conda_exe)
    except BaseException:
        shutil.rmtree(pkg, ignore_errors=True)  # no partial packages
        raise
    return pkg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("type", choices=["python", "cpp"])
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--runtime", default="torch-1.12.1-cu116")
    p.add_argument("--name")
    p.add_argument("--so", help="function .so, relative to source (cpp)")
    p.add_argument("--conda-exe")
    p.add_argument(
        "--force",
        action="store_true",
        help="replace an existing <dest>/<name> (otherwise refused)",
    )
    a = p.parse_args()
    try:
        print(
            pack(
                a.source, a.type, a.dest, a.runtime, a.name, a.so, a.conda_exe, a.force
            )
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: command failed ({e.returncode}): {' '.join(map(str, e.cmd))}")


if __name__ == "__main__":
    main()
