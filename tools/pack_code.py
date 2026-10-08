#!/usr/bin/env python3
"""Pack a function: <dest>/<name>/{code, env/ (python), torch-cache/ (python), package.json}.

Layout of the source dir (all optional except the handler code): requirements.txt (pip, python only),
prepare.py (python only). prepare.py runs at pack time with <pkg>/env/bin/python, cwd=<pkg>,
TORCH_HOME=<pkg>/torch-cache; it downloads/saves model weights (torchvision/torch.hub downloads land
in torch-cache). Packing then lists every model file (path relative to the package, size), checks that
each *.pt/*.pth loads with torch.load, and fails if prepare.py produced no model file. At runtime the
orchestrator sets TORCH_HOME=<pkg>/<package.json "torch-home">. prepare.py itself is copied like any file.

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

RUNTIMES = {"torch-1.12.1-cu116": ["python=3.9", "pytorch=1.12.1", "torchvision", "cudatoolkit=11.6", "numpy<2"]}  # torch 1.12 breaks with numpy 2
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


LOAD_CHECK = """
import sys, torch
bad = 0
for f in sys.argv[1:]:
    try:
        torch.load(f, map_location="cpu")
    except Exception as e:
        print("FAILED to load %s: %s: %s" % (f, type(e).__name__, e)); bad += 1
sys.exit(1 if bad else 0)
"""


def prepare_and_verify(pkg: Path, python: str):
    """Run <pkg>/prepare.py (if any), then list and verify the package's models."""
    cache = pkg / "torch-cache"
    cache.mkdir(exist_ok=True)
    prep = pkg / "prepare.py"
    if prep.is_file():
        r = subprocess.run([python, "prepare.py"], cwd=pkg, env={**os.environ, "TORCH_HOME": str(cache)})
        if r.returncode != 0:
            raise SystemExit(f"Error: prepare.py failed (exit {r.returncode})")
    files = {f for f in cache.rglob("*") if f.is_file()}
    files |= {f for ext in ("*.pt", "*.pth") for f in pkg.rglob(ext)
              if f.is_file() and not f.is_relative_to(pkg / "env")}
    if not files:
        if prep.is_file():
            raise SystemExit("Error: prepare.py produced no model files under torch-cache/ or the package")
        print("no models packaged")
        return
    total = 0
    for f in sorted(files):
        total += f.stat().st_size
        print(f"model: {f.relative_to(pkg)}  {f.stat().st_size / 1e6:.1f} MB")
    print(f"models: {len(files)} files, {total / 1e6:.1f} MB total in {pkg}")
    loadable = sorted(str(f) for f in files if f.suffix in (".pt", ".pth"))
    if loadable:
        r = subprocess.run([python, "-c", LOAD_CHECK, *loadable], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("Error: model not loadable with torch.load:\n" + r.stdout.strip() + r.stderr[-500:])
        print(f"verified: {len(loadable)} .pt/.pth files load with torch.load")


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
    if lang == "python":
        meta["torch-home"] = "torch-cache"
    (pkg / "package.json").write_text(json.dumps(meta, indent=2) + "\n")
    if lang == "python":
        prepare_and_verify(pkg, str(pkg / "env/bin/python"))


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
