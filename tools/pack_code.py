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

Python packages also get cubin-analysis.txt: gpuless's kernel parameter table for the env's CUDA libraries
(kernel names embed a per-build hash, so it must match the env). It comes from gpuless's cubin_analyzer
(--cubin-analyzer, from the MIGnificient build; cuobjdump from --cuda-home), is cached by the libraries' hash
under ~/.cache/mignificient/cubin/, and package.json "cubin-analysis" names it; the orchestrator uses it when a
request has no cubin-analysis. --no-cubin skips it.

Pillow: conda's Pillow links libjpeg 9 (JPEG decode ~2x slower than libjpeg-turbo); the same version's pip
wheel bundles libjpeg-turbo, so packing replaces it and checks the result.

The env is copied from conda's package cache unless the cache is on the same filesystem
as --dest; set CONDA_PKGS_DIRS=<dir on dest filesystem> to get hard links.
Speeds up the build and saves space for bare-metal packages.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import tempfile
import subprocess
import sys
from pathlib import Path

RUNTIMES = {
    "torch-1.12.1-cu116": [
        "python=3.9",
        "pytorch=1.12.1",
        "torchvision",
        "cudatoolkit=11.6",
        "numpy<2",
    ]
}  # torch 1.12 breaks with numpy 2
CHANNELS = ["pytorch", "conda-forge"]  # cudatoolkit 11.6 lives on conda-forge
TORCH_LIB = "lib/python3.9/site-packages/torch/lib/libc10_cuda.so"
# Libraries with the kernels gpuless needs parameter data for (inference: no cuDNN *_train).
CUBIN_LIBS = [
    f"lib/python3.9/site-packages/torch/lib/{n}"
    for n in (
        "libtorch_cuda_cu.so",
        "libtorch_cuda_cpp.so",
        "libtorch_cuda_linalg.so",
        "libcudnn_cnn_infer.so.8",
        "libcudnn_ops_infer.so.8",
        "libcudnn_adv_infer.so.8",
    )
] + ["lib/libcublas.so.11", "lib/libcublasLt.so.11"]
CUBIN_CACHE = Path.home() / ".cache" / "mignificient" / "cubin"


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


def cubin_analysis(env: Path, out: Path, analyzer: Path, arch: str, cuda_home: Path):
    """Write the cubin analysis of env's CUDA libraries to out (from the cache if the libraries are unchanged)."""
    libs = [env / l for l in CUBIN_LIBS if (env / l).is_file()]
    if not libs:
        raise SystemExit(
            f"Error: none of the CUDA libraries for the cubin analysis found in {env}"
        )
    h = hashlib.sha256(arch.encode())
    for lib in libs:
        with open(lib, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
    cached = CUBIN_CACHE / f"{h.hexdigest()[:32]}-sm{arch}.txt"
    if not cached.is_file():
        if not os.access(analyzer, os.X_OK):
            raise SystemExit(
                f"Error: cubin analyzer '{analyzer}' not found (--cubin-analyzer, or --no-cubin)"
            )
        CUBIN_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(f".tmp{os.getpid()}")
        print(f"cubin analysis of {len(libs)} libraries (takes ~1.5 min)")
        env_vars = clean_env(PATH=f"{cuda_home}/bin:" + os.environ["PATH"])
        r = subprocess.run(
            [str(analyzer), str(tmp), arch, ",".join(map(str, libs))],
            env=env_vars,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"Error: cubin_analyzer failed ({r.returncode}): {r.stderr[-500:]}"
            )
        tmp.rename(cached)
    else:
        print(f"cubin analysis from the cache: {cached}")
    shutil.copyfile(cached, out)


PILLOW_TURBO = "from PIL import features; import sys; sys.exit(0 if features.check_feature('libjpeg_turbo') else 1)"


def pillow_turbo(python: str, pip: str):
    """Replace a Pillow without libjpeg-turbo (conda's) with the same version's pip wheel, then check it."""
    r = subprocess.run(
        [python, "-c", "import PIL; print(PIL.__version__)"],
        capture_output=True,
        text=True,
        env=clean_env(),
    )
    if r.returncode != 0:
        return  # no Pillow in the env
    if subprocess.run([python, "-c", PILLOW_TURBO], env=clean_env()).returncode == 0:
        return
    version = r.stdout.strip()
    print(f"replacing Pillow {version} (no libjpeg-turbo) with its pip wheel")
    subprocess.run(
        [
            pip,
            "install",
            "--no-deps",
            "--force-reinstall",
            "--only-binary",
            ":all:",
            f"pillow=={version}",
        ],
        check=True,
        env=clean_env(),
    )
    if subprocess.run([python, "-c", PILLOW_TURBO], env=clean_env()).returncode != 0:
        raise SystemExit(
            f"Error: Pillow {version} in the env still decodes JPEG without libjpeg-turbo"
        )


def numpy_ok(python: str) -> bool:
    """True if the env's numpy is < 2 (torch 1.12 breaks with numpy 2)."""
    r = subprocess.run(
        [python, "-c", "import numpy; print(numpy.__version__)"],
        capture_output=True,
        text=True,
    )
    return (
        r.returncode != 0 or int(r.stdout.split(".")[0]) < 2
    )  # no numpy at all is fine


def clean_env(**extra):
    """Caller env without preload/library/python-path and gpuless variables."""
    drop = ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in drop and not k.startswith(("GPULESS_", "MIGNIFICIENT_"))
    }
    return {**env, **extra}


def check_reserved(src: Path):
    for reserved in ("env", "package.json", "cubin-analysis.txt"):
        if (src / reserved).exists():
            raise SystemExit(
                f"Error: '{reserved}' is reserved in the package; remove it from '{src}'"
            )


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


def prepare_and_verify(pkg: Path, python: str, timeout=1800):
    """Run <pkg>/prepare.py (if any), then list and verify the package's models."""
    cache = pkg / "torch-cache"
    cache.mkdir(exist_ok=True)
    prep = pkg / "prepare.py"
    if prep.is_file():
        proc = subprocess.Popen(
            [python, "prepare.py"],
            cwd=pkg,
            env=clean_env(TORCH_HOME=str(cache)),
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise SystemExit(
                f"Error: prepare.py timed out after {timeout} s (--prepare-timeout)"
            )
        if rc != 0:
            raise SystemExit(f"Error: prepare.py failed (exit {rc})")
    files = {f for f in cache.rglob("*") if f.is_file()}
    for d, dirs, names in os.walk(pkg):
        dirs[:] = [x for x in dirs if Path(d, x) != pkg / "env"]  # prune env/
        files |= {
            Path(d, n)
            for n in names
            if n.endswith((".pt", ".pth")) and Path(d, n).is_file()
        }
    if not files:
        if prep.is_file():
            raise SystemExit(
                "Error: prepare.py produced no model files under torch-cache/ or the package"
            )
        print("no models packaged")
        return
    total = 0
    for f in sorted(files):
        total += f.stat().st_size
        print(f"model: {f.relative_to(pkg)}  {f.stat().st_size / 1e6:.1f} MB")
    print(f"models: {len(files)} files, {total / 1e6:.1f} MB total in {pkg}")
    loadable = sorted(str(f) for f in files if f.suffix in (".pt", ".pth"))
    if loadable:
        r = subprocess.run(
            [python, "-c", LOAD_CHECK, *loadable],
            capture_output=True,
            text=True,
            env=clean_env(CUDA_VISIBLE_DEVICES=""),
        )
        if r.returncode != 0:
            raise SystemExit(
                "Error: model not loadable with torch.load:\n"
                + r.stdout.strip()
                + r.stderr[-500:]
            )
        print(f"verified: {len(loadable)} .pt/.pth files load with torch.load")


def build(src, lang, pkg, runtime, so, conda_exe, prepare_timeout=1800, cubin=None):
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
            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt"
            ) as c:  # pip must not upgrade numpy to 2
                c.write("numpy<2\n")
                c.flush()
                subprocess.run(
                    [str(pkg / "env/bin/pip"), "install", "-r", str(req), "-c", c.name],
                    check=True,
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
        if not numpy_ok(str(pkg / "env/bin/python")):
            raise SystemExit(
                "Error: the env has numpy >= 2, which breaks torch 1.12 ('Numpy is not available'); "
                "pin 'numpy<2' in requirements.txt"
            )
        pillow_turbo(str(pkg / "env/bin/python"), str(pkg / "env/bin/pip"))
        if cubin:
            cubin_analysis(pkg / "env", pkg / "cubin-analysis.txt", *cubin)
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
        if cubin:
            meta["cubin-analysis"] = "cubin-analysis.txt"
    (pkg / "package.json").write_text(json.dumps(meta, indent=2) + "\n")
    if lang == "python":
        prepare_and_verify(pkg, str(pkg / "env/bin/python"), prepare_timeout)


def pack(
    src: Path,
    lang: str,
    dest: Path,
    runtime: str,
    name=None,
    so=None,
    conda_exe=None,
    force=False,
    prepare_timeout=1800,
    cubin=None,
) -> Path:
    """cubin: (analyzer, arch, cuda_home) for python packages, None to skip the cubin analysis."""
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
        build(src, lang, pkg, runtime, so, conda_exe, prepare_timeout, cubin)
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
    p.add_argument(
        "--prepare-timeout",
        type=int,
        default=1800,
        metavar="SECONDS",
        help="kill prepare.py after this long",
    )
    p.add_argument(
        "--cubin-analyzer",
        type=Path,
        help="gpuless cubin_analyzer of the MIGnificient build "
        "(<build>/gpuless/cubin_analyzer); required for python packages unless --no-cubin",
    )
    p.add_argument(
        "--cubin-arch",
        default="86",
        help="sm version of the cubins to analyze (default 86)",
    )
    p.add_argument(
        "--cuda-home",
        type=Path,
        default=Path(os.environ.get("CUDA_HOME", "/opt/cuda/cuda-11.6")),
        help="CUDA toolkit with cuobjdump (default $CUDA_HOME or /opt/cuda/cuda-11.6)",
    )
    p.add_argument("--no-cubin", action="store_true", help="skip the cubin analysis")
    a = p.parse_args()
    cubin = None
    if a.type == "python" and not a.no_cubin:
        if not a.cubin_analyzer:
            p.error("--cubin-analyzer is required for python packages (or --no-cubin)")
        cubin = (a.cubin_analyzer.resolve(), a.cubin_arch, a.cuda_home)
    try:
        print(
            pack(
                a.source,
                a.type,
                a.dest,
                a.runtime,
                a.name,
                a.so,
                a.conda_exe,
                a.force,
                a.prepare_timeout,
                cubin,
            )
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: command failed ({e.returncode}): {' '.join(map(str, e.cmd))}")


if __name__ == "__main__":
    main()
