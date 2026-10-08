#!/usr/bin/env python3
"""Test fixture: pack a test function into <build>/test-packages with tools/pack_code.py, unless the package
is newer than every source file.  pack.py <build> <source dir> python|cpp [pack_code args...]"""
import os, subprocess, sys
from pathlib import Path

build, src, lang, extra = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
meta = build / "test-packages" / src.name / "package.json"
newest = max(f.stat().st_mtime for f in src.rglob("*") if f.is_file())
if meta.is_file() and meta.stat().st_mtime > newest:
    print(f"{meta.parent} is up to date")
    sys.exit(0)
tool = Path(__file__).resolve().parents[2] / "tools" / "pack_code.py"
args = [sys.executable, str(tool), str(src), lang, "--dest", str(build / "test-packages"), "--force", *extra]
if lang == "python":
    args += ["--cubin-analyzer", str(build / "gpuless" / "cubin_analyzer")]
sys.exit(subprocess.run(args).returncode)
