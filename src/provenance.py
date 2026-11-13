"""Run provenance without relying on a clean or committed working tree."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import platform
import subprocess
import sys


def environment_manifest():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    paths = sorted(set(root.glob("src/*.py")) | set(root.glob("experiments/*.py")) |
                   set(root.glob("tests/*.py")) | {root / "papers/Budd_Castro.pdf"})
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "matplotlib", "pytest")},
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "source_sha256": {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                          for path in paths if path.exists()},
        "primary_source": "Budd and Castro, arXiv:2207.05355v1, 12 July 2022",
        "random_generator": "numpy.random.Generator(PCG64), explicit seeds",
    }
