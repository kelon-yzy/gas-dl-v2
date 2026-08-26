"""Build and exercise a non-editable GIB wheel outside the source tree."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIG = "configs/p2_data_schema.json"


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gib-installed-smoke-") as temporary:
        root = Path(temporary)
        source = root / "source"
        wheel_dir = root / "wheel"
        target = root / "target"
        source.mkdir()
        wheel_dir.mkdir()
        target.mkdir()

        shutil.copy2(PROJECT_ROOT / "pyproject.toml", source / "pyproject.toml")
        shutil.copy2(PROJECT_ROOT / "README.md", source / "README.md")
        shutil.copytree(PROJECT_ROOT / "gib", source / "gib")
        shutil.copytree(PROJECT_ROOT / "configs", source / "configs")

        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(source),
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
            ],
            cwd=root,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, got {len(wheels)}")
        with zipfile.ZipFile(wheels[0]) as archive:
            if EXPECTED_CONFIG not in archive.namelist():
                raise RuntimeError(f"wheel is missing {EXPECTED_CONFIG}")

        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                str(wheels[0]),
                "--no-deps",
                "--target",
                str(target),
            ],
            cwd=root,
        )
        probe = (
            "import json,sys;"
            f"sys.path.insert(0,{str(target)!r});"
            "from gib.contract import load_contracts;"
            "from gib.audit.s2_s3 import load_profile;"
            "from gib.s5_contract import load_s5_contracts;"
            "from gib.common.io import sha256_bytes;"
            "from gib.pipeline.raw_dsp import dsp_config_sha256;"
            "from gib.sim.packaging.arrays import array_npy_bytes;"
            "import numpy as np;"
            "registry,discrepancy=load_s5_contracts();"
            "print(json.dumps({"
            "'contract':load_contracts()['data']['contract_status'],"
            "'audit':load_profile()['audit_id'],"
            "'source':registry['verdict'],"
            "'discrepancy':discrepancy['default_profile'],"
            "'common_hash':len(sha256_bytes(b'gib')),"
            "'dsp_hash':len(dsp_config_sha256({'frame_length':2,'hop_length':1})),"
            "'array_magic':array_npy_bytes(np.array([1.0])).startswith(b'\\x93NUMPY')},sort_keys=True))"
        )
        completed = _run([sys.executable, "-I", "-c", probe], cwd=root)
        payload = json.loads(completed.stdout)
        expected = {
            "audit": "GIB-S2-S3-v1",
            "contract": "contract_frozen",
            "discrepancy": "off",
            "common_hash": 64,
            "dsp_hash": 64,
            "array_magic": True,
            "source": "source_complete",
        }
        if payload != expected:
            raise RuntimeError(f"installed package probe mismatch: {payload}")
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
