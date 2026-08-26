from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "p2_s6_ownership_registry.json"
EXPECTED_CAPABILITIES = {
    "id_primary_keys",
    "schema_manifest",
    "array_storage",
    "file_io",
    "split_grouping",
    "dataset_validation",
    "fisher_crb_rank",
    "jacobian_angle",
    "varpro_solver",
    "raw_dsp_derivation",
    "append_only_freeze",
    "efficiency_statistics",
    "run_reporting",
}
HISTORICAL_PACKAGES = {"hg", "sg", "tv3", "rcdw"}


def test_every_audited_capability_has_one_gib_owner() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    capabilities = registry["capabilities"]
    assert {item["capability_id"] for item in capabilities} == EXPECTED_CAPABILITIES
    assert len(capabilities) == len(EXPECTED_CAPABILITIES)
    for item in capabilities:
        assert isinstance(item["owner"], str)
        assert item["owner"].startswith("gib.")
        assert item["implementation_status"] in {"active", "reserved"}


def test_gib_has_no_historical_private_package_imports() -> None:
    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "gib").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            if roots & HISTORICAL_PACKAGES:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []
