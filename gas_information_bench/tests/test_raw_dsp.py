import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gib.contract import ContractError, validate_dsp_provenance
from gib.pipeline.dataset import load_deployment_records
from gib.pipeline.raw_dsp import build_dsp_provenance, derive_dsp, dsp_config_sha256


HASH_A = "A" * 64
HASH_B = "B" * 64


def test_raw_to_dsp_is_deterministic_and_config_bound():
    raw = np.arange(64, dtype=np.float64).reshape(2, 32)
    config = {"frame_length": 8, "hop_length": 4, "features": ["mean", "standard_deviation", "slope"]}
    assert np.array_equal(derive_dsp(raw, config), derive_dsp(raw, config))
    changed = dict(config)
    changed["hop_length"] = 8
    assert dsp_config_sha256(config) != dsp_config_sha256(changed)


def test_three_provenance_hashes_are_independently_enforced():
    provenance = build_dsp_provenance(
        source_raw_manifest_id="GIB-MANIFEST-0123456789ABCDEF",
        raw_manifest_sha256=HASH_A,
        dsp_config_sha256_value=HASH_B,
        code_sha256=HASH_A,
    )
    for field in ("raw_manifest_sha256", "dsp_config_sha256", "code_sha256"):
        expected = {
            "raw_manifest_sha256": HASH_A,
            "dsp_config_sha256": HASH_B,
            "code_sha256": HASH_A,
        }
        expected[field] = "C" * 64
        with pytest.raises(ContractError, match="mismatch"):
            validate_dsp_provenance(provenance, **expected)


def test_deployment_loader_rejects_a_real_oracle_field(tmp_path: Path):
    path = tmp_path / "deployment.jsonl"
    row = {"mixture_id": "GIB-M-0123456789ABCDEF", "oracle_results": {}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="oracle fields"):
        load_deployment_records(path)
