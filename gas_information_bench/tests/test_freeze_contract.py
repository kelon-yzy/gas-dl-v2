import json
from pathlib import Path

import pytest

from gib.cli import main
from gib.freeze import FreezeContractError, freeze_attempt, verify_evidence_manifest


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, status: str = "complete") -> dict[str, object]:
    workspace = tmp_path / "workspace"
    attempt = tmp_path / "attempts" / "GIB-ATTEMPT-001"
    freezes = tmp_path / "freezes"
    _write(attempt / "attempt_manifest.json", json.dumps({"status": status}))
    _write(attempt / "metrics.json", '{"metric": 1}\n')
    input_files = {
        "config": [_write(workspace / "configs" / "profile.json", "{}\n")],
        "schema": [_write(workspace / "schemas" / "data.json", "{}\n")],
        "gate": [_write(workspace / "gates" / "gate.json", "{}\n")],
        "code": [_write(workspace / "gib" / "module.py", "VALUE = 1\n")],
        "source_registry": [_write(workspace / "sources" / "registry.json", "{}\n")],
    }
    source = _write(workspace / "docs" / "source.md", "source\n")
    return {
        "workspace_root": workspace,
        "attempt_dir": attempt,
        "freeze_root": freezes,
        "freeze_id": "GIB-FREEZE-P2-001",
        "input_files": input_files,
        "source_snapshots": [source],
    }


def test_complete_attempt_freezes_with_hash_bound_inputs_and_snapshots(tmp_path):
    fixture = _fixture(tmp_path)
    target = freeze_attempt(**fixture)

    assert fixture["attempt_dir"].is_dir()
    summary = verify_evidence_manifest(target)
    assert summary == {
        "freeze_id": "GIB-FREEZE-P2-001",
        "input_count": 5,
        "source_snapshot_count": 1,
        "evidence_file_count": 2,
    }
    manifest = json.loads((target / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert {record["role"] for record in manifest["inputs"]} == {
        "config",
        "schema",
        "gate",
        "code",
        "source_registry",
    }
    assert all(not Path(record["logical_path"]).is_absolute() for record in manifest["inputs"])


def test_freeze_is_append_only_and_tamper_is_visible(tmp_path):
    fixture = _fixture(tmp_path)
    target = freeze_attempt(**fixture)
    original_manifest = (target / "evidence_manifest.json").read_bytes()

    with pytest.raises(FreezeContractError, match="cannot be overwritten"):
        freeze_attempt(**fixture)
    assert (target / "evidence_manifest.json").read_bytes() == original_manifest

    (target / "attempt" / "metrics.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FreezeContractError, match="hash mismatch"):
        verify_evidence_manifest(target)


def test_incomplete_attempt_and_overlapping_roots_never_promote(tmp_path):
    fixture = _fixture(tmp_path, status="failed")
    with pytest.raises(FreezeContractError, match="status=complete"):
        freeze_attempt(**fixture)
    assert not (fixture["freeze_root"] / fixture["freeze_id"]).exists()

    complete = _fixture(tmp_path / "second")
    complete["freeze_root"] = complete["attempt_dir"] / "freezes"
    with pytest.raises(FreezeContractError, match="physically separate"):
        freeze_attempt(**complete)


def test_cli_verifies_a_frozen_directory(tmp_path, capsys):
    target = freeze_attempt(**_fixture(tmp_path))
    assert main(["verify-freeze", str(target)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["freeze_id"] == "GIB-FREEZE-P2-001"


def test_verifier_rejects_unregistered_and_escaping_files(tmp_path):
    target = freeze_attempt(**_fixture(tmp_path))
    (target / "injected.txt").write_text("not registered\n", encoding="utf-8")
    with pytest.raises(FreezeContractError, match="file set does not match"):
        verify_evidence_manifest(target)

    (target / "injected.txt").unlink()
    manifest_path = target / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"][0]["snapshot_path"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FreezeContractError, match="must stay inside"):
        verify_evidence_manifest(target)


def test_verifier_rejects_missing_required_role_even_when_an_unknown_role_is_added(tmp_path):
    target = freeze_attempt(**_fixture(tmp_path))
    manifest_path = target / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_registry = next(
        record for record in manifest["inputs"] if record["role"] == "source_registry"
    )
    source_registry["role"] = "unexpected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FreezeContractError, match="input roles mismatch"):
        verify_evidence_manifest(target)


def test_verifier_rejects_manifest_field_and_hash_policy_drift(tmp_path):
    target = freeze_attempt(**_fixture(tmp_path))
    manifest_path = target / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FreezeContractError, match="fields mismatch"):
        verify_evidence_manifest(target)

    del manifest["unexpected"]
    manifest["hash_algorithm"] = "MD5"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FreezeContractError, match="must be SHA256"):
        verify_evidence_manifest(target)
