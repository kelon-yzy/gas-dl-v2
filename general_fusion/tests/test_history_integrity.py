import json
import re
from pathlib import Path
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = PROJECT_ROOT / "general_fusion" / "docs" / "history"

REGISTRIES = {
    "source_registry": ("source_id", {
        "source_id",
        "project",
        "path",
        "source_type",
        "version_or_commit",
        "date",
        "owner",
        "status",
        "notes",
    }),
    "task_contract": ("task_id", {
        "task_id",
        "project",
        "source_refs",
        "input",
        "output",
        "sample_and_window",
        "sensor_definition",
        "label_definition",
        "group_definition",
        "split_protocol",
        "preprocessing",
        "deployment_assumption",
        "identifiability_note",
    }),
    "algorithm_ledger": ("algorithm_id", {
        "algorithm_id",
        "lineage_parent",
        "task_id",
        "source_refs",
        "input_representation",
        "fusion_form",
        "architecture",
        "output_head",
        "objective",
        "claimed_mechanism",
        "required_observable",
        "evidence_level",
        "conclusion_status",
        "conclusion_detail",
        "limitation",
        "disposition",
        "disposition_condition",
        "confidence",
        "claim_scope",
        "reopen_condition",
    }),
    "experiment_ledger": ("experiment_id", {
        "experiment_id",
        "algorithm_id",
        "task_id",
        "record_status",
        "source_refs",
        "data_version",
        "split_and_group",
        "seed_and_repeat",
        "config_ref",
        "metric_definition",
        "result_ref",
        "resource",
        "failure_or_exit",
        "reproducibility",
        "claim_scope",
        "notes",
    }),
    "failure_ledger": ("failure_id", {
        "failure_id",
        "algorithm_id",
        "experiment_id",
        "observation_type",
        "symptom",
        "layer",
        "hypothesis",
        "discriminator",
        "evidence",
        "root_cause",
        "root_cause_status",
        "residual_ceiling",
        "scope",
        "corrective_action",
        "stop_condition",
        "general_fusion_impact",
    }),
}

CONCLUSION_STATUSES = {
    "planned",
    "implemented",
    "smoke_only",
    "formal_pass",
    "formal_fail",
    "blocked",
    "invalidated",
    "superseded",
    "parked",
    "negative_control",
    "unknown",
}
DISPOSITIONS = {"inherit", "rewrite", "negative_control", "park", "close", "unknown"}
SOURCE_CLASSES = {"doc", "code", "config", "test", "log", "output", "freeze", "commit"}
PATH_PREFIXES = {
    "general_fusion",
    "hydrogen_ng",
    "syngas",
    "tunnel_ventilation",
    "rcdw_mgda",
    "gas_information_bench",
    "数据集",
}


def load_registry(name: str) -> dict:
    return json.loads((HISTORY_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_registry_required_fields_and_unique_ids() -> None:
    for name, (id_field, required_fields) in REGISTRIES.items():
        entries = load_registry(name)["entries"]
        ids = [entry[id_field] for entry in entries]
        assert len(ids) == len(set(ids)), f"duplicate {id_field} in {name}"
        for entry in entries:
            missing = required_fields - entry.keys()
            assert not missing, f"{entry.get(id_field)} missing {sorted(missing)}"


def test_source_types_have_one_canonical_class() -> None:
    registry = load_registry("source_registry")
    groups = registry["source_type_groups"]
    assert set(groups) == SOURCE_CLASSES

    mapped_types = [source_type for values in groups.values() for source_type in values]
    assert len(mapped_types) == len(set(mapped_types)), "source_type belongs to multiple classes"
    actual_types = {entry["source_type"] for entry in registry["entries"]}
    assert set(mapped_types) == actual_types


def test_cross_registry_references_and_semantic_alignment() -> None:
    sources = load_registry("source_registry")["entries"]
    tasks = load_registry("task_contract")["entries"]
    algorithms = load_registry("algorithm_ledger")["entries"]
    experiments = load_registry("experiment_ledger")["entries"]
    failures = load_registry("failure_ledger")["entries"]

    source_ids = {entry["source_id"] for entry in sources}
    task_ids = {entry["task_id"] for entry in tasks}
    algorithm_by_id = {entry["algorithm_id"]: entry for entry in algorithms}
    experiment_by_id = {entry["experiment_id"]: entry for entry in experiments}

    for entry in tasks:
        assert set(entry["source_refs"]) <= source_ids
    for entry in algorithms:
        assert entry["task_id"] in task_ids
        assert set(entry["source_refs"]) <= source_ids
        parent = entry["lineage_parent"]
        assert parent is None or parent in algorithm_by_id
    for entry in experiments:
        assert entry["task_id"] in task_ids
        assert entry["algorithm_id"] in algorithm_by_id
        assert entry["task_id"] == algorithm_by_id[entry["algorithm_id"]]["task_id"]
        assert set(entry["source_refs"]) <= source_ids
    for entry in failures:
        experiment = experiment_by_id[entry["experiment_id"]]
        assert entry["algorithm_id"] == experiment["algorithm_id"]
        assert set(entry["evidence"]) <= source_ids
        assert set(entry.get("related_algorithm_ids", [])) <= algorithm_by_id.keys()


def test_algorithm_vocabularies_and_evidence_coverage() -> None:
    algorithms = load_registry("algorithm_ledger")["entries"]
    experiments = load_registry("experiment_ledger")["entries"]
    algorithms_with_experiments = {entry["algorithm_id"] for entry in experiments}

    for entry in algorithms:
        assert entry["evidence_level"] in {"E0", "E1", "E2", "E3", "E4"}
        assert entry["conclusion_status"] in CONCLUSION_STATUSES
        assert entry["disposition"] in DISPOSITIONS
        if entry["conclusion_status"] in {"formal_pass", "formal_fail"}:
            assert entry["evidence_level"] in {"E3", "E4"}
        if entry["evidence_level"] in {"E3", "E4"}:
            assert entry["algorithm_id"] in algorithms_with_experiments


def test_registered_paths_and_experiment_artifacts_resolve() -> None:
    for source in load_registry("source_registry")["entries"]:
        path = PROJECT_ROOT / source["path"]
        if source["status"] != "missing":
            assert path.exists(), source["source_id"]

    for experiment in load_registry("experiment_ledger")["entries"]:
        for field in ("config_ref", "result_ref"):
            value = experiment[field]
            if not isinstance(value, str):
                continue
            first_segment = value.replace("\\", "/").split("/", 1)[0]
            if first_segment in PATH_PREFIXES:
                assert (PROJECT_ROOT / value).exists(), f"{experiment['experiment_id']} {field}"


def test_history_markdown_relative_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in HISTORY_DIR.glob("*.md"):
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith("#") or "://" in target or target.startswith("mailto:"):
                continue
            relative_path = unquote(target.split("#", 1)[0])
            assert (document.parent / relative_path).exists(), f"{document.name}: {target}"
