from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tv3.audit.mrs_ei_mei4_mc import C3Task, _load_context, aggregate_ppc_results, build_ppc_tasks
from tv3.audit.mrs_ei_mei4_parallel import (
    C3RuntimeConfig,
    _execute_local_shard,
    _execute_parallel_phase,
    _prepare_attempt,
    _read_existing_shards,
    _shard_path,
    _task_plan,
    _write_shard,
    load_runtime_config,
    plan_task_shards,
)


_ROOT = Path(__file__).resolve().parents[1]
_B4 = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei3_varpro_audit"
    / "freezes"
    / "20260729T120958962354Z_cf7ed57312d9"
)
_CONTRACT = _ROOT / "configs" / "tv3_mrs_ei" / "mei4_execution_contract.json"
_RUNTIME = _ROOT / "configs" / "tv3_mrs_ei" / "mei4_c3_runtime.json"


def _runtime(*, workers: int = 1) -> C3RuntimeConfig:
    return C3RuntimeConfig(
        workers=workers,
        blas_threads=1,
        max_inflight_per_worker=2,
        shard_sizes={"sbc": 2, "ppc": 1, "m2b": 1},
    )


def _dummy_tasks() -> dict[str, list[C3Task]]:
    return {
        "sbc": [
            C3Task("sbc", "test", index, f"{index + 1:04d}", None)  # type: ignore[arg-type]
            for index in range(3)
        ],
        "ppc": [],
        "m2b": [],
    }


def _fake_result(task: C3Task) -> dict:
    return {
        "task_id": task.task_id,
        "phase": task.phase,
        "domain": task.domain,
        "order": task.order,
        "item_id": task.item_id,
        "payload": {"value": task.order},
    }


def test_runtime_config_is_operational_only_and_supports_worker_override():
    runtime = load_runtime_config(_RUNTIME, workers_override=4)
    assert runtime.workers == 4
    assert runtime.blas_threads == 1
    assert runtime.shard_sizes == {"sbc": 16, "ppc": 8, "m2b": 1}


def test_attempt_resume_reads_only_validated_shards(tmp_path: Path):
    tasks = _dummy_tasks()
    runtime = _runtime()
    plan = _task_plan(tasks)
    attempt_dir = tmp_path / "attempt-a"
    binding = {"contract_sha256": "abc", "source_sha256": {"engine": "def"}}
    _prepare_attempt(
        attempt_dir=attempt_dir,
        binding=binding,
        runtime=runtime,
        task_plan=plan,
        resume=False,
    )
    shards = plan_task_shards(tasks, runtime)
    first = shards[0]
    _write_shard(
        attempt_dir=attempt_dir,
        attempt_id=attempt_dir.name,
        shard=first,
        execution={"results": [_fake_result(task) for task in first.tasks], "worker": {"pid": 1}},
    )

    completed, missing = _read_existing_shards(
        attempt_dir=attempt_dir,
        attempt_id=attempt_dir.name,
        shards=shards,
    )
    assert set(completed) == {first.shard_id}
    assert [shard.shard_id for shard in missing] == [shards[1].shard_id]

    _prepare_attempt(
        attempt_dir=attempt_dir,
        binding=binding,
        runtime=runtime,
        task_plan=plan,
        resume=True,
    )
    with pytest.raises(RuntimeError, match="binding"):
        _prepare_attempt(
            attempt_dir=attempt_dir,
            binding={**binding, "contract_sha256": "changed"},
            runtime=runtime,
            task_plan=plan,
            resume=True,
        )


def test_corrupt_or_unknown_shards_fail_explicitly(tmp_path: Path):
    tasks = _dummy_tasks()
    runtime = _runtime()
    attempt_dir = tmp_path / "attempt-b"
    _prepare_attempt(
        attempt_dir=attempt_dir,
        binding={"contract_sha256": "abc"},
        runtime=runtime,
        task_plan=_task_plan(tasks),
        resume=False,
    )
    shards = plan_task_shards(tasks, runtime)
    first = shards[0]
    _write_shard(
        attempt_dir=attempt_dir,
        attempt_id=attempt_dir.name,
        shard=first,
        execution={"results": [_fake_result(task) for task in first.tasks], "worker": {"pid": 1}},
    )
    path = _shard_path(attempt_dir, first)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["payload"]["value"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        _read_existing_shards(attempt_dir=attempt_dir, attempt_id=attempt_dir.name, shards=shards)

    path.unlink()
    unknown = attempt_dir / "shards" / "sbc" / "unknown.json"
    unknown.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unknown shard"):
        _read_existing_shards(attempt_dir=attempt_dir, attempt_id=attempt_dir.name, shards=shards)


def test_two_worker_ppc_matches_local_execution_and_resumes_without_missing_tasks(tmp_path: Path):
    contract = copy.deepcopy(json.loads(_CONTRACT.read_text(encoding="utf-8")))
    contract["mc_protocol"]["ppc"]["y_rep_per_frozen_mixture"] = 2
    _, solver_config, calibration, _, records = _load_context(_B4)
    selected = [
        next(record for record in records if record.mixture_id == "M000651"),
        next(record for record in records if record.mixture_id == "M001355"),
    ]
    tasks_by_phase = {"sbc": [], "ppc": build_ppc_tasks(selected), "m2b": []}
    shards = [shard for shard in plan_task_shards(tasks_by_phase, _runtime(workers=2)) if shard.phase == "ppc"]
    expected = {
        shard.shard_id: _execute_local_shard(
            shard,
            solver_config=solver_config,
            calibration=calibration,
            contract=contract,
        )["results"]
        for shard in shards
    }
    actual: dict[str, dict] = {}

    def collect(shard, execution):
        actual[shard.shard_id] = dict(execution)

    _execute_parallel_phase(
        phase_shards=shards,
        b4_dir=_B4,
        contract=contract,
        runtime=_runtime(workers=2),
        on_complete=collect,
    )
    assert {name: execution["results"] for name, execution in actual.items()} == expected
    for execution in actual.values():
        assert execution["worker"]["blas"]
        assert all(int(pool["num_threads"]) <= 1 for pool in execution["worker"]["blas"])

    attempt_dir = tmp_path / "ppc-attempt"
    binding = {"contract_sha256": "smoke", "source_sha256": {"engine": "smoke"}}
    runtime = _runtime(workers=2)
    _prepare_attempt(
        attempt_dir=attempt_dir,
        binding=binding,
        runtime=runtime,
        task_plan=_task_plan(tasks_by_phase),
        resume=False,
    )
    for shard in shards:
        _write_shard(
            attempt_dir=attempt_dir,
            attempt_id=attempt_dir.name,
            shard=shard,
            execution=actual[shard.shard_id],
        )
    completed, missing = _read_existing_shards(
        attempt_dir=attempt_dir,
        attempt_id=attempt_dir.name,
        shards=shards,
    )
    assert missing == []
    ordered = [result for shard in shards for result in completed[shard.shard_id]]
    expected_ordered = [result for shard in shards for result in expected[shard.shard_id]]
    assert aggregate_ppc_results(ordered, contract) == aggregate_ppc_results(expected_ordered, contract)

    _prepare_attempt(
        attempt_dir=attempt_dir,
        binding=binding,
        runtime=runtime,
        task_plan=_task_plan(tasks_by_phase),
        resume=True,
    )
    _, missing_after_resume = _read_existing_shards(
        attempt_dir=attempt_dir,
        attempt_id=attempt_dir.name,
        shards=shards,
    )
    assert missing_after_resume == []
