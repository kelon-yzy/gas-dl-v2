"""Deterministic, resumable execution for MEI-4 C3 Monte Carlo tasks."""
from __future__ import annotations

import os
import platform
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4

from threadpoolctl import threadpool_info, threadpool_limits

from tv3.audit.mrs_ei_mei4_mc import (
    C3Task,
    _load_context,
    aggregate_c3_results,
    build_c3_tasks,
    execute_c3_task,
)
from tv3.audit.mrs_ei_mei4_formal import load_frozen_c2_inputs
from tv3.audit.mrs_ei_registry import dumps_stable, load_json, sha256_bytes


ATTEMPT_SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-mei4-c3-attempt-1"
SHARD_SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-mei4-c3-shard-1"
RUNTIME_SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-mei4-c3-runtime-1"
PHASES = ("sbc", "ppc", "m2b")
DOMAINS = ("test", "ood")
_BLAS_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_WORKER_CONTEXT: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None
_WORKER_THREAD_LIMIT: Any = None
_WORKER_BLAS_INFO: list[dict[str, Any]] = []


@dataclass(frozen=True)
class C3RuntimeConfig:
    workers: int
    blas_threads: int
    max_inflight_per_worker: int
    shard_sizes: Mapping[str, int]

    @property
    def max_inflight(self) -> int:
        return self.workers * self.max_inflight_per_worker


@dataclass(frozen=True)
class TaskShard:
    phase: str
    domain: str
    index: int
    tasks: tuple[C3Task, ...]

    @property
    def shard_id(self) -> str:
        first = self.tasks[0].order + 1
        last = self.tasks[-1].order + 1
        return f"{self.phase}-{self.domain}-{first:06d}-{last:06d}"

    @property
    def task_ids(self) -> list[str]:
        return [task.task_id for task in self.tasks]


def load_runtime_config(path: Path, *, workers_override: int | None = None) -> C3RuntimeConfig:
    payload = load_json(path)
    if payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported C3 runtime schema: {payload.get('schema_version')}")
    config = C3RuntimeConfig(
        workers=int(payload["workers"]),
        blas_threads=int(payload["blas_threads"]),
        max_inflight_per_worker=int(payload["max_inflight_per_worker"]),
        shard_sizes={phase: int(payload["shard_sizes"][phase]) for phase in PHASES},
    )
    if workers_override is not None:
        config = replace(config, workers=workers_override)
    if config.workers < 1:
        raise ValueError("C3 workers must be positive")
    if config.blas_threads != 1:
        raise ValueError("C3 requires exactly one BLAS thread per worker")
    if config.max_inflight_per_worker < 1:
        raise ValueError("C3 max_inflight_per_worker must be positive")
    if any(size < 1 for size in config.shard_sizes.values()):
        raise ValueError("C3 shard sizes must be positive")
    return config


def _sha256_payload(payload: Any) -> str:
    return sha256_bytes(dumps_stable(payload).encode("utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, replace_existing: bool) -> None:
    if path.exists() and not replace_existing:
        raise FileExistsError(f"refuse overwrite of existing C3 file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    encoded = dumps_stable(payload).encode("utf-8")
    temporary.write_bytes(encoded)
    if temporary.read_bytes() != encoded:
        raise OSError(f"C3 atomic write verification failed: {temporary}")
    temporary.replace(path)


def _task_plan(tasks_by_phase: Mapping[str, Sequence[C3Task]]) -> dict[str, Any]:
    task_ids = [task.task_id for phase in PHASES for task in tasks_by_phase[phase]]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("C3 task plan contains duplicate task IDs")
    return {
        "counts": {phase: len(tasks_by_phase[phase]) for phase in PHASES},
        "task_ids_sha256": _sha256_payload(task_ids),
        "total": len(task_ids),
    }


def _attempt_immutable(
    *, attempt_id: str, binding: Mapping[str, Any], runtime: C3RuntimeConfig, task_plan: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "binding": dict(binding),
        "runtime_config": asdict(runtime),
        "task_plan": dict(task_plan),
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "threadpoolctl": version("threadpoolctl"),
        },
    }


def _prepare_attempt(
    *,
    attempt_dir: Path,
    binding: Mapping[str, Any],
    runtime: C3RuntimeConfig,
    task_plan: Mapping[str, Any],
    resume: bool,
) -> Mapping[str, Any]:
    manifest_path = attempt_dir / "attempt_manifest.json"
    expected = _attempt_immutable(
        attempt_id=attempt_dir.name,
        binding=binding,
        runtime=runtime,
        task_plan=task_plan,
    )
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"C3 resume attempt manifest is missing: {manifest_path}")
        manifest = load_json(manifest_path)
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError("C3 resume attempt binding, runtime, environment, or task plan changed")
        return manifest
    if attempt_dir.exists():
        raise FileExistsError(f"refuse overwrite of existing C3 attempt: {attempt_dir}")
    for phase in PHASES:
        (attempt_dir / "shards" / phase).mkdir(parents=True, exist_ok=False)
    manifest = {**expected, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_write_json(manifest_path, manifest, replace_existing=False)
    return manifest


def plan_task_shards(
    tasks_by_phase: Mapping[str, Sequence[C3Task]], runtime: C3RuntimeConfig
) -> list[TaskShard]:
    shards: list[TaskShard] = []
    for phase in PHASES:
        size = int(runtime.shard_sizes[phase])
        for domain in DOMAINS:
            tasks = sorted(
                (task for task in tasks_by_phase[phase] if task.domain == domain),
                key=lambda task: task.order,
            )
            if [task.order for task in tasks] != list(range(len(tasks))):
                raise RuntimeError(f"{phase} {domain} tasks are incomplete or duplicated")
            for index, start in enumerate(range(0, len(tasks), size)):
                shards.append(TaskShard(phase, domain, index, tuple(tasks[start : start + size])))
    return shards


def _shard_path(attempt_dir: Path, shard: TaskShard) -> Path:
    return attempt_dir / "shards" / shard.phase / f"{shard.shard_id}.json"


def _validate_shard_payload(payload: Mapping[str, Any], *, attempt_id: str, shard: TaskShard) -> list[Mapping[str, Any]]:
    if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise RuntimeError(f"invalid schema for C3 shard {shard.shard_id}")
    if payload.get("attempt_id") != attempt_id or payload.get("shard_id") != shard.shard_id:
        raise RuntimeError(f"attempt or shard identity mismatch for {shard.shard_id}")
    if payload.get("phase") != shard.phase or payload.get("domain") != shard.domain:
        raise RuntimeError(f"phase or domain mismatch for {shard.shard_id}")
    if payload.get("task_ids") != shard.task_ids:
        raise RuntimeError(f"task list mismatch for {shard.shard_id}")
    results = payload.get("results")
    if not isinstance(results, list) or [row.get("task_id") for row in results] != shard.task_ids:
        raise RuntimeError(f"result list mismatch for {shard.shard_id}")
    for result, task in zip(results, shard.tasks, strict=True):
        identity = (
            result.get("phase"),
            result.get("domain"),
            result.get("order"),
            result.get("item_id"),
        )
        if identity != (task.phase, task.domain, task.order, task.item_id):
            raise RuntimeError(f"result identity mismatch for {task.task_id}")
    if payload.get("payload_sha256") != _sha256_payload(results):
        raise RuntimeError(f"payload checksum mismatch for {shard.shard_id}")
    return results


def _read_existing_shards(
    *, attempt_dir: Path, attempt_id: str, shards: Sequence[TaskShard]
) -> tuple[dict[str, list[Mapping[str, Any]]], list[TaskShard]]:
    expected = {_shard_path(attempt_dir, shard): shard for shard in shards}
    actual = {
        path
        for phase in PHASES
        for path in (attempt_dir / "shards" / phase).glob("*.json")
    }
    unknown = sorted(str(path) for path in actual - set(expected))
    if unknown:
        raise RuntimeError(f"C3 attempt contains unknown shard files: {unknown}")
    completed: dict[str, list[Mapping[str, Any]]] = {}
    missing: list[TaskShard] = []
    seen_tasks: set[str] = set()
    for path, shard in expected.items():
        if not path.is_file():
            missing.append(shard)
            continue
        results = _validate_shard_payload(load_json(path), attempt_id=attempt_id, shard=shard)
        overlap = seen_tasks.intersection(shard.task_ids)
        if overlap:
            raise RuntimeError(f"C3 attempt contains duplicate task results: {sorted(overlap)}")
        seen_tasks.update(shard.task_ids)
        completed[shard.shard_id] = results
    return completed, missing


def _write_shard(
    *, attempt_dir: Path, attempt_id: str, shard: TaskShard, execution: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    results = execution.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"worker returned invalid results for {shard.shard_id}")
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "shard_id": shard.shard_id,
        "phase": shard.phase,
        "domain": shard.domain,
        "task_ids": shard.task_ids,
        "results": results,
        "payload_sha256": _sha256_payload(results),
        "worker": execution.get("worker"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _validate_shard_payload(payload, attempt_id=attempt_id, shard=shard)
    _atomic_write_json(_shard_path(attempt_dir, shard), payload, replace_existing=False)
    return results


def _init_worker(b4_dir: str, contract: Mapping[str, Any], blas_threads: int) -> None:
    global _WORKER_BLAS_INFO, _WORKER_CONTEXT, _WORKER_THREAD_LIMIT
    _WORKER_THREAD_LIMIT = threadpool_limits(limits=blas_threads)
    _WORKER_BLAS_INFO = [dict(row) for row in threadpool_info()]
    invalid = [row for row in _WORKER_BLAS_INFO if int(row.get("num_threads", 0)) > blas_threads]
    if invalid:
        raise RuntimeError(f"worker BLAS thread limit was not applied: {invalid}")
    _, solver_config, audit_tables = load_frozen_c2_inputs(Path(b4_dir))
    calibration = audit_tables["calibration"]
    _WORKER_CONTEXT = (solver_config, calibration, contract)


def _execute_worker_shard(tasks: Sequence[C3Task]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("C3 worker context is not initialized")
    solver_config, calibration, contract = _WORKER_CONTEXT
    results = [
        execute_c3_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
        for task in tasks
    ]
    return {
        "results": results,
        "worker": {"pid": os.getpid(), "blas": _WORKER_BLAS_INFO},
    }


def _execute_local_shard(
    shard: TaskShard,
    *,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        results = [
            execute_c3_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
            for task in shard.tasks
        ]
        info = [dict(row) for row in threadpool_info()]
    return {"results": results, "worker": {"pid": os.getpid(), "blas": info}}


@contextmanager
def _blas_environment(threads: int) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _BLAS_ENVIRONMENT}
    try:
        for name in _BLAS_ENVIRONMENT:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_status(
    attempt_dir: Path, *, attempt_id: str, shards: Sequence[TaskShard], completed_ids: set[str], status: str
) -> None:
    phase_counts = {
        phase: {
            "completed_shards": sum(shard.phase == phase and shard.shard_id in completed_ids for shard in shards),
            "total_shards": sum(shard.phase == phase for shard in shards),
            "completed_tasks": sum(
                len(shard.tasks) for shard in shards if shard.phase == phase and shard.shard_id in completed_ids
            ),
            "total_tasks": sum(len(shard.tasks) for shard in shards if shard.phase == phase),
        }
        for phase in PHASES
    }
    _atomic_write_json(
        attempt_dir / "attempt_status.json",
        {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "status": status,
            "phases": phase_counts,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        replace_existing=True,
    )


def _rejection_events(results: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for result in results:
        phase = result["phase"]
        payload = result["payload"]
        if phase in {"sbc", "ppc"}:
            count += sum(bool(row["rejected"]) for row in payload.values())
        elif phase == "m2b":
            count += int(bool(payload["row"]["rejected"]))
        else:
            raise RuntimeError(f"unsupported C3 result phase: {phase}")
    return count


def _execute_phase_with_executor(
    *,
    executor: ProcessPoolExecutor,
    phase_shards: Sequence[TaskShard],
    runtime: C3RuntimeConfig,
    on_complete: Callable[[TaskShard, Mapping[str, Any]], None],
) -> None:
    pending = iter(phase_shards)
    futures: dict[Future[dict[str, Any]], TaskShard] = {}
    try:
        while len(futures) < runtime.max_inflight:
            try:
                shard = next(pending)
            except StopIteration:
                break
            futures[executor.submit(_execute_worker_shard, shard.tasks)] = shard
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                shard = futures.pop(future)
                on_complete(shard, future.result())
                try:
                    next_shard = next(pending)
                except StopIteration:
                    continue
                futures[executor.submit(_execute_worker_shard, next_shard.tasks)] = next_shard
    except BaseException:
        for future in futures:
            future.cancel()
        raise


@contextmanager
def _worker_pool(
    *, b4_dir: Path, contract: Mapping[str, Any], runtime: C3RuntimeConfig
) -> Iterator[ProcessPoolExecutor]:
    with _blas_environment(runtime.blas_threads):
        with ProcessPoolExecutor(
            max_workers=runtime.workers,
            initializer=_init_worker,
            initargs=(str(b4_dir), contract, runtime.blas_threads),
        ) as executor:
            yield executor


def _execute_parallel_phase(
    *,
    phase_shards: Sequence[TaskShard],
    b4_dir: Path,
    contract: Mapping[str, Any],
    runtime: C3RuntimeConfig,
    on_complete: Callable[[TaskShard, Mapping[str, Any]], None],
) -> None:
    with _worker_pool(b4_dir=b4_dir, contract=contract, runtime=runtime) as executor:
        _execute_phase_with_executor(
            executor=executor,
            phase_shards=phase_shards,
            runtime=runtime,
            on_complete=on_complete,
        )


def run_c3_resumable(
    *,
    b4_dir: Path,
    contract: Mapping[str, Any],
    binding: Mapping[str, Any],
    runtime: C3RuntimeConfig,
    attempt_dir: Path,
    resume: bool,
    m2b_triggered: bool,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    if not m2b_triggered:
        raise RuntimeError("M2b must not run unless the registered PSIS trigger is present")
    templates, solver_config, calibration, audit_tables, records = _load_context(b4_dir)
    tasks_by_phase = build_c3_tasks(
        templates=templates,
        records=records,
        audit_tables=audit_tables,
        solver_config=solver_config,
        calibration=calibration,
        contract=contract,
    )
    plan = _task_plan(tasks_by_phase)
    manifest = _prepare_attempt(
        attempt_dir=attempt_dir,
        binding=binding,
        runtime=runtime,
        task_plan=plan,
        resume=resume,
    )
    attempt_id = str(manifest["attempt_id"])
    shards = plan_task_shards(tasks_by_phase, runtime)
    completed, missing = _read_existing_shards(attempt_dir=attempt_dir, attempt_id=attempt_id, shards=shards)
    completed_ids = set(completed)
    _write_status(attempt_dir, attempt_id=attempt_id, shards=shards, completed_ids=completed_ids, status="running")

    def execute_phases(executor: ProcessPoolExecutor | None) -> None:
        for phase in PHASES:
            phase_shards = [shard for shard in missing if shard.phase == phase]
            phase_total = sum(len(shard.tasks) for shard in shards if shard.phase == phase)
            phase_done = sum(
                len(shard.tasks) for shard in shards if shard.phase == phase and shard.shard_id in completed_ids
            )

            def on_complete(shard: TaskShard, execution: Mapping[str, Any]) -> None:
                nonlocal phase_done
                completed[shard.shard_id] = _write_shard(
                    attempt_dir=attempt_dir,
                    attempt_id=attempt_id,
                    shard=shard,
                    execution=execution,
                )
                completed_ids.add(shard.shard_id)
                phase_done += len(shard.tasks)
                _write_status(
                    attempt_dir,
                    attempt_id=attempt_id,
                    shards=shards,
                    completed_ids=completed_ids,
                    status="running",
                )
                if progress_callback is not None:
                    phase_completed = [
                        result
                        for completed_shard_id, shard_results in completed.items()
                        if completed_shard_id.startswith(f"{phase}-")
                        for result in shard_results
                    ]
                    completed_shards = sum(
                        shard_item.phase == phase and shard_item.shard_id in completed_ids
                        for shard_item in shards
                    )
                    total_shards = sum(shard_item.phase == phase for shard_item in shards)
                    progress_callback(
                        phase.upper(),
                        phase_done,
                        phase_total,
                        (
                            f"{shard.shard_id} shards={completed_shards}/{total_shards} "
                            f"rejection_events={_rejection_events(phase_completed)}"
                        ),
                    )

            if runtime.workers == 1:
                for shard in phase_shards:
                    on_complete(
                        shard,
                        _execute_local_shard(
                            shard,
                            solver_config=solver_config,
                            calibration=calibration,
                            contract=contract,
                        ),
                    )
            elif phase_shards and executor is not None:
                _execute_phase_with_executor(
                    executor=executor,
                    phase_shards=phase_shards,
                    runtime=runtime,
                    on_complete=on_complete,
                )

    try:
        if runtime.workers == 1:
            execute_phases(None)
        else:
            with _worker_pool(b4_dir=b4_dir, contract=contract, runtime=runtime) as executor:
                execute_phases(executor)
    except BaseException:
        _write_status(
            attempt_dir,
            attempt_id=attempt_id,
            shards=shards,
            completed_ids=completed_ids,
            status="failed",
        )
        raise

    completed, missing = _read_existing_shards(attempt_dir=attempt_dir, attempt_id=attempt_id, shards=shards)
    if missing:
        raise RuntimeError(f"C3 attempt remains incomplete: {[shard.shard_id for shard in missing]}")
    results_by_task: dict[str, Mapping[str, Any]] = {}
    for shard in shards:
        for result in completed[shard.shard_id]:
            task_id = str(result["task_id"])
            if task_id in results_by_task:
                raise RuntimeError(f"duplicate C3 task result during aggregation: {task_id}")
            results_by_task[task_id] = result
    expected_ids = [task.task_id for phase in PHASES for task in tasks_by_phase[phase]]
    if set(results_by_task) != set(expected_ids):
        raise RuntimeError("C3 result task set does not match the frozen task plan")
    ordered_results = [results_by_task[task_id] for task_id in expected_ids]
    reports = aggregate_c3_results(ordered_results, contract=contract, m2b_triggered=True)
    result_sha256 = _sha256_payload(reports)
    _write_status(attempt_dir, attempt_id=attempt_id, shards=shards, completed_ids=set(completed), status="complete")
    return {
        "reports": reports,
        "attempt": {
            "attempt_id": attempt_id,
            "attempt_dir": str(attempt_dir.resolve()),
            "attempt_manifest_sha256": _sha256_payload(load_json(attempt_dir / "attempt_manifest.json")),
            "result_sha256": result_sha256,
        },
    }


__all__ = ["C3RuntimeConfig", "load_runtime_config", "run_c3_resumable"]
