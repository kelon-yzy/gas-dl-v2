"""RCDW benchmark 生成入口脚本。

用法：
    cd rcdw_mgda && python -m scripts.generate_benchmark \\
        --config configs/smoke.yaml \\
        --dataset-slug rcdw-smoke \\
        --output-root data

或者从 default config 跑 formal benchmark：
    cd rcdw_mgda && python -m scripts.generate_benchmark \\
        --config configs/default.yaml \\
        --dataset-slug rcdw-formal \\
        --output-root data

对应方案 §13.1：可用 ``--precompute-cache-only`` 预计算 HITRAN cache 而
不生成完整 benchmark。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rcdw.sim.generation.benchmark import (
    BenchmarkGenerationSpec,
    generate_benchmark_dataset,
)
from rcdw.sim.generation.conditions import generate_condition_rows
from rcdw.sim.generation.optical_backend import (
    HITRAN_ABSORPTION_BACKEND,
    precompute_hitran_benchmark_cache,
)


def _build_spec_from_config(
    cfg: dict, dataset_slug: str
) -> BenchmarkGenerationSpec:
    gen_cfg = cfg.get("generation")
    if gen_cfg is None:
        raise ValueError(
            "config 缺少 generation 节;请按 configs/smoke.yaml 模板补充。"
        )
    path_lms = tuple(float(p) for p in gen_cfg.get("path_lms", (0.20, 0.25, 0.30, 0.35, 0.40)))
    train_modalities = tuple(
        cfg.get("data", {}).get("train_modalities", ("slow", "ultrasonic"))
    )
    return BenchmarkGenerationSpec(
        dataset_slug=dataset_slug,
        sequence_count=int(gen_cfg["sequence_count"]),
        seed=int(gen_cfg.get("seed", 42)),
        timesteps=int(gen_cfg.get("timesteps", 128)),
        dt_s=float(gen_cfg.get("dt_s", 0.5)),
        storage=str(gen_cfg.get("storage", "memmap")),
        multi_path_phase=str(gen_cfg.get("multi_path_phase", "steady")),
        stage_profile=str(gen_cfg.get("stage_profile", "standard_exposure")),
        stage_jitter=float(gen_cfg.get("stage_jitter", 0.0)),
        sampling_strategy=str(gen_cfg.get("sampling_strategy", "lhs")),
        path_lms=path_lms,
        optical_absorption_backend=str(
            gen_cfg.get("optical_absorption_backend", "hitran_hapi_v1")
        ),
        hitran_cache_root=str(
            gen_cfg.get("hitran_cache_root", "data/hitran_cache")
        ),
        train_modalities=train_modalities,
        num_workers=int(gen_cfg.get("num_workers", 1)),
        chunk_size=int(gen_cfg.get("chunk_size", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RCDW benchmark dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke.yaml",
        help="YAML 配置路径(需包含 generation 节)",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=None,
        help="数据集 slug(目录名),默认从 data.dataset_root 推导",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data",
        help="benchmark 输出根目录(实际目录 = {output_root}/{dataset_slug})",
    )
    parser.add_argument(
        "--precompute-cache-only",
        action="store_true",
        help="仅预计算 HITRAN cache, 不创建 benchmark 目录",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.dataset_slug:
        dataset_slug = args.dataset_slug
    else:
        dataset_root = cfg.get("data", {}).get("dataset_root", "data/rcdw-smoke")
        dataset_slug = Path(dataset_root).name

    spec = _build_spec_from_config(cfg, dataset_slug=dataset_slug)
    print(f"[generate_benchmark] dataset_slug={spec.dataset_slug}")
    print(f"[generate_benchmark] sequence_count={spec.sequence_count}")
    print(f"[generate_benchmark] timesteps={spec.timesteps} dt_s={spec.dt_s}")
    print(f"[generate_benchmark] backend={spec.optical_absorption_backend}")
    print(f"[generate_benchmark] output={args.output_root}/{spec.dataset_slug}")

    if args.precompute_cache_only:
        if spec.optical_absorption_backend != HITRAN_ABSORPTION_BACKEND:
            print(
                "[generate_benchmark] precompute-cache-only no-op: "
                f"backend={spec.optical_absorption_backend}"
            )
            return
        conditions = generate_condition_rows(
            spec.sequence_count,
            seed=spec.seed,
            sampling_strategy=spec.sampling_strategy,
        )
        stats = precompute_hitran_benchmark_cache(
            conditions, cache_root=spec.hitran_cache_root
        )
        print("[generate_benchmark] precompute-cache-only DONE:", stats)
        return

    result = generate_benchmark_dataset(args.output_root, spec)
    print("[generate_benchmark] DONE:", result)


if __name__ == "__main__":
    main()
