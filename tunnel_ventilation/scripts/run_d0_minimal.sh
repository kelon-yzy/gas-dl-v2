#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATASET="${DATASET:-tv3-formal-6000}"
DATA_DIR="data/${DATASET}"
SEED="${SEED:-20260704}"
SEQUENCES="${SEQUENCES:-6000}"
TIMESTEPS="${TIMESTEPS:-512}"
DT_S="${DT_S:-0.5}"
WORKERS="${WORKERS:-4}"

echo "==== D0 minimal: oracle / observed / tof-only ===="
echo "Python: $(python --version 2>&1)"
echo "Dataset: ${DATA_DIR}"
echo "Workers: ${WORKERS}"

if [[ -f "${DATA_DIR}/manifest.json" ]]; then
  echo "[SKIP] Dataset already exists: ${DATA_DIR}"
else
  python -m tv3.pipeline.generate_tunnel_ventilation_benchmark \
    --output-root data \
    --dataset "${DATASET}" \
    --sequences "${SEQUENCES}" \
    --seed "${SEED}" \
    --timesteps "${TIMESTEPS}" \
    --dt-s "${DT_S}" \
    --optical-absorption-backend empirical_v1 \
    --storage memmap \
    --workers "${WORKERS}" \
    --skip-fiber-mic
fi

for entry in \
  "D0-oracle|configs/tv3_d0_oracle_ridge.json" \
  "D0-observed|configs/tv3_d0_observed_ridge.json" \
  "D0-tof-only|configs/tv3_d0_tof_only_ridge.json"
do
  label="${entry%%|*}"
  config="${entry##*|}"
  echo "---- ${label} ----"
  python -m tv3.pipeline.run_tv3_rocket_baseline \
    --config "${config}" \
    --dataset-dir "${DATA_DIR}"
done

python - <<'PY'
import json
from pathlib import Path

for label, slug in (
    ("D0-oracle", "oracle_ridge"),
    ("D0-observed", "observed_ridge"),
    ("D0-tof-only", "tof_only_ridge"),
):
    metrics = json.loads((Path("outputs/tv3_d0") / slug / "metrics.json").read_text(encoding="utf-8"))
    values = metrics["evaluations"]
    print(f"{label}: features={metrics['feature_count']} alpha={metrics['diagnostics']['selected_alpha']}")
    for split in ("val", "test", "extrapolation"):
        components = values[split]["component_metrics"]
        print(
            f"  {split:13} CO2={components['x_CO2']['r2']:.4f} "
            f"O2={components['x_O2']['r2']:.4f} N2={components['x_N2']['r2']:.4f}"
        )
PY

echo "[DONE] D0 minimal completed."
