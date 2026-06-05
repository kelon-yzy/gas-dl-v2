from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl import cli as dl_cli
from dl.cli import build_parser as build_dl_cli_parser, run as run_dl_cli
from dl.training.losses import LOSS_REGISTRY, build_loss
from dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from dl.training.trainer import AmpConfig, EarlyStoppingConfig, Trainer, build_optimizer
from sim.core.schema import COMPONENT_FIELDS
from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset


def _make_smoke_dataset(tmp_path: Path, slug: str = "dl-train-smoke", sequences: int = 16) -> Path:
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=7,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
        ),
    )
    return tmp_path / slug


class TestLosses:
    def test_registry_contains_baseline_losses(self):
        assert {"mse", "mae", "smooth_l1", "huber"}.issubset(LOSS_REGISTRY)

    def test_build_loss_from_name(self):
        loss = build_loss("mse")
        value = loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
        assert torch.isclose(value, torch.tensor(2.0))

    def test_build_loss_from_config_passes_kwargs(self):
        loss = build_loss({"name": "smooth_l1", "beta": 0.5})
        assert loss.beta == 0.5

    def test_build_unknown_loss_raises(self):
        try:
            build_loss({"name": "imaginary"})
        except ValueError as exc:
            assert "imaginary" in str(exc)
        else:
            raise AssertionError("build_loss should reject unknown loss names")


class TestRegressionMetrics:
    def test_regression_metrics_for_perfect_prediction(self):
        y_true = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1]])
        metrics = regression_metrics(y_true, y_true)
        assert metrics == RegressionMetrics(mae=0.0, rmse=0.0, r2=1.0)

    def test_regression_metrics_for_nonperfect_prediction(self):
        y_true = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        y_pred = torch.tensor([[1.0, 1.0], [1.0, 5.0]])
        metrics = regression_metrics(y_pred, y_true)
        assert metrics.mae == 1.0
        assert round(metrics.rmse, 6) == round((6.0 / 4.0) ** 0.5, 6)
        assert metrics.r2 < 1.0

    def test_constant_target_r2_is_one_for_exact_match(self):
        y_true = torch.ones(3, 4)
        metrics = regression_metrics(y_true, y_true)
        assert metrics.r2 == 1.0

    def test_constant_target_r2_is_zero_for_mismatch(self):
        y_true = torch.ones(3, 4)
        y_pred = torch.zeros(3, 4)
        metrics = regression_metrics(y_pred, y_true)
        assert metrics.r2 == 0.0

    def test_component_metrics_use_v4_component_fields(self):
        y_true = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1]])
        y_pred = y_true + 0.1
        metrics = component_regression_metrics(y_pred, y_true)
        assert tuple(metrics) == COMPONENT_FIELDS
        for value in metrics.values():
            assert isinstance(value, RegressionMetrics)
            assert round(value.mae, 6) == 0.1

    def test_metric_shape_mismatch_raises(self):
        try:
            regression_metrics(torch.zeros(2, 4), torch.zeros(2, 3))
        except ValueError as exc:
            assert "shapes must match" in str(exc)
        else:
            raise AssertionError("regression_metrics should reject shape mismatch")

    def test_component_name_mismatch_raises(self):
        try:
            component_regression_metrics(torch.zeros(2, 4), torch.zeros(2, 4), component_names=("x",))
        except ValueError as exc:
            assert "component_names length" in str(exc)
        else:
            raise AssertionError("component_regression_metrics should reject name mismatch")


class TestDLCli:
    def test_cli_trains_and_writes_run_artifacts(self, tmp_path: Path, capsys):
        dataset_dir = _make_smoke_dataset(tmp_path)
        output_dir = tmp_path / "runs" / "cnn1d"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "cnn1d",
                "--model-kwargs",
                '{"hidden_channels":[4],"kernel_size":3,"dropout":0.0}',
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--lr",
                "0.001",
                "--eval-splits",
                "val,test",
                "--json",
            ]
        )

        payload = run_dl_cli(args)

        stdout_payload = json.loads(capsys.readouterr().out)
        metrics_path = output_dir / "metrics.json"
        config_path = output_dir / "run_config.json"
        checkpoint_path = output_dir / "checkpoint.pt"
        assert metrics_path.is_file()
        assert config_path.is_file()
        assert checkpoint_path.is_file()
        assert stdout_payload["checkpoint_path"] == str(checkpoint_path)
        assert payload["model_config"]["name"] == "cnn1d"
        assert payload["model_config"]["in_channels"] == 8
        assert payload["model_config"]["out_dim"] == 4
        assert set(payload["evaluations"]) == {"val", "test"}
        assert len(payload["history"]) == 1
        assert set(payload["evaluations"]["val"]["component_metrics"]) == set(COMPONENT_FIELDS)

    def test_cli_tcn_config_infers_target_timesteps(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="dl-tcn-config")
        output_dir = tmp_path / "runs" / "tcn"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "tcn",
                "--model-kwargs",
                '{"channels":[4,4],"dropout":0.0}',
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--eval-splits",
                "val",
            ]
        )

        payload = run_dl_cli(args)

        assert "target_timesteps" not in payload["model_config"]
        assert payload["input_format"] == "NCT"
        assert set(payload["evaluations"]) == {"val"}

    def test_cli_reads_json_config_and_writes_best_checkpoint(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="dl-config")
        output_dir = tmp_path / "runs" / "config-cnn1d"
        config_path = tmp_path / "dl_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "output_dir": str(output_dir),
                    "model": "cnn1d",
                    "model_kwargs": {"hidden_channels": [4], "kernel_size": 3, "dropout": 0.0},
                    "epochs": 1,
                    "batch_size": 4,
                    "eval_splits": ["val"],
                    "early_stopping": {"enabled": False},
                    "scheduler": {"name": "none"},
                }
            ),
            encoding="utf-8",
        )
        parser = build_dl_cli_parser()
        args = parser.parse_args(["--config", str(config_path)])

        payload = run_dl_cli(args)

        assert payload["model_config"]["name"] == "cnn1d"
        assert payload["best_checkpoint_path"] == str(output_dir / "best_checkpoint.pt")
        assert (output_dir / "best_checkpoint.pt").is_file()
        assert payload["learning_rates"] == [0.001]

    def test_cli_writes_live_progress_jsonl_without_polluting_json_stdout(self, tmp_path: Path, capsys):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="dl-progress")
        output_dir = tmp_path / "runs" / "progress-cnn1d"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "cnn1d",
                "--model-kwargs",
                '{"hidden_channels":[4],"kernel_size":3,"dropout":0.0}',
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--eval-splits",
                "val",
                "--json",
            ]
        )

        payload = run_dl_cli(args)

        captured = capsys.readouterr().out
        assert "[epoch]" not in captured
        stdout_payload = json.loads(captured)
        progress_path = output_dir / "metrics_live.jsonl"
        metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
        assert progress_path.is_file()
        assert payload["progress_log_path"] == str(progress_path)
        assert stdout_payload["progress_log_path"] == str(progress_path)
        assert metrics_payload["progress_log_path"] == str(progress_path)
        assert events[0]["event"] == "epoch_end"
        assert events[0]["model"] == "cnn1d"
        assert events[0]["epoch"] == 1
        assert events[0]["epochs"] == 1
        assert "train_loss" in events[0]
        assert "val_loss" in events[0]
        assert "learning_rate" in events[0]
        assert "epoch_seconds" in events[0]
        assert "train_seconds" in events[0]
        assert "val_seconds" in events[0]
        assert "train_samples_per_second" in events[0]
        assert "gpu_memory_allocated_mb" in events[0]
        assert "gpu_memory_reserved_mb" in events[0]
        assert "best_epoch" in events[0]
        assert events[-1]["event"] == "training_completed"

    def test_build_loader_omits_worker_only_kwargs_when_num_workers_zero(self, monkeypatch):
        captured = {}

        def fake_loader(dataset, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(dl_cli, "DataLoader", fake_loader)
        dataset = TensorDataset(torch.ones(2, 1), torch.ones(2, 4))

        dl_cli._build_loader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            seed=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        assert captured["pin_memory"] is True
        assert "persistent_workers" not in captured
        assert "prefetch_factor" not in captured

    def test_build_loader_passes_worker_kwargs_when_num_workers_positive(self, monkeypatch):
        captured = {}

        def fake_loader(dataset, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(dl_cli, "DataLoader", fake_loader)
        dataset = TensorDataset(torch.ones(2, 1), torch.ones(2, 4))

        dl_cli._build_loader(
            dataset,
            batch_size=1,
            num_workers=2,
            shuffle=False,
            seed=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        assert captured["pin_memory"] is True
        assert captured["persistent_workers"] is True
        assert captured["prefetch_factor"] == 2


class TestTrainerControl:
    def test_early_stopping_stops_when_val_loss_does_not_improve(self, tmp_path: Path):
        model = nn.Linear(1, 4)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(1.0)
        loss_fn = nn.MSELoss()
        optimizer = build_optimizer(model, {"name": "sgd", "lr": 0.0})
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
        loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4, 4)), batch_size=2)

        history = trainer.fit(
            loader,
            val_loader=loader,
            epochs=5,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=1),
            best_checkpoint_path=tmp_path / "best.pt",
        )

        assert history.stopped_early is True
        assert len(history.epochs) == 2
        assert "did not improve" in str(history.stop_reason)
        assert (tmp_path / "best.pt").is_file()

    def test_reduce_on_plateau_scheduler_reduces_lr(self):
        model = nn.Linear(1, 4)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(1.0)
        loss_fn = nn.MSELoss()
        optimizer = build_optimizer(model, {"name": "sgd", "lr": 0.1})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
        loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4, 4)), batch_size=2)

        trainer.fit(loader, val_loader=loader, epochs=3, scheduler=scheduler)

        assert optimizer.param_groups[0]["lr"] < 0.1

    def test_fit_calls_epoch_callback_with_progress_fields(self):
        model = nn.Linear(1, 4)
        loss_fn = nn.MSELoss()
        optimizer = build_optimizer(model, {"name": "sgd", "lr": 0.01})
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
        loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4, 4)), batch_size=2)
        events = []

        def callback(epoch, history, total_epochs):
            events.append(
                {
                    "epoch": epoch.epoch,
                    "epochs": total_epochs,
                    "train_loss": epoch.train_loss,
                    "val_loss": epoch.val_loss,
                    "learning_rate": epoch.learning_rate,
                    "best_epoch": history.best_epoch.epoch,
                }
            )

        trainer.fit(loader, val_loader=loader, epochs=2, epoch_callback=callback)

        assert len(events) == 2
        assert events[0]["epoch"] == 1
        assert events[0]["epochs"] == 2
        assert events[0]["val_loss"] is not None
        assert events[0]["learning_rate"] == 0.01
        assert events[0]["train_loss"] >= 0.0
        assert events[0]["best_epoch"] in {1, 2}

    def test_amp_enabled_requires_cuda_device(self):
        model = nn.Linear(1, 4)
        loss_fn = nn.MSELoss()
        optimizer = build_optimizer(model, {"name": "sgd", "lr": 0.01})
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device="cpu")
        loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4, 4)), batch_size=2)

        with pytest.raises(ValueError, match="requires a CUDA device"):
            trainer.fit(loader, epochs=1, amp=AmpConfig(enabled=True))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_amp_cuda_smoke_train_one_epoch(self):
        model = nn.Linear(1, 4)
        loss_fn = nn.MSELoss()
        optimizer = build_optimizer(model, {"name": "sgd", "lr": 0.01})
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device="cuda")
        loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4, 4)), batch_size=2)

        history = trainer.fit(loader, epochs=1, amp=AmpConfig(enabled=True))

        assert len(history.epochs) == 1
