from __future__ import annotations

import numpy as np

from hg.common.composition import (
    ALR_CH4_TRANSFORM,
    DEFAULT_ZERO_REPLACEMENT_EPSILON,
    ILR_N2_FIRST_TRANSFORM,
    TRAIN_MIN_POSITIVE_HALF_EPSILON,
    TargetTransformSpec,
    alr_forward,
    alr_inverse,
    close_to_100,
    close_to_unit_interval,
    ilr_forward,
    ilr_inverse,
    replace_zeros_multiplicative,
    resolve_target_transform_for_training,
    resolve_zero_replacement_epsilon,
    transform_composition_targets,
    inverse_transform_composition_targets,
)
from hg.sim.core.schema import COMPONENT_FIELDS


def test_closure_helpers_preserve_row_totals():
    values = np.array([[10.0, 80.0, 5.0, 5.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

    unit = close_to_unit_interval(values)
    percent = close_to_100(values)

    np.testing.assert_allclose(unit.sum(axis=1), np.ones(2), atol=1e-7)
    np.testing.assert_allclose(percent.sum(axis=1), np.full(2, 100.0), atol=1e-6)


def test_zero_replacement_is_explicit_and_closed():
    values = np.array([[0.0, 0.80, 0.05, 0.15], [0.10, 0.70, 0.0, 0.20]], dtype=np.float32)

    replaced, audit = replace_zeros_multiplicative(values, epsilon=DEFAULT_ZERO_REPLACEMENT_EPSILON)

    np.testing.assert_allclose(replaced.sum(axis=1), np.ones(2), atol=1e-7)
    assert audit.component_names == COMPONENT_FIELDS
    assert audit.replaced_counts == (1, 0, 1, 0)
    assert audit.total_rows == 2
    assert replaced[0, 0] == DEFAULT_ZERO_REPLACEMENT_EPSILON
    assert replaced[1, 2] == DEFAULT_ZERO_REPLACEMENT_EPSILON


def test_zero_replacement_allows_empty_2d_split_audit():
    values = np.empty((0, 4), dtype=np.float32)

    replaced, audit = replace_zeros_multiplicative(values, epsilon=DEFAULT_ZERO_REPLACEMENT_EPSILON)

    assert replaced.shape == (0, 4)
    assert audit.component_names == COMPONENT_FIELDS
    assert audit.replaced_counts == (0, 0, 0, 0)
    assert audit.max_abs_delta_percent == (0.0, 0.0, 0.0, 0.0)
    assert audit.total_rows == 0


def test_zero_replacement_rejects_empty_1d_input():
    try:
        replace_zeros_multiplicative(np.array([], dtype=np.float32), epsilon=DEFAULT_ZERO_REPLACEMENT_EPSILON)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("1D empty input should be rejected")


def test_train_min_positive_half_epsilon_resolves_from_train_labels():
    train_percent = np.array(
        [
            [0.0, 98.0, 1.0, 1.0],
            [0.5, 90.0, 5.0, 4.5],
        ],
        dtype=np.float32,
    )

    epsilon = resolve_zero_replacement_epsilon(TRAIN_MIN_POSITIVE_HALF_EPSILON, train_percent)
    spec = resolve_target_transform_for_training(
        {"name": ALR_CH4_TRANSFORM, "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
        train_percent,
    )

    assert epsilon == 0.0025
    assert spec is not None
    assert spec.epsilon == epsilon


def test_transform_rejects_unresolved_epsilon_strategy():
    values = np.array([[10.0, 75.0, 5.0, 10.0]], dtype=np.float32)
    spec = TargetTransformSpec(name=ALR_CH4_TRANSFORM, epsilon=TRAIN_MIN_POSITIVE_HALF_EPSILON)

    try:
        transform_composition_targets(values, spec)
    except ValueError as exc:
        assert "resolved" in str(exc)
    else:
        raise AssertionError("unresolved epsilon strategy should be rejected")


def test_alr_ch4_roundtrip_returns_original_composition():
    values = np.array([[10.0, 75.0, 5.0, 10.0], [25.0, 55.0, 12.0, 8.0]], dtype=np.float32)
    unit = close_to_unit_interval(values)

    transformed = alr_forward(unit, reference_index=1)
    restored = alr_inverse(transformed, reference_index=1)

    assert transformed.shape == (2, 3)
    np.testing.assert_allclose(restored, unit, atol=1e-6)


def test_alr_inverse_handles_large_coordinates_without_overflow():
    transformed = np.array([[1000.0, 0.0, -1000.0], [-1000.0, 0.0, 1000.0]], dtype=np.float32)

    restored = alr_inverse(transformed, reference_index=1)

    assert np.isfinite(restored).all()
    np.testing.assert_allclose(restored.sum(axis=1), np.ones(2), atol=1e-6)
    assert restored[0, 0] > 0.999
    assert restored[1, 3] > 0.999


def test_ilr_n2_first_roundtrip_returns_original_composition():
    values = np.array([[10.0, 75.0, 5.0, 10.0], [25.0, 55.0, 12.0, 8.0]], dtype=np.float32)
    unit = close_to_unit_interval(values)

    transformed = ilr_forward(unit)
    restored = ilr_inverse(transformed)

    assert transformed.shape == (2, 3)
    np.testing.assert_allclose(restored, unit, atol=1e-6)


def test_ilr_rejects_non_isometric_basis():
    values = np.array([[10.0, 75.0, 5.0, 10.0]], dtype=np.float32)
    bad_basis = np.ones((3, 4), dtype=np.float64)

    try:
        ilr_forward(close_to_unit_interval(values), basis=bad_basis)
    except ValueError as exc:
        assert "basis" in str(exc)
    else:
        raise AssertionError("non-isometric ILR basis should be rejected")


def test_target_transform_applies_zero_replacement_before_log_ratio():
    values = np.array([[0.0, 80.0, 5.0, 15.0], [10.0, 70.0, 0.0, 20.0]], dtype=np.float32)
    spec = TargetTransformSpec(name=ALR_CH4_TRANSFORM)

    transformed, audit = transform_composition_targets(values, spec)
    restored = inverse_transform_composition_targets(transformed, spec)

    assert transformed.shape == (2, 3)
    assert audit.replaced_counts == (1, 0, 1, 0)
    np.testing.assert_allclose(restored.sum(axis=1), np.full(2, 100.0), atol=1e-5)
    assert np.all(restored > 0.0)


def test_ilr_target_transform_uses_n2_first_basis():
    values = np.array([[10.0, 75.0, 5.0, 10.0], [10.0, 75.0, 5.0, 20.0]], dtype=np.float32)
    spec = TargetTransformSpec(name=ILR_N2_FIRST_TRANSFORM)

    transformed, _audit = transform_composition_targets(values, spec)

    assert transformed[1, 0] > transformed[0, 0]
