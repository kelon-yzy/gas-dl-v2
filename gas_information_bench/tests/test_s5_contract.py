import copy

import numpy as np
import pytest

from gib.s5_contract import (
    DiscrepancyUnavailable,
    S5ContractError,
    convert_unit,
    delta,
    load_s5_contracts,
    validate_discrepancy_contract,
    validate_source_registry,
)


CONDITION = {
    "T_K": 298.15,
    "P_kPa": 101.325,
    "RH_frac": 0.5,
    "L_m": 0.2,
    "gain": 1.0,
    "baseline": 0.0,
    "delay_s": 0.0,
    "crosstalk": 0.03,
    "q_flow": 1.0,
}


def test_source_registry_is_complete_for_controlled_synthetic_benchmark():
    registry, _ = load_s5_contracts()
    summary = validate_source_registry(registry)

    assert summary["verdict"] == "source_complete"
    assert summary["missing_key_source_count"] == 0
    assert registry["scope"]["benchmark_mode"] == "controlled_synthetic"
    assert registry["scope"]["real_hardware_fidelity_claim_allowed"] is False
    assert registry["policy"]["cross_method_profile_identity_required"] is True
    assert summary["inventory_count"] == len(registry["forward_inventory"])
    for entry in registry["entries"]:
        if entry["source_type"] == "engineering_assumption":
            assert entry["verification_status"] != "verified"


def test_source_registry_rejects_hardware_claim_scope_drift():
    registry, _ = load_s5_contracts()
    invalid = copy.deepcopy(registry)
    invalid["scope"]["real_hardware_fidelity_claim_allowed"] = True

    with pytest.raises(S5ContractError, match="cannot claim hardware fidelity"):
        validate_source_registry(invalid)


def test_discrepancy_contract_freezes_signature_and_profiles():
    _, contract = load_s5_contracts()
    summary = validate_discrepancy_contract(contract)

    assert summary["default_profile"] == "off"
    assert set(summary["profile_ids"]) == {"off", "p5_reserved"}


@pytest.mark.parametrize(
    ("value", "from_unit", "to_unit", "expected"),
    [
        (25.0, "C", "K", 298.15),
        (101.325, "kPa", "Pa", 101325.0),
        (2.0, "ms", "s", 0.002),
        (0.5, "mol/mol", "%", 50.0),
        (50.0, "%RH", "fraction", 0.5),
        (25.83, "mW/(m*K)", "W/(m*K)", 0.02583),
    ],
)
def test_registered_unit_conversions(value, from_unit, to_unit, expected):
    assert convert_unit(value, from_unit, to_unit) == pytest.approx(expected)


def test_unit_conversion_rejects_unregistered_pair():
    with pytest.raises(S5ContractError, match="unsupported unit conversion"):
        convert_unit(1.0, "bar", "kPa")


def test_off_profile_is_elementwise_identical_and_does_not_mutate_input():
    observation = np.array([0.25, -1.5, 3.0], dtype=float)
    before = copy.deepcopy(observation)

    result = delta(observation, CONDITION, "ndir")

    assert np.array_equal(result, before)
    result[0] = 99.0
    assert np.array_equal(observation, before)


def test_reserved_profile_is_not_injected_in_p2():
    with pytest.raises(DiscrepancyUnavailable, match="not executable during P2"):
        delta([0.1, 0.2], CONDITION, "thermal", "p5_reserved")


def test_unknown_profile_and_missing_condition_fail_explicitly():
    with pytest.raises(S5ContractError, match="unknown discrepancy profile"):
        delta([0.1], CONDITION, "slow", "future_profile")
    incomplete = dict(CONDITION)
    del incomplete["P_kPa"]
    with pytest.raises(S5ContractError, match="missing discrepancy condition field: P_kPa"):
        delta([0.1], incomplete, "slow")
