from importlib.resources import files

from gib.audit.s2_s3 import load_profile
from gib.contract import load_contracts
from gib.s5_contract import load_s5_contracts


EXPECTED_CONFIG_RESOURCES = {
    "p2_data_schema.json",
    "p2_manifest_schema.json",
    "p2_s1_grid.json",
    "p2_s2_s3_audit.json",
    "p2_s2_s3_frozen_evidence.json",
    "p2_s4_metric_registry.json",
    "p2_s5_discrepancy_contract.json",
    "p2_s5_source_registry.json",
    "p2_s6_ownership_registry.json",
    "p2_split_contract.json",
    "p3_execution_registry.json",
    "p3_g3_1_forward.json",
    "p3_pilot_plan.json",
    "p3_pilot_plan_v2.json",
    "p3_baseline_plan.json",
    "p3_c4_multiview_plan.json",
    "p3_c2_solver_plan.json",
    "p3_c5a_sampling_plan.json",
    "p3_c5b_data_efficiency_plan.json",
    "p3_c2_ic_rdu_vp_plan.json",
    "p3_c5c_crpkd_plan.json",
    "p3_c5d_figs_plan.json",
}


def test_frozen_configs_are_importable_package_resources() -> None:
    resource_names = {
        resource.name
        for resource in files("configs").iterdir()
        if resource.name.endswith(".json")
    }
    assert resource_names == EXPECTED_CONFIG_RESOURCES
    assert load_contracts()["data"]["contract_status"] == "contract_frozen"
    assert load_profile()["audit_id"] == "GIB-S2-S3-v1"
    registry, discrepancy = load_s5_contracts()
    assert registry["verdict"] == "source_complete"
    assert discrepancy["default_profile"] == "off"
