from pathlib import Path

from hg.pipeline.layout import CONFIG_GROUPS, OUTPUT_GROUPS, TOP_LEVEL_DIRS, ensure_project_layout
from hg.sim.generation.benchmark import DEFAULT_HITRAN_CACHE_ROOT


def test_layout_constants_match_target_architecture():
    assert TOP_LEVEL_DIRS == ("hg", "configs", "data", "outputs", "docs", "scripts", "tests")
    assert CONFIG_GROUPS == ("data",)
    assert OUTPUT_GROUPS == ("runs", "summary", "reports", "archive")


def test_ensure_project_layout_creates_expected_directories(tmp_path):
    ensure_project_layout(tmp_path)

    for dirname in TOP_LEVEL_DIRS:
        assert (tmp_path / dirname).is_dir()
    for group in CONFIG_GROUPS:
        assert (tmp_path / "configs" / group).is_dir()
    for group in OUTPUT_GROUPS:
        assert (tmp_path / "outputs" / group).is_dir()


def test_default_hitran_cache_root_uses_workspace_shared_cache():
    workspace_root = Path(__file__).resolve().parents[2]

    assert Path(DEFAULT_HITRAN_CACHE_ROOT) == workspace_root / "data" / "hitran_cache"
