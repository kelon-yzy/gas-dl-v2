from hg.pipeline.layout import CONFIG_GROUPS, OUTPUT_GROUPS, TOP_LEVEL_DIRS, ensure_project_layout


def test_layout_constants_match_target_architecture():
    assert TOP_LEVEL_DIRS == ("src", "configs", "data", "outputs", "docs", "experiments", "tests")
    assert CONFIG_GROUPS == ("data", "model", "train", "eval", "experiment")
    assert OUTPUT_GROUPS == ("runs", "summary", "reports", "archive")


def test_ensure_project_layout_creates_expected_directories(tmp_path):
    ensure_project_layout(tmp_path)

    for dirname in TOP_LEVEL_DIRS:
        assert (tmp_path / dirname).is_dir()
    for group in CONFIG_GROUPS:
        assert (tmp_path / "configs" / group).is_dir()
    for group in OUTPUT_GROUPS:
        assert (tmp_path / "outputs" / group).is_dir()
