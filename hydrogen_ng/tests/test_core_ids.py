from hg.sim.core.ids import (
    BenchmarkDatasetId,
    MixtureId,
    SequenceId,
    make_mixture_id,
    make_sequence_id,
)


def test_stable_ids_use_distinct_prefixes():
    mixture_id = make_mixture_id(12)
    sequence_id = make_sequence_id(12)

    assert mixture_id == MixtureId("M000012")
    assert sequence_id == SequenceId("Q000012")
    assert mixture_id != sequence_id


def test_benchmark_dataset_id_rejects_empty_slug():
    try:
        BenchmarkDatasetId("")
    except ValueError as exc:
        assert "dataset slug must not be empty" in str(exc)
    else:
        raise AssertionError("empty dataset slug was accepted")
