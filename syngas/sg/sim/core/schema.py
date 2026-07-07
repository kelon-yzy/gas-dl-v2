"""Default schema surface for the isolated syngas package.

The old monorepo kept hydrogen-ng defaults in ``sim.core.schema`` and placed
syngas-specific fields in ``syngas_schema``.  After scenario isolation, ``sg``
has one default schema: syngas.  Shared copied modules can keep importing
``sg.sim.core.schema`` without reintroducing hydrogen-ng target semantics.
"""
from __future__ import annotations

from sg.sim.core.syngas_schema import (
    ALL_COMPONENT_FIELDS,
    BACKGROUND_FIELDS,
    COMPONENT_FIELDS,
    COMPOSITION_SCHEME,
    CONDITION_GRID_FIELDS,
    SCHEMA_VERSION,
    SEQUENCE_INDEX_FIELDS,
    SEQUENCE_LABEL_FIELDS,
    SLOW_CHANNELS,
    SLOW_DYNAMIC_CHANNELS,
    SLOW_MODAL_GROUPS,
    SLOW_SEQUENCE_FIELDS,
    SPLIT_FIELDS,
    SPLIT_NAMES,
)


PHASE_NAMES = ("baseline", "exposure", "steady", "recovery")
MULTI_PATH_PHASES = ("off", "baseline", "steady")
VALID_STORAGE_FORMATS = ("memmap", "npz", "both")
LEGACY_CONDITION_FIELDS: tuple[str, ...] = ()
