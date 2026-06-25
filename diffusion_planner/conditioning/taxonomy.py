"""Closed set of nuPlan scenario types used for the VLM scenario-id channel.

The list mirrors the ten scenario types that appear in the 47-scenario
``one_of_each_scenario_type`` filter and in :mod:`scripts.plot_pdms_breakdown`.
A trailing ``unknown`` slot is reserved so that out-of-vocabulary VLM
outputs always have a well-defined id.

The id assignment is **append-only**; reordering or inserting types in the
middle would silently invalidate every checkpoint that was trained with
the scenario-id embedding. New types may only be appended before
``unknown`` (which is required to remain last).
"""
from __future__ import annotations

from typing import List

NUPLAN_SCENARIO_TYPES: List[str] = [
    "waiting_for_pedestrian_to_cross",
    "accelerating_at_traffic_light_without_lead",
    "changing_lane_to_left",
    "starting_unprotected_cross_turn",
    "starting_left_turn",
    "starting_protected_noncross_turn",
    "near_multiple_vehicles",
    "on_pickup_dropoff",
    "stopping_at_stop_sign_with_lead",
    "following_lane_with_lead",
    # append new types here, before 'unknown'
    "unknown",
]

NUM_SCENARIO_TYPES: int = len(NUPLAN_SCENARIO_TYPES)
UNKNOWN_SCENARIO_ID: int = NUPLAN_SCENARIO_TYPES.index("unknown")

_NAME_TO_ID = {name: i for i, name in enumerate(NUPLAN_SCENARIO_TYPES)}


def scenario_name_to_id(name: str) -> int:
    """Map a scenario name to its integer id.

    Unknown / free-text names map to :data:`UNKNOWN_SCENARIO_ID`.
    Matching is case-insensitive and tolerates a leading ``scenario_`` prefix
    (common in nuPlan logs).
    """
    if name is None:
        return UNKNOWN_SCENARIO_ID
    key = str(name).strip().lower()
    if key.startswith("scenario_"):
        key = key[len("scenario_"):]
    return _NAME_TO_ID.get(key, UNKNOWN_SCENARIO_ID)


def scenario_id_to_name(scenario_id: int) -> str:
    """Inverse of :func:`scenario_name_to_id`.

    Out-of-range ids resolve to the ``unknown`` slot rather than raising.
    """
    if 0 <= scenario_id < NUM_SCENARIO_TYPES:
        return NUPLAN_SCENARIO_TYPES[scenario_id]
    return NUPLAN_SCENARIO_TYPES[UNKNOWN_SCENARIO_ID]


# Tags that the safety-tier hook (Milestone 4) treats as elevated-risk.
HIGH_RISK_SCENARIO_NAMES = frozenset({
    "waiting_for_pedestrian_to_cross",
    "starting_unprotected_cross_turn",
    "near_multiple_vehicles",
})


def is_high_risk(name_or_id) -> bool:
    """Return True iff the scenario is in :data:`HIGH_RISK_SCENARIO_NAMES`.

    Accepts either a string name or an integer id.
    """
    if isinstance(name_or_id, int):
        name_or_id = scenario_id_to_name(name_or_id)
    return str(name_or_id).strip().lower() in HIGH_RISK_SCENARIO_NAMES
