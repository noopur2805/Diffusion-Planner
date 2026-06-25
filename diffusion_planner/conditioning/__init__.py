"""Scenario-type conditioning for the DiT planner.

Provides a closed-set integer id for every nuPlan scenario type. The id is
used as an optional categorical conditioning signal for the DiT, mixed into
the global ``y`` vector alongside the diffusion timestep embedding. Off by
default; enabled via ``--use_scenario_tag`` on the training entrypoints.

The id is derived directly from nuPlan's ground-truth ``scenario_type``
field, so no auxiliary perception model is required at inference time.
"""
from diffusion_planner.conditioning.taxonomy import (
    NUPLAN_SCENARIO_TYPES,
    NUM_SCENARIO_TYPES,
    UNKNOWN_SCENARIO_ID,
    HIGH_RISK_SCENARIO_NAMES,
    scenario_name_to_id,
    scenario_id_to_name,
    is_high_risk,
)

__all__ = [
    "NUPLAN_SCENARIO_TYPES",
    "NUM_SCENARIO_TYPES",
    "UNKNOWN_SCENARIO_ID",
    "HIGH_RISK_SCENARIO_NAMES",
    "scenario_name_to_id",
    "scenario_id_to_name",
    "is_high_risk",
]
