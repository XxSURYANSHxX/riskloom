"""Cost-aware three-tier ALLOW / REVIEW / DENY policy banding.

This package is deliberately isolated from ground-truth labels. It never imports the simulation
label module, directly or transitively, and never receives a scenario type, campaign identifier or
any other field absent from a real production checkout-preflight event. Routing decisions are a
function of the locked model's calibrated probability alone.
"""

from riskloom.policy.bands import (
    ALLOW,
    DENY,
    REVIEW,
    BandPolicy,
    CostPolicy,
    band_decisions,
    evaluate_band,
    evaluate_single_threshold,
    select_band,
)
from riskloom.policy.config import PolicyConfig, load_policy_config

__all__ = [
    "ALLOW",
    "DENY",
    "REVIEW",
    "BandPolicy",
    "CostPolicy",
    "PolicyConfig",
    "band_decisions",
    "evaluate_band",
    "evaluate_single_threshold",
    "load_policy_config",
    "select_band",
]
