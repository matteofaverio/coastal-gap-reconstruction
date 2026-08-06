"""The chlorophyll engineered-hybrid method-selection rule ("Rule D"):
a deterministic, length-routed assignment of which component method
produced the `engineered_hybrid_reconstruction` candidate for each real
gap.

**This is a lookup over an already-frozen output, not a new model.** No GP,
Kalman, or ExtraTrees model is fit here -- `method_for_length` reproduces
the *routing decision* the original engineered-hybrid pipeline
made, so a caller can explain and verify (not regenerate) which method
produced each real gap's `results/chlorophyll/
chlorophyll_reconstruction_engineered_hybrid.csv` row.

**Component methods** (validated on the canonical artificial-gap pool):
- L=1-3: GP M1 (Matern, time-only) -- validated significantly better than
  interpolation at these lengths specifically.
- L=4-29: state-space Kalman smoother -- root-caused as numerically
  degenerate (behaves like interpolation in 93% of artificial-pool gaps;
  see `probabilistic_models.py`) -- included here because it is
  what the released file actually used, not because its own rationale is
  fully evidenced; this module states that plainly rather than silently
  presenting the routing as if every segment were equally well-supported.
- L>=30: gap-edge residual model (ExtraTrees, residual-over-interpolation).

**Why a selected value remains a candidate, not observed truth**: Rule D's
per-length component methods were each validated on the *canonical
artificial-gap pool* (where the true value is known and withheld), which
establishes their *general* skill at a given gap length. Applying the
length-appropriate method to a *real* gap produces a plausible reconstruction
for that specific interval, but the real gap's true value was never observed
by anyone -- so the output is exactly as much a candidate as any other
method's output, never validated ground truth for that particular gap.

**Context-constrained gaps**: one gap in the 128-gap real-gap inventory
(`REAL_OPEN_20260515`, open-ended at the end of the available series, no
post-edge observation) is excluded from Rule D entirely -- every component
method requires bracketing context on both sides. `route_real_gaps` reports
this gap's method as `None`, matching the released file's own `NaN` value
for it, not a silently-substituted fallback.

**Gaps longer than the validated support**: Rule D still assigns a method
to every context-available gap regardless of length (including the
256-day scenario, L>=30 -> gap-edge residual model) -- this is an
extrapolation past the artificial-gap pool's own validated envelope
(max validated length is well under 256 days for any component method),
stated explicitly, not implied to be validated.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["RULE_D", "RULE_C", "method_for_length", "route_real_gaps"]

# Public method-name strings, matching the values actually present in
# results/chlorophyll/chlorophyll_reconstruction_engineered_hybrid.csv's
# own `method` column -- not the original internal short codes
# (gp_m1/m4_kalman/tier_ch).
METHOD_GP = "gaussian_process"
METHOD_KALMAN = "state_space_kalman"
METHOD_GAP_EDGE_RESIDUAL = "gap_edge_residual_model"

RULE_D = "D"  # primary rule: L1-3 GP, L4-29 Kalman, L>=30 gap-edge residual
RULE_C = "C"  # sensitivity variant: L1-3 GP, L4-13 Kalman, L>=14 gap-edge residual


def method_for_length(length_days: int, rule: str = RULE_D) -> str:
    """The Rule D (or Rule C sensitivity) method assignment for a gap of
    `length_days`, assuming bracketing context is available (see
    `route_real_gaps` for the context-availability exclusion)."""
    if rule == RULE_D:
        if length_days <= 3:
            return METHOD_GP
        if length_days <= 29:
            return METHOD_KALMAN
        return METHOD_GAP_EDGE_RESIDUAL
    if rule == RULE_C:
        if length_days <= 3:
            return METHOD_GP
        if length_days <= 13:
            return METHOD_KALMAN
        return METHOD_GAP_EDGE_RESIDUAL
    raise ValueError(f"Unknown rule {rule!r}; choose 'D' (primary) or 'C' (sensitivity)")


def route_real_gaps(inventory: pd.DataFrame, rule: str = RULE_D) -> pd.DataFrame:
    """Apply the length-routed method assignment to every gap in a real-gap
    inventory (as returned by `real_gap_inventory.detect_real_gaps`).

    A gap without post-edge context (`post_edge_available == False` --
    exactly the one open-ended gap running to the end of the series, in the
    released 128-gap inventory) gets `method = None` -- verified directly
    against the released output, not assumed: `post_edge_available` is the
    exact discriminator (126/127 gaps missing *pre*-edge context still get a
    routed method -- the Kalman/GP component methods use the full observed
    series, not a strict pre/post interpolation anchor, so lacking pre-edge
    alone does not exclude a gap; only lacking post-edge does). This must
    never be silently substituted with interpolation or any other fallback.

    Returns `inventory` with an added `assigned_method` column.
    """
    out = inventory.copy()
    post_available = out["post_edge_available"].fillna(False).astype(bool)
    out["assigned_method"] = [
        method_for_length(int(length), rule) if ok else None
        for length, ok in zip(out["length_days"], post_available)
    ]
    return out
