"""Percentile curricula for ERE-gated self-training.

A curriculum is fully described by two numbers:

    num_stages       how many curriculum stages (1 = "flat", i.e. a single
                     constant gate; 2/4/8 = progressively finer ramps).
    percentile_cap   the percentile reached at the final stage, in [0, 100].
                     The cap is the largest fraction of residual samples ever
                     admitted. cap=100 admits every sample at the last stage
                     (vanilla self-training); cap<100 never trusts the noisiest
                     (100 - cap)% of samples; cap=0 admits no pool sample at all,
                     i.e. oracle-only finetuning ("oracle-ft", the no-pool
                     control reported in the ablation tables).

Both the per-stage percentiles and the epoch boundaries are derived
deterministically from these two numbers, so a config only needs to specify
``num_stages`` and ``percentile_cap``.

Percentiles are evenly spaced up to the cap: for num_stages=n and cap=C the
schedule is ``[C/n, 2C/n, ..., C]`` (the paper's linear ramp, Eq. 4). Epoch
boundaries split the total training budget evenly across stages, with the last
boundary pinned to the budget.

An optional third number, ``schedule_gamma`` (default 1), bends the ramp between
the same endpoints, ``pi_t = C * (t/n) ** gamma`` (Suppl. "Shape of the admission
ramp"): gamma=1/2 is concave (front-loaded), gamma=1 linear (canonical PACER),
gamma=2 convex (keeps the gate tight for longer). Epoch boundaries are unchanged.
"""

from typing import List, Tuple

TOTAL_EPOCHS: int = 120


def build_percentile_schedule(
    num_stages: int,
    percentile_cap: float,
    total_epochs: int = TOTAL_EPOCHS,
    gamma: float = 1.0,
) -> Tuple[List[float], List[int]]:
    """Return ``(percentiles, epoch_boundaries)`` for a curriculum.

    ``percentiles`` is monotone increasing and ends at ``percentile_cap``.
    ``epoch_boundaries`` is strictly increasing, cumulative, and ends at
    ``total_epochs``.
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if not 0.0 <= percentile_cap <= 100.0:
        raise ValueError(f"percentile_cap must be in [0, 100], got {percentile_cap}")
    if total_epochs < num_stages:
        raise ValueError(
            f"total_epochs ({total_epochs}) must be >= num_stages ({num_stages})"
        )
    if gamma <= 0:
        raise ValueError(f"schedule_gamma must be > 0, got {gamma}")

    if float(gamma) == 1.0:
        # Kept in this exact form so the canonical linear ramp is bit-identical.
        percentiles = [percentile_cap * i / num_stages for i in range(1, num_stages + 1)]
    else:
        percentiles = [percentile_cap * (i / num_stages) ** gamma for i in range(1, num_stages + 1)]

    boundaries = [round(total_epochs * i / num_stages) for i in range(1, num_stages + 1)]
    boundaries[-1] = total_epochs
    # Guarantee strict monotonicity even if rounding collapses two boundaries.
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 1
    if boundaries[-1] != total_epochs:
        raise ValueError(
            f"Cannot fit {num_stages} stages into {total_epochs} epochs without collisions"
        )

    return percentiles, boundaries


def _num_str(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x).replace(".", "p")


def schedule_tag(num_stages: int, percentile_cap: float, gamma: float = 1.0) -> str:
    """Path-safe tag for a curriculum, e.g. 'flat_cap100', 'step-2_cap60' or
    'step-4_cap100_gamma0p5' (the gamma suffix appears only for non-linear ramps)."""
    shape = "flat" if num_stages == 1 else f"step-{num_stages}"
    tag = f"{shape}_cap{_num_str(percentile_cap)}"
    if float(gamma) != 1.0:
        tag += f"_gamma{_num_str(gamma)}"
    return tag


if __name__ == "__main__":
    # Self-check: linear default, and the ramp-shape percentiles of the paper.
    assert build_percentile_schedule(4, 100.0) == ([25.0, 50.0, 75.0, 100.0], [30, 60, 90, 120])
    assert build_percentile_schedule(4, 100.0, gamma=2.0)[0] == [6.25, 25.0, 56.25, 100.0]
    concave = build_percentile_schedule(4, 100.0, gamma=0.5)[0]
    assert [round(p, 2) for p in concave] == [50.0, 70.71, 86.6, 100.0], concave
    assert schedule_tag(4, 100.0) == "step-4_cap100"
    assert schedule_tag(4, 100.0, 0.5) == "step-4_cap100_gamma0p5"
    print("schedules.py self-check OK")
