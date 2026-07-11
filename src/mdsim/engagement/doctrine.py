"""How many rounds a decision commits to each threat.

Phase 1 fires one. Shoot-look-shoot falls out of that for free: a threat whose round
misses becomes assignable again at the next decision, and if stock remains it draws
another shot. Salvo -- committing k rounds up front against one threat -- is the case
this file does not yet cover.
"""

from __future__ import annotations

from torch import Tensor

from mdsim.engagement.assignment import greedy


def single_shot(priority: Tensor, assignable: Tensor, available: Tensor) -> Tensor:
    """One round per threat per decision, [N, I] int64 with -1 for unassigned.

    Single-shot is the Phase 1 choice because it is what makes the saturation curve
    mean anything: with one round per threat, leakage against inventory measures the
    defence running out of rounds. A salvo policy folds a second variable -- how many
    rounds each threat draws -- into the same curve and the knee stops being readable.

    Salvo hook: `greedy` binds the j-th available slot to the j-th ranked threat, so
    firing k rounds each needs the threat ranking repeated k times before the pairing
    -- repeat_interleave on the ranked order, with `assignable` recomputed so a threat
    stays eligible until it holds k rounds. The engagement loop around it is unchanged,
    which is why the doctrine choice lives here and not inside the assignment solver.
    """
    return greedy(priority, assignable, available)
