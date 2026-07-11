"""Greedy weapon-target assignment, batched across environments.

Greedy is the Phase 1 baseline and the thing the saturation curve is measured
against. An optimal solver replaces `greedy` in Phase 2 behind the same signature.
"""

from __future__ import annotations

import torch
from torch import Tensor

UNASSIGNED = -1


def greedy(priority: Tensor, assignable: Tensor, available: Tensor) -> Tensor:
    """Bind available interceptor slots to the highest-priority threats.

        priority   [N, T] float, higher is more urgent
        assignable [N, T] bool, threat is a legal target for a new round
        available  [N, I] bool, slot holds a round and is not already committed

    Returns [N, I] int64 giving each slot its bound threat index, or -1.

    The whole batch is solved with a sort and a gather rather than a loop over
    environments. Threats are ranked by priority; available slots are ranked by index;
    the j-th slot takes the j-th threat. That is exactly the greedy result, because
    every interceptor here is interchangeable -- there is no slot-dependent cost that
    would make a different pairing of the same two sets cheaper.

    Ties in priority break toward the lower threat index, via a stable sort. The
    tie-break is not arbitrary detail: an unstable sort would make assignment
    non-deterministic across backends and the oracle comparison meaningless.

    Guarantees, all asserted in tests: no threat is bound twice in one call, no
    unavailable slot is bound, no non-assignable threat is bound.
    """
    n_threats = priority.shape[1]

    # Non-assignable threats sort below every real candidate, so the ranked prefix of
    # length n_assignable contains exactly the legal targets.
    floor = torch.full_like(priority, float("-inf"))
    ranked = torch.argsort(
        torch.where(assignable, priority, floor), dim=1, descending=True, stable=True
    )
    n_assignable = assignable.sum(dim=1, keepdim=True)

    # Position of each available slot among the available ones, 0-based.
    slot_rank = available.to(torch.int64).cumsum(dim=1) - 1

    bound = available & (slot_rank >= 0) & (slot_rank < n_assignable)

    # Clamp keeps the gather in bounds where there are more slots than threats; those
    # entries are masked out by `bound` immediately afterwards.
    picked = torch.gather(ranked, 1, slot_rank.clamp(0, n_threats - 1))
    return torch.where(bound, picked, torch.full_like(picked, UNASSIGNED))
