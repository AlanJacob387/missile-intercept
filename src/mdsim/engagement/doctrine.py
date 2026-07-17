"""How many rounds a decision commits to each threat.

Phase 1 fires one. Shoot-look-shoot falls out of that for free: a threat whose round
misses becomes assignable again at the next decision, and if stock remains it draws
another shot. Salvo -- committing up to k rounds up front against one threat in a
single decision -- is `salvo` below; shoot-look-shoot within a salvo is not yet
implemented, see its docstring.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mdsim.engagement.assignment import UNASSIGNED, greedy


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


def salvo(
    priority: Tensor,
    assignable: Tensor,
    available: Tensor,
    salvo_size: int,
    assign_fn=greedy,
) -> Tensor:
    """Up to `salvo_size` rounds per threat per decision, [N, I] int64 with -1 for
    unassigned.

    Runs `assign_fn` `salvo_size` times against the same `priority` and `assignable`,
    shrinking `available` between rounds by the slots the previous round bound, so
    each round competes for whatever slots are left rather than re-binding the same
    ones. A slot's final value is whatever it was bound to in the LAST round that
    bound it, via torch.where the same way `inventory.commit` layers a new proposal
    over an old one.

    `priority` and `assignable` are not touched between rounds, so a threat that
    already won a round earlier in this call is still fully eligible for the next
    one -- this function draws up to `salvo_size` rounds per threat with no memory
    of prior decisions. Keeping a threat that already holds a full salvo out of
    `assignable` in the first place is the caller's job, done by widening the
    `salvo_size` argument to `inventory.engaged_threats` before it feeds `assignable`
    here; this function has no way to tell "won earlier in this call" apart from
    "already fully engaged from a prior decision" and does not try to.

    The loop is `salvo_size` iterations over a fixed, host-side integer -- typically 1
    or 2, set once from config -- not a loop over environments or steps. Every
    iteration still runs the whole batch through `assign_fn` in one call, so this
    carries none of the performance cost a per-environment or per-step Python loop
    would in this codebase; do not read it as that kind of loop.

    `assign_fn` must share `greedy`'s exact (priority, assignable, available) ->
    assignment contract. This function does not special-case `greedy`: any solver
    meeting that contract runs here unchanged, including one this file never imports.

    Property: `salvo(priority, assignable, available, salvo_size=1)` is identical,
    element for element, to `greedy(priority, assignable, available)` -- the loop
    body runs once with `remaining_available` equal to `available`, which is exactly
    the single-shot call.

    Shoot-look-shoot -- rebinding a fresh round to a threat whose earlier round
    already missed, instead of committing a whole salvo up front -- is a deliberate
    extension point this function does not implement. It would replace the
    unconditional k-fold repeat here with a per-decision draw that reads which rounds
    already resolved, and belongs in this file when it lands, not in the assignment
    solver.
    """
    binding = torch.full_like(available, UNASSIGNED, dtype=torch.int64)
    remaining_available = available
    for _ in range(salvo_size):
        round_binding = assign_fn(priority, assignable, remaining_available)
        bound_this_round = round_binding >= 0
        binding = torch.where(bound_this_round, round_binding, binding)
        remaining_available = remaining_available & ~bound_this_round
    return binding
