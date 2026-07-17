"""Magazine state and the binding between interceptor slots and threats.

Commitment lives in `interceptor_target` on EnvState, not in a structure of its own.
Two reasons. A commitment is per-slot and must survive every step alongside the slot's
position and velocity, so keeping it in the same batched object means one thing moves
between devices, one thing is sliced when an environment is extracted, and one thing
can fall out of sync -- none. And guidance already gathers per-slot track indices; a
separate container would have to be threaded through the same call sites to say the
same thing.

Stock is `interceptor_enabled & ~interceptor_committed` rather than a counter. A
counter is a second source of truth about the same fact and can disagree with the
slots it is supposed to describe; a mask cannot.
"""

from __future__ import annotations

import torch
from torch import Tensor

UNASSIGNED = -1


def available(interceptor_enabled: Tensor, interceptor_committed: Tensor) -> Tensor:
    """Slots holding an unfired round, [N, I] bool."""
    return interceptor_enabled & ~interceptor_committed


def commit(interceptor_target: Tensor, assignment: Tensor) -> Tensor:
    """Write a fresh assignment over the current bindings, [N, I] int64.

    A slot with no proposal keeps what it had. A slot WITH a proposal takes it,
    whatever it was bound to before, so this function does not by itself protect an
    in-flight round from being re-aimed. The protection is upstream: `greedy` only
    ever proposes for slots reported by `available`, and a committed slot is not
    available. Commitment therefore holds until the round kills, misses, or its
    target resolves -- as a property of the pipeline, not of this call.

    Guarding here as well would be redundant today and would hide a caller that
    started proposing for committed slots, which is a bug worth surfacing.
    """
    return torch.where(assignment >= 0, assignment, interceptor_target)


def release(interceptor_target: Tensor, released: Tensor) -> Tensor:
    """Clear bindings for slots whose engagement has ended, [N, I] int64."""
    return torch.where(
        released, torch.full_like(interceptor_target, UNASSIGNED), interceptor_target
    )


def engaged_threats(
    interceptor_target: Tensor, n_threats: int, salvo_size: int = 1
) -> Tensor:
    """Threats already holding a full salvo of bound shooters, [N, T] bool.

    Scatters into a buffer one column wider than needed and drops the extra column,
    so unbound slots have somewhere harmless to write. Pointing them at index 0
    instead would let an unbound slot overwrite a genuine binding on threat 0.

    Counts bound slots per threat rather than testing for any, via scatter_add_
    instead of a boolean scatter_, so a threat stays assignable until it holds
    `salvo_size` rounds rather than dropping out after the first. `salvo_size=1`
    makes "count >= 1" the same test as "any bound", so this is an exact
    generalisation of the Phase 1 behaviour, not a change to it.
    """
    n_envs = interceptor_target.shape[0]
    sentinel = torch.full_like(interceptor_target, n_threats)
    index = torch.where(interceptor_target >= 0, interceptor_target, sentinel)

    counts = torch.zeros(
        (n_envs, n_threats + 1), dtype=torch.int64, device=interceptor_target.device
    )
    counts.scatter_add_(1, index, torch.ones_like(index, dtype=torch.int64))
    return counts[:, :n_threats] >= salvo_size
