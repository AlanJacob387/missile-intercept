"""Greedy and Hungarian weapon-target assignment, batched across environments.

Greedy is the Phase 1 baseline and the thing the saturation curve is measured
against. `hungarian` is the Phase 2 optimal alternative: it does not replace greedy's
signature, because it needs a strictly richer input to have anything to be optimal
over (see its docstring).
"""

from __future__ import annotations

import torch
from torch import Tensor

UNASSIGNED = -1

# Cost assigned to an illegal pairing in the Hungarian solve: a real threat that
# fails `assignable`, or a decline column not owned by the row's slot. Bounded well
# above any achievable value, so the minimisation never picks one while a legal
# option (worst case, the slot's own free decline column) remains.
_LARGE_COST = 1e12

# Deterministic tie-break, added to every entry before solving. Two or more slots at
# the same battery see identical value for a given threat, so a genuine multi-battery
# scenario has real, frequent ties -- multiple assignments achieving the identical
# optimal total. Kuhn-Munkres and scipy's solver (used independently in the oracle)
# are not guaranteed to resolve a tie the same way, and the batch-equivalence gate
# compares the discrete assignment exactly, so an unresolved tie is a spurious parity
# failure between two equally-correct answers, not a bug in either.
#
# The perturbation must be non-separable in (slot, threat): an additive form
# f(slot) + g(threat) is invariant under swapping which of two tied slots takes which
# of two tied threats -- sum f(s1)+g(t1)+f(s2)+g(t2) equals f(s1)+g(t2)+f(s2)+g(t1)
# for any f, g -- so it fails to break EXACTLY the class of tie this matters for.
# A product term does not have that symmetry: (s1+1)(t1+2)+(s2+1)(t2+2) differs from
# (s1+1)(t2+2)+(s2+1)(t1+2) by (t1-t2)(s1-s2), zero only when s1==s2 or t1==t2.
# _TIE_EPS is small enough relative to any real value difference this engine produces
# that it can only ever decide between costs that were already equal.
_TIE_EPS = 1.0e-9


def _tie_break(slot: int, threat_or_decline: int) -> float:
    return _TIE_EPS * (slot + 1) * (threat_or_decline + 1)


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


def _solve_min_cost(cost: list[list[float]], n: int, m: int) -> list[int]:
    """Kuhn-Munkres for an n x m cost matrix, n <= m, 0-indexed.
    Returns, for each row i in [0, n), its assigned column in [0, m)."""
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = [0] * (n + 1)
    for j in range(1, m + 1):
        if p[j] != 0:
            result[p[j]] = j
    return [result[i] - 1 for i in range(1, n + 1)]


def hungarian(value: Tensor, assignable: Tensor, available: Tensor) -> Tensor:
    """Optimal weapon-target assignment via the Hungarian algorithm, per env.

        value      [N, I, T] float, per-(slot, threat) value, higher is better
        assignable [N, T] bool, threat is a legal target for a new round
        available  [N, I] bool, slot holds a round and is not already committed

    Returns [N, I] int64, -1 for a slot that declines every option.

    Unlike `greedy`, this needs a per-(slot, threat) value rather than a per-threat
    priority: every slot is interchangeable in the current single-battery engine, so
    today `value` is typically the same [N, T] priority broadcast to every slot, and
    the two functions tie exactly -- that is an expected, honest result of there
    being no slot-threat coupling to exploit yet, not a bug. The richer signature
    exists because Hungarian's entire value proposition is a genuine value matrix
    with slot-dependent variation, and a solver that could not accept one would be
    unable to ever prove its own optimality over greedy.

    Solved one environment at a time in a plain Python loop, which is an explicit
    exception to the no-host-loop rule the rest of this codebase follows. It is safe
    here specifically because assignment runs on a fixed decision cadence -- once
    per several dozen physics steps, not once per step -- so the loop's wall-clock
    cost is bounded and paid rarely. The per-step physics, sensing, and guidance code
    has no such cadence to hide behind and must stay batched.

    Per env, only rows for available slots are built at all; an unavailable slot's
    result is -1 directly. The cost matrix is [n_available, n_threats + n_available]:
    columns 0..n_threats-1 are real threats (cost -value if assignable, else a large
    penalty so the solver never picks one over a free decline); the remaining
    n_available columns are one dedicated decline option per available slot, free
    (cost 0) only in its own row and heavily penalised in every other row, so "assign
    nothing" is always available to a slot without one slot's decline being taken by
    another. Solving this rectangular minimisation is equivalent to maximising total
    value subject to a legal one-to-one matching. Because every row can always reach
    cost 0 via its own decline column, the optimum never uses a penalised entry, so
    masking is respected exactly: no non-assignable threat and no unavailable slot is
    ever bound, and no threat is bound twice.

    scipy is not used here: it is reserved for the independent oracle, and the point
    of this function is a hand-rolled solver that scipy can be checked against.
    """
    n_envs, n_slots, n_threats = value.shape
    device = value.device

    result = torch.full((n_envs, n_slots), UNASSIGNED, dtype=torch.int64)

    value_cpu = value.detach().to("cpu")
    assignable_cpu = assignable.detach().to("cpu")
    available_cpu = available.detach().to("cpu")

    for env in range(n_envs):
        slots = torch.nonzero(available_cpu[env], as_tuple=True)[0].tolist()
        n_avail = len(slots)
        if n_avail == 0:
            continue

        legal = assignable_cpu[env].tolist()
        cost = [[0.0] * (n_threats + n_avail) for _ in range(n_avail)]
        for row, slot in enumerate(slots):
            for threat in range(n_threats):
                base = -float(value_cpu[env, slot, threat]) if legal[threat] else _LARGE_COST
                cost[row][threat] = base + _tie_break(slot, threat)
            for k in range(n_avail):
                base = 0.0 if k == row else _LARGE_COST
                cost[row][n_threats + k] = base + _tie_break(slot, n_threats)

        assigned = _solve_min_cost(cost, n_avail, n_threats + n_avail)
        for row, slot in enumerate(slots):
            col = assigned[row]
            if col < n_threats:
                result[env, slot] = col

    return result.to(device=device, dtype=torch.int64)
