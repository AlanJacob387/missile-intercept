"""Batched engine step: one fixed-dt tick of threat physics across all environments.

`step` is the torch.compile target; it stays free of host-side control flow over
environments so the whole step compiles to device work. Compilation is applied later
-- eager first, once parity with the oracle holds.

Sensing, guidance and interceptor motion attach to this same step function as they
land.
"""

from __future__ import annotations

from dataclasses import replace

from mdsim.core.dynamics import PhysicsParams
from mdsim.core.integrator import integrate
from mdsim.core.state import EnvState
from mdsim.core.threat_models import get_threat_model


def step(state: EnvState, params: PhysicsParams) -> EnvState:
    """Advance every environment by one fixed dt.

    Pure: returns a new EnvState and never writes into the input's tensors. Both the
    parity tests and torch.compile depend on that -- an in-place update would alias
    a recorded trajectory into a single mutating buffer.
    """
    accel_fn = get_threat_model(params.threat_model)
    accel = accel_fn(state.threat_pos, state.threat_vel, params)

    threat_pos, threat_vel = integrate(
        state.threat_pos, state.threat_vel, accel, params.dt, params.integrator
    )

    return replace(
        state,
        threat_pos=threat_pos,
        threat_vel=threat_vel,
        t=state.t + params.dt,
    )
