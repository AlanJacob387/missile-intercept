"""Interacting Multiple Model filter bank: constant-velocity + constant-acceleration.

A single constant-velocity (CV) filter (kalman.py) treats every deviation from
straight-line flight as process noise. That is fine for a ballistic threat but
underestimates the covariance during an active maneuver and overestimates it in
between, biasing the gain the wrong way in both regimes. IMM instead runs two
filters in parallel -- CV and constant-acceleration (CA) -- and lets a Markov
mode probability decide, step by step, how much each one is trusted.

Both models share one 9-dimensional augmented state per track,
x = [px, py, pz, vx, vy, vz, ax, ay, az], so they can be mixed in a common space
even though only the CA model drives position/velocity off the acceleration
component. The CV model carries that component as dead weight: its transition
matrix never reads it, and its process noise pins it down with a tiny fixed
variance instead of letting it drift. This is the standard trick for building an
IMM bank out of models with different order.

Known acceleration (gravity) is a control input, exactly as in kalman.py: it is
added to the (position, velocity) block after F @ x, never touches the state's
acceleration component, and does not appear in either F. The state's a-block
represents unmodelled/maneuver acceleration, which is a different thing from a
deterministic force the defender already knows about.

The combined output exported as TrackState.x_est / .P is position + velocity
only (6-dim), matching the single-KF contract exactly. The acceleration
component and the per-model (mu, x_models, P_models) bookkeeping exist solely
for the filter's own continuity from one step to the next; nothing downstream of
the tracker reads them.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from mdsim.sensing import kalman
from mdsim.sensing.tracks import TrackState

STATE_DIM = 9
N_MODELS = 2

# Row i = "from model i" transition probability to model j. CV (index 0) is
# stickier than CA (index 1): threats spend most of flight time unmaneuvered, so
# a track is more likely to stay in CV than to stay in an active maneuver.
DEFAULT_TRANSITION_MATRIX = ((0.95, 0.05), (0.10, 0.90))

DEFAULT_MU0 = (0.9, 0.1)

# Matches kalman.py's INITIAL_VELOCITY_VARIANCE convention: wide enough that a
# single position fix does not bias the velocity estimate.
INITIAL_VELOCITY_VARIANCE = 1.0e6

# (100 m/s^2)^2 -- roughly 10g standard deviation. Wide enough to cover any
# physically plausible threat maneuver without the CA model's prior biasing the
# first few updates toward a particular acceleration.
INITIAL_ACCEL_VARIANCE = 1.0e4

# Fixed variance pinning the CV model's inert acceleration component so its
# (a, a) covariance block never goes singular under the Joseph-form recursion,
# without letting that unused state drift or feed back into (p, v).
CV_ACCEL_EPSILON = 1.0e-8


def _eye3(dtype: torch.dtype, device: torch.device | str) -> Tensor:
    return torch.eye(3, dtype=dtype, device=device)


def transition_cv(dt: float, dtype: torch.dtype, device: torch.device | str) -> Tensor:
    """CV transition on the augmented 9-state: position advances by velocity, [9, 9].

    The acceleration block is untouched by p/v propagation and persists (F's
    identity block on rows/cols 6:9) so it can still be mixed with the CA model.
    """
    eye3 = _eye3(dtype, device)
    F = torch.eye(9, dtype=dtype, device=device)
    F[0:3, 3:6] = eye3 * dt
    return F


def transition_ca(dt: float, dtype: torch.dtype, device: torch.device | str) -> Tensor:
    """CA transition on the augmented 9-state: standard constant-acceleration kinematics, [9, 9]."""
    eye3 = _eye3(dtype, device)
    F = torch.eye(9, dtype=dtype, device=device)
    F[0:3, 3:6] = eye3 * dt
    F[0:3, 6:9] = eye3 * (0.5 * dt**2)
    F[3:6, 6:9] = eye3 * dt
    return F


def process_noise_cv(
    dt: float, q_cv: float, dtype: torch.dtype, device: torch.device | str
) -> Tensor:
    """CV process noise on the augmented 9-state, [9, 9].

    The (p, v) block is exactly kalman.process_noise's existing 6-state formula,
    scaled by q_cv. The (a, a) block gets CV_ACCEL_EPSILON on the diagonal only
    -- zero cross terms with (p, v) -- so the inert acceleration state has a
    non-singular covariance without drifting or being fed by p/v.
    """
    Q = torch.zeros((9, 9), dtype=dtype, device=device)
    Q[0:6, 0:6] = kalman.process_noise(dt, q_cv, dtype, device)
    Q[6:9, 6:9] = _eye3(dtype, device) * CV_ACCEL_EPSILON
    return Q


def process_noise_ca(
    dt: float, q_ca: float, dtype: torch.dtype, device: torch.device | str
) -> Tensor:
    """CA process noise: discretized white-jerk ("nearly constant acceleration") model, [9, 9].

    Assembled per-axis with 3x3 identity blocks, the same way kalman.py's Q is
    assembled, scaled by q_ca (a jerk spectral density).
    """
    eye3 = _eye3(dtype, device)
    Q = torch.zeros((9, 9), dtype=dtype, device=device)
    Q[0:3, 0:3] = eye3 * (dt**5 / 20.0)
    Q[0:3, 3:6] = eye3 * (dt**4 / 8.0)
    Q[3:6, 0:3] = eye3 * (dt**4 / 8.0)
    Q[0:3, 6:9] = eye3 * (dt**3 / 6.0)
    Q[6:9, 0:3] = eye3 * (dt**3 / 6.0)
    Q[3:6, 3:6] = eye3 * (dt**3 / 3.0)
    Q[3:6, 6:9] = eye3 * (dt**2 / 2.0)
    Q[6:9, 3:6] = eye3 * (dt**2 / 2.0)
    Q[6:9, 6:9] = eye3 * dt
    return Q * q_ca


def measurement_matrix(dtype: torch.dtype, device: torch.device | str) -> Tensor:
    """Position-only observation on the augmented 9-state, [3, 9].

    kalman.py's existing 3x6 H padded with a zero 3x3 block over the
    acceleration columns.
    """
    H6 = kalman.measurement_matrix(dtype, device)
    zero = torch.zeros((3, 3), dtype=dtype, device=device)
    return torch.cat((H6, zero), dim=-1)


def _predict(
    x: Tensor, P: Tensor, dt: float, F: Tensor, Q: Tensor, known_accel: Tensor | None
) -> tuple[Tensor, Tensor]:
    """Joseph-recursion predict, generalized to whatever state dim F/Q carry.

    known_accel enters the same way kalman.predict's control input does: added
    to (position, velocity) after F @ x, never to any other state component.
    """
    x_pred = x @ F.transpose(-1, -2)
    if known_accel is not None:
        delta_v = known_accel * dt
        delta = torch.zeros_like(x_pred)
        delta[..., 0:3] = delta_v * dt
        delta[..., 3:6] = delta_v
        x_pred = x_pred + delta
    P_pred = F @ P @ F.transpose(-1, -2) + Q
    return x_pred, P_pred


def _update(
    x: Tensor, P: Tensor, H: Tensor, z: Tensor, R: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Joseph-form update, generalized to whatever state dim H carries.

    Returns (x_new, P_new, innovation, S) -- innovation and S are needed by the
    model-likelihood step and would otherwise be recomputed.
    """
    Ht = H.transpose(-1, -2)
    innovation = z - x @ Ht
    PHt = P @ Ht
    S = H @ PHt + R

    K = torch.linalg.solve(S, PHt.transpose(-1, -2)).transpose(-1, -2)
    x_new = x + (K @ innovation.unsqueeze(-1)).squeeze(-1)

    dim = x.shape[-1]
    eye = torch.eye(dim, dtype=x.dtype, device=x.device)
    A = eye - K @ H
    P_new = A @ P @ A.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    return x_new, P_new, innovation, S


def _log_likelihood(innovation: Tensor, S: Tensor) -> Tensor:
    """Log density of a 3-dim Gaussian innovation, [...].

    logL = -0.5 innovation^T S^-1 innovation - 0.5 logdet(S) - 1.5 log(2*pi).
    Solve/slogdet, never an explicit inverse or determinant.
    """
    sol = torch.linalg.solve(S, innovation.unsqueeze(-1))
    quad = (innovation.unsqueeze(-2) @ sol).squeeze(-1).squeeze(-1)
    _, logdet = torch.linalg.slogdet(S)
    return -0.5 * quad - 0.5 * logdet - 1.5 * math.log(2.0 * math.pi)


def _resolve_transition_matrix(
    transition_matrix: Tensor | None, dtype: torch.dtype, device: torch.device | str
) -> Tensor:
    if transition_matrix is not None:
        return transition_matrix.to(dtype=dtype, device=device)
    return torch.tensor(DEFAULT_TRANSITION_MATRIX, dtype=dtype, device=device)


def _mix(
    mu_prev: Tensor, x_prev: Tensor, P_prev: Tensor, Pi: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """IMM mixing step. mu_prev [..., 2], x_prev [..., 2, 9], P_prev [..., 2, 9, 9].

    Returns (c, w, x0, P0): predicted mode probabilities [..., 2], mixing
    weights [..., i, j], and the mixed initial condition per model [..., 2, 9]
    / [..., 2, 9, 9].
    """
    c = torch.einsum("ij,...i->...j", Pi, mu_prev)
    w = torch.einsum("ij,...i->...ij", Pi, mu_prev) / c.unsqueeze(-2)

    x0 = torch.einsum("...ij,...ik->...jk", w, x_prev)

    diff = x_prev.unsqueeze(-2) - x0.unsqueeze(-3)  # [..., i, j, 9]
    outer = torch.einsum("...ija,...ijb->...ijab", diff, diff)
    term = P_prev.unsqueeze(-3) + outer  # [..., i, j, 9, 9]
    P0 = torch.einsum("...ij,...ijab->...jab", w, term)

    return c, w, x0, P0


def _combine(x_models: Tensor, P_models: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
    """Mode-probability-weighted combination. x_models [..., 2, 9], mu [..., 2]."""
    x_comb = torch.einsum("...j,...ja->...a", mu, x_models)
    diff = x_models - x_comb.unsqueeze(-2)
    outer = torch.einsum("...ja,...jb->...jab", diff, diff)
    term = P_models + outer
    P_comb = torch.einsum("...j,...jab->...ab", mu, term)
    return x_comb, P_comb


def imm_initialize(
    measured_cart: Tensor,
    R: Tensor,
    initial_velocity_variance: float = INITIAL_VELOCITY_VARIANCE,
    initial_accel_variance: float = INITIAL_ACCEL_VARIANCE,
    mu0: tuple[float, float] = DEFAULT_MU0,
) -> TrackState:
    """First detection: both models start from the same measurement.

    Position = measurement, velocity = 0, acceleration = 0, all wide on the
    components a single fix says nothing about.
    """
    dtype = measured_cart.dtype
    device = measured_cart.device
    lead_shape = measured_cart.shape[:-1]

    zeros3 = torch.zeros_like(measured_cart)
    x9 = torch.cat((measured_cart, zeros3, zeros3), dim=-1)

    P9 = torch.zeros((*lead_shape, 9, 9), dtype=dtype, device=device)
    P9[..., 0:3, 0:3] = R
    eye3 = _eye3(dtype, device)
    P9[..., 3:6, 3:6] = eye3 * initial_velocity_variance
    P9[..., 6:9, 6:9] = eye3 * initial_accel_variance

    x_models = torch.stack((x9, x9), dim=-2)
    P_models = torch.stack((P9, P9), dim=-3)

    mu = torch.tensor(mu0, dtype=dtype, device=device).expand(*lead_shape, N_MODELS).clone()

    detected = torch.ones(lead_shape, dtype=torch.bool, device=device)
    age = torch.zeros(lead_shape, dtype=dtype, device=device)
    updated = torch.ones(lead_shape, dtype=torch.bool, device=device)

    return TrackState(
        x_est=x9[..., :6],
        P=P9[..., :6, :6],
        detected=detected,
        age=age,
        updated=updated,
        mu=mu,
        x_models=x_models,
        P_models=P_models,
    )


def imm_step(
    tracks: TrackState,
    measured_cart: Tensor,
    R: Tensor,
    detected: Tensor,
    dt: float,
    q_cv: float,
    q_ca: float,
    known_accel: Tensor,
    transition_matrix: Tensor | None = None,
) -> TrackState:
    """One IMM predict/(update|coast) tick for every held track.

    detected is this step's radar detection (already truth-gated by the
    caller), independent of tracks.detected (which means "this track has been
    initiated and is held" and is simply carried forward here). Where detected
    is False the models coast: x_upd = x_pred, P_upd = P_pred, and the mode
    probabilities are the Markov-predicted c with no likelihood correction.
    """
    x_prev = tracks.x_models
    P_prev = tracks.P_models
    mu_prev = tracks.mu
    dtype = x_prev.dtype
    device = x_prev.device

    Pi = _resolve_transition_matrix(transition_matrix, dtype, device)
    c, _w, x0, P0 = _mix(mu_prev, x_prev, P_prev, Pi)

    F_cv = transition_cv(dt, dtype, device)
    F_ca = transition_ca(dt, dtype, device)
    Q_cv = process_noise_cv(dt, q_cv, dtype, device)
    Q_ca = process_noise_ca(dt, q_ca, dtype, device)

    x_pred_cv, P_pred_cv = _predict(x0[..., 0, :], P0[..., 0, :, :], dt, F_cv, Q_cv, known_accel)
    x_pred_ca, P_pred_ca = _predict(x0[..., 1, :], P0[..., 1, :, :], dt, F_ca, Q_ca, known_accel)

    H = measurement_matrix(dtype, device)
    x_upd_cv, P_upd_cv, innov_cv, S_cv = _update(x_pred_cv, P_pred_cv, H, measured_cart, R)
    x_upd_ca, P_upd_ca, innov_ca, S_ca = _update(x_pred_ca, P_pred_ca, H, measured_cart, R)

    logL_cv = _log_likelihood(innov_cv, S_cv)
    logL_ca = _log_likelihood(innov_ca, S_ca)
    logL = torch.stack((logL_cv, logL_ca), dim=-1)  # [..., 2]

    max_logL = logL.max(dim=-1, keepdim=True).values
    weights = c * torch.exp(logL - max_logL)
    mu_from_update = weights / weights.sum(dim=-1, keepdim=True)

    x_pred = torch.stack((x_pred_cv, x_pred_ca), dim=-2)
    P_pred = torch.stack((P_pred_cv, P_pred_ca), dim=-3)
    x_upd = torch.stack((x_upd_cv, x_upd_ca), dim=-2)
    P_upd = torch.stack((P_upd_cv, P_upd_ca), dim=-3)

    det_mu = detected.unsqueeze(-1)
    mu_new = torch.where(det_mu, mu_from_update, c)

    det_x = detected[..., None, None]
    x_final = torch.where(det_x, x_upd, x_pred)
    det_P = detected[..., None, None, None]
    P_final = torch.where(det_P, P_upd, P_pred)

    x_comb, P_comb = _combine(x_final, P_final, mu_new)

    return TrackState(
        x_est=x_comb[..., :6],
        P=P_comb[..., :6, :6],
        detected=tracks.detected,
        age=tracks.age + dt,
        updated=detected,
        mu=mu_new,
        x_models=x_final,
        P_models=P_final,
    )
