"""Euclidean counterparts of the operations in `lorentz.py`.

Entailment cones are the Euclidean cones of Ganea et al. (2018): the cone at
`x` has half-aperture `asin(K / ||x||)`, with `K` a fixed constant standing in
for the `2 * min_radius / sqrt(curv)` of `lorentz.py`. Curvature does not
exist in this geometry and is not simulated.
"""

from __future__ import annotations

import torch
from torch import Tensor

# acos/asin have infinite derivatives at +-1, and 1 - 1e-8 is not representable in
# float32, so the clamp bound cannot be tied to the much smaller `eps` below.
_TRIG_BOUND = 1.0 - 1e-6


def exp_map0(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Identity: Euclidean space is its own tangent space."""
    return x


def exp_map0_batch(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Identity, for inputs of shape `(B, N, D)`."""
    return x


def pairwise_dist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """`(B1, D)`, `(B2, D)` -> `(B1, B2)`."""
    sq_dist = (
        torch.sum(x**2, dim=-1, keepdim=True)
        + torch.sum(y**2, dim=-1, keepdim=True).transpose(-1, -2)
        - 2 * x @ y.transpose(-1, -2)
    )
    # The sqrt gradient is undefined at coincident points.
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def pairwise_dist_batch(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """`(B1, N, D)`, `(B2, N, D)` -> `(N, B1, B2)`."""
    return pairwise_dist(x.transpose(0, 1), y.transpose(0, 1), eps=eps)


def get_space_norm(x: Tensor) -> Tensor:
    """`(..., D)` -> `(..., 1)`."""
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def get_distance_from_origin(x: Tensor) -> Tensor:
    """`(B, N, D)` -> `(B, N, 1)`."""
    return get_space_norm(x)


def half_aperture(x: Tensor, k: float, eps: float = 1e-8) -> Tensor:
    """Half aperture of the entailment cone at `x`, in `(0, pi/2)`."""
    asin_input = k / (torch.norm(x, dim=-1) + eps)
    return torch.asin(torch.clamp(asin_input, min=-_TRIG_BOUND, max=_TRIG_BOUND))


def half_aperture_batch(x: Tensor, k: float, eps: float = 1e-8) -> Tensor:
    """`(B, N, D)` -> `(N, B)`."""
    x_norms = torch.norm(x.transpose(0, 1), dim=-1)
    asin_input = k / (x_norms + eps)

    return torch.asin(torch.clamp(asin_input, min=-_TRIG_BOUND, max=_TRIG_BOUND))


def oxy_angle(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Exterior angle at `x` in the triangle `Oxy`, in `(0, pi)`."""
    x_norm = torch.norm(x, dim=-1)
    y_norm = torch.norm(y, dim=-1)
    xy_dist = torch.norm(y - x, dim=-1)

    acos_numer = y_norm**2 - x_norm**2 - xy_dist**2
    # `lorentz.py` floors sinh^2 of the separation, so its guard on the separation
    # itself is sqrt(eps); against an unsquared `xy_dist` the same constant floors
    # four orders lower and the gradient diverges as the two points coincide.
    acos_denom = 2 * x_norm * torch.clamp(xy_dist, min=eps**0.5) + eps

    return torch.acos(torch.clamp(acos_numer / acos_denom, min=-_TRIG_BOUND, max=_TRIG_BOUND))


def oxy_angle_batch(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """`(B, N, D)`, `(B, N, D)` -> `(N, B)`."""
    x_t = x.transpose(0, 1)
    y_t = y.transpose(0, 1)

    x_norm = torch.norm(x_t, dim=-1)
    y_norm = torch.norm(y_t, dim=-1)
    xy_dist = torch.norm(y_t - x_t, dim=-1)

    acos_numer = y_norm**2 - x_norm**2 - xy_dist**2
    acos_denom = 2 * x_norm * torch.clamp(xy_dist, min=eps**0.5) + eps

    return torch.acos(torch.clamp(acos_numer / acos_denom, min=-_TRIG_BOUND, max=_TRIG_BOUND))
