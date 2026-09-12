"""Phase 0 physics validation of the nvalchemiops PME engine.

These tests pin down the two things the whole long-range design rests on:
the PME energy is correct in absolute terms, and it is twice differentiable
(needed to train against force labels through an autograd-derived force).

They also guard the `spline_order` hazard: at order >= 7 nvalchemiops silently
drops the entire reciprocal-space sum and returns only the self-energy, with
no error raised.
"""

import itertools
import math

import pytest
import torch
from nvalchemiops.torch.interactions.electrostatics.pme import (
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.neighbors import neighbor_list

MADELUNG_NACL = 1.7475645946331821
SPLINE_ORDER = 6  # validated; do NOT raise above 6 (see module docstring)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PME validation runs on GPU"
)


def nacl(reps: int, a: float = 1.0, device: str = "cuda"):
    """Rocksalt lattice: `reps`^3 conventional cells, alternating unit charges."""
    dt = torch.float64
    pos, q = [], []
    for s in itertools.product(range(reps), repeat=3):
        for i, j, k in itertools.product(range(2), repeat=3):
            pos.append([(s[0] + i / 2) * a, (s[1] + j / 2) * a, (s[2] + k / 2) * a])
            q.append(1.0 if (i + j + k) % 2 == 0 else -1.0)
    pos = torch.tensor(pos, dtype=dt, device=device)
    q = torch.tensor(q, dtype=dt, device=device)
    cell = torch.eye(3, dtype=dt, device=device) * (a * reps)
    return pos, q, cell


def madelung_from_energy(energy: float, n_atoms: int, a: float = 1.0) -> float:
    """Madelung constant from the total electrostatic energy of the lattice."""
    return -energy * 2.0 / n_atoms * (a / 2)


def test_pme_reproduces_madelung_constant():
    """Absolute PME energy is right, not merely self-consistent."""
    pos, q, cell = nacl(reps=3)
    a, reps = 1.0, 3
    box = a * reps
    pbc = torch.tensor([True] * 3, device=pos.device)
    nl, ptr, shifts = neighbor_list(
        pos, 0.49 * box, cell=cell, pbc=pbc,
        return_neighbor_list=True, max_neighbors=4096,
    )
    energy = particle_mesh_ewald(
        pos, q, cell,
        neighbor_list=nl, neighbor_ptr=ptr, neighbor_shifts=shifts,
        energy_reduction="system", alpha=2.5,
        mesh_dimensions=(96, 96, 96), spline_order=SPLINE_ORDER,
    )
    m = madelung_from_energy(float(energy), pos.shape[0], a)
    assert abs(m - MADELUNG_NACL) / MADELUNG_NACL < 1e-5


@pytest.mark.parametrize("order", [3, 4, 5, 6])
def test_reciprocal_sum_is_present_for_supported_spline_orders(order):
    """Guard the silent-failure mode: recip must not collapse to -self_energy."""
    pos, q, cell = nacl(reps=2)
    alpha = 2.0
    self_energy = alpha / math.sqrt(math.pi) * float((q * q).sum())
    got = float(
        pme_reciprocal_space(
            pos, q, cell, alpha=alpha, mesh_dimensions=(64, 64, 64),
            spline_order=order, energy_reduction="system",
        )
    )
    # If the k-space sum were dropped, `got` would equal -self_energy exactly.
    assert abs(got + self_energy) > 1e-3, (
        f"reciprocal sum vanished at spline_order={order}: got {got}, "
        f"-self_energy is {-self_energy}"
    )


def test_energy_is_twice_differentiable():
    """Force training backprops through an autograd force: needs double backward."""
    dt, dev = torch.float64, "cuda"
    torch.manual_seed(0)
    box, n = 8.0, 12
    pos = torch.rand((n, 3), dtype=dt, device=dev) * box * 0.8 + box * 0.1
    q = torch.randn(n, dtype=dt, device=dev)
    q[-1] = -q[:-1].sum()  # neutral
    cell = torch.eye(3, dtype=dt, device=dev) * box
    pbc = torch.tensor([True] * 3, device=dev)
    nl, ptr, shifts = neighbor_list(
        pos, 3.5, cell=cell, pbc=pbc, return_neighbor_list=True, max_neighbors=512
    )

    def energy(p, c):
        return particle_mesh_ewald(
            p, c, cell,
            neighbor_list=nl, neighbor_ptr=ptr, neighbor_shifts=shifts,
            energy_reduction="system", alpha=1.0,
            mesh_dimensions=(32, 32, 32), spline_order=SPLINE_ORDER,
        )

    p = pos.clone().requires_grad_(True)
    c = q.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: energy(x, q), (p,), eps=1e-6, atol=1e-6)
    assert torch.autograd.gradcheck(lambda x: energy(pos, x), (c,), eps=1e-6, atol=1e-6)

    # force -> loss -> gradient w.r.t. charges, the actual training path
    p = pos.clone().requires_grad_(True)
    c = q.clone().requires_grad_(True)
    force = -torch.autograd.grad(energy(p, c), p, create_graph=True)[0]
    grad_q = torch.autograd.grad((force**2).sum(), c)[0]
    assert torch.isfinite(grad_q).all()
    assert float(grad_q.norm()) > 0.0
