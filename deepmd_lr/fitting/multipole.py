# SPDX-License-Identifier: MIT
r"""Atomic dipoles and quadrupoles from the descriptor's equivariant features.

A dipole is a vector and a quadrupole a traceless symmetric tensor of the
atom's surroundings, so they cannot come from the invariant features the
energy and the charge are fitted on. The descriptor already carries what is
needed: per atom, a set of channels that rotate as vectors (:math:`l = 1`)
and as traceless tensors (:math:`l = 2`). The head mixes them,

.. math::
    \mu_i = \frac1{C_1} \sum_c a_c(\mathcal D_i)\, X^{(1)}_{i,c},
    \qquad
    Q_i = \frac1{C_2} \sum_c b_c(\mathcal D_i)\, X^{(2)}_{i,c},

with coefficients :math:`a`, :math:`b` fitted on the invariant features
:math:`\mathcal D_i`. An invariant number times an equivariant tensor is
equivariant, so the moments rotate with the frame by construction and nothing
about rotations has to be learned.

The network is DeePMD-kit's invariant fitting net with one output per
channel, so the per-type nets, the precision, the seed and the serialization
are its own. What this module adds is the map from a descriptor's packed
harmonic coefficients to Cartesian components
(:func:`cartesian_features`) and the contraction (:func:`contract`).

Two descriptors hand out such features. DPA4 keeps a node state with every
degree at full width (:math:`C_1 = C_2`, its channel count) through its
message passing. DPA4C reduces its edges once into degree-wise moments, a
few channels per degree (8 vectors and 4 tensors at 32 channels), before its
invariant readout; only its uncompressed evaluation builds them in Python,
the compressed kernel keeps them on the device.
"""

import functools
from collections.abc import (
    Callable,
)
from typing import (
    Any,
    NamedTuple,
)

import torch
from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.descriptor.dpa4_nn.cartesian import (
    build_cartesian_basis,
)
from deepmd.dpmodel.descriptor.dpa4c_nn.geometry import (
    packed_l2_to_stf,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)

#: The name of the head's output, and of its fitting type.
MULTIPOLE_VAR_NAME = "multipole"

#: The highest moment the head can give: 1 a dipole, 2 a quadrupole too.
MAX_MOMENT = 2


def _dpa4_channels(descriptor: Any, lmax: int) -> list[int]:
    last = int(descriptor.node_l_schedule[-1])
    if last < lmax:
        raise ValueError(
            f"multipole_net.lmax is {lmax}, but this DPA4 ends with features "
            f"of degree {last} (the last entry of its l schedule); raise the "
            "schedule or lower lmax"
        )
    return [int(descriptor.channels)] * lmax


def _dpa4c_channels(descriptor: Any, lmax: int) -> list[int]:
    # DPA4C has no form below degree 2, the highest moment a head takes.
    return [int(width) for width in descriptor.degree_channels[1 : lmax + 1]]


@functools.cache
def _dpa4_l2_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """DeePMD-kit's Cartesian basis of DPA4's five ``l = 2`` coefficients,
    ``(5, 3, 3)``."""
    basis = build_cartesian_basis(2, dtype="float64")[4:9]
    return torch.as_tensor(basis, dtype=dtype, device=device)


def _dpa4_cartesian(
    descriptor: Any, state: torch.Tensor, lmax: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del descriptor
    packed = state[:, :, 0, :]
    vectors = torch.stack([packed[:, 3], -packed[:, 1], -packed[:, 2]], dim=1)
    if lmax < 2:
        return vectors, None
    basis = _dpa4_l2_basis(state.dtype, state.device)
    return vectors, torch.einsum("ndc,dab->nabc", packed[:, 4:9], basis)


def _dpa4c_cartesian(
    descriptor: Any, state: torch.Tensor, lmax: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    offsets = descriptor.degree_offsets
    n = state.shape[0]
    vectors = state[:, offsets[1] : offsets[2]].reshape(n, 3, -1)
    if lmax < 2:
        return vectors, None
    packed = state[:, offsets[2] : offsets[3]].reshape(n, 5, -1)
    tensors = packed_l2_to_stf(packed.transpose(1, 2)).permute(0, 2, 3, 1)
    return vectors, tensors


class _Source(NamedTuple):
    """How one descriptor class hands out its equivariant features."""

    #: The method whose return value carries them, and where in that value.
    method: str
    position: int
    #: ``(descriptor, lmax)`` to the channel count of each degree.
    channels: Callable[[Any, int], list[int]]
    #: ``(descriptor, state, lmax)`` to Cartesian vectors and tensors.
    cartesian: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


_SOURCES = {
    "DescrptDPA4": _Source("_run_graph", 1, _dpa4_channels, _dpa4_cartesian),
    "DescrptDPA4C": _Source("aggregate_moments", 0, _dpa4c_channels, _dpa4c_cartesian),
}


def state_source(descriptor: Any) -> _Source:
    """How ``descriptor`` hands out its equivariant features (the method that
    computes them and the index of them in what it returns, first); refused
    for a descriptor that has none."""
    try:
        return _SOURCES[type(descriptor).__name__]
    except KeyError:
        raise NotImplementedError(
            "atomic dipoles and quadrupoles need the descriptor's equivariant "
            f"features, which {type(descriptor).__name__} does not hand out; "
            'use `descriptor.type: "dpa4"` or `"dpa4c"`'
        ) from None


def equivariant_channels(descriptor: Any, lmax: int) -> list[int]:
    """How many channels of degree ``1 .. lmax`` ``descriptor`` offers a
    head; refused for a descriptor that offers none."""
    return state_source(descriptor).channels(descriptor, lmax)


def cartesian_features(
    descriptor: Any, state: torch.Tensor, lmax: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The descriptor's equivariant features in Cartesian components:
    vectors ``(N, 3, C_1)`` and, for ``lmax`` 2, traceless symmetric tensors
    ``(N, 3, 3, C_2)``.

    DPA4 packs its node state as ``(N, (l+1)^2, 1, C)`` in the global frame,
    degree by degree with ``m`` ascending. Its vectors are
    ``(c_-1, c_0, c_+1) = (-y, -z, x)`` up to a scale; its ``l = 2``
    coefficients expand in the basis DeePMD-kit builds for the same layout.

    DPA4C keeps its moments flat, ``(N, S)``: degree after degree, each one
    component after component with its channels together. Its vectors are
    ``(x, y, z)``; its ``l = 2`` components are the ones DeePMD-kit's own
    ``packed_l2_to_stf`` turns into a traceless matrix.
    """
    return state_source(descriptor).cartesian(descriptor, state, lmax)


def contract(
    coefficient: torch.Tensor,
    vectors: torch.Tensor,
    tensors: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dipoles ``(N, 3)`` and quadrupoles ``(N, 3, 3)`` (None without
    ``tensors``): the channels mixed with the head's coefficients ``(N, C_1)``
    or ``(N, C_1 + C_2)``, each moment the mean over its channels."""
    width = vectors.shape[-1]
    dipole = torch.einsum(
        "nc,nac->na", coefficient[:, :width] / width, vectors.to(coefficient)
    )
    if tensors is None:
        return dipole, None
    quadrupole = torch.einsum(
        "nc,nabc->nab",
        coefficient[:, width:] / tensors.shape[-1],
        tensors.to(coefficient),
    )
    return dipole, quadrupole


class MultipoleFittingNetDP(InvarFitting):
    __doc__ = r"""Fit the coefficients that mix a descriptor's equivariant
        channels into atomic dipoles and quadrupoles.

        One coefficient per channel of each moment, ``lmax`` moments, the
        dipole's first. ``channels`` is the count per moment, or one number
        for a descriptor as wide in every degree.

        """ + (InvarFitting.__doc__ or "")

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        lmax: int = 1,
        channels: int | list[int] = 1,
        # Mutable defaults mirror deepmd's own fitting signatures verbatim.
        neuron: list[int] = [128, 128, 128],  # noqa: B006
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        rcond: float | None = None,
        trainable: list[bool] | None = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        exclude_types: list[int] = [],  # noqa: B006
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        default_fparam: list | None = None,
        **kwargs: Any,
    ) -> None:
        if lmax not in range(1, MAX_MOMENT + 1):
            raise ValueError(
                f"multipole_net.lmax must be 1 (dipoles) or 2 (and "
                f"quadrupoles), got {lmax!r}"
            )
        self.lmax = int(lmax)
        if isinstance(channels, int):
            channels = [channels] * self.lmax
        if len(channels) != self.lmax:
            raise ValueError(
                f"`channels` needs one count per moment ({self.lmax}), got "
                f"{list(channels)}"
            )
        self.channels = [int(width) for width in channels]
        super().__init__(
            var_name=MULTIPOLE_VAR_NAME,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=sum(self.channels),
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            rcond=rcond,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
            default_fparam=default_fparam,
            **kwargs,
        )

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "type": MULTIPOLE_VAR_NAME,
            "lmax": self.lmax,
            "channels": self.channels,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "MultipoleFittingNetDP":
        """``var_name`` and ``dim_out`` are constants of this class, as in
        :class:`~deepmd_lr.fitting.charge.ChargeFittingNetDP`."""
        data = data.copy()
        data.pop("var_name", None)
        data.pop("dim_out", None)
        return super().deserialize(data)


@BaseFitting.register(MULTIPOLE_VAR_NAME)
@torch_module
class MultipoleFittingNet(MultipoleFittingNetDP):
    """PyTorch (pt_expt) binding of :class:`MultipoleFittingNetDP`."""
