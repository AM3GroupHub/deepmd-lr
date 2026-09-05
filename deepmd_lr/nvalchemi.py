# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run a long-range model under nvalchemi's MD engine.

nvalchemi ships the ensembles (``NVE``, ``NVTLangevin``, ``NVTNoseHoover``,
``NPT``, ``NPH``) and DeePMD-kit ships ``DPA4Wrapper`` to adapt a DPA4 model to
them. That wrapper cannot be used unchanged here: it evaluates
``model.forward_lower``, the neighbour-list interface, which carries no cell --
and this model refuses that path precisely because the reciprocal-space term
needs one (see ``LAMMPS.md``).

So the wrapper is subclassed and only ``forward`` is replaced, with a call
through the model's full ``forward``. Everything else -- the element mapping,
the neighbour config, the output plumbing -- is DeePMD-kit's.

Set ``devices`` to evaluate over spatial partitions instead
(:mod:`deepmd_lr.distributed`); the result is identical, only the placement
changes.
"""

from typing import (
    Any,
)

import torch
from deepmd.pt.nvalchemi import (
    DPA4Wrapper,
)
from nvalchemi.data import (
    AtomicData,
    Batch,
)

from deepmd_lr.distributed.driver import (
    evaluate_partitioned,
)

__all__ = ["LongRangeNVAlchemiModel"]


class LongRangeNVAlchemiModel(DPA4Wrapper):
    """Adapt a ``dp_lr`` model to nvalchemi's dynamics engine.

    Parameters
    ----------
    model
        A constructed ``LongRangeEnergyModel``.
    devices
        ``None`` evaluates the whole cell at once. A list of devices (repeats
        allowed) evaluates over that many spatial partitions.

    Notes
    -----
    One frame per call. The reciprocal sum runs over a whole periodic cell, so
    batching several independent systems into one graph would mix them; that
    is refused rather than silently summed.
    """

    def __init__(self, model: Any, devices: list[Any] | None = None, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.devices = devices

    def forward(self, data: AtomicData | Batch, **kwargs: Any):
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        if data.num_graphs != 1:
            raise NotImplementedError(
                "the long-range term is defined per periodic cell, so this "
                f"wrapper takes one frame at a time (got {data.num_graphs}). "
                "Run separate systems as separate trajectories."
            )

        positions = data.positions
        cell = data.cell.reshape(1, 3, 3)
        atype = self._atype(data.atomic_numbers).reshape(1, -1)

        if self.devices is None:
            out = self.model(
                positions.reshape(1, -1, 3).to(self._dtype),
                atype,
                box=cell.to(self._dtype),
            )
            energy = out["energy"].reshape(1)
            forces = out["force"].reshape(-1, 3)
        else:
            # The partitioned path returns numpy; MD only needs values, and
            # keeping it that way avoids holding an autograd graph across
            # devices for no reason.
            result = evaluate_partitioned(
                self.model,
                positions.detach().cpu().numpy(),
                atype.reshape(-1).cpu().numpy(),
                cell.reshape(3, 3).detach().cpu().numpy(),
                n_partitions=len(self.devices),
                devices=self.devices,
            )
            energy = torch.tensor(
                [result["energy"]], dtype=positions.dtype, device=positions.device
            )
            forces = torch.tensor(
                result["force"], dtype=positions.dtype, device=positions.device
            )

        return {
            "energy": energy.to(positions.dtype),
            "forces": forces.to(positions.dtype),
        }
