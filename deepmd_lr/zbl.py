# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fixed, smoothly truncated ZBL repulsion on the DPA4C neighbour graph.

This is an additive prior, not DeePMD's ``use_srtab`` interpolation or DPA4's
Zone/descriptor-clamping construction. Training fits the residual energy and
forces with this term present; adding it to existing weights changes the model.
"""

import math
from numbers import Real
from typing import Any

import torch
from ase.data import atomic_numbers
from deepmd.dpmodel.utils.neighbor_graph import NeighborGraph
from deepmd.pt_expt.model.edge_transform_output import fit_output_to_model_output_graph


class ZBL(torch.nn.Module):
    """Analytical screened nuclear repulsion, with no trainable parameters."""

    def __init__(self, type_map: list[str], settings: dict[str, Any], device=None):
        super().__init__()
        if not isinstance(settings, dict):
            raise ValueError("zbl must be a configuration dictionary")
        settings = {k: v for k, v in settings.items() if not k.startswith("_")}
        if set(settings) - {"r_inner", "r_outer"}:
            raise ValueError("zbl accepts only r_inner and r_outer (Angstrom)")
        settings = {"r_inner": 0.5, "r_outer": 0.8, **settings}
        for key, value in settings.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"zbl.{key} must be a finite real distance")
        if not 0 < settings["r_inner"] < settings["r_outer"]:
            raise ValueError("zbl requires 0 < r_inner < r_outer")
        if not type_map or any(
            s not in atomic_numbers or atomic_numbers[s] <= 0 for s in type_map
        ):
            raise ValueError("zbl requires real element symbols in type_map")
        self.r_inner = float(settings["r_inner"])
        self.r_outer = float(settings["r_outer"])
        # Derived entirely from serialized type_map; not an optimizer parameter.
        self.register_buffer(
            "numbers",
            torch.tensor(
                [atomic_numbers[s] for s in type_map],
                dtype=torch.float64,
                device=device,
            ),
            persistent=False,
        )

    def serialize(self):
        return {"r_inner": self.r_inner, "r_outer": self.r_outer}

    def atomic_energy(self, atype, edge_index, edge_vec, edge_mask):
        src, dst = edge_index.long()
        valid_type = (atype >= 0) & (atype < self.numbers.numel())
        valid = edge_mask.bool() & valid_type[src] & valid_type[dst]
        # Avoid singular padded edges without regularizing real close contacts.
        r2 = edge_vec.to(torch.float64).square().sum(-1)
        r = torch.sqrt(torch.where(valid, r2, torch.ones_like(r2)))
        z = self.numbers.to(device=edge_vec.device)[
            atype.clamp(0, self.numbers.numel() - 1).long()
        ]
        zi, zj = z[src], z[dst]
        a = 0.46850 / (zi.pow(0.23) + zj.pow(0.23))
        x = r / a
        screening = (
            0.18175 * torch.exp(-3.19980 * x)
            + 0.50986 * torch.exp(-0.94229 * x)
            + 0.28022 * torch.exp(-0.40290 * x)
            + 0.02817 * torch.exp(-0.20162 * x)
        )
        t = ((r - self.r_inner) / (self.r_outer - self.r_inner)).clamp(0, 1)
        switch = 1 - 10 * t.pow(3) + 15 * t.pow(4) - 6 * t.pow(5)
        pair = 14.3996454784255 * zi * zj * screening / r * switch
        pair = torch.where(valid & (r < self.r_outer), pair, torch.zeros_like(pair))
        # Directed graph: each destination owns half of each pair energy.
        atomic = edge_vec.new_zeros(atype.shape[0], dtype=torch.float64)
        return atomic.index_add(0, dst, 0.5 * pair).unsqueeze(-1)

    def graph_output(
        self,
        model,
        atype,
        n_node,
        n_local,
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        **kwargs,
    ):
        leaf = edge_vec.detach().requires_grad_(True)
        graph = NeighborGraph(
            n_node=n_node,
            n_local=n_local,
            edge_index=edge_index,
            edge_vec=leaf,
            edge_mask=edge_mask,
            destination_order=destination_order,
            destination_row_ptr=destination_row_ptr,
            source_order=kwargs.get("source_order"),
            source_row_ptr=kwargs.get("source_row_ptr"),
            destination_sorted=kwargs.get("destination_sorted", False),
        )
        energy = self.atomic_energy(atype, edge_index, leaf, edge_mask)
        return fit_output_to_model_output_graph(
            {"energy": energy},
            model.atomic_output_def(),
            graph,
            do_atomic_virial=kwargs.get("do_atomic_virial", False),
            # No parameters and a detached leaf: nothing upstream needs a
            # second-order graph, in training or out of it.
            create_graph=False,
            force_precision=edge_vec.dtype,
            node_capacity=atype.shape[0],
            n_local=n_local,
        )
