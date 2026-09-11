# SPDX-License-Identifier: LGPL-3.0-or-later
"""An ASE calculator that keeps the long-range term.

Why not DeePMD-kit's own calculator
-----------------------------------
``deepmd.calculator.DP`` goes through ``DeepPot``, which evaluates the
*lower* interface: a neighbour graph of edge vectors, with no cell. That is
enough for a strictly local model and structurally insufficient here -- the
reciprocal-space term is a function of the cell and absolute positions. Used
with a ``dp_lr`` model it would return only the short-range energy, and
quietly: on the Phase 5 benchmark the frozen artifact came back at -93.74 eV
where the eager model gives -55.12.

This calculator drives the model's full ``forward`` instead, so the cell is
present and the long-range term is evaluated. It is eager PyTorch rather than
a compiled artifact, so it is slower than a frozen short-range model -- the
price of computing physics the exported path cannot express.
"""

from typing import (
    Any,
)

import numpy as np
import torch
from ase.calculators.calculator import (
    Calculator,
    all_changes,
)
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

__all__ = ["LongRangeCalculator", "load_model", "select_neighbor_graph_method"]

#: eV/A^3 is what ASE expects for stress; DeePMD-kit reports the virial in eV.
_STATE_PREFIX = "model.Default."


def load_model(
    checkpoint: str, device: str | torch.device = DEVICE, *, head: str | None = None
) -> Any:
    """Rebuild a trained model from a training checkpoint.

    Reads the architecture from the checkpoint's own ``model_params`` rather
    than asking the caller to supply the input file again, so a checkpoint is
    self-describing.
    Multi-head checkpoints require ``head`` (branch name or unique alias).
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = state["model"]["_extra_state"]["model_params"]
    prefix = _STATE_PREFIX
    if "model_dict" in params:
        branches = params["model_dict"]
        if head is None:
            raise ValueError(f"Select a checkpoint head from {list(branches)}")
        matches = [
            name
            for name, branch in branches.items()
            if head == name or head in branch.get("model_branch_alias", [])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Unknown or ambiguous head {head!r}; available: {list(branches)}"
            )
        name = matches[0]
        params = branches[name]
        prefix = f"model.{name}."
    elif head not in (None, "Default"):
        raise ValueError(
            f"Single-head checkpoint has only head 'Default', got {head!r}"
        )
    model = get_model(params)
    weights = {
        key[len(prefix) :]: value
        for key, value in state["model"].items()
        if key.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError(
            "checkpoint weights do not match the recorded model config: "
            f"{len(missing)} missing (e.g. {missing[:3]}) and "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})"
        )
    model = model.to(device).eval()
    select_neighbor_graph_method(model)
    return model


def select_neighbor_graph_method(model: Any) -> str:
    """Point the short-range path at the GPU cell list where possible.

    DeePMD-kit resolves ``neighbor_graph_method`` once at *training* startup
    and does not record it in the checkpoint, so a model rebuilt for
    inference has no such attribute and
    ``make_model._resolve_graph_method`` falls back to ``"dense"`` -- its
    O(N^2) all-pairs builder. That is invisible until the system is large:
    on the 6520-atom LiFSI frame the dense path peaks at **48.2 GB** and
    takes 724 ms, against **4.0 GB** and 201 ms through ``"nv"``, for a
    bit-identical energy. 48 GB does not fit on a 48 GB card, which is how
    this first showed up -- as an out-of-memory error on an A40.

    Resolved through DeePMD-kit's own helper, so a CPU model or a missing
    nvalchemiops degrades to ``"dense"`` rather than raising.
    """
    from deepmd.pt_expt.utils.graph_builder import (
        resolve_neighbor_graph_method,
    )

    device = next(model.parameters()).device
    method = resolve_neighbor_graph_method("auto", device)
    model.neighbor_graph_method = method
    return method


class LongRangeCalculator(Calculator):
    """ASE calculator for a ``dp_lr`` model, long-range term included.

    Parameters
    ----------
    model
        A path to a training checkpoint (``model.ckpt.pt``), or an already
        constructed model.
    device
        Torch device; defaults to DeePMD-kit's own choice.
    charge_spin
        Optional fixed (total_charge, spin_multiplicity) condition for the
        trajectory. Defaults to the model's neutral/default state.
    head
        Required branch name or unique alias for a multi-head checkpoint.
        Omit when passing an already constructed model.

    Examples
    --------
    >>> from ase.md.verlet import VelocityVerlet
    >>> atoms.calc = LongRangeCalculator("run_joint/model.ckpt.pt")
    >>> VelocityVerlet(atoms, timestep=0.5 * units.fs).run(1000)

    Notes
    -----
    ``atoms.calc.results`` carries ``energy_short_range``,
    ``energy_long_range`` and ``charges`` alongside the usual keys, so the
    split can be inspected during a trajectory without re-running the model.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "charges"]

    def __init__(
        self,
        model: str | Any,
        device: str | torch.device = DEVICE,
        charge_spin: tuple[float, float] | list[float] | torch.Tensor | None = None,
        head: str | None = None,
        efield: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        # A uniform external field in V/angstrom, or None. The force it adds is
        # `F_i = Z_i E` with Z the Born effective charge tensor. That is a
        # response, NOT the gradient of a `-M.E` energy: under periodic
        # boundaries M is not even defined, and for an isolated system the two
        # differ by a centroid-dependent term (see `deepmd_lr/born.py`). So the
        # energy reported here is left untouched, and a constant-field NVE run
        # will not conserve `E - M.E`. That is by construction, not a bug.
        self.efield = (
            None
            if efield is None
            else np.asarray(efield, dtype=np.float64).reshape(3)
        )
        if head is not None and not isinstance(model, str):
            raise ValueError("head selects a checkpoint branch; pass a checkpoint path")
        self.model = (
            load_model(model, device, head=head)
            if isinstance(model, str)
            else model.to(device).eval()
        )
        if self.efield is not None:
            if not self.model.has_long_range:
                raise ValueError(
                    "an external field needs Born effective charges, which "
                    "need the charge head -- this model has no `long_range` "
                    "block, so it predicts no charges."
                )
            self.model.compute_born = True
        self.type_map = list(self.model.get_type_map())
        self.charge_spin = None
        if charge_spin is not None:
            condition = torch.as_tensor(
                charge_spin, dtype=torch.float64, device=self.device
            )
            if condition.numel() != 2:
                raise ValueError(
                    "charge_spin must contain (total_charge, spin_multiplicity)"
                )
            self.charge_spin = condition.reshape(1, 2)

    def _atom_types(self, atoms: Any) -> torch.Tensor:
        """Map ASE chemical symbols onto the model's type indices."""
        try:
            indices = [self.type_map.index(s) for s in atoms.get_chemical_symbols()]
        except ValueError as exc:
            unknown = sorted(set(atoms.get_chemical_symbols()) - set(self.type_map))
            raise ValueError(
                f"element(s) {unknown} are not in the model's type_map {self.type_map}"
            ) from exc
        return torch.tensor(indices, dtype=torch.int64, device=self.device)

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties or ["energy"], system_changes)
        atoms = self.atoms

        # Only a term that actually sums over periodic images needs a cell.
        # This check used to reject every non-periodic frame unconditionally,
        # which was wrong twice over: a short-range model has no such
        # requirement at all, and a slab configured as pbc=[T, T, F] is
        # exactly what `slab_correction` exists to handle.
        periodic = bool(atoms.pbc.any())
        needs_cell = (
            self.model.has_long_range and self.model.long_range_energy.needs_cell
        )
        if needs_cell and not periodic:
            raise ValueError(
                "this model's long-range term uses method "
                f"'{self.model.long_range_energy.method}', which sums over "
                f"periodic images, but the Atoms object has pbc={atoms.pbc} "
                "and so no cell to repeat. For an isolated molecule build the "
                "model with `long_range.reciprocal.method: 'isolated'`."
            )

        if self.model.has_long_range:
            term = self.model.long_range_energy
            expected_pbc = term.pbc if term.needs_cell else (False, False, False)
            if not np.array_equal(atoms.pbc, expected_pbc):
                raise ValueError(
                    f"Atoms pbc={atoms.pbc.tolist()} does not match the "
                    f"selected long-range branch pbc={list(expected_pbc)}"
                )
        elif periodic and not atoms.pbc.all():
            raise ValueError(
                "Partial pbc is unsupported by the short-range graph interface"
            )

        coord = torch.tensor(
            np.asarray(atoms.get_positions()),
            dtype=torch.float64,
            device=self.device,
        ).unsqueeze(0)
        # A non-periodic Atoms carries a zero cell, which is not a cell --
        # passing it would trip the degenerate-box check. Hand the model None,
        # which is the representation DeePMD-kit itself uses for `nopbc` data.
        cell = (
            torch.tensor(
                np.asarray(atoms.get_cell()), dtype=torch.float64, device=self.device
            ).unsqueeze(0)
            if periodic
            else None
        )
        atype = self._atom_types(atoms).unsqueeze(0)

        out = self.model(coord, atype, box=cell, charge_spin=self.charge_spin)

        energy = float(out["energy"].reshape(-1)[0].detach())
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        forces = (
            (out["forces"] if "forces" in out else out["force"])[0]
            .detach()
            .cpu()
            .numpy()
        )
        if self.efield is not None:
            born = out["born_charge"][0].detach().cpu().numpy()
            # F_ia = Z_iab E_b. Sums to Q.E over the cell, because the Born
            # tensors sum to Q.I -- so a neutral system feels no net force and
            # does not drift.
            forces = forces + np.einsum("iab,b->ia", born, self.efield)
            self.results["born_charge"] = born
        self.results["forces"] = forces

        # Stress is force per area, so it needs a volume to divide by. The
        # virial itself is still meaningful without a cell -- it is the pair
        # sum -- but there is nothing to normalize it against.
        if "virial" in out and periodic and self.efield is not None:
            # The field's own contribution to the stress is a separate
            # derivation and is not attempted here. Reporting the field-free
            # stress under a field would be quietly wrong, so report none.
            pass
        elif "virial" in out and periodic:
            virial = out["virial"].reshape(3, 3).detach().cpu().numpy()
            # ASE wants stress = -virial / volume, in Voigt order.
            stress = -virial / atoms.get_volume()
            self.results["stress"] = np.array(
                [
                    stress[0, 0],
                    stress[1, 1],
                    stress[2, 2],
                    0.5 * (stress[1, 2] + stress[2, 1]),
                    0.5 * (stress[0, 2] + stress[2, 0]),
                    0.5 * (stress[0, 1] + stress[1, 0]),
                ]
            )
        # Diagnostics. The energy split is per-frame and reduces to a scalar;
        # charges are per-atom and keep their axis.
        for key, name in (
            ("energy_short_range", "energy_short_range"),
            ("energy_long_range", "energy_long_range"),
        ):
            if key in out:
                self.results[name] = float(out[key].reshape(-1)[0].detach())
        if "charge" in out:
            self.results["charges"] = out["charge"].reshape(-1).detach().cpu().numpy()
