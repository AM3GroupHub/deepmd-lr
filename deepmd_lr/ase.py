# SPDX-License-Identifier: MIT
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

from deepmd_lr.born import (
    screening_factor,
)
from deepmd_lr.compress import (
    compress_model,
)
from deepmd_lr.fitting.charge import check_charge_filter

__all__ = ["LongRangeCalculator", "load_model", "select_neighbor_graph_method"]

#: The prefix every parameter carries in a checkpoint's state dict.
_STATE_PREFIX = "model.Default."


def load_model(
    checkpoint: str,
    device: str | torch.device = DEVICE,
    *,
    head: str | None = None,
    long_range: dict | None = None,
) -> Any:
    """Rebuild a trained model from a training checkpoint.

    Reads the architecture from the checkpoint's own ``model_params`` rather
    than asking the caller to supply the input file again, so a checkpoint is
    self-describing.
    Multi-head checkpoints require ``head`` (branch name or unique alias).

    ``long_range`` overrides the reciprocal-space settings the model was
    trained with -- see
    :meth:`~deepmd_lr.model.lr_model.LongRangeEnergyModel.set_long_range`. The
    long-range term holds no weights, so this changes how the electrostatics
    are summed without touching a single trained parameter.
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
        hint = ""
        if any(".compress_" in key for key in unexpected):
            # Tables from `enable_compression` are a function of the weights
            # and are rebuilt at load, never restored from a state dict.
            hint = (
                " The `compress_*` buffers of a compressed DPA4C are not "
                "persisted; save the uncompressed model and pass compress=True "
                "to LongRangeCalculator or the evaluator instead."
            )
        raise ValueError(
            "checkpoint weights do not match the recorded model config: "
            f"{len(missing)} missing (e.g. {missing[:3]}) and "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})." + hint
        )
    if long_range is not None:
        model.set_long_range(long_range)
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
        constructed model. A model passed in is used, not copied, and before
        each frame its long-range term is adapted to that frame's cell and
        pbc. GPU evaluators build terms of their own, so they may share it.
    device
        Torch device; defaults to DeePMD-kit's own choice.
    charge_spin
        Optional fixed (total_charge, spin_multiplicity) condition for the
        trajectory. Defaults to the model's neutral/default state.
    head
        Required branch name or unique alias for a multi-head checkpoint.
        Omit when passing an already constructed model.
    long_range
        Optional replacement for the reciprocal-space settings, e.g.
        ``{"method": "ewald"}`` or ``{"isolated_kernel": "dense"}``. The
        frame's pbc already picks the 3D, the slab or the isolated sum. The
        long-range term carries no weights, so the boundary treatment is a
        property of the simulation rather than of the checkpoint.
    compress
        ``True`` to compress the DPA4C backbone for inference at DeePMD-kit's
        default table stride, or a float stride in angstrom. Needs a
        deepmd-kit built from source with the CUDA operators -- refused
        otherwise -- and folds ``charge_spin`` into the tables; see
        :meth:`~deepmd_lr.model.lr_model.LongRangeEnergyModel.enable_compression`.
    eps_inf
        High-frequency dielectric constant of the medium, default ``1.0``
        (vacuum). ``results["charges"]`` and ``results["born_charge"]`` are
        multiplied by ``sqrt(eps_inf)``, the factor that turns the model's
        Coulomb charges into the screened ones for which the energy reads
        ``q~_i q~_j / (4 pi eps_0 eps_inf r)``. A field force built from those
        tensors scales with it; the energy and field-free forces do not.
    charge_filter
        ``(c_t, delta)`` in e; ``False`` for none; None (the default) for the
        model's own ``long_range.charge_filter``. As an override, simulation
        only: charges below ``c_t - delta`` in magnitude are set exactly to
        zero, those above
        ``c_t + delta`` kept, with a C^3 polynomial between, and the total
        restored on the atoms that keep charge; see
        :func:`~deepmd_lr.fitting.charge.filter_charges`. Refused with a field.

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
        long_range: dict | None = None,
        compress: bool | float = False,
        eps_inf: float = 1.0,
        charge_filter: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        # A uniform external field in V/angstrom, or None. The force it adds is
        # `F_i = Z_i.T E` with Z the Born effective charge tensor. That is a
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
            load_model(model, device, head=head, long_range=long_range)
            if isinstance(model, str)
            else model.to(device).eval()
        )
        if long_range is not None and not isinstance(model, str):
            self.model.set_long_range(long_range)
        # A model trained WITHOUT `type: dp_lr` -- the baseline twin the
        # README asks for -- is a plain EnergyModel and simply has no
        # `has_long_range` attribute. The distributed driver already reads it
        # defensively; the calculator used not to, so loading a baseline into
        # it raised AttributeError instead of just running short-range.
        self._has_long_range = bool(getattr(self.model, "has_long_range", False))
        if self.efield is not None:
            if not self._has_long_range:
                raise ValueError(
                    "an external field needs Born effective charges, which "
                    "need the charge head -- this model has no `long_range` "
                    "block, so it predicts no charges."
                )
            self.model.compute_born = True
        # Screened charges: `results["charges"]` and `results["born_charge"]`
        # -- and so the field force -- carry sqrt(eps_inf); the energy and the
        # field-free forces do not. Validated now rather than on the first
        # step. A short-range model reports no charges, so nothing to scale.
        screening_factor(eps_inf)
        self.eps_inf = float(eps_inf)
        if self._has_long_range:
            self.model.eps_inf = self.eps_inf
        # None: the model's own `long_range.charge_filter`; False: none.
        self.charge_filter = charge_filter
        effective = (
            getattr(self.model, "trained_charge_filter", None)
            if charge_filter is None
            else check_charge_filter(charge_filter)
        )
        if effective is not None:
            if not self._has_long_range:
                raise ValueError("charge_filter needs a model with a long_range block")
            if self.efield is not None:
                raise ValueError(
                    "charge_filter cannot be combined with an efield yet: the "
                    "field force is built from unfiltered Born charges"
                )
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
        # Last: the model is on its device and in eval mode, and the run's
        # charge state is known -- compression folds it into the tables.
        compress_model(self.model, compress, charge_spin=self.charge_spin)

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

    def _check_slab_gap(self, atoms) -> None:
        """Refuse a slab whose vacuum a cutoff sphere could cross.

        This calculator hands the model the full cell, so the descriptor sees
        a slab as 3D-periodic. With a gap narrower than the cutoff, atoms at
        one face would see the next image's opposite face, an interaction the
        slab does not have.
        """
        axis = [bool(p) for p in atoms.pbc].index(False)
        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        a, b = (cell[j] for j in range(3) if j != axis)
        normal = np.cross(a, b)
        normal /= np.linalg.norm(normal)
        depth = abs(float(cell[axis] @ normal))
        gap = depth - float(np.ptp(atoms.get_positions() @ normal))
        rcut = float(self.model.get_rcut())
        if gap < rcut:
            raise ValueError(
                f"the slab leaves {gap:.2f} A of vacuum along its open axis, "
                f"less than the cutoff {rcut} A: atoms would see their images "
                "across it. Pad the cell and centre the slab, e.g. "
                "deepmd_lr.nvalchemi.read_structure(path) does both."
            )

    def _check_pbc(self, atoms) -> bool:
        """Refuse a boundary this model cannot evaluate; return whether any
        axis is periodic.

        The long-range term follows the frame's pbc (:meth:`calculate`); what
        is left to refuse is a slab whose vacuum a cutoff sphere could cross,
        and a partial pbc for a short-range model, whose graph has no slab
        mode.
        """
        periodic = bool(atoms.pbc.any())
        if self._has_long_range:
            if sum(bool(p) for p in atoms.pbc) == 2:
                self._check_slab_gap(atoms)
        elif periodic and not atoms.pbc.all():
            raise ValueError(
                "Partial pbc is unsupported by the short-range graph interface"
            )
        return periodic

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties or ["energy"], system_changes)
        atoms = self.atoms
        periodic = self._check_pbc(atoms)
        if self._has_long_range:
            # The frame decides how the sum is taken: its pbc picks the 3D, the
            # slab or the isolated sum, its cell the PME mesh. The term holds
            # no weights, so this changes no parameter, and each frame gets the
            # mesh its own cell needs.
            self.model.adapt_long_range(np.asarray(atoms.get_cell()), atoms.pbc)

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

        if self._has_long_range:
            # Re-asserted every call: the model object may be shared between
            # calculators built with different media, and the last one to be
            # constructed must not win over the one actually computing.
            self.model.eps_inf = self.eps_inf
            self.model.charge_filter = self.charge_filter     # re-asserted, as eps_inf
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
            # F_ib = E_a Z_iab. Sums to Q.E over the cell, because the Born
            # tensors sum to Q.I -- so a neutral system feels no net force and
            # does not drift.
            forces = forces + np.einsum("iab,a->ib", born, self.efield)
            self.results["born_charge"] = born
        self.results["forces"] = forces

        # Stress is force per area, so it needs a volume to divide by. The
        # virial itself is still meaningful without a cell -- it is the pair
        # sum -- but there is nothing to normalize it against.
        # Under a field, none: the field's own contribution to the stress is a
        # separate derivation not attempted here, and reporting the field-free
        # stress would be quietly wrong.
        if "virial" in out and periodic and self.efield is None:
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
        for key in ("energy_short_range", "energy_long_range"):
            if key in out:
                self.results[key] = float(out[key].reshape(-1)[0].detach())
        if "charge" in out:
            self.results["charges"] = out["charge"].reshape(-1).detach().cpu().numpy()
