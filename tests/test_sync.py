# SPDX-License-Identifier: MIT
"""One evaluation, one download.

The MD engine's step is meant to leave the host free: every device result
stays on the device, and the only copy back is the small word at the end of
``DistributedEvaluator.__call__`` -- the energy, the status flags and, when
asked for, the virial -- because the call returns the energy as a float. A
neighbour search, a graph build or a long-range term that reads the host in
the middle of a step stalls every kernel queued after it, which is what made
the compressed path host-bound at a few thousand atoms. The counts below are
the contract: `Memcpy DtoH` once per step, no pageable upload, no `nonzero`
or `bincount`, and `cudaStreamSynchronize` only for that one copy.

Measured with torch.profiler on a warm step. `aten::item` is not counted:
the readback's own `float()`/`int()` calls on the downloaded CPU tensor
record as `aten::item` too, and they are not synchronisations. The same
contract holds for a barostat step, which used to download and upload the
cell twice and re-upload the virial every step.
"""

import collections

import pytest
import torch
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.nvalchemi import DistributedEvaluator

from .test_joint_model import build, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the evaluator runs on the GPU"
)


def compressed_available() -> bool:
    from deepmd.kernels.cuda.dpa4c.graph_compress import op_available

    return bool(op_available())


def count_syncs(step, *args) -> dict[str, int]:
    """Device-to-host copies, pageable uploads, stream synchronisations and
    the two syncing torch ops, in one warm step."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        step(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(*args)
    torch.cuda.synchronize()
    names = collections.Counter()
    for e in prof.events():
        n = e.name
        if n.startswith("Memcpy DtoH"):
            names["dtoh"] += 1
        elif n.startswith("Memcpy HtoD") and "Pageable" in n:
            names["pageable_htod"] += 1
        elif n == "cudaStreamSynchronize":
            names["stream_sync"] += 1
        elif n in ("cudaDeviceSynchronize", "aten::nonzero", "aten::bincount"):
            names[n] += 1
    return dict(names)


def system():
    coord, atype, cell = frame(n_mol=16, box=14.0)
    c, a, h = (t[0].cpu().numpy() for t in (coord, atype, cell))
    return torch.as_tensor(c, device=DEVICE), a, h


def assert_downloads(counts, downloads):
    """Exactly ``downloads`` copies to the host, each at most one stream
    synchronisation, and no pageable upload or syncing torch op."""
    assert counts.get("dtoh", 0) == downloads, counts
    assert counts.get("stream_sync", 0) <= downloads, counts
    assert counts.get("pageable_htod", 0) == 0, counts
    assert "aten::nonzero" not in counts and "aten::bincount" not in counts, counts


class TestOneDevice:
    def test_uncompressed_short_range(self):
        """Uncompressed models get a compact edge axis, which costs one more
        copy (the edge count) by design: their descriptor processes every
        edge it is handed and must not pay for a padded tail."""
        x, a, h = system()
        ev = DistributedEvaluator(
            build(long_range=False), a, h, devices=[DEVICE], memory_headroom=None
        )
        assert_downloads(count_syncs(ev, x), 2)

    def test_uncompressed_long_range_with_virial(self):
        x, a, h = system()
        ev = DistributedEvaluator(
            build(), a, h, devices=[DEVICE], memory_headroom=None, compute_virial=True
        )
        assert_downloads(count_syncs(ev, x), 2)

    @pytest.mark.skipif(not compressed_available(), reason="needs the CUDA kernel")
    @pytest.mark.parametrize("long_range", [False, True])
    def test_compressed(self, long_range):
        from .test_compress import build32

        x, a, h = system()
        model = build32(long_range=long_range)
        model.enable_compression()
        ev = DistributedEvaluator(model, a, h, devices=[DEVICE], memory_headroom=None)
        assert_downloads(count_syncs(ev, x), 1)


class TestBarostat:
    """An NPT step downloads what an NVE step does, nothing more: the
    barostat integrates the cell on the device and hands it over there
    (`Thermo.set_cell_device`), the pressure kernel reads the evaluator's
    device virial, and the host copy of the cell is refreshed only when a log
    row or a checkpoint asks for it. Search radii follow cell changes on the
    device, including in small periodic boxes."""

    @staticmethod
    def npt(model, compress, box=32.0):
        from ase import Atoms, units

        from deepmd_lr.nvalchemi import MTKNPT

        coord, atype, cell = frame(n_mol=128 if box > 30 else 16, box=box)
        c, a, h = (t[0].cpu().numpy() for t in (coord, atype, cell))
        atoms = Atoms(
            numbers=[8 if t == 0 else 1 for t in a], positions=c, cell=h, pbc=True
        )
        ev = DistributedEvaluator(
            model,
            a,
            h,
            devices=[DEVICE],
            memory_headroom=None,
            compute_virial=True,
            compress=compress,
        )
        return MTKNPT(
            ev,
            atoms,
            0.5 * units.fs,
            temperature_K=300.0,
            pressure_au=1e-4 * units.GPa,
            tdamp=100 * units.fs,
            pdamp=1000 * units.fs,
            cell="isotropic",
            logfile=None,
            loginterval=10**6,
        )

    def test_uncompressed_long_range(self):
        md = self.npt(build(), False)
        try:
            md.run(0)  # Initialise forces; count_syncs warms three bare steps.
            assert_downloads(count_syncs(md.step), 2)
        finally:
            md.close()

    @pytest.mark.skipif(not compressed_available(), reason="needs the CUDA kernel")
    @pytest.mark.parametrize("box", [14.0, 32.0])
    def test_compressed_long_range(self, box):
        from .test_compress import build32

        md = self.npt(build32(), 0.01, box=box)
        try:
            md.run(0)
            assert_downloads(count_syncs(md.step), 1)
        finally:
            md.close()


class TestTwoPartitions:
    """Several devices need the host to decide a repartition: one bool more."""

    def test_uncompressed(self):
        x, a, h = system()
        ev = DistributedEvaluator(
            build(long_range=False), a, h, devices=[DEVICE] * 2, memory_headroom=None
        )
        # the halo flag, each partition's compact edge count, the word
        assert_downloads(count_syncs(ev, x), 4)

    @pytest.mark.skipif(not compressed_available(), reason="needs the CUDA kernel")
    def test_compressed(self):
        from .test_compress import build32

        x, a, h = system()
        model = build32()
        model.enable_compression()
        ev = DistributedEvaluator(
            model, a, h, devices=[DEVICE] * 2, memory_headroom=None
        )
        # the halo flag and the word
        assert_downloads(count_syncs(ev, x), 2)
