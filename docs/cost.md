# Cost

The charge head reads the descriptor features the energy head produced, so the
backbone runs once, on the serial and the partitioned path alike (the model's
`_capturing_backbone`; a second pass remains as a fallback for when the capture
cannot carry the gradient, never for a compressed model).

These are recorded measurements, not performance guarantees across hardware
or model configurations. Against an identically configured short-range twin
(`long_range` deleted, nothing else changed):

| | short-range | long-range | ratio |
|---|---:|---:|---:|
| training, 192-atom water | 0.0431 s/batch | 0.0967 s/batch | **2.24×** |
| MD, 6,000 atoms | 34.6 ms/step | 55.5 ms/step | **1.60×** |
| MD, 65,856 atoms | 256.7 ms/step | 403.8 ms/step | **1.57×** |

Training pays more than MD because the force loss differentiates a second time,
through the charge head too.

The reciprocal sum alone took about 1.5 ms in the recorded runs; the charge
head and charge-response backward accounted for most of the overhead.
PME cost depends on both atom count and mesh size, so refining the mesh or
enlarging the cell at fixed mesh spacing is not free
([choosing the mesh](alpha-and-mesh.md#choosing-alpha-and-the-mesh)).

The recorded long-range model used ~0.65 MiB/atom (42.6 GiB at 65,856 atoms),
with or without a field. For the current compressed Born-charge path and its
shared backward, see [implementation notes](born-charges.md#implementation-notes).

### With the compressed backbone

GB10, source-built CUDA operators, 6,144-atom water, 1,000 NVE steps.
The ns/day figures use a 0.5 fs timestep. This is a different machine from
the table above, so compare within this table only.

| | ms/step | ns/day | peak GiB |
|---|---:|---:|---:|
| short-range | 226.8 | 0.190 | 6.28 |
| short-range, compressed | **20.9** | **2.069** | **0.32** |
| long-range | 373.3 | 0.116 | 6.41 |
| long-range, compressed | **27.2** | **1.590** | **0.34** |

**13.7× faster and 19× less memory** for the long-range model, with the NVE
drift unchanged. The LR/SR ratio falls from 1.66× to 1.30× because the
charge-response backward runs through the fused kernel too. Against the
uncompressed model on 50 water frames the tabulation error is 1e-5 eV/Å in
force and 1e-7 e in charge and Born tensors, three orders below the model's
error against DFT; stride 0.002 Å gains nothing over the default 0.01.
