# Cost

The charge head reads the descriptor features the energy head produced, so the
backbone runs once, on the serial and the partitioned path alike (the model's
`_capturing_backbone`; a second pass remains as a fallback for when the capture
cannot carry the gradient, never for a compressed model).

Against an identically configured short-range twin (`long_range` deleted,
nothing else changed):

| | short-range | long-range | ratio |
|---|---:|---:|---:|
| training, 192-atom water | 0.0431 s/batch | 0.0967 s/batch | **2.24×** |
| MD, 6,000 atoms | 34.6 ms/step | 55.5 ms/step | **1.60×** |
| MD, 65,856 atoms | 256.7 ms/step | 403.8 ms/step | **1.57×** |

Training pays more than MD because the force loss differentiates a second time,
through the charge head too.

**The reciprocal sum is not what costs.** On its own it is 1.5 ms, 2.3% of a
6,000-atom step and 0.3% of a 65,856-atom one, and it barely grows with the
system because the FFT follows the mesh rather than the atom count. Refining
the mesh is close to free
([choosing it](alpha-and-mesh.md#choosing-alpha-and-the-mesh)). The overhead
is the charge head and the charge-response backward, which is work rather than
duplication.

An external field adds one more backward for `Z`: +25–30% on the step and no
extra memory, since it reuses the captured features. The long-range model
takes ~0.65 MiB/atom (42.6 GiB at 65,856 atoms), with or without a field.

### With the compressed backbone

GB10, source-built CUDA operators, 6,144-atom water, 1,000 NVE steps. A
different machine from the table above, so compare within this table only.

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
