#!/usr/bin/env bash
# The three stages in order, each in its own folder. Run from this directory:
#     ./train.sh
# The dataset is example 01's (../01-bulk/data); build it there first. The
# inputs name it relative to the stage folders (../../01-bulk/data).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
graft="$here/../../scripts/graft_short_range.py"
mkdir -p sr lr ft

# Stage 1, SR: the short-range model alone.
(cd sr && cp "$here/sr.json" input.json &&
    dp --pt-expt train input.json)

# Stage 2, LR: the charge head on the frozen short-range model. The graft
# gives `--init-model` a checkpoint that has the charge head's keys.
(cd lr && cp "$here/lr.json" input.json &&
    python "$graft" --short-range ../sr/model.ckpt.pt --input input.json \
        --out init.ckpt.pt &&
    dp --pt-expt train input.json --init-model init.ckpt)

# Stage 3, FT: everything, from the LR model. The live checkpoint, not the
# EMA one, is what `--init-model` continues from.
(cd ft && cp "$here/ft.json" input.json &&
    dp --pt-expt train input.json --init-model ../lr/model.ckpt)

echo "final model: ft/model_ema.ckpt.pt"
