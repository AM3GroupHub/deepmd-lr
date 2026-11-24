#!/bin/bash
# SPDX-License-Identifier: MIT
# Build deepmd-kit 3.2.0 from source with the CUDA operators, on VANDA or HOPPER.
# Same recipe as the local GB10 build, with the cluster's CUDA toolkit module,
# CMake from a module, and the system gcc 11 pinned as host compiler (the CMake
# module pulls GCCcore 14, which nvcc 12.4 rejects).
#
#   bash scripts/cluster/build_deepmd_cuda_cluster.sh vanda   # A40  -> sm_86, cuda12.4
#   bash scripts/cluster/build_deepmd_cuda_cluster.sh hopper  # H200 -> sm_90, cuda12.9
set -o pipefail
CLUSTER=${1:?vanda|hopper}
case "$CLUSTER" in
  vanda)  CUDA_MOD=cuda12.4/toolkit/12.4.1; ARCH=86; TORCH_ARCH=8.6 ;;
  hopper) CUDA_MOD=cuda12.4/toolkit/12.4.1; ARCH=90; TORCH_ARCH=9.0 ;;
  *) echo "unknown cluster $CLUSTER"; exit 2 ;;
esac
ENVS=/scratch/$USER/conda/envs
ENV=$ENVS/dpmd-cuda
SRC=$HOME/repo/deepmd-kit
PLUGIN=$HOME/repo/deepmd-lr
JOBS=${JOBS:-8}
stage() { echo; echo "##### [$(date '+%H:%M:%S')] $*"; }

stage "modules"
module load "$CUDA_MOD" CMake/4.0.3-GCCcore-14.3.0 || exit 3
echo "nvcc : $(nvcc --version | grep release)"; echo "cmake: $(cmake --version | head -1)"
echo "host compiler pinned: $(/usr/bin/g++ --version | head -1)"
export CUDA_HOME=${CUDA_HOME:-$CUDA_PATH}; echo "CUDA_HOME=$CUDA_HOME"

stage "A0 clone env -> $ENV"
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y --clone dpmd -p "$ENV" || { echo "ENV CLONE FAILED"; exit 10; }
fi
"$ENV/bin/python" -c "import sys,torch,deepmd;print(sys.prefix);print('torch',torch.__version__,torch.version.cuda);print(deepmd.__file__)" || exit 11

stage "A1 build deps"
"$ENV/bin/python" -m pip install -q "scikit-build-core>=1" setuptools_scm dependency_groups packaging ninja hatch-fancy-pypi-readme || exit 12

stage "A2 source -> $SRC"
mkdir -p "$HOME/repo"
if [ ! -d "$SRC/.git" ]; then
  git clone --branch v3.2.0 --depth 1 https://github.com/deepmodeling/deepmd-kit.git "$SRC" || exit 13
fi
cd "$SRC" && echo "dpa4c compress sources: $(ls source/op/pt | grep -c dpa4c_graph_compress)"

stage "A3 build (sm_$ARCH, $JOBS jobs)"
export PATH="$ENV/bin:$PATH"
export DP_VARIANT=cuda CUDAToolkit_ROOT="$CUDA_HOME" CUDACXX="$CUDA_HOME/bin/nvcc" \
       CC=/usr/bin/gcc CXX=/usr/bin/g++ \
       DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 \
       CMAKE_BUILD_PARALLEL_LEVEL=$JOBS TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" \
       SETUPTOOLS_SCM_PRETEND_VERSION=3.2.0 SKBUILD_BUILD_VERBOSE=true \
       SKBUILD_CMAKE_DEFINE="CMAKE_CUDA_ARCHITECTURES=$ARCH;DEEPMD_CUDA_PORTABLE_PTX=OFF;CMAKE_DISABLE_FIND_PACKAGE_MPI=ON;CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc;CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++;CMAKE_C_COMPILER=/usr/bin/gcc;CMAKE_CXX_COMPILER=/usr/bin/g++;CMAKE_CUDA_FLAGS=-gencode arch=compute_$ARCH,code=sm_$ARCH -gencode arch=compute_$ARCH,code=compute_$ARCH"
"$ENV/bin/python" -m pip install --no-build-isolation --no-deps -v . 2>&1 | tee "$SRC/build-$CLUSTER.log" \
  | grep -E --line-buffered "CUDA compiler identification|Found Torch|CMake Error|error:|nvcc fatal|dpa4c_graph_compress_c128.cu -o|Successfully|not compatible with"
rc=${PIPESTATUS[0]}; echo "pip install exit=$rc"; [ $rc -eq 0 ] || exit 14

stage "A4 plugin"
"$ENV/bin/python" -m pip install -q --no-deps -e "$PLUGIN" || exit 15

stage "CHECKPOINTS"
cd "$HOME"
SP=$("$ENV/bin/python" -c "import deepmd,os;print(os.path.dirname(deepmd.__file__))")
echo "deepmd at      : $SP"
echo "symbol hits    : $(strings $SP/lib/libdeepmd_op_pt.so | grep -c dpa4c_graph_compress)"
echo "ldd            : $(ldd $SP/lib/libdeepmd_op_pt.so | grep -oE 'libcudart[^ ]*|libcublas[^ ]*|libmpi[^ ]*' | sort -u | tr '\n' ' ')"
echo "SASS archs     : $(cuobjdump --list-elf $SP/lib/libdeepmd_op_pt.so 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
DP_CUDA_INFER=1 "$ENV/bin/python" - <<'PYEOF' 2>&1 | grep -v recommended
import deepmd.pt_expt, torch
from deepmd.kernels.cuda.dpa4c.graph_compress import op_available
print("op_available   :", op_available())
print("torch.ops.deepmd:", [n for n in dir(torch.ops.deepmd) if "dpa4c" in n])
print("cuda visible   :", torch.cuda.is_available())
PYEOF
stage "DONE"
