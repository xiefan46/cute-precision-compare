import os
import torch
from torch.utils.cpp_extension import load
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler


def set_seed(seed=42):
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == "__main__":

    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


    torch.manual_seed(0)
    # Load the CUDA kernel as a python module
    myflash = load(name='myflash',
                   sources=[
                       'main.cpp',
                       'cute_precision_f16.cu',
                   ],
                   extra_cuda_cflags=[
                       '-O2',
                       '-lcublas',
                       '-lcublasLt',
                       '-std=c++17',
                       '-I/root/cutlass/include',
                       '-I/root/cutlass/tools/util/include',
                   ],
                   )


    set_seed(10086)

    # NOTE: remember to change c++ template if you want to change N and d
    N = 128
    d = 64

    k = torch.randn(N, d).cuda().half()
    v = torch.randn(N, d).cuda().half()

    # torch_kv = torch.matmul(k.transpose(-1, -2), v)


    torch_kv = torch.matmul(k, v.transpose(-1, -2))



    cute_kv = myflash.cute_gemm(k, v)

    assert torch_kv.shape == (N, N)
    assert cute_kv.shape == (N, N)

    torch.testing.assert_close(
        torch_kv,
        cute_kv,
        rtol=1e-3,
        atol=1e-5,
    )