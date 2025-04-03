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


def test_kv_match_cute_f16_torch_f16(k, v, myflash):
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16
    BLOCK = 64
    B, H, N, d = k.shape
    num_block = (N + BLOCK - 1) // BLOCK

    def torch_compute_kv_f16():

        kv = torch.zeros(d, d).to(torch.float16).to(q.device)
        kv_output = torch.zeros(num_block, d, d).to(torch.float16).to(q.device)

        for i in range(num_block):
            si = i * BLOCK
            ei = min(si + BLOCK, N)
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()

            new_kv = torch.matmul(ki.transpose(-1, -2), vi)
            kv = kv + new_kv
            kv_output[i] = kv.detach().clone()
            # print(f"data types. ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
        return kv_output

    torch_kv_output = torch_compute_kv_f16()
    cute_kv_output = myflash.cute_compute_kv_F16F16F16F16(k, v)

    assert torch_kv_output.dtype == torch.float16
    assert cute_kv_output.dtype == torch.float16

    print(f"test_kv_match. num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")

        print(f"block: {i}, torch_kv_output: {torch_kv_output[i]}, cute_kv_output: {cute_kv_output[i]}")

        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
            rtol=1e-3,
            atol=1e-5,
        )

        print(f"✅ block : {i}, kv results match")

    print("✅ ✅  kv results match")


def test_kv_match_cute_F32F16F16F32_torch_f16(k, v, myflash):
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16
    BLOCK = 64
    B, H, N, d = k.shape
    num_block = (N + BLOCK - 1) // BLOCK


    def torch_compute_kv_f32_acc():

        kv = torch.zeros(d, d).to(torch.float32).to(q.device)
        kv_output = torch.zeros(num_block, d, d).to(torch.float32).to(q.device)

        for i in range(num_block):
            si = i * BLOCK
            ei = min(si + BLOCK, N)
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()

            new_kv = torch.matmul(ki.transpose(-1, -2), vi).to(torch.float32)
            kv = kv + new_kv
            kv_output[i] = kv.detach().clone()
            # print(f"data types. ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
        return kv_output

    torch_kv_output = torch_compute_kv_f32_acc()
    cute_kv_output = myflash.cute_compute_kv_F32F16F16F32(k, v)

    assert torch_kv_output.dtype == torch.float32
    assert cute_kv_output.dtype == torch.float32

    print(f"test_kv_match. num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")

        print(f"block: {i}, torch_kv_output: {torch_kv_output[i]}, cute_kv_output: {cute_kv_output[i]}")

        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
            rtol=1e-3,
            atol=1e-5,
        )

        print(f"✅ block : {i}, kv results match")

    print("✅ ✅  kv results match")


def test_kv_match_cute_F32F16F16F32_torch_f32(k, v, myflash):
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16
    BLOCK = 64
    B, H, N, d = k.shape
    num_block = (N + BLOCK - 1) // BLOCK

    def torch_compute_kv_all_f32():

        kv = torch.zeros(d, d).to(torch.float32).to(q.device)
        kv_output = torch.zeros(num_block, d, d).to(torch.float32).to(q.device)

        for i in range(num_block):
            si = i * BLOCK
            ei = min(si + BLOCK, N)
            ki = k[:, :, si:ei].contiguous().to(torch.float32)
            vi = v[:, :, si:ei].contiguous().to(torch.float32)

            new_kv = torch.matmul(ki.transpose(-1, -2), vi).to(torch.float32)
            kv = kv + new_kv
            kv_output[i] = kv.detach().clone()
            # print(f"data types. ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
        return kv_output

    torch_kv_output = torch_compute_kv_all_f32()
    cute_kv_output = myflash.cute_compute_kv_F32F16F16F32(k, v)

    assert torch_kv_output.dtype == torch.float32
    assert cute_kv_output.dtype == torch.float32

    print(f"test_kv_match. num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")

        print(f"block: {i}, torch_kv_output: {torch_kv_output[i]}, cute_kv_output: {cute_kv_output[i]}")

        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
            rtol=1e-3,
            atol=1e-5,
        )

        print(f"✅ block : {i}, kv results match")

    print("✅ ✅  kv results match")


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
                       'cute_precision_F32F16F16F32.cu'
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
    B = 1
    H = 1
    N = 512
    # NOTE: we only support d = 64!
    d = 64


    q = torch.randn(B, N, H, d).cuda().half()
    k = torch.randn(B, N, H, d).cuda().half()
    v = torch.randn(B, N, H, d).cuda().half()
    q1 = q.transpose(1, 2).contiguous()
    k1 = k.transpose(1, 2).contiguous()
    v1 = v.transpose(1, 2).contiguous()

    # This test only support B=1 and H=1 and d==64
    assert B == 1 and H == 1 and d == 64


    # case1: cute F16F16F16F16, torch all F16, not match！
    # test_kv_match_cute_f16_torch_f16(k1, v1, myflash)

    # case2 cute F32F16F16F32, torch F16 matmul, then convert to F32 to accumulate，not match!
    # test_kv_match_cute_F32F16F16F32_torch_f16(k1, v1, myflash)


    # case3 cute F32F16F16F32, torch all f32, match
    test_kv_match_cute_F32F16F16F32_torch_f32(k1, v1, myflash)
