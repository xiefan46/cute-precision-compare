import os
import torch
from torch.utils.cpp_extension import load
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# def torch_compute_kv(k, v, BLOCK = 64):
#     B, H, N, d = k.shape
#     NUM_BLOCK = (N + BLOCK - 1) // BLOCK
#
#     kv = torch.zeros(d, d).to(torch.float32).to(q.device)
#     kv_output = torch.zeros(NUM_BLOCK, d, d).to(torch.float32).to(q.device)
#
#     for i in range(NUM_BLOCK):
#         si = i * BLOCK
#         ei = min(si + BLOCK, N)
#         ki = k[:, :, si:ei].contiguous().to(torch.float32)
#         vi = v[:, :, si:ei].contiguous().to(torch.float32)
#
#         new_kv = torch.matmul(ki.transpose(-1, -2), vi).to(torch.float32)
#         kv = kv + new_kv
#         kv_output[i] = kv.detach().clone()
#         print(f"data types.ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
#     return kv_output
#
# def torch_compute_amp(k, v, BLOCK = 64):
#
#     assert k.dtype == torch.float16
#     assert v.dtype == torch.float16
#
#     B, H, N, d = k.shape
#     NUM_BLOCK = (N + BLOCK - 1) // BLOCK
#
#     kv = torch.zeros(d, d).to(torch.float32).to(q.device)
#     kv_output = torch.zeros(NUM_BLOCK, d, d).to(torch.float32).to(q.device)
#
#     for i in range(NUM_BLOCK):
#         si = i * BLOCK
#         ei = min(si + BLOCK, N)
#         ki = k[:, :, si:ei].contiguous()
#         vi = v[:, :, si:ei].contiguous()
#         with autocast():
#             new_kv = torch.matmul(ki.transpose(-1, -2), vi)
#         new_kv = new_kv.to(torch.float32)
#         kv = kv + new_kv
#         kv_output[i] = kv.detach().clone()
#         print(f"data types.ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
#     return kv_output




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


def test_kv_match_f16(k, v, myflash):

    def torch_compute_kv_f16(k, v, BLOCK = 64):
        B, H, N, d = k.shape
        NUM_BLOCK = (N + BLOCK - 1) // BLOCK

        kv = torch.zeros(d, d).to(torch.float16).to(q.device)
        kv_output = torch.zeros(NUM_BLOCK, d, d).to(torch.float16).to(q.device)

        for i in range(NUM_BLOCK):
            si = i * BLOCK
            ei = min(si + BLOCK, N)
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()

            new_kv = torch.matmul(ki.transpose(-1, -2), vi)
            kv = kv + new_kv
            kv_output[i] = kv.detach().clone()
            print(f"data types. ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
        return kv_output

    torch_kv_output = torch_compute_kv_f16(k, v)
    cute_kv_output = myflash.cute_compute_kv_F16F16F16F16(k, v).to(torch.float32)

    assert torch_kv_output.dtype == torch.float16
    assert cute_kv_output.dtype == torch.float16

    BLOCK = 64
    B, H, N, d = k.shape
    num_block = (N + BLOCK - 1) // BLOCK

    print(f"test_kv_match. num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")

        print(f"block: {i}, torch_kv_output: {torch_kv_output[i]}, cute_kv_output: {cute_kv_output[i]}")

        # torch.testing.assert_close(
        #     torch_kv_output[i],
        #     cute_kv_output[i],
        #     rtol=1e-3,
        #     atol=1e-5,
        #     msg=f"block : {i}, KV results are different.torch_kv_output: {torch_kv_output[i]}. cute_kv_output: {cute_kv_output[i]}",
        # )

        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
        )

        print(f"✅ block : {i}, kv results match")

    print("✅ kv results match")


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

    test_kv_match_f16(k1, v1, myflash)