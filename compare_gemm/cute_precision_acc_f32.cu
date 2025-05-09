#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <tuple>

using namespace cute;

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

#define PRINT_TENSOR(name, content) \
    print(name);             \
    print(" : ");            \
    print_tensor(content);          \
    print("\n");


namespace config {
using namespace cute;

template <typename T_, int kHeadDim_ = 64, int N_ = 128>
struct FlashConfig {
  using T = T_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int N = N_;

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaEURepeatM = 1; // 4 -> 1
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 2 -> 1
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using TiledMMA =
      decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
  static constexpr int kThreadNum = size(TiledMMA{});

};

}  // namespace config




template <typename config>
__global__ void gemm_kernel(const half_t* k, const half_t* v, half_t* kv_out)
{
    using namespace cute;
    using TiledMMA = typename config::TiledMMA;

    constexpr int kHeadDim = config::kHeadDim;
    constexpr int N = config::N;

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    Tensor gK = make_tensor(make_gmem_ptr<half_t>(k), make_shape(Int<N>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d
    Tensor gV = make_tensor(make_gmem_ptr<half_t>(v), make_shape(Int<N>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d
    Tensor gKV = make_tensor(make_gmem_ptr<half_t>(kv_out),
                             make_shape(Int<N>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{})); // N x N

    TiledMMA mma;
    ThrMMA thr_mma = mma.get_slice(tx);


    Tensor tAgK = thr_mma.partition_A(gK);
    Tensor tArK = thr_mma.partition_fragment_A(gK);
    Tensor tBgV = thr_mma.partition_B(gV);
    Tensor tBrV = thr_mma.partition_fragment_B(gV);

    cute::copy(tAgK, tArK);
    cute::copy(tBgV, tBrV);

    Tensor tCgKV = thr_mma.partition_C(gKV);
    Tensor tCrKV = thr_mma.partition_fragment_C(gKV);
    clear(tCrKV);

    __syncthreads();
    cute::gemm(mma, tArK, tBrV, tCrKV);
    __syncthreads();
    cute::copy(tCrKV, tCgKV);
}


torch::Tensor cute_gemm_f32_acc(torch::Tensor k, torch::Tensor v) {
    int N = k.size(0);
    int d = k.size(1);

    int BLOCK = 64;
    int num_block = (N + BLOCK - 1) / BLOCK;

    PRINT("num_block", num_block);

    auto kv_out = torch::zeros({N, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::Device(torch::kCUDA, 0)));

    // only for head_dim=64
    config::FlashConfig<cute::half_t> config;
    dim3 block = config.kThreadNum;
    dim3 grid(1);
    auto kernel = gemm_kernel<decltype(config)>;
    PRINT("grid", grid);
    PRINT("block", block);

    kernel<<<grid, block>>>((cute::half_t*) k.data_ptr(), (cute::half_t*)v.data_ptr(), (cute::half_t*)kv_out.data_ptr());
    cudaDeviceSynchronize();

    return kv_out;
}