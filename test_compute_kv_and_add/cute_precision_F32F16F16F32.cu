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

template <typename T_, int kHeadDim_ = 64, int BLOCK_ = 64>
struct FlashConfig {
  using T = T_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int BLOCK = BLOCK_;

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


template<typename Tensor>
__forceinline__ __device__ auto fp32_to_fp16(Tensor& src_fp32) {
    using namespace cute;
    auto dest_fp16 = make_tensor_like<half_t>(src_fp32);
    auto src_fp32x2 = recast<float2>(src_fp32);
    auto dest_fp16x2 = recast<half2>(dest_fp16);
#pragma unroll
    for (int si = 0; si < size(dest_fp16x2); si++) {
        dest_fp16x2(si) = __float22half2_rn(src_fp32x2(si));
    }
    return dest_fp16;
}

template <typename config>
__global__ void compute_kv_kernel_f32_acc(const half_t* k, const half_t* v, float* kv_out, const int B, const int H, const int N)
{
  using namespace cute;
  using TiledMMA = typename config::TiledMMA;


  constexpr int BLOCK = config::BLOCK;
  constexpr int kHeadDim = config::kHeadDim;


  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int bs_head_offset = bx * N * kHeadDim;
  const int num_block = (N + BLOCK - 1) / BLOCK;

  __shared__ float smem_kv[kHeadDim * kHeadDim];
  Tensor Kt = make_tensor(make_gmem_ptr<half_t>(k + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N
  Tensor Vt = make_tensor(make_gmem_ptr<half_t>(v + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N

  Tensor sKV = make_tensor(make_smem_ptr<float>(&smem_kv), make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{},Int<1>{}));

  TiledMMA mma;
  ThrMMA thr_mma = mma.get_slice(tx);

  Tensor tCsKV = thr_mma.partition_C(sKV);
  clear(tCsKV);


  if (thread0()) {
    PRINT("mma size", size(mma));
    PRINT("num_block", num_block);
  }

  for (int block_id = 0; block_id < num_block; block_id++) {

    Tensor gKt = local_tile(Kt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); // d x BLOCK
    Tensor gVt = local_tile(Vt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); //d x BLOCK

    Tensor tAgKt = thr_mma.partition_A(gKt);
    Tensor tArKt = thr_mma.partition_fragment_A(gKt);
    Tensor tBgVt = thr_mma.partition_B(gVt);
    Tensor tBrVt = thr_mma.partition_fragment_B(gVt);

    cute::copy(tAgKt, tArKt);
    cute::copy(tBgVt, tBrVt);

    Tensor tCrNewKV = thr_mma.partition_fragment_C(sKV);
    clear(tCrNewKV);
    cute::gemm(mma, tArKt, tBrVt, tCrNewKV);


    tCrNewKV_f16 = fp32_to_fp16(tCrNewKV);

    float one = 1.0f;

    cute::axpby(one, tCrNewKV_f16, one, tCsKV);

    __syncthreads();

    Tensor gKV = make_tensor(make_gmem_ptr<float>(kv_out + block_id * kHeadDim * kHeadDim),
                             make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // d x d

    Tensor tCgKV = thr_mma.partition_C(gKV);
    // copy kv result to global
    cute::copy(tCsKV, tCgKV);

    __syncthreads();
  }

}


torch::Tensor cute_compute_kv_F32F16F16F32(torch::Tensor k, torch::Tensor v) {
    int B = k.size(0);
    int H = k.size(1);
    int N = k.size(2);
    int d = k.size(3);

    int BLOCK = 64;
    int num_block = (N + BLOCK - 1) / BLOCK;

    PRINT("num_block", num_block);

    auto kv_out = torch::zeros({num_block, d, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, 0)));

    // only for head_dim=64
    config::FlashConfig<cute::half_t> config;
    dim3 block = config.kThreadNum;
    dim3 grid(B * H);
    auto partition_kernel = compute_kv_kernel_f32_acc<decltype(config)>;
    PRINT("grid", grid);
    PRINT("block", block);

    partition_kernel<<<grid, block>>>((cute::half_t*)k.data_ptr(), (cute::half_t*)v.data_ptr(), (float*)kv_out.data_ptr(), B, H, N);
    cudaDeviceSynchronize();

    return kv_out;
}