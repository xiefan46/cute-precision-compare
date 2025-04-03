#include <torch/extension.h>
#include <tuple>


torch::Tensor cute_gemm(torch::Tensor k, torch::Tensor v);
torch::Tensor cute_gemm_f32_acc(torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cute_gemm", torch::wrap_pybind_function(cute_gemm), "cute_gemm");
  m.def("cute_gemm_f32_acc", torch::wrap_pybind_function(cute_gemm_f32_acc), "cute_gemm_f32_acc");
}