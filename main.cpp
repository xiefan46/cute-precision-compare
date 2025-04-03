#include <torch/extension.h>
#include <tuple>


torch::Tensor cute_compute_kv_F16F16F16F16(torch::Tensor k, torch::Tensor v);
torch::Tensor cute_compute_kv_F32F16F16F32(torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cute_compute_kv_F16F16F16F16", torch::wrap_pybind_function(cute_compute_kv_F16F16F16F16), "cute_compute_kv_F16F16F16F16");
  m.def("cute_compute_kv_F32F16F16F32", torch::wrap_pybind_function(cute_compute_kv_F32F16F16F32), "cute_compute_kv_F32F16F16F32");
}