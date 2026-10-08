#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                   \
  do {                                                                \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
      fprintf(stderr, "%s\n", cudaGetErrorString(err));               \
    }                                                                 \
  } while (0)

namespace kernels {

// Adds two vectors element by element.
__global__ void add(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

__device__ float square(float x) { return x * x; }

template <typename T>
__global__ void scale(T* data, T factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= factor;
}

}  // namespace kernels

int main() {
  const int n = 1 << 20;
  float *a, *b, *out;
  CHECK(cudaMallocManaged(&a, n * sizeof(float)));
  CHECK(cudaMallocManaged(&b, n * sizeof(float)));
  CHECK(cudaMallocManaged(&out, n * sizeof(float)));
  kernels::add<<<(n + 255) / 256, 256>>>(a, b, out, n);
  CHECK(cudaDeviceSynchronize());
  cudaFree(a);
  cudaFree(b);
  cudaFree(out);
  return 0;
}
