#ifndef NEURALNET_CUDAB11FFN_H_
#define NEURALNET_CUDAB11FFN_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Optional B11-only output functor, never a vendor/global Sigmoid override.
// The original CudaFusedFFN implementation and out-major weight layout remain
// available for all generic launches, including a partial batch in a B11 handle.
namespace CudaB11FFN {
  // Setup only: verifies the exact RTX5090 device and executes this kernel's
  // own tiny poisoned-output probe. The backend also retains the generic probe.
  bool supportedOnCurrentDevice();
  bool supportsShape(int N, int K);

  // FP16 tensor-core GEMM and FP16 tanh-based SwiGLU; inputs and output use the
  // same layout/alignment as CudaFusedFFN. N is the FFN hidden width and K the
  // inner width; both must be multiples of 8 and at most 4096 (b11c768 uses a
  // single hidden width, b15 prunes it per block). M is 361*(13 or 16) only.
  // Caller must pass the full model/device/layout/profile and actual-batch gates
  // and have prepared successfully. No setup or fallback inside a hot launch.
  void runSwiGLU(const half* A, const half* w1, const half* wGate, half* output,
    int M, int N, int K, cudaStream_t stream);
}

#endif
