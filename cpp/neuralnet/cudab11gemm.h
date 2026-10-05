#ifndef NEURALNET_CUDAB11GEMM_H_
#define NEURALNET_CUDAB11GEMM_H_

#include "cudaincludes.h"

// Dense, contiguous, row-major FP16 matrices:
//   output[M,N] = input[M,K] * weights[K,N] + (betaIsOne ? output[M,N] : 0).
// Weights MUST already be transposed from the model's output-major layout.
// All pointers require 16-byte alignment. Input/weights may not alias output;
// betaIsOne deliberately uses output itself as the residual source.
// N and K must be multiples of 8 (RowMajor leading dimensions) and at most 4096.
// The supported (N,K) are the profile's projection shapes: trunk<->inner 1x1
// convs, the attention outProj, the combined QKV projection, and the FFN
// down-projection whose K is the per-block pruned hidden width.
// Model/device/board/batch dispatch belongs to the caller. No allocation or
// synchronization is performed; failures are returned as CUDA error codes.
// Accumulation and epilogue arithmetic use FP16, matching the existing fast
// FP16 CUDA path's precision class, but not promising bitwise cuBLAS identity.
cudaError_t launchB11Gemm(
  const half* input, const half* weights, half* output,
  int M, int N, int K, bool betaIsOne, cudaStream_t stream);

#endif
