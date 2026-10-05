#pragma once
#include <cuda_runtime_api.h>

// Geometry-dispatched entry points over the compiled-in schema-2 FA4 static
// bundles. Exactly one bundle exists per shape-specialization profile; the
// engine selects by the packed combined-QKV row width it is about to feed in.
//
// No device queries, allocation, locks or setup happen inside these calls: the
// caller runs prepareStaticFa4 once per handle during warmup, before timing or
// graph capture, and then only launchStaticFa4 on the hot path.
namespace b11aot {

// True when this build carries a static bundle able to serve `qkvStride`.
// Mirrors static_fa4_profile.h so callers can pick the fast path up front.
bool hasStaticFa4For(int qkvStride);

// Configures kernel attributes for the current device. Returns
// cudaErrorInvalidValue when `qkvStride` has no compiled-in bundle.
cudaError_t prepareStaticFa4(int qkvStride, bool masked);

// masked == false requires mask == nullptr; masked == true requires a nonnull
// FP16 [batch,361] key mask. Returns cudaErrorInvalidValue when `qkvStride` has
// no compiled-in bundle, or when the batch is not the bundle's fixed capacity.
cudaError_t launchStaticFa4(void* q, void* k, void* v, void* out, void* mask,
  int batch, float scale, int qkvStride, bool masked, cudaStream_t stream);

} // namespace b11aot
