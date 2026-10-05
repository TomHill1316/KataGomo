// Optional schema-2 adapter. No device queries, setup, locks or allocation at launch.
// Compiled only when at least one static bundle is present, so the per-geometry
// blocks below are the only definitions of these symbols in such a build.
#include "static_fa4_profile.h"
#include "static_fa4_dispatch.h"

#if defined(USE_B11_STATIC_FA4)
#include "b11_fa4_static_b13_exact_m128_n96_s1_api.h"
#include "b11_fa4_static_b13_mask_m128_n96_s1_api.h"
#include "b11_fa4_static_b16_exact_m128_n96_s1_api.h"
#include "b11_fa4_static_b16_mask_m128_n96_s1_api.h"
#endif
#if defined(USE_B15_STATIC_FA4)
#include "b15_fa4_static_b13_exact_m128_n96_s1_api.h"
#include "b15_fa4_static_b13_mask_m128_n96_s1_api.h"
#include "b15_fa4_static_b16_exact_m128_n96_s1_api.h"
#include "b15_fa4_static_b16_mask_m128_n96_s1_api.h"
#endif

#if defined(USE_B11_STATIC_FA4)
// b11c768, packed row 1152. Prepared per device, so both fixed batches are
// brought up before the first timed launch.
extern "C" cudaError_t b11_fa4_exact_m128_n96_s1_prepare() {
  const auto status=b11_fa4_static_b13_exact_m128_n96_s1_prepare();
  return status==cudaSuccess?b11_fa4_static_b16_exact_m128_n96_s1_prepare():status;
}
extern "C" cudaError_t b11_fa4_mask_m128_n96_s1_prepare() {
  const auto status=b11_fa4_static_b13_mask_m128_n96_s1_prepare();
  return status==cudaSuccess?b11_fa4_static_b16_mask_m128_n96_s1_prepare():status;
}
extern "C" cudaError_t b11_fa4_exact_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packed,cudaStream_t stream) {
  if(!b11aot::supportsStaticFa4(batch,packed)) return cudaErrorInvalidValue;
  if(batch==13) return b11_fa4_static_b13_exact_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
  return b11_fa4_static_b16_exact_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
}
extern "C" cudaError_t b11_fa4_mask_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packed,cudaStream_t stream) {
  if(!b11aot::supportsStaticFa4(batch,packed)) return cudaErrorInvalidValue;
  if(batch==13) return b11_fa4_static_b13_mask_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
  return b11_fa4_static_b16_mask_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
}
#endif

#if defined(USE_B15_STATIC_FA4)
// b15 ffn-pruned, packed row 1536. Same contract as the b11c768 set.
extern "C" cudaError_t b15_fa4_exact_m128_n96_s1_prepare() {
  const auto status=b15_fa4_static_b13_exact_m128_n96_s1_prepare();
  return status==cudaSuccess?b15_fa4_static_b16_exact_m128_n96_s1_prepare():status;
}
extern "C" cudaError_t b15_fa4_mask_m128_n96_s1_prepare() {
  const auto status=b15_fa4_static_b13_mask_m128_n96_s1_prepare();
  return status==cudaSuccess?b15_fa4_static_b16_mask_m128_n96_s1_prepare():status;
}
extern "C" cudaError_t b15_fa4_exact_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packed,cudaStream_t stream) {
  if(!b11aot::supportsStaticFa4(batch,packed)) return cudaErrorInvalidValue;
  if(batch==13) return b15_fa4_static_b13_exact_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
  return b15_fa4_static_b16_exact_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
}
extern "C" cudaError_t b15_fa4_mask_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packed,cudaStream_t stream) {
  if(!b11aot::supportsStaticFa4(batch,packed)) return cudaErrorInvalidValue;
  if(batch==13) return b15_fa4_static_b13_mask_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
  return b15_fa4_static_b16_mask_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
}
#endif

namespace {

using StaticLaunch = cudaError_t (*)(void*,void*,void*,void*,void*,int,float,bool,cudaStream_t);

// One geometry's four artifacts behind a uniform signature. The generated
// launchers take (..., int batch, float scale, bool packedQKV, cudaStream_t);
// every static artifact is packed-only, so the flag is always true here and is
// still re-checked by the generated bridge.
struct StaticGeometry {
  int qkvStride;
  cudaError_t (*prepareExact)();
  cudaError_t (*prepareMask)();
  StaticLaunch launchExact;
  StaticLaunch launchMask;
};

const StaticGeometry* findGeometry(int qkvStride) {
  static const StaticGeometry kGeometries[] = {
#if defined(USE_B11_STATIC_FA4)
    {1152,
     &b11_fa4_exact_m128_n96_s1_prepare, &b11_fa4_mask_m128_n96_s1_prepare,
     &b11_fa4_exact_m128_n96_s1_launch, &b11_fa4_mask_m128_n96_s1_launch},
#endif
#if defined(USE_B15_STATIC_FA4)
    {1536,
     &b15_fa4_exact_m128_n96_s1_prepare, &b15_fa4_mask_m128_n96_s1_prepare,
     &b15_fa4_exact_m128_n96_s1_launch, &b15_fa4_mask_m128_n96_s1_launch},
#endif
  };
  for(const StaticGeometry& geometry : kGeometries) {
    if(geometry.qkvStride == qkvStride) return &geometry;
  }
  return nullptr;
}

} // namespace

namespace b11aot {

bool hasStaticFa4For(int qkvStride) {
  return findGeometry(qkvStride) != nullptr;
}

cudaError_t prepareStaticFa4(int qkvStride, bool masked) {
  const StaticGeometry* geometry = findGeometry(qkvStride);
  if(geometry == nullptr) return cudaErrorInvalidValue;
  return masked ? geometry->prepareMask() : geometry->prepareExact();
}

cudaError_t launchStaticFa4(void* q, void* k, void* v, void* out, void* mask,
  int batch, float scale, int qkvStride, bool masked, cudaStream_t stream) {
  const StaticGeometry* geometry = findGeometry(qkvStride);
  if(geometry == nullptr) return cudaErrorInvalidValue;
  if(!supportsStaticFa4(batch,true)) return cudaErrorInvalidValue;
  if(masked && mask == nullptr) return cudaErrorInvalidValue;
  if(!masked && mask != nullptr) return cudaErrorInvalidValue;
  return masked
    ? geometry->launchMask(q,k,v,out,mask,batch,scale,true,stream)
    : geometry->launchExact(q,k,v,out,mask,batch,scale,true,stream);
}

} // namespace b11aot
