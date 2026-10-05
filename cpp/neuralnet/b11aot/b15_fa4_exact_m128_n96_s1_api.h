#pragma once
#include <cuda_runtime_api.h>
// Common engine ABI for the b15 ffn-pruned geometry (packed QKV row 1536).
// Schema 2 static bundles implement it through static_fa4_dispatch.cpp.
extern "C" cudaError_t b15_fa4_exact_m128_n96_s1_prepare();
extern "C" cudaError_t b15_fa4_exact_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packedQKV,cudaStream_t stream);
