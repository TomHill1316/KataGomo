#ifndef NEURALNET_CUDAB11_H_
#define NEURALNET_CUDAB11_H_
#include "cudab11profile.h"
#include "cudaincludes.h"
#include "cudab11affine.h"
// Shape-specialized profiles and the geometry-dispatched AOT attention entry
// points. Included when any FA4 artifact set is compiled in.
#ifdef USE_B11_FA4
#include "b11aot/static_fa4_profile.h"
#endif
#ifdef USE_B11_AOT_STATIC
#include "b11aot/static_fa4_dispatch.h"
#endif

void launchB11RmsNorm(half* trunk,const half* delta,half* output,const half* gamma,
  const half* mask,int rows,float epsilon,int channels,cudaStream_t stream);
// `pairs` is the number of (x,y) frequency pairs, i.e. numKVHeads*headDim/2.
void launchB11Rope(half* q,half* k,const float* freqs,int batch,int stride,int pairs,cudaStream_t stream);
// Schema-1 dynamic-B artifacts expose these symbols directly. A build with a
// schema-2 static bundle routes through static_fa4_dispatch.h instead.
#if defined(USE_B11_FA4) && !defined(USE_B11_AOT_STATIC)
#include "b11_fa4_exact_m128_n96_s1_api.h"
#endif
#if defined(USE_B11_MASKED_FA4) && !defined(USE_B11_AOT_STATIC)
#include "b11_fa4_mask_m128_n96_s1_api.h"
#endif
#endif
