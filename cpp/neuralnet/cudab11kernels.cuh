#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Reuse the same RoPE phase across the batch. Rotation arithmetic stays FP32,
// as in the generic kernel; only the load/store is vectorized into half2.
// `pairs` is the number of (x,y) frequency pairs, i.e. midChannels/2, and also
// the number of participating threads (192 for inner384, 256 for inner512).
__global__ void b11BatchSharedRope(half2* q,half2* k,const float* freqs,int batch,int stride2,int pairs) {
  const int xy=blockIdx.x,hp=threadIdx.x;
  if(hp>=pairs)return;
  const float angle=float(xy%19)*freqs[2*hp]+float(xy/19)*freqs[2*hp+1];
  float s,c;__sincosf(angle,&s,&c);
  for(int n=0;n<batch;n++) {
    const size_t offset=(size_t)(n*361+xy)*stride2+hp;
    float2 a=__half22float2(q[offset]);
    float2 b=__half22float2(k[offset]);
    q[offset]=__floats2half2_rn(a.x*c-a.y*s,a.x*s+a.y*c);
    k[offset]=__floats2half2_rn(b.x*c-b.y*s,b.x*s+b.y*c);
  }
}
void launchB11Rope(half* q,half* k,const float* freqs,int batch,int stride,int pairs,cudaStream_t stream) {
  const int threads=pairs<1?1:(pairs>1024?1024:pairs);
  b11BatchSharedRope<<<361,threads,0,stream>>>((half2*)q,(half2*)k,freqs,batch,stride/2,pairs);
}

// Four independent rows per CTA. Unlike the generic norm this needs no
// shared-memory reduction or block-wide barriers.
//
// The original b11c768 kernel below is kept byte-for-byte: inner384 is not a
// multiple of 256, so it uses one uint4 (8 halfs) plus one uint2 (4 halfs) per
// lane. Wider multiples of 256 (inner512, trunk1024) use the templated kernel,
// which walks Channels/256 uint4 chunks with the same arithmetic order.
template<bool Residual, bool Masked>
__global__ void b11RmsNorm384(half* trunk,const half* delta,half* output,
  const half* gamma,const half* mask,int rows,float epsilon) {
  const int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  const float m=Masked?__half2float(mask[row]):1.0f;
  half* x=trunk+(size_t)row*384;
  const half* d=Residual?delta+(size_t)row*384:nullptr;
  half* y=output+(size_t)row*384;
  uint4 xm=reinterpret_cast<const uint4*>(x)[lane];
  uint2 xt=reinterpret_cast<const uint2*>(x+256)[lane];
  uint4 gm=reinterpret_cast<const uint4*>(gamma)[lane];
  uint2 gt=reinterpret_cast<const uint2*>(gamma+256)[lane];
  uint4 dm;uint2 dt;
  if(Residual){dm=reinterpret_cast<const uint4*>(d)[lane];dt=reinterpret_cast<const uint2*>(d+256)[lane];}
  float values[12],sum=0.0f;
  #pragma unroll
  for(int i=0;i<12;i++){
    half& v=i<8?reinterpret_cast<half*>(&xm)[i]:reinterpret_cast<half*>(&xt)[i-8];
    if(Residual){
      const half dv=i<8?reinterpret_cast<const half*>(&dm)[i]:reinterpret_cast<const half*>(&dt)[i-8];
      v=__float2half(__half2float(v)+__half2float(dv)*m);
    }
    values[i]=__half2float(v)*m;
  }
  // Pair the squares before accumulation, as in the generic half2 path.
  #pragma unroll
  for(int i=0;i<6;i++)sum+=values[2*i]*values[2*i]+values[2*i+1]*values[2*i+1];
  if(Residual){reinterpret_cast<uint4*>(x)[lane]=xm;reinterpret_cast<uint2*>(x+256)[lane]=xt;}
  #pragma unroll
  for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,offset);
  const float inv=rsqrtf(sum/384.0f+epsilon);
  uint4 om;uint2 ot;
  #pragma unroll
  for(int i=0;i<12;i++){
    const half g=i<8?reinterpret_cast<const half*>(&gm)[i]:reinterpret_cast<const half*>(&gt)[i-8];
    half& o=i<8?reinterpret_cast<half*>(&om)[i]:reinterpret_cast<half*>(&ot)[i-8];
    o=__float2half(values[i]*inv*__half2float(g)*m);
  }
  reinterpret_cast<uint4*>(y)[lane]=om;reinterpret_cast<uint2*>(y+256)[lane]=ot;
}

// Wide variant for channel counts that are multiples of 256 (512, 1024).
// Each of the 32 lanes owns Channels/32 halfs, read as Channels/256 uint4
// chunks of 256 halfs each. Same arithmetic order as the 384 kernel: residual
// add and mask scaling in FP16, square-pair accumulation in FP32, warp
// butterfly reduction, then a single FP32 rsqrt.
template<int Channels, bool Residual, bool Masked>
__global__ void b11RmsNormWarp(half* trunk,const half* delta,half* output,
  const half* gamma,const half* mask,int rows,float epsilon) {
  static_assert(Channels % 256 == 0,"b11RmsNormWarp requires a multiple of 256 channels");
  constexpr int kChunks=Channels/256;
  constexpr int kPerLane=Channels/32;
  const int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  const float m=Masked?__half2float(mask[row]):1.0f;
  half* x=trunk+(size_t)row*Channels;
  const half* d=Residual?delta+(size_t)row*Channels:nullptr;
  half* y=output+(size_t)row*Channels;
  float values[kPerLane];
  #pragma unroll
  for(int c=0;c<kChunks;c++){
    uint4 xr=reinterpret_cast<const uint4*>(x)[c*32+lane];
    if(Residual){
      const uint4 dr=reinterpret_cast<const uint4*>(d)[c*32+lane];
      #pragma unroll
      for(int i=0;i<8;i++){
        half& v=reinterpret_cast<half*>(&xr)[i];
        v=__float2half(__half2float(v)+__half2float(reinterpret_cast<const half*>(&dr)[i])*m);
      }
      reinterpret_cast<uint4*>(x)[c*32+lane]=xr;
    }
    #pragma unroll
    for(int i=0;i<8;i++)
      values[c*8+i]=__half2float(reinterpret_cast<const half*>(&xr)[i])*m;
  }
  float sum=0.0f;
  #pragma unroll
  for(int i=0;i<kPerLane/2;i++)sum+=values[2*i]*values[2*i]+values[2*i+1]*values[2*i+1];
  #pragma unroll
  for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,offset);
  const float inv=rsqrtf(sum/float(Channels)+epsilon);
  #pragma unroll
  for(int c=0;c<kChunks;c++){
    const uint4 gr=reinterpret_cast<const uint4*>(gamma)[c*32+lane];
    uint4 o;
    #pragma unroll
    for(int i=0;i<8;i++)
      reinterpret_cast<half*>(&o)[i]=__float2half(
        values[c*8+i]*inv*__half2float(reinterpret_cast<const half*>(&gr)[i])*m);
    reinterpret_cast<uint4*>(y)[c*32+lane]=o;
  }
}

template<int Channels>
static void launchRmsNormWide(half* trunk,const half* delta,half* output,const half* gamma,
  const half* mask,int rows,float epsilon,int blocks,cudaStream_t stream) {
  if(delta){
    if(mask)b11RmsNormWarp<Channels,true,true><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
    else b11RmsNormWarp<Channels,true,false><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
  }else{
    if(mask)b11RmsNormWarp<Channels,false,true><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
    else b11RmsNormWarp<Channels,false,false><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
  }
}

void launchB11RmsNorm(half* trunk,const half* delta,half* output,const half* gamma,
  const half* mask,int rows,float epsilon,int channels,cudaStream_t stream){
  const int blocks=(rows+3)/4;
  if(channels==512){launchRmsNormWide<512>(trunk,delta,output,gamma,mask,rows,epsilon,blocks,stream);return;}
  if(channels==1024){launchRmsNormWide<1024>(trunk,delta,output,gamma,mask,rows,epsilon,blocks,stream);return;}
  if(delta){
    if(mask)b11RmsNorm384<true,true><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
    else b11RmsNorm384<true,false><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
  }else{
    if(mask)b11RmsNorm384<false,true><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
    else b11RmsNorm384<false,false><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
  }
}
