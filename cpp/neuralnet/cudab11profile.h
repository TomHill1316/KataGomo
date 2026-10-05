#ifndef NEURALNET_CUDAB11PROFILE_H_
#define NEURALNET_CUDAB11PROFILE_H_

// Pure dispatch predicates. Deliberately independent of CUDA headers/runtime so
// the profile and synthetic device cases can be tested on CPU-only machines.
#include "desc.h"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Shape-specialized transformer profiles.
//
// Every kernel of the RTX5090 specialization is compiled for one concrete
// geometry. Two geometries are recognized. Both share model version 17,
// 22/19 input channels, no meta encoder, sequence 361 (19x19), head dim 32,
// learned RoPE, SwiGLU FFN, and the 6-nested-block (3 attention + 3 FFN)
// nested bottleneck layout:
//
//   * tf3-b11c768     : 11 blocks, trunk 768, inner 384, FFN 384->1152->384, H12
//   * b15 ffn-pruned  : 15 blocks, trunk1024, inner 512, H16 D32, and an FFN
//                       512->w->512 whose hidden width w is pruned *per block*:
//                       45 distinct widths from 16 to 1160, each a multiple of 8.
//
// The b15 case is why the FFN width is stored as an envelope plus a uniformity
// flag rather than as a single constant. The attention, trunk and inner
// projections are uniform; only the FFN hidden width varies, and the GEMM
// dispatch is width-generic (CUTLASS validates alignment itself), so no extra
// kernel instantiation is needed per width.
//
// Kernels are instantiated once per geometry (see the dispatch tables in
// cudab11gemm.cu / cudab11kernels.cuh / cudab11qkvrope.cu / cudab11affine.cuh
// / cudab11ffn.cu). Anything outside those two geometries falls back to the
// original generic backend; there is no partial activation.
//
// Do not inspect weight-vector sizes: callers may retain the descriptor after
// releasing its weights. The model loader remains responsible for validating
// disk data.
// ---------------------------------------------------------------------------
struct B11SpecProfile {
  int trunkChannels = 0;
  int midChannels = 0;
  int numHeads = 0;
  int numKVHeads = 0;
  int headDim = 0;
  int numBlocks = 0;
  // FFN hidden-width envelope over every FFN block in the model.
  int ffnMinChannels = 0;
  int ffnMaxChannels = 0;
  bool ffnUniform = false;

  bool valid() const { return trunkChannels > 0 && midChannels > 0 && ffnMaxChannels > 0; }
  // Combined packed QKV row width: q|k|v concatenated per input channel.
  int qkvN() const { return 3 * midChannels; }
  // Number of (x,y) frequency pairs in the learned-RoPE table.
  int ropePairs() const { return numKVHeads * (headDim / 2); }
  // The single FFN width when the model uses one everywhere, else 0.
  int ffnChannels() const { return ffnUniform ? ffnMaxChannels : 0; }
  // True when a launch-time FFN width belongs to this profile.
  bool acceptsFfnWidth(int channels) const {
    return channels >= ffnMinChannels && channels <= ffnMaxChannels && (channels % 8) == 0;
  }
};

namespace b11profile_detail {
struct SpecExpectation {
  int trunkChannels, midChannels, numBlocks;
  int numHeads, numKVHeads, headDim, ropeNumPairs;
  // 0 accepts any 8-aligned per-block FFN hidden width (b15 ffn-pruned).
  int uniformFfnChannels;
};

// Walks the descriptor and reports the FFN envelope through ffnMin/ffnMax/
// ffnUniform. Returns false on the first structural mismatch.
inline bool matchesSpecGeometry(const ModelDesc& d, const SpecExpectation& e,
  int& ffnMin, int& ffnMax, bool& ffnUniform) {
  ffnMin = INT_MAX;
  ffnMax = 0;
  ffnUniform = true;
  if(d.modelVersion != 17 || d.numInputChannels != 22 || d.numInputGlobalChannels != 19 ||
     d.numInputMetaChannels != 0 || d.trunk.trunkNumChannels != e.trunkChannels ||
     (int)d.trunk.blocks.size() != e.numBlocks || d.trunk.numBlocks != e.numBlocks)
    return false;
  const auto pointwise = [](const ConvLayerDesc& c) {
    return c.convXSize == 1 && c.convYSize == 1 && c.dilationX == 1 && c.dilationY == 1;
  };
  int ffnSeen = 0;
  for(const auto& outer : d.trunk.blocks) {
    if(outer.first != NESTED_BOTTLENECK_BLOCK_KIND || outer.second == nullptr) return false;
    const auto& n = *static_cast<const NestedBottleneckResidualBlockDesc*>(outer.second.get());
    if(n.numBlocks != 6 || n.blocks.size() != 6 ||
       n.preBN.numChannels != e.trunkChannels || n.postBN.numChannels != e.midChannels ||
       n.preConv.inChannels != e.trunkChannels || n.preConv.outChannels != e.midChannels ||
       n.postConv.inChannels != e.midChannels || n.postConv.outChannels != e.trunkChannels ||
       !pointwise(n.preConv) || !pointwise(n.postConv)) return false;
    for(std::size_t i = 0; i < n.blocks.size(); i++) {
      const auto& item = n.blocks[i];
      if(item.second == nullptr) return false;
      if(i % 2 == 0) {
        if(item.first != TRANSFORMER_ATTENTION_BLOCK_KIND) return false;
        const auto& a = *static_cast<const TransformerAttentionDesc*>(item.second.get());
        if(a.numHeads != e.numHeads || a.numKVHeads != e.numKVHeads ||
           a.qHeadDim != e.headDim || a.vHeadDim != e.headDim ||
           !a.useRope || !a.learnableRope ||
           a.ropeNumKVHeads != e.numKVHeads || a.ropeNumPairs != e.ropeNumPairs ||
           a.preLN.numChannels != e.midChannels ||
           a.qProj.inChannels != e.midChannels || a.qProj.outChannels != e.midChannels ||
           a.kProj.inChannels != e.midChannels || a.kProj.outChannels != e.midChannels ||
           a.vProj.inChannels != e.midChannels || a.vProj.outChannels != e.midChannels ||
           a.outProj.inChannels != e.midChannels || a.outProj.outChannels != e.midChannels) return false;
      }
      else {
        if(item.first != TRANSFORMER_FFN_BLOCK_KIND) return false;
        const auto& f = *static_cast<const TransformerFFNDesc*>(item.second.get());
        // Every specialized GEMM requires CUTLASS's 8-half operand alignment, so
        // a hidden width that is not a multiple of 8 cannot be specialized.
        if(f.ffnChannels <= 0 || (f.ffnChannels % 8) != 0) return false;
        if(e.uniformFfnChannels != 0 && f.ffnChannels != e.uniformFfnChannels) return false;
        if(f.numChannels != e.midChannels || !f.useSwiGLU ||
           f.preLN.numChannels != e.midChannels ||
           f.linear1.inChannels != e.midChannels || f.linear1.outChannels != f.ffnChannels ||
           f.linearGate.inChannels != e.midChannels || f.linearGate.outChannels != f.ffnChannels ||
           f.linear2.inChannels != f.ffnChannels || f.linear2.outChannels != e.midChannels) return false;
        ffnMin = std::min(ffnMin, f.ffnChannels);
        ffnMax = std::max(ffnMax, f.ffnChannels);
        ffnSeen++;
      }
    }
  }
  if(ffnSeen == 0) return false;
  ffnUniform = (ffnMin == ffnMax);
  return true;
}
} // namespace b11profile_detail

inline B11SpecProfile detectB11SpecProfile(const ModelDesc& d) {
  static constexpr b11profile_detail::SpecExpectation kProfiles[] = {
    // tf3-b11c768      : trunk 768, inner 384, uniform FFN 384->1152->384, H12 D32, 11 blocks
    {768, 384, 11, 12, 12, 32, 16, 1152},
    // b15 ffn-pruned   : trunk1024, inner 512, per-block pruned FFN 512->w->512, H16 D32, 15 blocks
    {1024, 512, 15, 16, 16, 32, 16, 0},
  };
  for(const auto& e : kProfiles) {
    int ffnMin = 0;
    int ffnMax = 0;
    bool ffnUniform = false;
    if(!b11profile_detail::matchesSpecGeometry(d, e, ffnMin, ffnMax, ffnUniform))
      continue;
    B11SpecProfile profile;
    profile.trunkChannels = e.trunkChannels;
    profile.midChannels = e.midChannels;
    profile.numHeads = e.numHeads;
    profile.numKVHeads = e.numKVHeads;
    profile.headDim = e.headDim;
    profile.numBlocks = e.numBlocks;
    profile.ffnMinChannels = ffnMin;
    profile.ffnMaxChannels = ffnMax;
    profile.ffnUniform = ffnUniform;
    return profile;
  }
  return B11SpecProfile{};
}

// Kernels are instantiated only for the geometries listed above. This is a
// second, cheap guard so that adding a profile above without adding the matching
// kernel instantiations fails closed instead of launching a wrong-geometry kernel.
// It mirrors the dispatch tables verbatim:
//   GEMM   (N,K): trunk/mid/inner/FFN projections, any 8-aligned width
//   norm         : 384 (mixed-width), 512 and 1024 (multiple-of-256 wide)
//   affine       : 384, 512, 768, 1024
//   QKV-RoPE     : 384, 512
//   FA4 AOT      : packed QKV row 1152 (H12) or 1536 (H16), head dim 32
inline bool isB11SpecGeometryInstanced(const B11SpecProfile& p) {
  if(!p.valid() || p.headDim != 32) return false;
  // Every projection is square in the head dimension, so the packed QKV row is
  // exactly three times the inner width.
  if(p.numHeads * p.headDim != p.midChannels) return false;
  if(p.numKVHeads * p.headDim != p.midChannels) return false;
  if(p.qkvN() != 3 * p.midChannels) return false;
  if(p.ffnMinChannels <= 0 || p.ffnMaxChannels < p.ffnMinChannels) return false;
  if((p.ffnMinChannels % 8) != 0 || (p.ffnMaxChannels % 8) != 0) return false;
  if(p.numBlocks == 11 && p.trunkChannels == 768 && p.midChannels == 384)
    return p.ffnUniform && p.ffnMaxChannels == 1152;
  if(p.numBlocks == 15 && p.trunkChannels == 1024 && p.midChannels == 512)
    return p.ffnMaxChannels <= 4096;
  return false;
}

// Structural profile, independent of checkpoint name and weight values.
inline bool isOfficialB11C768(const ModelDesc& d) {
  return detectB11SpecProfile(d).trunkChannels == 768;
}

inline bool isB11CudaHardwareLayoutEligible(const char* deviceName, int major, int minor,
  bool useFP16, bool useNHWC, int nnXLen, int nnYLen) {
  return deviceName != nullptr && std::strcmp(deviceName,"NVIDIA GeForce RTX 5090") == 0 &&
    major == 12 && minor == 0 && useFP16 && useNHWC && nnXLen == 19 && nnYLen == 19;
}

// The supported set is an explicit caller decision, not an implicit profile
// constant. This permits benchmark screening followed by a narrower final set.
// These predicates do not replace an actualBatch == capacity check when a
// particular launch path requires that invariant.
inline bool isB11SupportedBatch(int batch, int inclusiveMin, int inclusiveMax) {
  return inclusiveMin >= 1 && inclusiveMax >= inclusiveMin &&
    batch >= inclusiveMin && batch <= inclusiveMax;
}

inline bool isB11SupportedBatch(int batch, const int* supported, std::size_t count) {
  if(batch < 1 || supported == nullptr || count == 0) return false;
  for(std::size_t i = 0; i < count; i++)
    if(batch == supported[i]) return true;
  return false;
}

template<std::size_t N>
inline bool isB11SupportedBatch(int batch, const int (&supported)[N]) {
  return isB11SupportedBatch(batch,supported,N);
}

// Kept deliberately narrow: the two strongest measured throughput profiles.
// New batch sizes need numerical and performance validation before adding here.
inline bool isB11OptimizedBatch(int batch) {
  constexpr int supported[] = {13,16};
  return isB11SupportedBatch(batch,supported);
}

// Widths with an instantiated B11 RMSNorm kernel (see launchB11RmsNorm).
inline bool isB11RmsNormWidth(int channels) {
  return channels == 384 || channels == 512 || channels == 1024;
}

// Widths with an instantiated B11 flat affine+SiLU kernel.
inline bool isB11AffineWidth(int channels) {
  return channels == 384 || channels == 512 || channels == 768 || channels == 1024;
}

// A prepared handle still has to reject a partial batch at every FFN launch.
// Do not switch its packed weight layout: fallback uses the original fused FFN.
// The FFN hidden width is checked against the profile envelope, since b15
// prunes it per block and therefore has no single value to compare against.
inline bool isB11TanhFFNLaunchEligible(bool prepared,int actualBatch,int capacity,
  int nnXLen,int nnYLen,int channels,int ffnChannels,const B11SpecProfile& profile,
  bool useFP16,bool useNHWC) {
  return prepared && profile.valid() && isB11OptimizedBatch(capacity) &&
    actualBatch == capacity && nnXLen == 19 && nnYLen == 19 &&
    channels == profile.midChannels && profile.acceptsFfnWidth(ffnChannels) &&
    useFP16 && useNHWC;
}

// A complete planned mapping, not createComputeContext's deduplicated device
// inventory. Match the CUDA backend's explicit rule that -1 selects GPU 0.
// Zero means unknown/invalid/not used and must never be replaced by a guessed
// stream count. No CUDA calls or inspection of unrelated devices is needed.
inline int getB11ExpectedConcurrentGpuThreads(const std::vector<int>& mapping, int actualDevice) {
  if(actualDevice < 0 || mapping.empty() || mapping.size() > static_cast<std::size_t>(INT_MAX))
    return 0;
  int count = 0;
  for(int requestedDevice : mapping) {
    if(requestedDevice < -1) return 0;
    const int resolvedDevice = requestedDevice == -1 ? 0 : requestedDevice;
    if(resolvedDevice == actualDevice) count++;
  }
  return count;
}

// Only measured per-GPU concurrency profiles enable the automatic cache policy.
// The complete model/device/layout gate and actualBatch == capacity check are
// separate requirements. Deliberately unaffected by research batch expansion.
inline bool isB11AutoL2Eligible(int batch, int expectedConcurrentGpuThreads) {
  return (batch == 13 || batch == 16) && expectedConcurrentGpuThreads == 3;
}

#endif
