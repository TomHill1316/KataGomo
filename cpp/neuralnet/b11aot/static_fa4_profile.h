#pragma once
// The engine must also gate the model, device and S361/H*/D32 geometry.
// `qkvStride` is the packed combined-QKV row width, i.e. 3*innerChannels:
//   1152 = tf3-b11c768      (H12 D32, output row 384)
//   1536 = b15 ffn-pruned   (H16 D32, output row 512)
// Which of those actually have a compiled-in static bundle is decided by the
// per-bundle build defines, so a build carrying only one of them fails closed
// instead of asking the linker for symbols it does not have.
namespace b11aot {

// Batch and layout half of the eligibility test. Geometry-independent.
constexpr bool supportsStaticFa4(int batch, bool packedQKV) {
  return packedQKV && (batch == 13 || batch == 16);
}

// True when a static bundle for this packed-QKV row width was compiled in.
constexpr bool staticFa4GeometryCompiled(int qkvStride) {
  bool compiled = false;
#if defined(USE_B11_STATIC_FA4)
  compiled = compiled || qkvStride == 1152;
#endif
#if defined(USE_B15_STATIC_FA4)
  compiled = compiled || qkvStride == 1536;
#endif
  (void)qkvStride;
  return compiled;
}

// Full compile-time eligibility: batch, packed layout, and a matching bundle.
constexpr bool supportsStaticFa4Geometry(int batch, bool packedQKV, int qkvStride) {
  return supportsStaticFa4(batch, packedQKV) && staticFa4GeometryCompiled(qkvStride);
}

} // namespace b11aot
