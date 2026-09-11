// ==========================================================================
// GT-VBAO: Ground Truth Visibility Bitmask AO + optional GI Color Bleeding
// ==========================================================================
// Based on:
//   - GTAO (Jimenez et al. 2016) for the horizon-based slice framework
//   - SSILVB (Therrien, Levesque, Gilet 2023) for the visibility bitmask
//   - GT-VBAO (Salm 2024) for SliceRelCDF cosine remapping
// SM 3.0 compatible: bitmask emulated via 16 float sectors in 4x float4
// ==========================================================================

#define GTAO_LOW 0
#define GTAO_MEDIUM 1
#define GTAO_HIGH 2
#define GTAO_ULTRA 3

#include "common_ps_fxc.h"
#include "common_ps_packdepth.h"

#define PACKDEPTH

sampler depthSampler  : register(s0);
sampler BumpBuffer    : register(s2);

const float4 vScreenSize : register(c0); // {1/w, 1/h, w, h}
const float4 vSettings   : register(c1); // {NegInvR2, AngleBiasRadians, intensity, R}
const float4 vSettings2  : register(c2); // {R2, texelAspect, MaxRadPx, fogMode}
const float4 vSettings3  : register(c3); // {thicknessRatio, giStrength, tanHalfFovX, tanHalfFovY}

static const float M_PI   = 3.14159265f;
static const float M_PI_2 = 1.57079632f;
static const float RcpPi  = 0.31830989f;

#if GTAO_SETTINGS == GTAO_ULTRA
  static const int NUM_SLICES = 6;
  static const int NUM_STEPS  = 7;
#elif GTAO_SETTINGS == GTAO_HIGH
  static const int NUM_SLICES = 4;
  static const int NUM_STEPS  = 5;
#elif GTAO_SETTINGS == GTAO_MEDIUM
  static const int NUM_SLICES = 3;
  static const int NUM_STEPS  = 5;
#elif GTAO_SETTINGS == GTAO_LOW
  static const int NUM_SLICES = 2;
  static const int NUM_STEPS  = 4;
#endif

#define g_NegInvR2       vSettings.x
#define g_AngleBias      vSettings.y
#define intensivity       vSettings.z
#define g_R              vSettings.w
#define g_R2             vSettings2.x
#define size_wx          vSettings2.y
#define g_MaxRadiusPixels vSettings2.z
#define PIXELFOGTYPE     vSettings2.w
#define g_ThicknessRatio vSettings3.x

static const float INVALID_DEPTH = 0.00025f;
static const float EPSILON       = 1e-5f;

// =========================================================================
// Fast acos() — improved GTAOFastAcos from GT-VBAO reference (mode 2)
// Max abs error: ~0.0016 rad. 5 MAD + 1 sqrt vs ~20 ALU for hardware acos.
// =========================================================================
float ACos_Fast(float x) {
  float ax = abs(x);
  float poly = 1.5707963f + (-0.20491203f + 0.04832927f * ax) * ax;
  float r = poly * sqrt(1.0f - ax);
  return (x >= 0.0f) ? r : M_PI - r;
}

struct PS_INPUT {
  float2 ppos     : VPOS;
  float2 texCoord : TEXCOORD0;
  float3 fogParams: TEXCOORD1;
};

// =========================================================================
// Utilities
// =========================================================================

float InterleavedGradientNoise(float2 pos) {
  float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
  return frac(magic.z * frac(dot(pos, magic.xy)));
}

float GTAOFastMultiBounce(float visibility, float albedo) {
  float a = 2.0404f * albedo - 0.3324f;
  float b = -4.7951f * albedo + 0.6417f;
  float c = 2.7552f * albedo + 0.6903f;
  return max(visibility, ((visibility * a + b) * visibility + c) * visibility);
}

bool IsValidUV(float2 uv) {
  return uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f;
}

bool IsValidDepth(float depth) {
  return depth > INVALID_DEPTH;
}

float FetchDepth(float2 uv) {
#if defined(DOWNSAMPLE)
  return tex2Dlod(depthSampler, float4(uv, 0, 0)).r;
#else
  return tex2Dlod(depthSampler, float4(uv, 0, 0)).a;
#endif
}

float FetchEyeZ(float2 uv) {
  return 1.0f / max(FetchDepth(uv), EPSILON);
}

float3 UVToEye(float2 uv, float eye_z) {
  uv = uv * 2.0f - 1.0f;
  uv.x *= vSettings3.z;
  uv.y *= vSettings3.w;
  return float3(uv * eye_z, eye_z);
}

float3 FetchEyePos(float2 uv) {
  return UVToEye(uv, FetchEyeZ(uv));
}

bool FetchEyePosSafe(float2 uv, out float3 pos, out float depth) {
  pos = 0.0f;
  depth = 0.0f;

  if (!IsValidUV(uv)) {
    return false;
  }

  depth = FetchDepth(uv);
  if (!IsValidDepth(depth)) {
    return false;
  }

  pos = UVToEye(uv, 1.0f / depth);
  return true;
}

float Length2(float3 v) { return dot(v, v); }

float3 MinDiff(float3 P, float3 Pr, float3 Pl) {
  float3 V1 = Pr - P;
  float3 V2 = P - Pl;
  return (Length2(V1) < Length2(V2)) ? V1 : V2;
}

float2 SnapUVOffset(float2 uv) {
  return round(uv * vScreenSize.zw) * vScreenSize.xy;
}

float2 RotateDirections(float2 Dir, float2 CosSin) {
  float4 DirCosSin = float4(Dir * CosSin, Dir * CosSin.yx);
  return float2(DirCosSin.x - DirCosSin.y, DirCosSin.z + DirCosSin.w);
}

void ComputeSteps(inout float2 step_size_uv, inout float numSteps,
                  float ray_radius_pix, float rand, float viewZ) {
  float depth_fade = saturate(viewZ / 5000.0f);
  float adapted_max = lerp((float)NUM_STEPS, max(2.0f, (float)NUM_STEPS - 2.0f), depth_fade);
  numSteps = min(adapted_max, ray_radius_pix);
  float step_size_pix = ray_radius_pix / (numSteps + 1.0f);
  float maxNumSteps = g_MaxRadiusPixels / step_size_pix;
  if (maxNumSteps < numSteps) {
    numSteps = floor(maxNumSteps + rand);
    numSteps = max(numSteps, 1.0f);
    step_size_pix = g_MaxRadiusPixels / numSteps;
  }
  step_size_uv = step_size_pix * vScreenSize.xy;
}

// =========================================================================
// SM 3.0 Visibility Bitmask: 16 sectors in 4x float4
// =========================================================================
// Each sector center is at (i + 0.5) / 16.0 in [0..1] hemisphere space.
// Sector is "hit" when its center falls within [minH, maxH].
//
// GT-VBAO: Instead of weighting the output with cosine weights,
// the input sector boundaries are remapped through SliceRelCDF_Cos,
// which is the CDF of the cosine-weighted hemisphere PDF.
// After this remapping, simple uniform counting (1 - count/16)
// IS ground truth � the cosine weighting is baked into the distribution.
// =========================================================================

#define NUM_SECTORS 24
#define INV_SECTORS (1.0f / 24.0f)

static const float4 kIdx0 = float4( 0.5f,  1.5f,  2.5f,  3.5f);
static const float4 kIdx1 = float4( 4.5f,  5.5f,  6.5f,  7.5f);
static const float4 kIdx2 = float4( 8.5f,  9.5f, 10.5f, 11.5f);
static const float4 kIdx3 = float4(12.5f, 13.5f, 14.5f, 15.5f);
static const float4 kIdx4 = float4(16.5f, 17.5f, 18.5f, 19.5f);
static const float4 kIdx5 = float4(20.5f, 21.5f, 22.5f, 23.5f);

// Mark sectors whose center is in [lo, hi]. max() prevents un-occluding.
void MarkSectors(float minH, float maxH,
                 inout float4 s0, inout float4 s1,
                 inout float4 s2, inout float4 s3,
                 inout float4 s4, inout float4 s5) {
  float center = (minH + maxH) * (NUM_SECTORS * 0.5f);
  float radius = (maxH - minH) * (NUM_SECTORS * 0.5f);
  s0 = max(s0, step(abs(kIdx0 - center), radius));
  s1 = max(s1, step(abs(kIdx1 - center), radius));
  s2 = max(s2, step(abs(kIdx2 - center), radius));
  s3 = max(s3, step(abs(kIdx3 - center), radius));
  s4 = max(s4, step(abs(kIdx4 - center), radius));
  s5 = max(s5, step(abs(kIdx5 - center), radius));
}

// Count occluded sectors (uniform)
float SumSectors(float4 s0, float4 s1, float4 s2, float4 s3, float4 s4, float4 s5) {
  return dot(s0, 1) + dot(s1, 1) + dot(s2, 1) + dot(s3, 1) + dot(s4, 1) + dot(s5, 1);
}

// =========================================================================
// SliceRelCDF_Cos2: Vectorized Cosine-weighted CDF remapping (GT-VBAO)
// =========================================================================
// Maps a linear [0,1] sector position through the analytic CDF of the
// cosine-weighted hemisphere PDF. Vectorized and branchless for SM 3.0 performance.

// Positive direction (d=+1)
float2 SliceRelCDF_Cos2_Pos(float2 x, float angN, float cosN, float sinN, float t1_rcp) {
  float2 phi = x * M_PI - M_PI_2;
  float2 sinPhi, cosPhi;
  sincos(phi, sinPhi, cosPhi);
  float2 cos2phi = 1.0f - 2.0f * sinPhi * sinPhi;
  float2 sin2phi = 2.0f * sinPhi * cosPhi;
  float2 cos_ang_2phi = cosN * cos2phi + sinN * sin2phi;
  float2 t0 = 3.0f * cosN - cos_ang_2phi + (4.0f * angN - 2.0f * phi + M_PI) * sinN;
  float2 res = saturate(t0 * t1_rcp);
  res = lerp(res, 0.0f, step(x, 0.0f));
  res = lerp(res, 1.0f, step(1.0f, x));
  return res;
}

// Negative direction (d=-1)
float2 SliceRelCDF_Cos2_Neg(float2 x, float angN, float cosN, float sinN, float t1_rcp) {
  float2 phi = x * M_PI - M_PI_2;
  float2 sinPhi, cosPhi;
  sincos(phi, sinPhi, cosPhi);
  float2 cos2phi = 1.0f - 2.0f * sinPhi * sinPhi;
  float2 sin2phi = 2.0f * sinPhi * cosPhi;
  float2 cos_ang_2phi = cosN * cos2phi + sinN * sin2phi;
  float2 t0 = 1.0f * cosN + cos_ang_2phi + (0.0f * angN + 2.0f * phi + M_PI) * sinN;
  float2 res = saturate(t0 * t1_rcp);
  res = lerp(res, 0.0f, step(x, 0.0f));
  res = lerp(res, 1.0f, step(1.0f, x));
  return res;
}

// =========================================================================
// GT-VBAO Slice: Visibility Bitmask with CDF remapping + optional GI
// =========================================================================

float ComputeVBAOSlice(
    float2 dir, float2 texCoord, float3 P, float3 viewNormal,
    float3 V, float3 dPdu, float3 dPdv, float numSteps,
    float randstep, float2 step_size_uv)
{
  float2 deltaUV = dir * step_size_uv;

  // Tangent & projected normal (same as classic GTAO)
  float3 T = dPdu * dir.x + dPdv * dir.y;
  T -= V * dot(T, V);

  float tLen2 = dot(T, T);
  if (tLen2 < 1e-8f) {
    return 1.0f;
  }
  T *= rsqrt(tLen2);

  float3 sliceN = normalize(cross(V, T));
  float3 projN = viewNormal - sliceN * dot(viewNormal, sliceN);
  float projLen2 = dot(projN, projN);
  if (projLen2 < 1e-8f) {
    return 1.0f;
  }

  float projNRcpLen = rsqrt(projLen2);
  float proj_len = 1.0f / projNRcpLen;
  projN *= projNRcpLen;

  // Normal angle in the slice
  float cos_n = clamp(dot(projN, V), -1.0f, 1.0f);
  float3 sliceT = cross(sliceN, projN);
  float sgn = (dot(V, sliceT) < 0.0f) ? -1.0f : 1.0f;
  float n_angle = sgn * ACos_Fast(cos_n);

  // Precompute sin(angN) for CDF
  float sinAngN, cosAngN;
  sincos(n_angle, sinAngN, cosAngN);

  float angOff = n_angle * RcpPi + 0.5f;

  // Precompute t1 reciprocal for CDF (constant per slice)
  float t1 = 4.0f * (cosAngN + n_angle * sinAngN);
  float t1_rcp = (abs(t1) > EPSILON) ? (1.0f / t1) : 0.0f;

  // Thickness in view-space units
  float thickness = g_R * g_ThicknessRatio;

  // Initialize sector arrays (all unoccluded)
  float4 sec0 = 0, sec1 = 0, sec2 = 0, sec3 = 0, sec4 = 0, sec5 = 0;

  // March in BOTH directions along the slice.
  for (float j = 1; j <= NUM_STEPS; ++j) {
    if (j > numSteps) {
      break;
    }

    float t = j / max(numSteps, 1.0f);
    float qt = t * (0.5f + 0.5f * t);
    float2 offset_uv = deltaUV * (qt * numSteps);
    float step_jitter = frac(randstep + j * 0.3819660113f);
    float sector_jitter = (frac(randstep * 1.6180339887f + j * 0.754877666f) - 0.5f) * (INV_SECTORS * 0.35f);
    float2 snapped = SnapUVOffset(offset_uv + step_jitter * deltaUV);

    // --- Positive direction ---
    {
      float2 uv = texCoord + snapped;
      float3 S;
      float sampleDepth;
      if (FetchEyePosSafe(uv, S, sampleDepth)) {
      float3 dPos = S - P;
      float dLen2 = dot(dPos, dPos);
      float distAtten = saturate(1.0f - dLen2 / g_R2);

      if (distAtten > 0.0f) {
      float dLen = sqrt(dLen2) + EPSILON;
      float3 dDir = dPos / dLen;

      // Horizon angles: front surface, back = surface shifted by thickness
      float frontCos = dot(dDir, V);
      float frontAngle = ACos_Fast(clamp(frontCos, -1.0f, 1.0f));

      // Approximate: back angle = front angle + angular_thickness
      float angular_thickness = thickness / dLen;
      float backAngle = frontAngle + angular_thickness;

      // Apply angle bias to prevent self-occlusion from coplanar surfaces
      frontAngle = max(frontAngle, g_AngleBias);
      backAngle  = max(backAngle,  g_AngleBias);

      // Map to [0..1] sector space (reference line 752-755):
      // hor01 = acos(cos) * d / PI + angOff
      // For positive direction (d=+1): h * (+1) / PI + angOff
      float2 hor01;
      hor01.x = saturate(frontAngle * RcpPi + angOff);
      hor01.y = saturate(backAngle  * RcpPi + angOff);

      // GT-VBAO: Apply cosine CDF remapping (d > 0 branch)
      hor01 = saturate(SliceRelCDF_Cos2_Pos(hor01, n_angle, cosAngN, sinAngN, t1_rcp) + sector_jitter);

      float sMin = min(hor01.x, hor01.y);
      float sMax = max(hor01.x, hor01.y);



      MarkSectors(sMin, sMax, sec0, sec1, sec2, sec3, sec4, sec5);

      }
      }
    }

    // --- Negative direction ---
    {
      float2 uv = texCoord - snapped;
      float3 S;
      float sampleDepth;
      if (FetchEyePosSafe(uv, S, sampleDepth)) {
      float3 dPos = S - P;
      float dLen2 = dot(dPos, dPos);
      float distAtten = saturate(1.0f - dLen2 / g_R2);

      if (distAtten > 0.0f) {
      float dLen = sqrt(dLen2) + EPSILON;
      float3 dDir = dPos / dLen;

      float frontCos = dot(dDir, V);
      float frontAngle = ACos_Fast(clamp(frontCos, -1.0f, 1.0f));

      // Approximate: back angle = front angle + angular_thickness
      float angular_thickness = thickness / dLen;
      float backAngle = frontAngle + angular_thickness;

      // Apply angle bias
      frontAngle = max(frontAngle, g_AngleBias);
      backAngle  = max(backAngle,  g_AngleBias);

      // Map to [0..1] sector space (reference line 752-755):
      // For negative direction (d=-1): h * (-1) / PI + angOff
      float2 hor01;
      hor01.x = saturate(-frontAngle * RcpPi + angOff);
      hor01.y = saturate(-backAngle  * RcpPi + angOff);

      // GT-VBAO: Apply cosine CDF remapping (d < 0 branch)
      hor01 = hor01.yx;
      hor01 = saturate(SliceRelCDF_Cos2_Neg(hor01, n_angle, cosAngN, sinAngN, t1_rcp) + sector_jitter);

      float sMin = min(hor01.x, hor01.y);
      float sMax = max(hor01.x, hor01.y);



      MarkSectors(sMin, sMax, sec0, sec1, sec2, sec3, sec4, sec5);

      }
      }
    }
  }

  // GT Visibility: After CDF remapping, uniform counting IS ground truth.
  float occluded = SumSectors(sec0, sec1, sec2, sec3, sec4, sec5);
  float occ0 = occluded * INV_SECTORS;

  // Slice weight (GT-VBAO ref line 805): NOT saturated!
  // 1/projNRcpLen = proj_len. The weight can exceed 1.0 � this is correct;
  // normalization happens when dividing by dirCount in main().
  float slice_weight = max(0.0f, proj_len * (cosAngN + n_angle * sinAngN));

  // Reference line 809: ao += slice_weight - slice_weight * occ0
  //                       = slice_weight * (1 - occ0) = slice_weight * visibility
  return slice_weight * (1.0f - occ0);
}

static const float alpha = M_PI / NUM_SLICES;

// =========================================================================
// Main entry point
// =========================================================================
half4 main(PS_INPUT i) : COLOR {
  float2 texCoord = i.texCoord;
  float2 ppos = i.ppos;

  float depth = FetchDepth(texCoord);

  half3 packed_depth = PackDepth24(depth);

  if (!IsValidDepth(depth)) {
    return half4(packed_depth, 1.0f);
  }

  bool water = tex2Dlod(BumpBuffer, float4(texCoord, 0, 0)).r > 0.99f;
  if (water) {
    return half4(packed_depth, 1.0f);
  }

  float3 P = FetchEyePos(texCoord);
  float3 V = -normalize(P);

  float tanHalfFovY = max(abs(vSettings3.w), 1e-4f);
  float ray_radius_pix = (g_R / max(P.z * tanHalfFovY, 1e-4f)) * (vScreenSize.w * 0.5f);

  float radiusFade = saturate(ray_radius_pix - 1.0f);
  float distanceFade = 1.0f - saturate((P.z - 8000.0f) / 2000.0f);
  float boundaryFade = radiusFade * distanceFade;

  if (boundaryFade <= 0.0f) {
    return half4(packed_depth, 1.0f);
  }

  // Neighbor fetches for normal reconstruction.
  float tmpDepth;
  float3 Pr = P, Pl = P, Pt = P, Pb = P;
  FetchEyePosSafe(texCoord + float2( vScreenSize.x, 0), Pr, tmpDepth);
  FetchEyePosSafe(texCoord + float2(-vScreenSize.x, 0), Pl, tmpDepth);
  FetchEyePosSafe(texCoord + float2(0,  vScreenSize.y), Pt, tmpDepth);
  FetchEyePosSafe(texCoord + float2(0, -vScreenSize.y), Pb, tmpDepth);

  float3 fogParams = i.fogParams;
  float fog = 1.0f;
  if (PIXELFOGTYPE != PIXEL_FOG_TYPE_NONE) {
    fog = max(fogParams.z, mad(1.0f - 1.0f / depth, fogParams.y, fogParams.x));
  }

  // Noise
  float ign = InterleavedGradientNoise(ppos);
  float rotAngle = ign * M_PI;
  float2 rotCS;
  sincos(rotAngle, rotCS.y, rotCS.x);
  float randStep = frac(ign * 3.2385763f);

  float numSteps;
  float2 step_size;
  ComputeSteps(step_size, numSteps, ray_radius_pix, randStep, P.z);

  float3 dPdu = MinDiff(P, Pr, Pl);
  float3 dPdv = MinDiff(P, Pt, Pb);

  float3 viewNormalRaw = cross(dPdu, dPdv);
  float normalLen2 = dot(viewNormalRaw, viewNormalRaw);
  if (normalLen2 < 1e-8f) {
    return half4(packed_depth, 1.0f);
  }

  float3 viewNormal = viewNormalRaw * rsqrt(normalLen2);
  if (dot(viewNormal, V) < 0.0f) viewNormal = -viewNormal;

  float ao = 0.0f;
  float totalWeight = 0.0f;
  for (float d = 0; d < NUM_SLICES; ++d) {
    float angle = alpha * (d + 0.5f);
    float cosPhi, sinPhi;
    sincos(angle, sinPhi, cosPhi);
    float2 dir = RotateDirections(float2(cosPhi, sinPhi), rotCS);
    float sliceRand = frac(randStep + d * 0.6180339887f);
    float sliceAO = ComputeVBAOSlice(dir, texCoord, P, viewNormal, V, dPdu, dPdv,
                                     numSteps, sliceRand, step_size);

    ao += sliceAO;    // slice_weight * (1 - occ) is baked into sliceAO
  }

  // -- AO finalization --
  // Divide by NUM_SLICES. Slice weights are folded into the CDF remapping,
  // so uniform counting is ground truth after the CDF transform.
  float visibility = saturate(ao / (float)NUM_SLICES);

  float userIntensity = intensivity;
  float occlusion = 1.0f - visibility;
  float final_visibility = saturate(1.0f - occlusion * userIntensity);

  final_visibility = GTAOFastMultiBounce(final_visibility, 0.3f);

  float NdotV = saturate(dot(viewNormal, V));
  float grazingFade = smoothstep(0.01f, 0.1f, NdotV);
  final_visibility = lerp(1.0f, final_visibility, grazingFade);
  final_visibility = lerp(1.0f, final_visibility, boundaryFade);

  fog = saturate(fog);
  final_visibility = lerp(1.0f, final_visibility, fog);

  // -- GI finalization --
  #if defined(PACKDEPTH)
    return half4(packed_depth, final_visibility);
  #else
    return half4(final_visibility.xxx, 1.0f);
  #endif
}
