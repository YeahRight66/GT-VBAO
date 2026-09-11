#include "common_ps_packdepth.h"

sampler TexSampler : register(s0);
sampler WPDepthBuffer : register(s1);
float2 TexBaseSize : register(c4);

// ������������� ��� ������������� �������
// #define Vertical_
// #define PACKDEPTH12
#define PACKEDDEPTH_

const float size : register(c0);
const float sigma : register(c1);
const float2 ppp : register(c2);
#define sigma_pow ppp.x
#define sigma_min ppp.y

static const float INVALID_DEPTH = 0.00025f;

float wy_sq_cashed(float distance_sq, float sigma2) {
  return exp(sigma2 * distance_sq);
}

bool IsValidUV(float2 uv) {
  return uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f;
}

bool IsValidDepth(float depth) { return depth > INVALID_DEPTH; }

struct PS_IN {
  float2 vTexCoord : TEXCOORD0;
  float2 pos : VPOS;
};

float3 cEyePos : register(c3);
const float4x4 g_invViewProjMatrix : register(c15);

float3 reconstructPosition(float2 uv, float z) {
  return mad(mul(float4(mad(uv, 2, -1), 0, 1), g_invViewProjMatrix).xyz, z,
             cEyePos);
}

// 8-tap bilateral blur � wider kernel to clean up non-TAA GTAO noise
static const int NUM_SAMPLES = 10;
static const float OFFSETS[10] = {-4.5, -3.5, -2.5, -1.5, -0.5,
                                  0.5,  1.5,  2.5,  3.5,  4.5};
// Gaussian spatial weights (sigma ~2.0, normalized)
static const float GAUSS_W[10] = {0.0330, 0.0626, 0.1012, 0.1394, 0.1636,
                                  0.1636, 0.1394, 0.1012, 0.0626, 0.0330};

#define WORLDPOS_

half4 main(PS_IN i) : COLOR {
  float2 TexCoords = i.vTexCoord;

  float4 orig = tex2Dlod(TexSampler, float4(TexCoords, 0, 0));
  float3 orig_depth = orig.rgb;

  float depth = tex2Dlod(WPDepthBuffer, float4(TexCoords, 0, 0)).a;
  if (!IsValidDepth(depth))
    discard;

  float powed_si = sigma * pow(saturate(1.0f - depth),
                               sigma_pow); // sigma * (pow(1-depth, ppp));
  float sigma2 = -1 / (powed_si * powed_si);

  float3 worldPos = reconstructPosition(TexCoords, 1.0f / depth);
  half result = 0.0;
  float total_weight = 0.0;

  float2 pixel_size = TexBaseSize * size;

  [unroll] for (int x = 0; x < NUM_SAMPLES; ++x) {
#if defined(Vertical_)
// ������������ ������
#if defined(DOWNSAMPLE)
    float2 offset = TexCoords + float2(0, (OFFSETS[x] + 0.5) * pixel_size.y);
#else
    float2 offset = TexCoords + float2(0, OFFSETS[x] * pixel_size.y);
#endif
#else
    // �������������� ������
    float2 offset = TexCoords + float2(OFFSETS[x] * pixel_size.x, 0);
#endif

    if (!IsValidUV(offset)) {
      continue;
    }

#if defined(PACKEDDEPTH_)
    half4 temp = tex2Dlod(TexSampler, float4(offset, 0, 0));
    float sampleDepth = tex2Dlod(WPDepthBuffer, float4(offset, 0, 0)).a;
    if (!IsValidDepth(sampleDepth)) {
      continue;
    }

    float3 diff_vec =
        worldPos - reconstructPosition(offset.xy, 1.0f / sampleDepth);
    float diff = dot(diff_vec, diff_vec);
    float weight = wy_sq_cashed(diff, sigma2) * GAUSS_W[x];
    result += temp.a * weight;
    total_weight += weight;
#else
    float sampleDepth = tex2D(WPDepthBuffer, offset).a;
    if (!IsValidDepth(sampleDepth)) {
      continue;
    }

    float3 diff_vec =
        worldPos - reconstructPosition(offset.xy, 1.0f / sampleDepth);
    float diff = dot(diff_vec, diff_vec);
    float weight = wy_sq_cashed(diff, sigma2) * GAUSS_W[x];
    result += tex2D(TexSampler, offset).r * weight;
    total_weight += weight;
#endif
  }

  // result /= max(total_weight, sigma_min); // ������� ������� �� 0
  result = (total_weight > sigma_min) ? (result / total_weight) : orig.a;

#if defined(DOWNSAMPLE)
  return half4(orig_depth, result);
#else
  // Non-DOWNSAMPLE path (hbao_blury_combine): output AO in RGB for compositing
  return half4(result, result, result, 1);
#endif
};
