#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "ops_func.h"
#include "ops_unary.h"

#define  C_PI  3.141592653589793f

// helpers
float __sigmoid(float x) { return 1.0f / ( 1.0f + expf(-x)); }
float __pdf(float x) { return (1.0f / sqrtf(2.0f * C_PI)) * expf(-0.5f * x * x); }
float __erf_approx(float x) {
  // constants
  float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
  float p  = 0.3275911f; int sign = x < 0 ? -1 : 1;
  x = fabsf(x);
  float t = 1.0f / (1.0f + p * x);
  float y = 1.0f - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * expf(-x * x);
  return sign * y;
}
float __cdf(float x) { return 0.5f * (1.0f + __erf_approx(x / sqrtf(2.0f))); }
float __gelu(float x) { return x * __cdf(x); }

// forward
void sigmoid_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = __sigmoid(a[i]); } }
void relu_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : 0; } }
void gelu_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = __gelu(a[i]); } }
void leaky_relu_ops(float* a, float eps, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = fmaxf(0.0f, a[i]) + eps * fminf(0.0f, a[i]); } }
void elu_ops(float* a, float alpha, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : alpha * (expf(a[i]) - 1.0f); } }
void swish_ops(float* a, float beta, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(beta * a[i]); } }
void silu_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(a[i]); } }
void softplus_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = logf(1.0f + expf(a[i])); } }

// backwards
void sin_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = cosf(a[i]); } }
void cos_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = -sinf(a[i]); } }
void sinh_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = coshf(a[i]); } }
void cosh_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = sinhf(a[i]); } }
void tanh_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = 1 - a[i] * a[i]; } }
void sigmoid_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] * (1 - a[i]); } }
void relu_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1 : 0; } }
void leaky_relu_backwards_ops(float* a, float eps, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1 : eps; } }
void elu_backwards_ops(float* a, float alpha, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1 : alpha * expf(a[i]); } }
void gelu_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = __cdf(a[i]) + a[i] * __pdf(a[i]); } }
void softplus_backwards_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = 1 / (1 + expf(-a[i])); } }
void swish_backwards_ops(float* a, float beta, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float x = a[i];
    float sb = __sigmoid(beta * x);
    out[i] = sb + beta * x * sb * (1 - sb);
  }
}
void silu_backwards_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float s = __sigmoid(a[i]);
    out[i] = s * (1 + a[i] * (1 - s));
  }
}
void tan_backwards_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float t = tanf(a[i]);
    out[i] = 1 + t * t;
  }
}