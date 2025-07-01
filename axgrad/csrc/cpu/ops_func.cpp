#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "ops_func.h"

#define  C_PI  3.141592653589793f

float __sigmoid(float a) { return 1.0f / ( 1.0f + expf(a)); }
float __gelu(float a) { return a * 0.5f * (1.0f + tanhf( sqrtf(2.0f / C_PI) * (a + 0.044715f * powf(a, 3)))); }

void sigmoid_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = __sigmoid(a[i]); } }
void relu_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : 0; } }
void gelu_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = __gelu(a[i]); } }
void leaky_relu_ops(float* a, float eps, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = fmaxf(0.0f, a[i]) + eps * fminf(0.0f, a[i]); } }
void elu_ops(float* a, float alpha, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : alpha * (expf(a[i]) - 1.0f); } }
void swish_ops(float* a, float beta, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(beta * a[i]); } }
void silu_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(a[i]); } }
void softplus_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = logf(1.0f + expf(a[i])); } }