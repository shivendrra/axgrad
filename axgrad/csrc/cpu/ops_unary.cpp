#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "ops_unary.h"

void exp_tensor_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = expf(a[i]); } }
void log_tensor_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = logf(a[i]); } }
void abs_tensor_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = fabsf(a[i]); } }
void sin_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = sinf(a[i]); } }
void cos_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = cosf(a[i]); } }
void tan_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = tanf(a[i]); } }
void sinh_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = sinhf(a[i]); } }
void cosh_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = coshf(a[i]); } }
void tanh_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = tanhf(a[i]); } }
void sqrt_tensor_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = sqrtf(a[i]); } }
void neg_tensor_ops(float* a, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = -a[i]; } }
void sign_tensor_ops(float* a, float* out, size_t size) {for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : ((a[i] < 0) ? -1.0f : 0.0f); } }