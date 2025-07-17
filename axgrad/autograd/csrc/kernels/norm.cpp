#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "norm.h"
#include "../../../csrc/cpu/ops_binary.h"
#include "../../../csrc/cpu/ops_unary.h"
#include "../../../csrc/cpu/ops_shape.h"

// out_grad = grad * (x <= max_val)
void clip_backwards_ops(float* a, float* grad, float* out, float max_val, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  smaller_equal_scalar_ops(a, max_val, temp, size);
  mul_ops(grad, temp, out, size);
  free(temp);
}

// out_grad = grad * (x <= max_val) * (x >= min_val)
void clamp_backwards_ops(float* a, float* grad, float* out, float max_val, float min_val, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float)), *temp3 = (float*)malloc(size * sizeof(float));
  smaller_equal_scalar_ops(a, max_val, temp1, size);
  greater_equal_scalar_ops(a, min_val, temp2, size);
  mul_ops(temp1, temp2, temp3, size);
  mul_ops(grad, temp3, out, size);
  free(temp1); free(temp2); free(temp3);
}

// out_grad = grad / range
void mm_norm_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float min_val = a[0], max_val = a[0];
  for (size_t i = 1; i < size; i++) {
    if (a[i] < min_val) min_val = a[i];
    if (a[i] > max_val) max_val = a[i];
  }
  float range = max_val - min_val;
  if (range == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else { div_scalar_ops(grad, range, out, size); }
}

// out_grad = grad * (1/std_dev - (x-mean)/(std_dev^3 * n))
void std_norm_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += a[i];
  float mean = sum / size;
  
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));
  sub_scalar_ops(a, mean, temp1, size);
  mul_ops(temp1, temp1, temp2, size);
  
  float var_sum = 0.0f;
  for (size_t i = 0; i < size; i++) var_sum += temp2[i];
  float std_dev = sqrtf(var_sum / size);

  if (std_dev == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else {
    float inv_std = 1.0f / std_dev, inv_std3 = 1.0f / (std_dev * std_dev * std_dev);
    mul_scalar_ops(temp1, inv_std3 / size, temp2, size);
    sub_scalar_ops(temp2, inv_std, temp2, size);
    mul_ops(grad, temp2, out, size);
  }
  free(temp1); free(temp2);
}

// out_grad = grad * (1/rms - x/(rms^3 * n))
void rms_norm_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));
  mul_ops(a, a, temp1, size);
  
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += temp1[i];
  float rms = sqrtf(sum / size);

  if (rms == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else {
    float inv_rms = 1.0f / rms, inv_rms3 = 1.0f / (rms * rms * rms);
    mul_scalar_ops(a, inv_rms3 / size, temp2, size);
    sub_scalar_ops(temp2, inv_rms, temp2, size);
    mul_ops(grad, temp2, out, size);
  }
  free(temp1); free(temp2);
}

// out_grad = grad * (sign(x)/l1_norm - x/(l1_norm^2))
void l1_norm_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));
  abs_tensor_ops(a, temp1, size);
  
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += temp1[i];

  if (sum == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else {
    sign_tensor_ops(a, temp1, size);
    div_scalar_ops(temp1, sum, temp1, size);
    div_scalar_ops(a, sum * sum, temp2, size);
    sub_ops(temp1, temp2, temp1, size);
    mul_ops(grad, temp1, out, size);
  }
  free(temp1); free(temp2);
}

// out_grad = grad * (1/l2_norm - x/(l2_norm^3))
void l2_norm_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));
  mul_ops(a, a, temp1, size);
  
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += temp1[i];
  float l2_norm = sqrtf(sum);

  if (l2_norm == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else {
    float inv_l2 = 1.0f / l2_norm, inv_l2_3 = 1.0f / (l2_norm * l2_norm * l2_norm);
    mul_scalar_ops(a, inv_l2_3, temp2, size);
    sub_scalar_ops(temp2, inv_l2, temp2, size);
    mul_ops(grad, temp2, out, size);
  }
  free(temp1); free(temp2);
}

// alias for l2_norm_backwards_ops
void unit_norm_backwards_ops(float* a, float* grad, float* out, size_t size) {
  l2_norm_backwards_ops(a, grad, out, size);
}

// robust norm backward - complex gradient involving median and mad
void robust_norm_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));

  // compute median
  for (size_t i = 0; i < size; i++) temp1[i] = a[i];
  for (size_t i = 0; i < size - 1; i++) {
    for (size_t j = i + 1; j < size; j++) { if (temp1[i] > temp1[j]) { float swap = temp1[i]; temp1[i] = temp1[j]; temp1[j] = swap; } }
  }
  float median = (size % 2 == 0) ? (temp1[size/2 - 1] + temp1[size/2]) / 2.0f : temp1[size/2];
  // compute mad
  sub_scalar_ops(a, median, temp1, size);
  abs_tensor_ops(temp1, temp1, size);
  for (size_t i = 0; i < size - 1; i++) {
    for (size_t j = i + 1; j < size; j++) { if (temp1[i] > temp1[j]) { float swap = temp1[i]; temp1[i] = temp1[j]; temp1[j] = swap; } }
  }
  float mad = (size % 2 == 0) ? (temp1[size/2 - 1] + temp1[size/2]) / 2.0f : temp1[size/2];
  if (mad == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else { div_scalar_ops(grad, mad, out, size); }
  free(temp1); free(temp2);
}