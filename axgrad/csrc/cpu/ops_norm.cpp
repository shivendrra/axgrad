#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "ops_norm.h"
#include "ops_redux.h"
#include "ops_binary.h"
#include "ops_unary.h"

void clip_tensor_ops(float* a, float* out, float max_val, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = (a[i] > max_val) ? max_val : ((a[i] < -max_val) ? -max_val : a[i]); } }
void clamp_tensor_ops(float* a, float* out, float min_val, float max_val, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = (a[i] > max_val) ? max_val : ((a[i] < min_val) ? min_val : a[i]); } }

void mm_norm_tensor_ops(float* a, float* out, size_t size) {
  float min_val = a[0], max_val = a[0];
  for (size_t i = 1; i < size; i++) {
    if (a[i] < min_val) min_val = a[i];
    if (a[i] > max_val) max_val = a[i];
  }
  float range = max_val - min_val;
  if (range == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else {
    sub_scalar_ops(a, min_val, out, size);
    div_scalar_ops(out, range, out, size);
  }
}

void std_norm_tensor_ops(float* a, float* out, size_t size) {
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += a[i];
  float mean = sum / size;

  float* temp = (float*)malloc(size * sizeof(float));
  sub_scalar_ops(a, mean, temp, size);
  mul_ops(temp, temp, temp, size);

  float var_sum = 0.0f;
  for (size_t i = 0; i < size; i++) var_sum += temp[i];
  float std_dev = sqrtf(var_sum / size);

  if (std_dev == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else {
    sub_scalar_ops(a, mean, out, size);
    div_scalar_ops(out, std_dev, out, size);
  }
  free(temp);
}

void rms_norm_tensor_ops(float* a, float* out, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  mul_ops(a, a, temp, size);

  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += temp[i];
  float rms = sqrtf(sum / size);
  if (rms == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else { div_scalar_ops(a, rms, out, size); }
  free(temp);
}

void l1_norm_tensor_ops(float* a, float* out, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  abs_tensor_ops(a, temp, size);
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += temp[i];
  if (sum == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else { div_scalar_ops(a, sum, out, size); }
  free(temp);
}

void l2_norm_tensor_ops(float* a, float* out, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  mul_ops(a, a, temp, size);

  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) sum += temp[i];
  float l2_norm = sqrtf(sum);

  if (l2_norm == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else { div_scalar_ops(a, l2_norm, out, size); }
  free(temp);
}

void unit_norm_tensor_ops(float* a, float* out, size_t size) { l2_norm_tensor_ops(a, out, size); }

void robust_norm_tensor_ops(float* a, float* out, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; i++) temp[i] = a[i];  
  for (size_t i = 0; i < size - 1; i++) {
    for (size_t j = i + 1; j < size; j++) {
      if (temp[i] > temp[j]) {
        float swap = temp[i];
        temp[i] = temp[j];
        temp[j] = swap;
      }
    }
  }

  float median = (size % 2 == 0) ? (temp[size/2 - 1] + temp[size/2]) / 2.0f : temp[size/2];
  sub_scalar_ops(a, median, temp, size);
  abs_tensor_ops(temp, temp, size);
  for (size_t i = 0; i < size - 1; i++) {
    for (size_t j = i + 1; j < size; j++) {
      if (temp[i] > temp[j]) {
        float swap = temp[i];
        temp[i] = temp[j];
        temp[j] = swap;
      }
    }
  }
  float mad = (size % 2 == 0) ? (temp[size/2 - 1] + temp[size/2]) / 2.0f : temp[size/2];
  if (mad == 0.0f) { for (size_t i = 0; i < size; i++) out[i] = 0.0f; }
  else {
    sub_scalar_ops(a, median, out, size);
    div_scalar_ops(out, mad, out, size);
  }
  free(temp);
}