#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include "ops_vector.h"
#include "ops_tensor.h"


void vector_dot_ops(float* a, float* b, float* out, size_t size) { dot_tensor_ops(a, b, out, size); }
void vector_inner_product_ops(float* a, float* b, float* out, size_t size) { dot_tensor_ops(a, b, out, size); }

// Vector-Matrix: out = vec(1 x size_v) * mat(size_v x cols)

void vector_matrix_dot_ops(float* vec, float* mat, float* out, size_t size_v, size_t size_m) {
  size_t cols = size_m / size_v;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t j = 0; j < cols; j++) {
    __m256 vacc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i <= size_v - 8; i += 8)
      vacc = _mm256_fmadd_ps(_mm256_loadu_ps(vec + i), _mm256_set_ps(mat[(i+7)*cols+j], mat[(i+6)*cols+j], mat[(i+5)*cols+j], mat[(i+4)*cols+j], mat[(i+3)*cols+j], mat[(i+2)*cols+j], mat[(i+1)*cols+j], mat[(i+0)*cols+j]), vacc);
    // horizontal sum
    __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < size_v; i++) sum += vec[i] * mat[i * cols + j];
    out[j] = sum;
  }
}

// Matrix-Vector: out = mat(rows x size_v) * vec(size_v x 1)

void matrix_vector_dot_ops(float* mat, float* vec, float* out, size_t size_m, size_t size_v) {
  size_t rows = size_m / size_v;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < rows; i++) {
    __m256 vacc = _mm256_setzero_ps();
    size_t j = 0;
    for (; j <= size_v - 8; j += 8)
      vacc = _mm256_fmadd_ps(_mm256_loadu_ps(mat + i * size_v + j), _mm256_loadu_ps(vec + j), vacc);
    __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; j < size_v; j++) sum += mat[i * size_v + j] * vec[j];
    out[i] = sum;
  }
}

// Outer Product: out(n x m) = a(n) (x) b(m)

void vector_outer_product_ops(float* a, float* b, float* out, size_t size_n, size_t size_m) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < size_n; i++) {
    __m256 va = _mm256_set1_ps(a[i]);
    size_t j = 0;
    for (; j <= size_m - 8; j += 8)
      _mm256_storeu_ps(out + i * size_m + j,
        _mm256_mul_ps(va, _mm256_loadu_ps(b + j)));
    for (; j < size_m; j++) out[i * size_m + j] = a[i] * b[j];
  }
}

// Cross Product (general N-D)

static void cross_product_ops(float* a, float* b, float* out, size_t* shape, size_t ndim, size_t axis, size_t* a_stride, size_t* b_stride) {
  size_t axis_size = shape[axis];
  size_t total_elements = 1;
  for (size_t i = 0; i < ndim; i++) { if (i != axis) total_elements *= shape[i]; }

  if (axis_size == 2) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < total_elements; i++) {
      size_t tmp = i, a_idx = 0, b_idx = 0;
      for (size_t dim = 0; dim < ndim; dim++) {
        if (dim != axis) {
          size_t coord = tmp % shape[dim]; tmp /= shape[dim];
          a_idx += coord * a_stride[dim];
          b_idx += coord * b_stride[dim];
        }
      }
      float a0 = a[a_idx], a1 = a[a_idx + a_stride[axis]];
      float b0 = b[b_idx], b1 = b[b_idx + b_stride[axis]];
      out[i] = a0 * b1 - a1 * b0;
    }
  } else if (axis_size == 3) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < total_elements; i++) {
      size_t tmp = i, a_idx = 0, b_idx = 0;
      for (size_t dim = 0; dim < ndim; dim++) {
        if (dim != axis) {
          size_t coord = tmp % shape[dim]; tmp /= shape[dim];
          a_idx += coord * a_stride[dim];
          b_idx += coord * b_stride[dim];
        }
      }
      float a0 = a[a_idx], a1 = a[a_idx +     a_stride[axis]], a2 = a[a_idx + 2 * a_stride[axis]];
      float b0 = b[b_idx], b1 = b[b_idx +     b_stride[axis]], b2 = b[b_idx + 2 * b_stride[axis]];
      out[i*3+0] = a1*b2 - a2*b1;
      out[i*3+1] = a2*b0 - a0*b2;
      out[i*3+2] = a0*b1 - a1*b0;
    }
  }
}

// Cross 1D

void cross_1d_ops(float* a, float* b, float* out, size_t size) {
  if      (size == 2) { out[0] = a[0]*b[1] - a[1]*b[0]; }
  else if (size == 3) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
  }
}

// Cross 2D

void cross_2d_ops(float* a, float* b, float* out, size_t rows, size_t cols, size_t axis) {
  size_t shape[2]  = {rows, cols};
  size_t stride[2] = {cols, 1};
  cross_product_ops(a, b, out, shape, 2, axis, stride, stride);
}

// Cross 3D

void cross_3d_ops(float* a, float* b, float* out, size_t dim0, size_t dim1, size_t dim2, size_t axis) {
  size_t shape[3]  = {dim0, dim1, dim2};
  size_t stride[3] = {dim1 * dim2, dim2, 1};
  cross_product_ops(a, b, out, shape, 3, axis, stride, stride);
}