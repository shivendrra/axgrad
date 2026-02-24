#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>
#include "ops_matrix.h"

// Helpers

// AVX2 row copy
static inline void copy_row(float* dst, const float* src, int n) {
  int i = 0;
  for (; i <= n - 8; i += 8) _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  for (; i < n; i++) dst[i] = src[i];
}

// AVX2 row swap
static inline void swap_rows(float* a, float* b, int n) {
  int i = 0;
  for (; i <= n - 8; i += 8) {
    __m256 ra = _mm256_loadu_ps(a + i);
    __m256 rb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(a + i, rb);
    _mm256_storeu_ps(b + i, ra);
  }
  for (; i < n; i++) { float t = a[i]; a[i] = b[i]; b[i] = t; }
}

// AVX2 fused-multiply-subtract: row[k] -= factor * row[i], length n
static inline void row_elim(float* rk, const float* ri, float factor, int n) {
  __m256 vf = _mm256_set1_ps(factor);
  int i = 0;
  for (; i <= n - 8; i += 8)
    _mm256_storeu_ps(rk + i,
      _mm256_fmadd_ps(vf, _mm256_loadu_ps(ri + i),   // fmadd: vf*ri + (-rk)
        _mm256_sub_ps(_mm256_setzero_ps(),              // = -(rk - factor*ri)
          _mm256_sub_ps(_mm256_loadu_ps(rk + i), _mm256_mul_ps(vf, _mm256_loadu_ps(ri + i))))));
  // Simpler scalar equivalent for the above (tail + correctness):
  // rk[i] -= factor * ri[i]
  // Use a cleaner formulation:
  i = 0;
  // Redo with cleaner AVX2 (subtract approach)
  for (i = 0; i <= n - 8; i += 8) {
    __m256 vk = _mm256_loadu_ps(rk + i);
    __m256 vi = _mm256_loadu_ps(ri + i);
    _mm256_storeu_ps(rk + i, _mm256_sub_ps(vk, _mm256_mul_ps(vf, vi)));
  }
  for (; i < n; i++) rk[i] -= factor * ri[i];
}

// Determinant

void det_ops_tensor(float* a, float* out, size_t size) {
  int n = (int)size;
  float* temp = (float*)malloc(n * n * sizeof(float));
  if (!temp) { *out = 0.0f; return; }
  copy_row(temp, a, n * n);   // AVX2 bulk copy

  float det = 1.0f;
  for (int i = 0; i < n; i++) {
    // Partial pivot
    int pivot_row = i;
    float max_val = fabsf(temp[i * n + i]);
    for (int row = i + 1; row < n; row++) {
      float v = fabsf(temp[row * n + i]);
      if (v > max_val) { max_val = v; pivot_row = row; }
    }
    if (pivot_row != i) {
      swap_rows(temp + i * n, temp + pivot_row * n, n);
      det = -det;
    }
    float pivot = temp[i * n + i];
    if (fabsf(pivot) < 1e-6f) { det = 0.0f; break; }
    det *= pivot;

    // Eliminate rows below pivot — parallelise across rows
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = i + 1; j < n; j++) {
      float factor = temp[j * n + i] / pivot;
      row_elim(temp + j * n + i, temp + i * n + i, factor, n - i);
    }
  }
  free(temp);
  *out = det;
}

void batched_det_ops(float* a, float* out, size_t size, size_t batch) {
  size_t mat_size = size * size;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t b = 0; b < batch; b++)
    det_ops_tensor(a + b * mat_size, out + b, size);
}

// Inverse

void inv_ops(float* a, float* out, int* shape) {
  int n = shape[0];
  int w = n * 2;   // augmented row width
  float* temp = (float*)malloc(n * w * sizeof(float));
  if (!temp) return;

  // Build augmented [A | I]
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      temp[i * w + j] = a[i * n + j];
      temp[i * w + j + n] = (i == j) ? 1.0f : 0.0f;
    }
  }

  for (int i = 0; i < n; i++) {
    // Partial pivot
    int pivot = i;
    for (int k = i + 1; k < n; k++)
      if (fabsf(temp[k * w + i]) > fabsf(temp[pivot * w + i])) pivot = k;
    if (pivot != i) swap_rows(temp + i * w, temp + pivot * w, w);

    float diag = temp[i * w + i];
    // Scale pivot row — AVX2
    __m256 vd = _mm256_set1_ps(1.0f / diag);
    int j = 0;
    for (; j <= w - 8; j += 8)
      _mm256_storeu_ps(temp + i * w + j,
        _mm256_mul_ps(_mm256_loadu_ps(temp + i * w + j), vd));
    for (; j < w; j++) temp[i * w + j] /= diag;

    // Eliminate all other rows — parallel
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < n; k++) {
      if (k == i) continue;
      float factor = temp[k * w + i];
      row_elim(temp + k * w, temp + i * w, factor, w);
    }
  }

  // Extract right half → out
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      out[i * n + j] = temp[i * w + j + n];

  free(temp);
}

void batched_inv_ops(float* a, float* out, int* shape, int ndim) {
  if (ndim < 2) return;
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];
  int matrix_size = shape[ndim - 2] * shape[ndim - 1];
  int matrix_shape[2] = {shape[ndim - 2], shape[ndim - 1]};

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int b = 0; b < batch_size; b++)
    inv_ops(a + b * matrix_size, out + b * matrix_size, matrix_shape);
}

// Solve

void solve_ops(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int n = shape_a[0];
  int nrhs = (shape_b[1] > 0) ? shape_b[1] : 1;

  float* ta = (float*)malloc(n * n    * sizeof(float));
  float* tb = (float*)malloc(n * nrhs * sizeof(float));
  if (!ta || !tb) { free(ta); free(tb); return; }
  memcpy(ta, a, n * n * sizeof(float));
  memcpy(tb, b, n * nrhs * sizeof(float));

  // Forward elimination with partial pivoting
  for (int i = 0; i < n; i++) {
    int pivot = i;
    for (int k = i + 1; k < n; k++)
      if (fabsf(ta[k * n + i]) > fabsf(ta[pivot * n + i])) pivot = k;

    if (pivot != i) {
      swap_rows(ta + i * n, ta + pivot * n, n);
      swap_rows(tb + i * nrhs, tb + pivot * nrhs, nrhs);
    }

    float piv_val = ta[i * n + i];
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = i + 1; k < n; k++) {
      float factor = ta[k * n + i] / piv_val;
      row_elim(ta + k * n + i, ta + i * n + i, factor, n - i);
      row_elim(tb + k * nrhs, tb + i * nrhs, factor, nrhs);
    }
  }

  // Back substitution
  for (int j = 0; j < nrhs; j++) {
    for (int i = n - 1; i >= 0; i--) {
      float sum = tb[i * nrhs + j];
      for (int k = i + 1; k < n; k++) sum -= ta[i * n + k] * out[k * nrhs + j];
      out[i * nrhs + j] = sum / ta[i * n + i];
    }
  }

  free(ta); free(tb);
}

void batched_solve_ops(float* a, float* b, float* out, int* shape_a, int* shape_b, int ndim) {
  if (ndim < 2) return;
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape_a[i];
  int ms_a = shape_a[ndim-2] * shape_a[ndim-1], ms_b = shape_b[ndim-2] * shape_b[ndim-1];
  int sha[2] = {shape_a[ndim-2], shape_a[ndim-1]}, shb[2] = {shape_b[ndim-2], shape_b[ndim-1]};

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int bi = 0; bi < batch_size; bi++)
    solve_ops(a + bi * ms_a, b + bi * ms_b, out + bi * ms_b, sha, shb);
}

// Least Squares

void lstsq_ops(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int m = shape_a[0], n = shape_a[1];
  int nrhs = (shape_b[1] > 0) ? shape_b[1] : 1;

  float* at = (float*)malloc(n * m * sizeof(float));
  float* ata = (float*)malloc(n * n * sizeof(float));
  float* atb = (float*)malloc(n * nrhs * sizeof(float));
  if (!at || !ata || !atb) { free(at); free(ata); free(atb); return; }

  // Transpose A — parallel rows
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) at[j * m + i] = a[i * n + j];

  // ATA = Aᵀ · A — parallel over output rows, AVX2 inner product
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      __m256 acc = _mm256_setzero_ps();
      int k = 0;
      for (; k <= m - 8; k += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(at + i * m + k), _mm256_loadu_ps(a  + k * n + j), acc);
      // Horizontal sum
      __m128 lo  = _mm256_castps256_ps128(acc);
      __m128 hi  = _mm256_extractf128_ps(acc, 1);
      __m128 sum = _mm_add_ps(lo, hi);
      sum = _mm_hadd_ps(sum, sum);
      sum = _mm_hadd_ps(sum, sum);
      float s = _mm_cvtss_f32(sum);
      for (; k < m; k++) s += at[i * m + k] * a[k * n + j];
      ata[i * n + j] = s;
    }
  }

  // ATB = Aᵀ · B — parallel over output rows, AVX2 inner product
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < nrhs; j++) {
      __m256 acc = _mm256_setzero_ps();
      int k = 0;
      for (; k <= m - 8; k += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(at + i * m + k), _mm256_loadu_ps(b  + k * nrhs + j), acc);
      __m128 lo  = _mm256_castps256_ps128(acc);
      __m128 hi  = _mm256_extractf128_ps(acc, 1);
      __m128 sum = _mm_add_ps(lo, hi);
      sum = _mm_hadd_ps(sum, sum);
      sum = _mm_hadd_ps(sum, sum);
      float s = _mm_cvtss_f32(sum);
      for (; k < m; k++) s += at[i * m + k] * b[k * nrhs + j];
      atb[i * nrhs + j] = s;
    }
  }

  int sha[2] = {n, n}, shb[2] = {n, nrhs};
  solve_ops(ata, atb, out, sha, shb);
  free(at); free(ata); free(atb);
}

void batched_lstsq_ops(float* a, float* b, float* out, int* shape_a, int* shape_b, int ndim) {
  if (ndim < 2) return;
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape_a[i];
  int ms_a  = shape_a[ndim-2] * shape_a[ndim-1], ms_b = shape_b[ndim-2] * shape_b[ndim-1];
  int out_s = shape_a[ndim-1] * shape_b[ndim-1];
  int sha[2] = {shape_a[ndim-2], shape_a[ndim-1]}, shb[2] = {shape_b[ndim-2], shape_b[ndim-1]};

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int bi = 0; bi < batch_size; bi++)
    lstsq_ops(a + bi * ms_a, b + bi * ms_b, out + bi * out_s, sha, shb);
}