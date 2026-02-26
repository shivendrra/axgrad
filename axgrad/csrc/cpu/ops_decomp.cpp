#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include "ops_decomp.h"
#include "ops_tensor.h"
#include "ops_shape.h"

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3")

// AVX2 Helpers

// Dot product of two float arrays of length n, AVX2 accelerated
static inline float avx2_dot(const float* a, const float* b, int n) {
  __m256 vacc = _mm256_setzero_ps();
  int i = 0;
  for (; i <= n - 8; i += 8)
    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), vacc);
  __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc, 1);
  __m128 s  = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
  float sum = _mm_cvtss_f32(s);
  for (; i < n; i++) sum += a[i] * b[i];
  return sum;
}

// row[i] -= factor * row[j], AVX2 accelerated
static inline void avx2_row_elim(float* rk, const float* ri, float factor, int n) {
  __m256 vf = _mm256_set1_ps(factor);
  int i = 0;
  for (; i <= n - 8; i += 8)
    _mm256_storeu_ps(rk + i, _mm256_sub_ps(_mm256_loadu_ps(rk + i), _mm256_mul_ps(vf, _mm256_loadu_ps(ri + i))));
  for (; i < n; i++) rk[i] -= factor * ri[i];
}

// AVX2 row swap
static inline void avx2_swap_rows(float* a, float* b, int n) {
  int i = 0;
  for (; i <= n - 8; i += 8) {
    __m256 ra = _mm256_loadu_ps(a + i), rb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(a + i, rb); _mm256_storeu_ps(b + i, ra);
  }
  for (; i < n; i++) { float t = a[i]; a[i] = b[i]; b[i] = t; }
}

// Square matrix multiply: C = A * B, all n x n, parallel + AVX2
static void matmul_nn(const float* A, const float* B, float* C, int n) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      __m256 vacc = _mm256_setzero_ps();
      int k = 0;
      for (; k <= n - 8; k += 8)
        vacc = _mm256_fmadd_ps(_mm256_loadu_ps(A + i*n + k), _mm256_loadu_ps(B + k*n + j + 0), vacc); // B column not contiguous — gather manually for small n, use dot helper
      // Fall back to scalar for non-contiguous column access
      float sum = 0.0f;
      for (k = 0; k < n; k++) sum += A[i*n + k] * B[k*n + j];
      C[i*n + j] = sum;
    }
  }
}

// Row-contiguous matmul: C(m x n) = A(m x k) * B(k x n), parallel + AVX2 inner
static void matmul_rect(const float* A, const float* B, float* C, int m, int kk, int n) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < kk; k++) sum += A[i*kk + k] * B[k*n + j];
      C[i*n + j] = sum;
    }
  }
}

// QR Decomposition

void qr_decomp_ops(float* a, float* q, float* r, int* shape) {
  int m = shape[0], n = shape[1];
  float* work = (float*)malloc(m * n * sizeof(float));
  memcpy(work, a, m * n * sizeof(float));

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m * m; i++) q[i] = 0.0f;
  for (int i = 0; i < m; i++) q[i * m + i] = 1.0f;
  memset(r, 0, m * n * sizeof(float));

  for (int k = 0; k < n && k < m; k++) {
    // Norm of column k — AVX2
    float norm = 0.0f;
    for (int i = 0; i < m; i++) { float v = work[i * n + k]; norm += v * v; }
    norm = sqrtf(norm);
    r[k * n + k] = norm;

    if (norm > 1e-6f) {
      float inv_norm = 1.0f / norm;
      __m256 vi = _mm256_set1_ps(inv_norm);
      // Normalize column k into q column k (non-contiguous — scalar)
#ifdef _OPENMP
      #pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < m; i++) q[i * m + k] = work[i * n + k] * inv_norm;
    }

    // Orthogonalize remaining columns — parallel over j
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = k + 1; j < n; j++) {
      // dot = q_k^T * work_j
      float dot = 0.0f;
      for (int i = 0; i < m; i++) dot += q[i * m + k] * work[i * n + j];
      r[k * n + j] = dot;
      // work_j -= dot * q_k
      for (int i = 0; i < m; i++) work[i * n + j] -= dot * q[i * m + k];
    }
  }
  free(work);
}

void batched_qr_decomp_ops(float* a, float* q, float* r, int* shape, int ndim) {
  if (ndim < 2) { fprintf(stderr, "error: qr requires at least 2 dimensions\n"); exit(EXIT_FAILURE); }
  int m = shape[ndim-2], n = shape[ndim-1], batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];
  int ms_a = m*n, ms_q = m*m, ms_r = m*n;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int b = 0; b < batch_size; b++) {
    int sh[2] = {m, n};
    qr_decomp_ops(a + b*ms_a, q + b*ms_q, r + b*ms_r, sh);
  }
}

// LU Decomposition

void lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape) {
  int n = shape[0];
  memcpy(u, a, n * n * sizeof(float));
  memset(l, 0, n * n * sizeof(float));
  for (int i = 0; i < n; i++) { l[i*n+i] = 1.0f; p[i] = i; }

  for (int k = 0; k < n - 1; k++) {
    // Partial pivot
    int pivot_row = k;
    float max_val = fabsf(u[k*n+k]);
    for (int i = k+1; i < n; i++) {
      float v = fabsf(u[i*n+k]);
      if (v > max_val) { max_val = v; pivot_row = i; }
    }
    if (pivot_row != k) {
      avx2_swap_rows(u + k*n, u + pivot_row*n, n);
      avx2_swap_rows(l + k*n, l + pivot_row*n, k);  // only lower part
      int tp = p[k]; p[k] = p[pivot_row]; p[pivot_row] = tp;
    }

    float piv = u[k*n+k];
    if (fabsf(piv) <= 1e-9f) continue;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = k+1; i < n; i++) {
      float factor = u[i*n+k] / piv;
      l[i*n+k] = factor;
      avx2_row_elim(u + i*n + k, u + k*n + k, factor, n - k);
    }
  }
}

void batched_lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape, int ndim) {
  if (ndim < 2) { fprintf(stderr, "error: lu requires at least 2 dimensions\n"); exit(EXIT_FAILURE); }
  int n = shape[ndim-1], batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];
  int ms = n * n;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int b = 0; b < batch_size; b++) {
    int sh[2] = {n, n};
    lu_decomp_ops(a + b*ms, l + b*ms, u + b*ms, p + b*n, sh);
  }
}

// Cholesky

static void compute_chol(float* a, float* l, int n) {
  memset(l, 0, n * n * sizeof(float));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      if (i == j) {
        // Diagonal: sum of squares of row i up to j — AVX2
        float sum = avx2_dot(l + i*n, l + i*n, j);
        float val = a[i*n+i] - sum;
        if (val <= 1e-12f) { l[i*n+j] = 0.0f; return; }
        l[i*n+j] = sqrtf(val);
      } else {
        // Off-diagonal: dot of row i and row j up to k — AVX2
        float sum = avx2_dot(l + i*n, l + j*n, j);
        float ljj = l[j*n+j];
        l[i*n+j] = (fabsf(ljj) < 1e-12f) ? 0.0f : (a[i*n+j] - sum) / ljj;
      }
    }
  }
}

void chol_ops(float* a, float* l, int* shape) { compute_chol(a, l, shape[0]); }

void batched_chol_ops(float* a, float* l, int* shape, int ndim) {
  if (ndim < 2) { fprintf(stderr, "error: cholesky requires at least 2 dimensions\n"); exit(EXIT_FAILURE); }
  int n = shape[ndim-1], batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];
  int ms = n * n;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int b = 0; b < batch_size; b++) {
    int sh[2] = {n, n};
    chol_ops(a + b*ms, l + b*ms, sh);
  }
}

// Jacobi Eigenvalue (Hermitian)

static void compute_eigenvals_h(float* a, float* eigenvals, size_t size) {
  size_t mat_size = size * size;
  float* temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { for (size_t i = 0; i < size; i++) eigenvals[i] = 0.0f; return; }
  memcpy(temp, a, mat_size * sizeof(float));

  for (size_t iter = 0; iter < 1000; iter++) {
    float max_val = 0.0f; size_t p = 0, q = 1;
    for (size_t i = 0; i < size; i++) {
      for (size_t j = i+1; j < size; j++) {
        float v = fabsf(temp[i*size+j]);
        if (v > max_val) { max_val = v; p = i; q = j; }
      }
    }
    if (max_val < 1e-14f) break;

    float app = temp[p*size+p], aqq = temp[q*size+q], apq = temp[p*size+q];
    float c, s;
    if (fabsf(apq) < 1e-15f) { c = 1.0f; s = 0.0f; }
    else {
      float theta = (aqq - app) / (2.0f * apq);
      float t = (theta >= 0.0f) ? 1.0f/(theta + sqrtf(theta*theta + 1.0f)) : 1.0f/(theta - sqrtf(theta*theta + 1.0f));
      c = 1.0f / sqrtf(t*t + 1.0f); s = t * c;
    }

    // k-loop is independent for k != p,q
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < size; k++) {
      if (k == p || k == q) continue;
      float akp = temp[k*size+p], akq = temp[k*size+q];
      temp[k*size+p] = temp[p*size+k] = c*akp - s*akq;
      temp[k*size+q] = temp[q*size+k] = s*akp + c*akq;
    }
    temp[p*size+p] = c*c*app + s*s*aqq - 2.0f*s*c*apq;
    temp[q*size+q] = s*s*app + c*c*aqq + 2.0f*s*c*apq;
    temp[p*size+q] = temp[q*size+p] = 0.0f;
  }

  for (size_t i = 0; i < size; i++) eigenvals[i] = temp[i*size+i];
  // Sort ascending — small n, serial OK
  for (size_t i = 0; i < size-1; i++) {
    for (size_t j = i+1; j < size; j++) {
      if (eigenvals[i] > eigenvals[j]) {
        float t = eigenvals[i]; eigenvals[i] = eigenvals[j]; eigenvals[j] = t;
      }
    }
  }
  free(temp);
}

static void compute_eigenvecs_h(float* a, float* eigenvecs, size_t size) {
  size_t mat_size = size * size;
  float* temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { for (size_t i = 0; i < mat_size; i++) eigenvecs[i] = 0.0f; return; }
  memcpy(temp, a, mat_size * sizeof(float));

  // Init eigenvecs as identity
  memset(eigenvecs, 0, mat_size * sizeof(float));
  for (size_t i = 0; i < size; i++) eigenvecs[i*size+i] = 1.0f;

  for (size_t iter = 0; iter < 1000; iter++) {
    float max_val = 0.0f; size_t p = 0, q = 1;
    for (size_t i = 0; i < size; i++) {
      for (size_t j = i+1; j < size; j++) {
        float v = fabsf(temp[i*size+j]);
        if (v > max_val) { max_val = v; p = i; q = j; }
      }
    }
    if (max_val < 1e-14f) break;

    float app = temp[p*size+p], aqq = temp[q*size+q], apq = temp[p*size+q];
    float c, s;
    if (fabsf(apq) < 1e-15f) { c = 1.0f; s = 0.0f; }
    else {
      float theta = (aqq - app) / (2.0f * apq);
      float t = (theta >= 0.0f) ? 1.0f/(theta + sqrtf(theta*theta + 1.0f))
                                 : 1.0f/(theta - sqrtf(theta*theta + 1.0f));
      c = 1.0f / sqrtf(t*t + 1.0f); s = t * c;
    }

    // Rotate temp — independent k
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < size; k++) {
      if (k == p || k == q) continue;
      float akp = temp[k*size+p], akq = temp[k*size+q];
      temp[k*size+p] = temp[p*size+k] = c*akp - s*akq;
      temp[k*size+q] = temp[q*size+k] = s*akp + c*akq;
    }
    temp[p*size+p] = c*c*app + s*s*aqq - 2.0f*s*c*apq;
    temp[q*size+q] = s*s*app + c*c*aqq + 2.0f*s*c*apq;
    temp[p*size+q] = temp[q*size+p] = 0.0f;

    // Accumulate rotation into eigenvecs — independent k
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < size; k++) {
      float vkp = eigenvecs[k*size+p], vkq = eigenvecs[k*size+q];
      eigenvecs[k*size+p] = c*vkp - s*vkq;
      eigenvecs[k*size+q] = s*vkp + c*vkq;
    }
  }

  // Sort eigenvectors by ascending eigenvalue
  float* eigenvals = (float*)malloc(size * sizeof(float));
  size_t* indices  = (size_t*)malloc(size * sizeof(size_t));
  for (size_t i = 0; i < size; i++) { eigenvals[i] = temp[i*size+i]; indices[i] = i; }
  for (size_t i = 0; i < size-1; i++) {
    for (size_t j = i+1; j < size; j++) {
      if (eigenvals[indices[i]] > eigenvals[indices[j]]) {
        size_t t = indices[i]; indices[i] = indices[j]; indices[j] = t;
      }
    }
  }

  float* tmp_vecs = (float*)malloc(mat_size * sizeof(float));
  memcpy(tmp_vecs, eigenvecs, mat_size * sizeof(float));
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t j = 0; j < size; j++) {
    size_t src = indices[j];
    for (size_t i = 0; i < size; i++) eigenvecs[i*size+j] = tmp_vecs[i*size+src];
  }

  // Sign canonicalization
  for (size_t j = 0; j < size; j++) {
    if (eigenvecs[j] < 0.0f) {
      __m256 vm1 = _mm256_set1_ps(-1.0f);
      size_t i = 0;
      for (; i + 8 <= size; i += 8)
        _mm256_storeu_ps(eigenvecs + i*size + j,
          _mm256_mul_ps(_mm256_loadu_ps(eigenvecs + i*size + j), vm1));
      // scalar tail — column not contiguous, do full scalar pass
      for (size_t ii = 0; ii < size; ii++) eigenvecs[ii*size+j] *= -1.0f;
      break;  // already negated above, avoid double-negation
    }
    // Correct sign canonicalization: column stride is non-contiguous
    if (eigenvecs[0*size+j] < 0.0f) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < size; i++) eigenvecs[i*size+j] *= -1.0f;
    }
  }

  free(eigenvals); free(indices); free(tmp_vecs); free(temp);
}

// Eigenvalues (General, via QR iteration)

static void compute_eigenvals(float* a, float* eigenvals, size_t size) {
  size_t mat_size = size * size;
  float* buf = (float*)malloc(3 * mat_size * sizeof(float));
  if (!buf) { for (size_t i = 0; i < size; i++) eigenvals[i] = 0.0f; return; }
  float *q = buf, *r = buf + mat_size, *curr = buf + 2*mat_size;
  memcpy(curr, a, mat_size * sizeof(float));

  for (size_t iter = 0; iter < 200; iter++) {
    // Wilkinson shift
    float shift = 0.0f;
    if (size > 1) {
      float a11 = curr[(size-2)*size+(size-2)], a12 = curr[(size-2)*size+(size-1)];
      float a21 = curr[(size-1)*size+(size-2)], a22 = curr[(size-1)*size+(size-1)];
      float tr = a11+a22, det = a11*a22 - a12*a21, disc = tr*tr - 4.0f*det;
      if (disc >= 0.0f) {
        float sd = sqrtf(disc);
        float l1 = (tr+sd)/2.0f, l2 = (tr-sd)/2.0f;
        shift = (fabsf(l1-a22) < fabsf(l2-a22)) ? l1 : l2;
      } else { shift = tr/2.0f; }
    }
    for (size_t i = 0; i < size; i++) curr[i*size+i] -= shift;

    int sh[2] = {(int)size, (int)size};
    qr_decomp_ops(curr, q, r, sh);

    // curr = R*Q (parallel matmul)
    memset(curr, 0, mat_size * sizeof(float));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < size; k++) sum += r[i*size+k] * q[k*size+j];
        curr[i*size+j] = sum;
      }
    }
    for (size_t i = 0; i < size; i++) curr[i*size+i] += shift;

    // Convergence check — parallel reduction
    float off = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:off)
#endif
    for (size_t i = 0; i < size; i++)
      for (size_t j = 0; j < size; j++) if (i != j) off += fabsf(curr[i*size+j]);
    if (off < 1e-8f) break;
  }
  for (size_t i = 0; i < size; i++) eigenvals[i] = curr[i*size+i];
  free(buf);
}

static void compute_eigenvecs(float* a, float* eigenvecs, size_t size) {
  size_t mat_size = size * size;
  float* buf = (float*)malloc(5 * mat_size * sizeof(float));
  if (!buf) { for (size_t i = 0; i < mat_size; i++) eigenvecs[i] = 0.0f; return; }
  float *q = buf, *r = buf+mat_size, *qt = buf+2*mat_size, *vacc = buf+3*mat_size, *curr = buf+4*mat_size;
  memcpy(curr, a, mat_size * sizeof(float));
  memset(eigenvecs, 0, mat_size * sizeof(float));
  memset(vacc, 0, mat_size * sizeof(float));
  for (size_t i = 0; i < size; i++) { eigenvecs[i*size+i] = 1.0f; vacc[i*size+i] = 1.0f; }

  for (size_t iter = 0; iter < 200; iter++) {
    float shift = 0.0f;
    if (size > 1) {
      float a11 = curr[(size-2)*size+(size-2)], a12 = curr[(size-2)*size+(size-1)];
      float a21 = curr[(size-1)*size+(size-2)], a22 = curr[(size-1)*size+(size-1)];
      float tr = a11+a22, det = a11*a22 - a12*a21, disc = tr*tr - 4.0f*det;
      if (disc >= 0.0f) {
        float sd = sqrtf(disc);
        float l1 = (tr+sd)/2.0f, l2 = (tr-sd)/2.0f;
        shift = (fabsf(l1-a22) < fabsf(l2-a22)) ? l1 : l2;
      } else { shift = tr/2.0f; }
    }
    for (size_t i = 0; i < size; i++) curr[i*size+i] -= shift;

    int sh[2] = {(int)size, (int)size};
    qr_decomp_ops(curr, q, r, sh);

    // vacc = vacc * Q (accumulate eigenvectors) — parallel
    memset(qt, 0, mat_size * sizeof(float));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < size; k++) sum += vacc[i*size+k] * q[k*size+j];
        qt[i*size+j] = sum;
      }
    }
    memcpy(vacc, qt, mat_size * sizeof(float));

    // curr = R * Q — parallel
    memset(curr, 0, mat_size * sizeof(float));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < size; k++) sum += r[i*size+k] * q[k*size+j];
        curr[i*size+j] = sum;
      }
    }
    for (size_t i = 0; i < size; i++) curr[i*size+i] += shift;

    float off = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:off)
#endif
    for (size_t i = 0; i < size; i++)
      for (size_t j = 0; j < size; j++) if (i != j) off += fabsf(curr[i*size+j]);
    if (off < 1e-8f) break;
  }
  memcpy(eigenvecs, vacc, mat_size * sizeof(float));
  free(buf);
}

// SVD

static void compute_svd(float* a, float* u, float* s, float* vt, int m, int n) {
  int min_mn = (m < n) ? m : n;
  float *aat = (float*)malloc(m*m*sizeof(float));
  float *ata = (float*)malloc(n*n*sizeof(float));
  float *tu  = (float*)malloc(m*m*sizeof(float));
  float *tv  = (float*)malloc(n*n*sizeof(float));

  // AAT = A * A^T — parallel rows, AVX2 dot
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++)
      aat[i*m+j] = avx2_dot(a + i*n, a + j*n, n);
  }

  // ATA = A^T * A — parallel rows
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < m; k++) sum += a[k*n+i] * a[k*n+j];
      ata[i*n+j] = sum;
    }
  }

  float *ev_u = (float*)malloc(m * sizeof(float));
  float *ev_v = (float*)malloc(n * sizeof(float));
  eigenvecs_h_ops_tensor(aat, tu, m); eigenvals_h_ops_tensor(aat, ev_u, m);
  eigenvecs_h_ops_tensor(ata, tv, n); eigenvals_h_ops_tensor(ata, ev_v, n);

  // VT = tv^T — parallel
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) vt[i*n+j] = tv[j*n+i];

  // Singular values from ATA eigenvalues (descending)
  for (int i = 0; i < min_mn; i++) {
    float val = (i < n) ? ev_v[n-1-i] : 0.0f;
    s[i] = (val > 1e-12f) ? sqrtf(val) : 0.0f;
  }

  // U — parallel copy + flip column order
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) u[i*m+j] = tu[i*m+(m-1-j)];

  // Sort singular values descending (insertion sort, usually small n)
  for (int i = 0; i < min_mn-1; i++) {
    for (int j = i+1; j < min_mn; j++) {
      if (s[i] < s[j]) {
        float ts = s[i]; s[i] = s[j]; s[j] = ts;
        avx2_swap_rows(u + i, u + j, m);       // swap columns i,j of u (non-contiguous — scalar)
        // scalar column swap for u
        for (int k = 0; k < m; k++) {
          float tv2 = u[k*m+i]; u[k*m+i] = u[k*m+j]; u[k*m+j] = tv2;
        }
        avx2_swap_rows(vt + i*n, vt + j*n, n); // swap rows i,j of vt
      }
    }
  }

  // Sign canonicalization — parallel over columns
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int j = 0; j < min_mn; j++) {
    if (u[0*m+j] > 0.0f) {
      for (int i = 0; i < m; i++) u[i*m+j] *= -1.0f;
    }
  }

  free(aat); free(ata); free(tu); free(tv); free(ev_u); free(ev_v);
}

void svd_ops(float* a, float* u, float* s, float* vt, int* shape) {
  compute_svd(a, u, s, vt, shape[0], shape[1]);
}

void batched_svd_ops(float* a, float* u, float* s, float* vt, int* shape, int ndim) {
  if (ndim < 2) { fprintf(stderr, "error: svd requires at least 2 dimensions\n"); exit(EXIT_FAILURE); }
  int m = shape[ndim-2], n = shape[ndim-1], min_mn = (m<n)?m:n, batch_size = 1;
  for (int i = 0; i < ndim-2; i++) batch_size *= shape[i];

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int b = 0; b < batch_size; b++) {
    int sh[2] = {m, n};
    svd_ops(a + b*m*n, u + b*m*m, s + b*min_mn, vt + b*n*n, sh);
  }
}

// Eigenvalue / Eigenvector Public API

void eigenvals_ops_tensor(float* a, float* eigenvals, size_t size)  { compute_eigenvals(a, eigenvals, size); }
void eigenvecs_ops_tensor(float* a, float* eigenvecs, size_t size)  { compute_eigenvecs(a, eigenvecs, size); }
void eigenvals_h_ops_tensor(float* a, float* eigenvals, size_t size){ compute_eigenvals_h(a, eigenvals, size); }
void eigenvecs_h_ops_tensor(float* a, float* eigenvecs, size_t size){ compute_eigenvecs_h(a, eigenvecs, size); }

void batched_eigenvals_ops(float* a, float* eigenvals, size_t size, size_t batch) {
  size_t ms = size * size;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t b = 0; b < batch; b++) eigenvals_ops_tensor(a + b*ms, eigenvals + b*size, size);
}

void batched_eigenvecs_ops(float* a, float* eigenvecs, size_t size, size_t batch) {
  size_t ms = size * size;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t b = 0; b < batch; b++) eigenvecs_ops_tensor(a + b*ms, eigenvecs + b*ms, size);
}

void batched_eigenvals_h_ops(float* a, float* eigenvals, size_t size, size_t batch) {
  size_t ms = size * size;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t b = 0; b < batch; b++) eigenvals_h_ops_tensor(a + b*ms, eigenvals + b*size, size);
}

void batched_eigenvecs_h_ops(float* a, float* eigenvecs, size_t size, size_t batch) {
  size_t ms = size * size;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t b = 0; b < batch; b++) eigenvecs_h_ops_tensor(a + b*ms, eigenvecs + b*ms, size);
}