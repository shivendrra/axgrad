#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "ops_decomp.h"
#include "ops_tensor.h"
#include "ops_shape.h"

void det_ops_tensor(float* a, float* out, size_t size) {
  float det = 1.0f;
  // creating a copy to avoid modifying original matrix
  float* temp = (float*)malloc(size * size * sizeof(float));
  if (!temp) { *out = 0.0f; return; }
  for (size_t i = 0; i < size * size; ++i) { temp[i] = a[i]; }
  // gaussian elimination with partial pivoting
  for (size_t i = 0; i < size; ++i) {
    // finding pivot
    size_t pivot_row = i;
    float max_val = fabsf(temp[i * size + i]);
    for (size_t row = i + 1; row < size; ++row) {
      float val = fabsf(temp[row * size + i]);
      if (val > max_val) { max_val = val; pivot_row = row; }
    }
    // swapping rows if needed
    if (pivot_row != i) {
      for (size_t col = 0; col < size; ++col) {
        float tmp = temp[i * size + col];
        temp[i * size + col] = temp[pivot_row * size + col];
        temp[pivot_row * size + col] = tmp;
      }
      det = -det; // row swap changes sign
    }
    float pivot = temp[i * size + i];
    if (fabsf(pivot) < 1e-6f) { det = 0.0f; break; }
    det *= pivot;
    // eliminating column
    for (size_t j = i + 1; j < size; ++j) {
      float factor = temp[j * size + i] / pivot;
      for (size_t k = i; k < size; ++k) { temp[j * size + k] -= factor * temp[i * size + k]; }
    }
  }
  free(temp);
  *out = det;
}

void batched_det_ops(float* a, float* out, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float* mat = &a[b * mat_size];
    det_ops_tensor(mat, &out[b], size);
  }
}

// improved qr decomposition using householder reflections
static void qr_decomp(float* a, float* q, float* r, size_t n) {
  size_t i, j, k;
  float *v, *w, norm, tau, sum;
  size_t mat_size = n * n;
  // allocate working memory
  v = (float*)malloc(n * sizeof(float)), w = (float*)malloc(n * sizeof(float));
  if (!v || !w) { 
    free(v); free(w); 
    return; 
  }
  // initialize q as identity, r as copy of a
  for (i = 0; i < mat_size; ++i) { q[i] = 0.0f; r[i] = a[i]; }
  for (i = 0; i < n; ++i) q[i * n + i] = 1.0f;
  // householder qr decomposition
  for (k = 0; k < n - 1; ++k) {
    // compute householder vector for column k
    norm = 0.0f;
    for (i = k; i < n; ++i) {
      v[i] = r[i * n + k];
      norm += v[i] * v[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-10f) continue;
    // adjust sign to avoid cancellation
    if (v[k] >= 0.0f) norm = -norm;
    v[k] -= norm;
    // compute tau
    tau = 0.0f;
    for (i = k; i < n; ++i) tau += v[i] * v[i];
    if (tau < 1e-10f) continue;
    tau = 2.0f / tau;
    // apply householder transformation to r
    for (j = k; j < n; ++j) {
      sum = 0.0f;
      for (i = k; i < n; ++i) sum += v[i] * r[i * n + j];
      sum *= tau;
      for (i = k; i < n; ++i) r[i * n + j] -= sum * v[i];
    }
    // apply householder transformation to q
    for (j = 0; j < n; ++j) {
      sum = 0.0f;
      for (i = k; i < n; ++i) sum += v[i] * q[j * n + i];
      sum *= tau;
      for (i = k; i < n; ++i) q[j * n + i] -= sum * v[i];
    }
  }
  free(v);
  free(w);
}

// compute eigenvalues using qr iteration with shifts
static void compute_eigenvals(float* a, float* eigenvals, size_t size) {
  float *temp, *q, *r, *qt;
  size_t i, j, iter, mat_size = size * size;
  temp = (float*)malloc(4 * mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; 
    return; 
  }
  q = temp;
  r = temp + mat_size;
  qt = temp + 2 * mat_size;
  float* curr_a = temp + 3 * mat_size;
  // copy input matrix
  for (i = 0; i < mat_size; ++i) curr_a[i] = a[i];
  // qr iteration with wilkinson shift
  for (iter = 0; iter < 200; ++iter) {
    // compute wilkinson shift for last 2x2 submatrix
    float shift = 0.0f;
    if (size > 1) {
      float a11 = curr_a[(size-2) * size + (size-2)], a12 = curr_a[(size-2) * size + (size-1)], a21 = curr_a[(size-1) * size + (size-2)], a22 = curr_a[(size-1) * size + (size-1)];
      float trace = a11 + a22;
      float det = a11 * a22 - a12 * a21;
      float disc = trace * trace - 4.0f * det;
      if (disc >= 0.0f) {
        float sqrt_disc = sqrtf(disc);
        float lambda1 = (trace + sqrt_disc) / 2.0f, lambda2 = (trace - sqrt_disc) / 2.0f;
        // choose shift closer to a22
        shift = (fabsf(lambda1 - a22) < fabsf(lambda2 - a22)) ? lambda1 : lambda2;
      } else { shift = trace / 2.0f; }
    }
    // apply shift: a = a - shift * i
    for (i = 0; i < size; ++i) curr_a[i * size + i] -= shift;
    qr_decomp(curr_a, q, r, size);      // qr decomposition
    // compute rq + shift * i
    for (i = 0; i < mat_size; ++i) curr_a[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { curr_a[i * size + j] += r[i * size + k] * q[k * size + j]; }
      }
    }
    // add shift back
    for (i = 0; i < size; ++i) curr_a[i * size + i] += shift;
    // check convergence
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(curr_a[i * size + j]); }
    }
    if (off_diag < 1e-8f) break;
  }
  // extract eigenvalues from diagonal
  for (i = 0; i < size; ++i) eigenvals[i] = curr_a[i * size + i];
  free(temp);
}

// compute eigenvectors using qr iteration
static void compute_eigenvecs(float* a, float* eigenvecs, size_t size) {
  float *temp, *q, *r, *qt, *v_acc;
  size_t i, j, iter, mat_size = size * size;
  temp = (float*)malloc(5 * mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; 
    return; 
  }
  q = temp;
  r = temp + mat_size;
  qt = temp + 2 * mat_size;
  v_acc = temp + 3 * mat_size;
  float* curr_a = temp + 4 * mat_size;

  // copy input matrix and initialize eigenvectors as identity
  for (i = 0; i < mat_size; ++i) {
    curr_a[i] = a[i];
    eigenvecs[i] = 0.0f;
    v_acc[i] = 0.0f;
  }
  for (i = 0; i < size; ++i) {
    eigenvecs[i * size + i] = 1.0f;
    v_acc[i * size + i] = 1.0f;
  }
  // qr iteration
  for (iter = 0; iter < 200; ++iter) {
    // compute wilkinson shift
    float shift = 0.0f;
    if (size > 1) {
      float a11 = curr_a[(size-2) * size + (size-2)], a12 = curr_a[(size-2) * size + (size-1)], a21 = curr_a[(size-1) * size + (size-2)], a22 = curr_a[(size-1) * size + (size-1)];
      float trace = a11 + a22;
      float det = a11 * a22 - a12 * a21;
      float disc = trace * trace - 4.0f * det;
      if (disc >= 0.0f) {
        float sqrt_disc = sqrtf(disc);
        float lambda1 = (trace + sqrt_disc) / 2.0f, lambda2 = (trace - sqrt_disc) / 2.0f;
        shift = (fabsf(lambda1 - a22) < fabsf(lambda2 - a22)) ? lambda1 : lambda2;
      } else { shift = trace / 2.0f; }
    }
    // applying shift
    for (i = 0; i < size; ++i) curr_a[i * size + i] -= shift;
    // qr decomposition
    qr_decomp(curr_a, q, r, size);
    // update accumulated eigenvectors: v_acc = v_acc * q
    for (i = 0; i < mat_size; ++i) qt[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { qt[i * size + j] += v_acc[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < mat_size; ++i) v_acc[i] = qt[i];
    // compute rq + shift * i
    for (i = 0; i < mat_size; ++i) curr_a[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { curr_a[i * size + j] += r[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < size; ++i) curr_a[i * size + i] += shift;     // add shift back
    // check convergence
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) { for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(curr_a[i * size + j]); } }
    if (off_diag < 1e-8f) break;
  }
  // copy accumulated eigenvectors
  for (i = 0; i < mat_size; ++i) eigenvecs[i] = v_acc[i];
  free(temp);
}

// compute eigenvalues for hermitian/symmetric matrices using jacobi method
static void compute_eigenvals_h(float* a, float* eigenvals, size_t size) {
  float *temp;
  size_t i, j, k, iter, mat_size = size * size;
  temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; 
    return; 
  }
  for (i = 0; i < mat_size; ++i) temp[i] = a[i];    // copy input matrix
  // jacobi iteration
  for (iter = 0; iter < 100; ++iter) {
    // find largest off-diagonal element
    float max_val = 0.0f;
    size_t p = 0, q = 1;
    for (i = 0; i < size; ++i) {
      for (j = i + 1; j < size; ++j) {
        if (fabsf(temp[i * size + j]) > max_val) {
          max_val = fabsf(temp[i * size + j]);
          p = i; q = j;
        }
      }
    }

    if (max_val < 1e-10f) break;
    // computing jacobi rotation
    float theta = (temp[q * size + q] - temp[p * size + p]) / (2.0f * temp[p * size + q]);
    float t = 1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    if (theta < 0.0f) t = -t;
    float c = 1.0f / sqrtf(t * t + 1.0f);
    float s = t * c;
    // apply rotation
    for (k = 0; k < size; ++k) {
      if (k != p && k != q) {
        float temp_kp = temp[k * size + p], temp_kq = temp[k * size + q];
        temp[k * size + p] = temp[p * size + k] = c * temp_kp - s * temp_kq;
        temp[k * size + q] = temp[q * size + k] = s * temp_kp + c * temp_kq;
      }
    }
    float temp_pp = temp[p * size + p], temp_qq = temp[q * size + q], temp_pq = temp[p * size + q];
    temp[p * size + p] = c * c * temp_pp + s * s * temp_qq - 2.0f * s * c * temp_pq;
    temp[q * size + q] = s * s * temp_pp + c * c * temp_qq + 2.0f * s * c * temp_pq;
    temp[p * size + q] = temp[q * size + p] = 0.0f;
  }
  for (i = 0; i < size; ++i) eigenvals[i] = temp[i * size + i]; // extract eigenvalues from diagonal
  free(temp);
}

// computing eigenvectors for hermitian/symmetric matrices using jacobi method
static void compute_eigenvecs_h(float* a, float* eigenvecs, size_t size) {
  float *temp;
  size_t i, j, k, iter, mat_size = size * size;
  temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; 
    return; 
  }
  // copying input matrix and initialize eigenvectors as identity
  for (i = 0; i < mat_size; ++i) {
    temp[i] = a[i];
    eigenvecs[i] = 0.0f;
  }
  for (i = 0; i < size; ++i) eigenvecs[i * size + i] = 1.0f;
  // jacobi iteration
  for (iter = 0; iter < 100; ++iter) {
    // finding largest off-diagonal element
    float max_val = 0.0f;
    size_t p = 0, q = 1;
    for (i = 0; i < size; ++i) {
      for (j = i + 1; j < size; ++j) {
        if (fabsf(temp[i * size + j]) > max_val) {
          max_val = fabsf(temp[i * size + j]);
          p = i; q = j;
        }
      }
    }
    if (max_val < 1e-10f) break;
    // compute jacobi rotation
    float theta = (temp[q * size + q] - temp[p * size + p]) / (2.0f * temp[p * size + q]);
    float t = 1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    if (theta < 0.0f) t = -t;
    float c = 1.0f / sqrtf(t * t + 1.0f);
    float s = t * c;
    // apply rotation to matrix
    for (k = 0; k < size; ++k) {
      if (k != p && k != q) {
        float temp_kp = temp[k * size + p], temp_kq = temp[k * size + q];
        temp[k * size + p] = temp[p * size + k] = c * temp_kp - s * temp_kq;
        temp[k * size + q] = temp[q * size + k] = s * temp_kp + c * temp_kq;
      }
    }
    float temp_pp = temp[p * size + p], temp_qq = temp[q * size + q], temp_pq = temp[p * size + q];
    temp[p * size + p] = c * c * temp_pp + s * s * temp_qq - 2.0f * s * c * temp_pq;
    temp[q * size + q] = s * s * temp_pp + c * c * temp_qq + 2.0f * s * c * temp_pq;
    temp[p * size + q] = temp[q * size + p] = 0.0f;

    // apply rotation to eigenvectors
    for (k = 0; k < size; ++k) {
      float temp_kp = eigenvecs[k * size + p], temp_kq = eigenvecs[k * size + q];
      eigenvecs[k * size + p] = c * temp_kp - s * temp_kq;
      eigenvecs[k * size + q] = s * temp_kp + c * temp_kq;
    }
  }
  free(temp);
}

void eigenvals_ops_tensor(float* a, float* eigenvals, size_t size) {
  compute_eigenvals(a, eigenvals, size);
}

void batched_eigenvals_ops(float* a, float* eigenvals, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vals = &eigenvals[b * size];
    eigenvals_ops_tensor(mat, vals, size);
  }
}

void eigenvecs_ops_tensor(float* a, float* eigenvecs, size_t size) {
  compute_eigenvecs(a, eigenvecs, size);
}

void batched_eigenvecs_ops(float* a, float* eigenvecs, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vecs = &eigenvecs[b * mat_size];
    eigenvecs_ops_tensor(mat, vecs, size);
  }
}

void eigenvals_h_ops_tensor(float* a, float* eigenvals, size_t size) { compute_eigenvals_h(a, eigenvals, size); }

void batched_eigenvals_h_ops(float* a, float* eigenvals, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vals = &eigenvals[b * size];
    eigenvals_h_ops_tensor(mat, vals, size);
  }
}

void eigenvecs_h_ops_tensor(float* a, float* eigenvecs, size_t size) { compute_eigenvecs_h(a, eigenvecs, size); }

void batched_eigenvecs_h_ops(float* a, float* eigenvecs, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vecs = &eigenvecs[b * mat_size];
    eigenvecs_h_ops_tensor(mat, vecs, size);
  }
}