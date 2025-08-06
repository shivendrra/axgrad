#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "ops_decomp.h"
#include "ops_tensor.h"
#include "ops_shape.h"

static void compute_svd(float* a, float* u, float* s, float* vt, int m, int n) {
  int min_mn = (m < n) ? m : n;
  float *aat = (float*)malloc(m * m * sizeof(float)), *ata = (float*)malloc(n * n * sizeof(float)), *temp_u = (float*)malloc(m * m * sizeof(float)), *temp_v = (float*)malloc(n * n * sizeof(float));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      aat[i * m + j] = 0.0f;
      for (int k = 0; k < n; ++k) aat[i * m + j] += a[i * n + k] * a[j * n + k];
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      ata[i * n + j] = 0.0f;
      for (int k = 0; k < m; ++k) ata[i * n + j] += a[k * n + i] * a[k * n + j];
    }
  }
  float *eigenvals_u = (float*)malloc(m * sizeof(float)), *eigenvals_v = (float*)malloc(n * sizeof(float));
  eigenvecs_h_ops_tensor(aat, temp_u, m);
  eigenvals_h_ops_tensor(aat, eigenvals_u, m);
  eigenvecs_h_ops_tensor(ata, temp_v, n);
  eigenvals_h_ops_tensor(ata, eigenvals_v, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) vt[i * n + j] = temp_v[j * n + i];
  }

  for (int i = 0; i < min_mn; ++i) {
    float val = (i < n) ? eigenvals_v[n - 1 - i] : 0.0f;
    s[i] = (val > 1e-12f) ? sqrtf(val) : 0.0f;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) u[i * m + j] = temp_u[i * m + (m - 1 - j)];
  }
  
  for (int i = 0; i < min_mn - 1; ++i) {
    for (int j = i + 1; j < min_mn; ++j) {
      if (s[i] < s[j]) {
        float temp_s = s[i]; s[i] = s[j]; s[j] = temp_s;
        for (int k = 0; k < m; ++k) {
          float temp_u_val = u[k * m + i]; u[k * m + i] = u[k * m + j]; u[k * m + j] = temp_u_val;
        }
        for (int k = 0; k < n; ++k) {
          float temp_v_val = vt[i * n + k]; vt[i * n + k] = vt[j * n + k]; vt[j * n + k] = temp_v_val;
        }
      }
    }
  }

  for (int j = 0; j < min_mn; ++j) {
    if (u[0 * m + j] > 0.0f) {
      for (int i = 0; i < m; ++i) u[i * m + j] *= -1.0f;
    }
  }
  free(aat); free(ata); free(temp_u); free(temp_v); free(eigenvals_u); free(eigenvals_v);
}

static void compute_eigenvecs_h(float* a, float* eigenvecs, size_t size) {
  float *temp;
  size_t i, j, k, iter, mat_size = size * size;
  temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; 
    return; 
  }
  for (i = 0; i < mat_size; ++i) {
    temp[i] = a[i];
    eigenvecs[i] = 0.0f;
  }
  for (i = 0; i < size; ++i) eigenvecs[i * size + i] = 1.0f;
  for (iter = 0; iter < 1000; ++iter) {
    float max_val = 0.0f;
    size_t p = 0, q = 1;
    for (i = 0; i < size; ++i) {
      for (j = i + 1; j < size; ++j) {
        float val = fabsf(temp[i * size + j]);
        if (val > max_val) {
          max_val = val;
          p = i; q = j;
        }
      }
    }
    if (max_val < 1e-14f) break;
    float app = temp[p * size + p], aqq = temp[q * size + q], apq = temp[p * size + q];
    float theta, t, c, s;
    if (fabsf(apq) < 1e-15f) {
      c = 1.0f; s = 0.0f;
    } else {
      theta = (aqq - app) / (2.0f * apq);
      t = (theta >= 0.0f) ? 1.0f / (theta + sqrtf(theta * theta + 1.0f)) : 1.0f / (theta - sqrtf(theta * theta + 1.0f));
      c = 1.0f / sqrtf(t * t + 1.0f);
      s = t * c;
    }
    for (k = 0; k < size; ++k) {
      if (k != p && k != q) {
        float akp = temp[k * size + p], akq = temp[k * size + q];
        temp[k * size + p] = temp[p * size + k] = c * akp - s * akq;
        temp[k * size + q] = temp[q * size + k] = s * akp + c * akq;
      }
    }
    temp[p * size + p] = c * c * app + s * s * aqq - 2.0f * s * c * apq;
    temp[q * size + q] = s * s * app + c * c * aqq + 2.0f * s * c * apq;
    temp[p * size + q] = temp[q * size + p] = 0.0f;
    for (k = 0; k < size; ++k) {
      float vkp = eigenvecs[k * size + p], vkq = eigenvecs[k * size + q];
      eigenvecs[k * size + p] = c * vkp - s * vkq;
      eigenvecs[k * size + q] = s * vkp + c * vkq;
    }
  }
  float* eigenvals = (float*)malloc(size * sizeof(float));
  size_t* indices = (size_t*)malloc(size * sizeof(size_t));
  for (i = 0; i < size; ++i) {
    eigenvals[i] = temp[i * size + i];
    indices[i] = i;
  }
  for (i = 0; i < size - 1; ++i) {
    for (j = i + 1; j < size; ++j) {
      if (eigenvals[indices[i]] > eigenvals[indices[j]]) {
        size_t tmp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_idx;
      }
    }
  }
  float* temp_vecs = (float*)malloc(mat_size * sizeof(float));
  for (i = 0; i < mat_size; ++i) temp_vecs[i] = eigenvecs[i];
  for (j = 0; j < size; ++j) {
    size_t src_col = indices[j];
    for (i = 0; i < size; ++i) eigenvecs[i * size + j] = temp_vecs[i * size + src_col];
  }

  for (j = 0; j < size; ++j) {
    if (eigenvecs[0 * size + j] < 0.0f) {
      for (i = 0; i < size; ++i) eigenvecs[i * size + j] *= -1.0f;
    }
  }

  free(eigenvals); free(indices); free(temp_vecs); free(temp);
}

static void compute_chol(float* a, float* l, int n) {
  memset(l, 0, n * n * sizeof(float));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (i == j) {
        float sum = 0.0f;
        for (int k = 0; k < j; ++k) sum += l[i * n + k] * l[i * n + k];
        float val = a[i * n + i] - sum;
        if (val <= 1e-12f) {
          l[i * n + j] = 0.0f;
          return;
        }
        l[i * n + j] = sqrtf(val);
      } else {
        float sum = 0.0f;
        for (int k = 0; k < j; ++k) sum += l[i * n + k] * l[j * n + k];
        if (fabsf(l[j * n + j]) < 1e-12f) l[i * n + j] = 0.0f;
        else l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
      }
    }
  }
}

void svd_ops(float* a, float* u, float* s, float* vt, int* shape) {
  int m = shape[0], n = shape[1];
  compute_svd(a, u, s, vt, m, n);
}

void batched_svd_ops(float* a, float* u, float* s, float* vt, int* shape, int ndim) {
  if (ndim < 2) {
    fprintf(stderr, "error: svd requires at least 2 dimensions\n");
    exit(EXIT_FAILURE);
  }

  int m = shape[ndim - 2], n = shape[ndim - 1];
  int min_mn = (m < n) ? m : n, batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];

  int a_matrix_size = m * n, u_matrix_size = m * m, s_vector_size = min_mn, vt_matrix_size = n * n;  
  for (int batch = 0; batch < batch_size; batch++) {
    float *a_batch = a + batch * a_matrix_size, *u_batch = u + batch * u_matrix_size;
    float *s_batch = s + batch * s_vector_size, *vt_batch = vt + batch * vt_matrix_size;
    int matrix_shape[2] = {m, n};
    svd_ops(a_batch, u_batch, s_batch, vt_batch, matrix_shape);
  }
}

void chol_ops(float* a, float* l, int* shape) {
  int n = shape[0];
  compute_chol(a, l, n);
}

void batched_chol_ops(float* a, float* l, int* shape, int ndim) {
  if (ndim < 2) {
    fprintf(stderr, "error: cholesky decomposition requires at least 2 dimensions\n");
    exit(EXIT_FAILURE);
  }

  int n = shape[ndim - 1], batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];
  int matrix_size = n * n;

  for (int batch = 0; batch < batch_size; batch++) {
    float *a_batch = a + batch * matrix_size, *l_batch = l + batch * matrix_size;
    int matrix_shape[2] = {n, n};
    chol_ops(a_batch, l_batch, matrix_shape);
  }
}

void qr_decomp_ops(float* a, float* q, float* r, int* shape) {
  int m = shape[0], n = shape[1];  // rows, cols
  float* work = (float*)malloc(m * n * sizeof(float));
  memcpy(work, a, m * n * sizeof(float));
  memset(q, 0, m * m * sizeof(float));  // initialize q as identity matrix (m x m)
  for (int i = 0; i < m; i++) q[i * m + i] = 1.0f;
  memset(r, 0, m * n * sizeof(float));    // initialize r as zero matrix (m x n)
  for (int k = 0; k < n && k < m; k++) {  // modified gram-schmidt process
    float norm = 0.0f;  // compute column norm for r[k][k]
    for (int i = 0; i < m; i++) {
      float val = work[i * n + k];
      norm += val * val;
    }
    norm = sqrtf(norm);
    r[k * n + k] = norm;

    // normalize column k to get q_k
    if (norm > 1e-6f) { for (int i = 0; i < m; i++) { q[i * m + k] = work[i * n + k] / norm; } }
    for (int j = k + 1; j < n; j++) { // orthogonalize remaining columns
      float dot = 0.0f; // compute r[k][j] = q_k^T * a_j (dot product)
      for (int i = 0; i < m; i++) dot += q[i * m + k] * work[i * n + j];
      r[k * n + j] = dot;
      for (int i = 0; i < m; i++) work[i * n + j] -= dot * q[i * m + k]; // subtract projection: a_j = a_j - r[k][j] * q_k
    }
  }  
  free(work);
}

// batched qr decomposition for n-dimensional arrays, processes matrices along the last two dimensions
void batched_qr_decomp_ops(float* a, float* q, float* r, int* shape, int ndim) {
  if (ndim < 2) {
    fprintf(stderr, "error: qr decomposition requires at least 2 dimensions\n");
    exit(EXIT_FAILURE);
  }
  // matrix dimensions are the last two
  int m = shape[ndim - 2], n = shape[ndim - 1]; // rows, cols
  int batch_size = 1; // compute batch size (product of all leading dimensions)
  for (int i = 0; i < ndim - 2; i++) { batch_size *= shape[i]; }
  int a_matrix_size = m * n, q_matrix_size = m * m, r_matrix_size = m * n;
  // process each matrix in the batch
  for (int batch = 0; batch < batch_size; batch++) {
    float *a_batch = a + batch * a_matrix_size, *q_batch = q + batch * q_matrix_size, *r_batch = r + batch * r_matrix_size;
    int matrix_shape[2] = {m, n};
    qr_decomp_ops(a_batch, q_batch, r_batch, matrix_shape);
  }
}

void lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape) {
  int n = shape[0];  // assuming square matrix n x n
  memcpy(u, a, n * n * sizeof(float));    // copy input to u matrix
  memset(l, 0, n * n * sizeof(float));    // initialize l as identity matrix
  for (int i = 0; i < n; i++) l[i * n + i] = 1.0f;
  for (int i = 0; i < n; i++) p[i] = i; // initialize permutation array
  // gaussian elimination with partial pivoting
  for (int k = 0; k < n - 1; k++) {
    int pivot_row = k;
    float max_val = fabsf(u[k * n + k]);
    for (int i = k + 1; i < n; i++) {
      if (fabsf(u[i * n + k]) > max_val) {
        max_val = fabsf(u[i * n + k]);
        pivot_row = i;
      }
    }
    if (pivot_row != k) { // swap rows in u matrix
      for (int j = 0; j < n; j++) {
        float temp = u[k * n + j];
        u[k * n + j] = u[pivot_row * n + j];
        u[pivot_row * n + j] = temp;
      }
      for (int j = 0; j < k; j++) { // swap rows in l matrix (only lower part)
        float temp = l[k * n + j];
        l[k * n + j] = l[pivot_row * n + j];
        l[pivot_row * n + j] = temp;
      }
      int temp_p = p[k];  // update permutation
      p[k] = p[pivot_row];
      p[pivot_row] = temp_p;
    }
    for (int i = k + 1; i < n; i++) {
      if (fabsf(u[k * n + k]) > 1e-9f) {
        float factor = u[i * n + k] / u[k * n + k];
        l[i * n + k] = factor;
        for (int j = k; j < n; j++) u[i * n + j] -= factor * u[k * n + j];
      }
    }
  }
}

void batched_lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape, int ndim) {
  if (ndim < 2) {
    fprintf(stderr, "error: lu decomposition requires at least 2 dimensions\n");
    exit(EXIT_FAILURE);
  }

  int n = shape[ndim - 1], batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) { batch_size *= shape[i]; }
  int matrix_size = n * n;
  for (int batch = 0; batch < batch_size; batch++) {
    float *a_batch = a + batch * matrix_size, *l_batch = l + batch * matrix_size, *u_batch = u + batch * matrix_size;
    int* p_batch = p + batch * n;
    int matrix_shape[2] = {n, n};
    lu_decomp_ops(a_batch, l_batch, u_batch, p_batch, matrix_shape);
  }
}

static void compute_eigenvals(float* a, float* eigenvals, size_t size) {
  float *temp, *q, *r;
  size_t i, j, iter, mat_size = size * size;
  temp = (float*)malloc(3 * mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; 
    return; 
  }
  q = temp;
  r = temp + mat_size;
  float* curr_a = temp + 2 * mat_size;
  for (i = 0; i < mat_size; ++i) curr_a[i] = a[i];
  for (iter = 0; iter < 200; ++iter) {
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
    for (i = 0; i < size; ++i) curr_a[i * size + i] -= shift;
    int qr_shape[2] = {(int)size, (int)size};
    qr_decomp_ops(curr_a, q, r, qr_shape);
    for (i = 0; i < mat_size; ++i) curr_a[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { curr_a[i * size + j] += r[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < size; ++i) curr_a[i * size + i] += shift;
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(curr_a[i * size + j]); }
    }
    if (off_diag < 1e-8f) break;
  }
  for (i = 0; i < size; ++i) eigenvals[i] = curr_a[i * size + i];
  free(temp);
}

static void compute_eigenvecs(float* a, float* eigenvecs, size_t size) {
  float *temp, *q, *r, *qt, *v_acc;
  size_t i, j, iter, mat_size = size * size;
  temp = (float*)malloc(5 * mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; 
    return;
  }
  q = temp, r = temp + mat_size, qt = temp + 2 * mat_size, v_acc = temp + 3 * mat_size;
  float* curr_a = temp + 4 * mat_size;
  for (i = 0; i < mat_size; ++i) {
    curr_a[i] = a[i];
    eigenvecs[i] = 0.0f;
    v_acc[i] = 0.0f;
  }
  for (i = 0; i < size; ++i) {
    eigenvecs[i * size + i] = 1.0f;
    v_acc[i * size + i] = 1.0f;
  }
  for (iter = 0; iter < 200; ++iter) {
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
    for (i = 0; i < size; ++i) curr_a[i * size + i] -= shift;
    int qr_shape[2] = {(int)size, (int)size};
    qr_decomp_ops(curr_a, q, r, qr_shape);
    for (i = 0; i < mat_size; ++i) qt[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { qt[i * size + j] += v_acc[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < mat_size; ++i) v_acc[i] = qt[i];
    for (i = 0; i < mat_size; ++i) curr_a[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { curr_a[i * size + j] += r[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < size; ++i) curr_a[i * size + i] += shift;
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) { for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(curr_a[i * size + j]); } }
    if (off_diag < 1e-8f) break;
  }
  for (i = 0; i < mat_size; ++i) eigenvecs[i] = v_acc[i];
  free(temp);
}

static void compute_eigenvals_h(float* a, float* eigenvals, size_t size) {
  float *temp;
  size_t i, j, k, iter, mat_size = size * size;
  temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; 
    return; 
  }
  for (i = 0; i < mat_size; ++i) temp[i] = a[i];
  for (iter = 0; iter < 1000; ++iter) {
    float max_val = 0.0f;
    size_t p = 0, q = 1;
    for (i = 0; i < size; ++i) {
      for (j = i + 1; j < size; ++j) {
        float val = fabsf(temp[i * size + j]);
        if (val > max_val) {
          max_val = val;
          p = i; q = j;
        }
      }
    }
    if (max_val < 1e-14f) break;
    float app = temp[p * size + p], aqq = temp[q * size + q], apq = temp[p * size + q];
    float theta, t, c, s;
    if (fabsf(apq) < 1e-15f) {
      c = 1.0f; s = 0.0f;
    } else {
      theta = (aqq - app) / (2.0f * apq);
      t = (theta >= 0.0f) ? 1.0f / (theta + sqrtf(theta * theta + 1.0f)) : 1.0f / (theta - sqrtf(theta * theta + 1.0f));
      c = 1.0f / sqrtf(t * t + 1.0f);
      s = t * c;
    }
    for (k = 0; k < size; ++k) {
      if (k != p && k != q) {
        float akp = temp[k * size + p], akq = temp[k * size + q];
        temp[k * size + p] = temp[p * size + k] = c * akp - s * akq;
        temp[k * size + q] = temp[q * size + k] = s * akp + c * akq;
      }
    }
    temp[p * size + p] = c * c * app + s * s * aqq - 2.0f * s * c * apq;
    temp[q * size + q] = s * s * app + c * c * aqq + 2.0f * s * c * apq;
    temp[p * size + q] = temp[q * size + p] = 0.0f;
  }
  for (i = 0; i < size; ++i) eigenvals[i] = temp[i * size + i];
  for (i = 0; i < size - 1; ++i) {
    for (j = i + 1; j < size; ++j) {
      if (eigenvals[i] > eigenvals[j]) {
        float tmp = eigenvals[i];
        eigenvals[i] = eigenvals[j];
        eigenvals[j] = tmp;
      }
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