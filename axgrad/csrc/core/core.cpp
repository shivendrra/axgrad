#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <immintrin.h>
#include "core.h"
#include "dtype.h"
#include "contiguous.h"

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3")

// -- Internal helpers --------------------------------------

// Allocate and zero a Tensor struct with shape + strides arrays.
// Returns NULL on any allocation failure (caller must handle).
static Tensor* alloc_tensor_shell(size_t ndim) {
  Tensor* t = (Tensor*)malloc(sizeof(Tensor));
  if (!t) return NULL;
  t->shape = (int*)malloc(ndim * sizeof(int));
  t->strides = (int*)malloc(ndim * sizeof(int));
  if (!t->shape || !t->strides) {
    if (t->shape)   free(t->shape);
    if (t->strides) free(t->strides);
    free(t);
    return NULL;
  }
  t->data = NULL;
  t->is_view = 0;
  t->ndim = ndim;
  t->size = 0;
  return t;
}

// Compute row-major contiguous strides from shape into t->strides.
static void fill_contiguous_strides(Tensor* t) {
  int s = 1;
  for (int i = (int)t->ndim - 1; i >= 0; i--) {
    t->strides[i] = s;
    s *= t->shape[i];
  }
}

// Fatal allocation failure handler — prints message and exits.
static void oom(const char* ctx) {
  fprintf(stderr, "Memory allocation failed: %s\n", ctx);
  exit(EXIT_FAILURE);
}

// -- create_tensor --------------------------------------──

Tensor* create_tensor(float* data, size_t ndim, int* shape, size_t size, dtype_t dtype) {
  if (!data || !ndim || !size) { fprintf(stderr, "Invalid input parameters!\n"); exit(EXIT_FAILURE); }

  Tensor* self = alloc_tensor_shell(ndim);
  if (!self) oom("create_tensor shell");

  self->size = size;
  self->dtype = dtype;
  self->data = allocate_dtype_tensor(dtype, size);
  if (!self->data) { free(self->shape); free(self->strides); free(self); oom("create_tensor data"); }

  // convert_from_float32 is already AVX2+OpenMP accelerated
  convert_from_float32(data, self->data, dtype, size);

  memcpy(self->shape, shape, ndim * sizeof(int));
  fill_contiguous_strides(self);
  return self;
}

// -- cast_tensor ----------------------------------------─
// cast_tensor_simple is the fast path: single allocation + direct dtype cast.
// cast_tensor goes via float32 intermediate — kept for API compat but simplified.

Tensor* cast_tensor_simple(Tensor* self, dtype_t new_dtype) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }

  void* new_data = cast_tensor_dtype(self->data, self->dtype, new_dtype, self->size);
  if (!new_data) oom("cast_tensor_simple");

  Tensor* result = alloc_tensor_shell(self->ndim);
  if (!result) { free(new_data); oom("cast_tensor_simple shell"); }

  result->data = new_data;
  result->dtype = new_dtype;
  result->size = self->size;
  memcpy(result->shape, self->shape, self->ndim * sizeof(int));
  memcpy(result->strides, self->strides, self->ndim * sizeof(int));
  return result;
}

Tensor* cast_tensor(Tensor* self, dtype_t new_dtype) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }
  // Delegate to the faster simple path — both do the same thing
  return cast_tensor_simple(self, new_dtype);
}

// -- Contiguous helpers ------------------------------------─

int is_contiguous_tensor(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }
  return is_contiguous(self);
}

// Core of both contiguous_tensor (already-contiguous and non-contiguous paths).
// Allocates new tensor, copies shape, computes strides, then either memcpy or
// contiguous_tensor_ops depending on whether src is already contiguous.
Tensor* contiguous_tensor(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }

  Tensor* result = alloc_tensor_shell(self->ndim);
  if (!result) oom("contiguous_tensor shell");

  result->dtype = self->dtype;
  result->size = self->size;

  size_t elem_size = get_dtype_size(self->dtype);
  result->data = malloc(self->size * elem_size);
  if (!result->data) { free(result->shape); free(result->strides); free(result); oom("contiguous_tensor data"); }

  memcpy(result->shape, self->shape, self->ndim * sizeof(int));
  fill_contiguous_strides(result);

  if (is_contiguous(self)) {
    // AVX2 bulk copy via conv_f32_to_f32 fast-path inside convert_from_float32,
    // or plain memcpy for non-float32 (already the fastest path).
    memcpy(result->data, self->data, self->size * elem_size);
  } else {
    // contiguous_tensor_ops is already AVX2+OpenMP parallelised
    contiguous_tensor_ops(self->data, result->data, self->strides, self->shape, self->ndim, elem_size);
  }
  return result;
}

void make_contiguous_inplace_tensor(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }
  make_contiguous_inplace(self);  // already AVX2+OpenMP
}

// -- View / Reshape / Slice ----------------------------------

Tensor* view_tensor(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }

  Tensor* view = alloc_tensor_shell(self->ndim);
  if (!view) oom("view_tensor");

  view->data = self->data;   // shared pointer
  view->dtype = self->dtype;
  view->size = self->size;
  view->is_view = 1;
  memcpy(view->shape, self->shape, self->ndim * sizeof(int));
  memcpy(view->strides, self->strides, self->ndim * sizeof(int));
  return view;
}

Tensor* reshape_view(Tensor* self, int* new_shape, size_t new_ndim) {
  if (!self || !new_shape) { fprintf(stderr, "Null pointer!\n"); exit(EXIT_FAILURE); }

  size_t new_size = 1;
  for (size_t i = 0; i < new_ndim; i++) {
    if (new_shape[i] <= 0) { fprintf(stderr, "Invalid shape dimension: %d\n", new_shape[i]); return NULL; }
    new_size *= new_shape[i];
  }
  if (new_size != self->size) {
    fprintf(stderr, "Cannot reshape tensor of size %zu into size %zu\n", self->size, new_size);
    return NULL;
  }
  if (!is_contiguous(self)) {
    fprintf(stderr, "Cannot reshape non-contiguous tensor. Use contiguous() first.\n");
    return NULL;
  }

  Tensor* r = alloc_tensor_shell(new_ndim);
  if (!r) oom("reshape_view");

  r->data = self->data;
  r->dtype = self->dtype;
  r->size = new_size;
  r->is_view = 1;
  memcpy(r->shape, new_shape, new_ndim * sizeof(int));
  fill_contiguous_strides(r);
  return r;
}

Tensor* slice_view(Tensor* self, int* start, int* end, int* step) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }

  Tensor* sl = alloc_tensor_shell(self->ndim);
  if (!sl) oom("slice_view");

  size_t data_offset = 0, new_size = 1;
  for (size_t i = 0; i < self->ndim; i++) {
    int s = (start && start[i] >= 0) ? start[i] : 0;
    int e = (end && end[i] >= 0) ? end[i] : self->shape[i];
    int st = (step && step[i] > 0) ? step[i] : 1;

    if (s >= self->shape[i]) s = self->shape[i] - 1;
    if (e > self->shape[i]) e = self->shape[i];
    if (s < 0) s = 0;

    sl->shape[i] = (e - s + st - 1) / st;
    sl->strides[i] = self->strides[i] * st;
    new_size *= sl->shape[i];
    data_offset += s * self->strides[i];
  }

  size_t elem_size = get_dtype_size(self->dtype);
  sl->data = (char*)self->data + data_offset * elem_size;
  sl->dtype = self->dtype;
  sl->size = new_size;
  sl->is_view = 1;
  return sl;
}

// -- copy_tensor ----------------------------------------─

Tensor* copy_tensor(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }

  Tensor* result = alloc_tensor_shell(self->ndim);
  if (!result) oom("copy_tensor shell");

  result->dtype = self->dtype;
  result->size = self->size;

  size_t elem_size = get_dtype_size(self->dtype);
  result->data = malloc(self->size * elem_size);
  if (!result->data) { free(result->shape); free(result->strides); free(result); oom("copy_tensor data"); }

  // AVX2 bulk copy for float32, or plain memcpy for other dtypes
  if (self->dtype == DTYPE_FLOAT32) {
    float* src = (float*)self->data;
    float* dst = (float*)result->data;
    size_t n = self->size, i = 0;
    for (; i + 8 <= n; i += 8)
      _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
    for (; i < n; i++) dst[i] = src[i];
  } else {
    memcpy(result->data, self->data, self->size * elem_size);
  }

  memcpy(result->shape, self->shape, self->ndim * sizeof(int));
  memcpy(result->strides, self->strides, self->ndim * sizeof(int));
  return result;
}

// -- Delete / Utility ------------------------------------──

void delete_tensor(Tensor* self) {
  if (!self) return;
  if (!self->is_view && self->data) free(self->data);
  if (self->shape) free(self->shape);
  if (self->strides) free(self->strides);
  free(self);
}

void delete_shape(Tensor* self) {
  if (self && self->shape) {
    free(self->shape);
    self->shape = NULL;
  }
}
void delete_data(Tensor* self) {
  if (self && self->data) {
    free(self->data);
    self->data= NULL;
  }
}

void delete_strides(Tensor* self) {
  if (self && self->strides) {
    free(self->strides);
    self->strides = NULL;
  }
}

float* out_data(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }
  return convert_to_float32(self->data, self->dtype, self->size); // AVX2 accelerated
}

int* out_shape(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }
  return self->shape;
}

int* out_strides(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }
  return self->strides;
}

int out_size(Tensor* self) {
  if (!self) { fprintf(stderr, "Null tensor!\n"); exit(EXIT_FAILURE); }
  return (int)self->size;
}

int is_view_tensor(Tensor* self) { return self ? self->is_view : 0; }

// -- Indexing ------------------------------------------─

int get_linear_index(Tensor* self, int* indices) {
  if (!self || !indices) { fprintf(stderr, "Null pointer!\n"); exit(EXIT_FAILURE); }
  int idx = 0;
  for (int i = 0; i < (int)self->ndim; i++) {
    if (indices[i] < 0) indices[i] += self->shape[i];
    if (indices[i] < 0 || indices[i] >= self->shape[i]) {
      fprintf(stderr, "Index %d out of bounds for dim %d (size %d)\n", indices[i], i, self->shape[i]);
      exit(EXIT_FAILURE);
    }
    idx += indices[i] * self->strides[i];
  }
  return idx;
}

float get_item_tensor(Tensor* self, int* indices) {
  if (!self || !indices) { fprintf(stderr, "Null pointer!\n"); exit(EXIT_FAILURE); }
  int li = get_linear_index(self, indices);
  return dtype_to_float32(self->data, self->dtype, li);  // single dispatch
}

void set_item_tensor(Tensor* self, int* indices, float value) {
  if (!self || !indices) { fprintf(stderr, "Null pointer!\n"); exit(EXIT_FAILURE); }
  int li = get_linear_index(self, indices);
  float32_to_dtype(value, self->data, self->dtype, li);  // single dispatch
}

// Print

static void format_element_by_dtype(void* data, dtype_t dtype, size_t index, char* buf) {
  switch (dtype) {
    case DTYPE_FLOAT32: sprintf(buf, "%.3f", ((float*)data)[index]); break;
    case DTYPE_FLOAT64: sprintf(buf, "%.4f", ((double*)data)[index]); break;
    case DTYPE_INT8: sprintf(buf, "%d.", ((int8_t*)data)[index]); break;
    case DTYPE_INT16: sprintf(buf, "%d.", ((int16_t*)data)[index]); break;
    case DTYPE_INT32: sprintf(buf, "%d.", ((int32_t*)data)[index]); break;
    case DTYPE_INT64: sprintf(buf, "%lld.", (long long)((int64_t*)data)[index]); break;
    case DTYPE_UINT8: sprintf(buf, "%u.", ((uint8_t*)data)[index]); break;
    case DTYPE_UINT16: sprintf(buf, "%u.", ((uint16_t*)data)[index]); break;
    case DTYPE_UINT32: sprintf(buf, "%u.", ((uint32_t*)data)[index]); break;
    case DTYPE_UINT64: sprintf(buf, "%llu.", (unsigned long long)((uint64_t*)data)[index]); break;
    case DTYPE_BOOL: sprintf(buf, "%s", ((uint8_t*) data)[index] ? "True" : "False"); break;
    default: sprintf(buf, "0"); break;
  }
}

static void truncate_row(Tensor* self, int row_offset, int length, int max_display, char* result) {
  strcat(result, "  [");
  char buf[32];
  if (length > max_display) {
    int half = max_display / 2;
    for (int i = 0; i < half; i++) {
      format_element_by_dtype(self->data, self->dtype, row_offset + i, buf);
      strcat(result, buf); strcat(result, ", ");
    }
    strcat(result, "...");
    for (int i = length - half; i < length; i++) {
      format_element_by_dtype(self->data, self->dtype, row_offset + i, buf);
      strcat(result, ", "); strcat(result, buf);
    }
    size_t len = strlen(result);
    if (len >= 2 && result[len - 2] == ',') result[len - 2] = '\0';
  } else {
    for (int i = 0; i < length; i++) {
      format_element_by_dtype(self->data, self->dtype, row_offset + i, buf);
      strcat(result, buf);
      if (i != length - 1) strcat(result, ", ");
    }
  }
  strcat(result, "]");
}

static void format_tensor(Tensor* self, const int* shape, int ndim, int level, int offset, char* result) {
  if (ndim == 1) { truncate_row(self, offset, shape[0], 8, result); return; }

  strcat(result, "[\n");
  int stride = 1;
  for (int i = 1; i < ndim; i++) stride *= shape[i];
  int show = shape[0] > 8 ? 4 : shape[0];

  for (int i = 0; i < show; i++) {
    if (i > 0) strcat(result, ",\n");
    for (int j = 0; j <= level; j++) strcat(result, "  ");
    format_tensor(self, shape + 1, ndim - 1, level + 1, offset + i * stride, result);
  }
  if (shape[0] > 8) {
    strcat(result, ",\n");
    for (int j = 0; j <= level; j++) strcat(result, "  ");
    strcat(result, "...");
    for (int i = shape[0] - 4; i < shape[0]; i++) {
      strcat(result, ",\n");
      for (int j = 0; j <= level; j++) strcat(result, "  ");
      format_tensor(self, shape + 1, ndim - 1, level + 1, offset + i * stride, result);
    }
  }
  strcat(result, "\n");
  for (int j = 0; j < level; j++) strcat(result, "  ");
  strcat(result, "]");
}

void print_tensor(Tensor* self) {
  if (!self) { printf("axon.tensor(NULL)\n"); return; }
  char result[8192] = "";
  format_tensor(self, self->shape, self->ndim, 0, 0, result);
  printf("axon.tensor(%s, dtype=%s)\n", result, get_dtype_name(self->dtype));
}