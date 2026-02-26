#ifndef __MATMUL__H__
#define __MATMUL__H__

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static inline float* aligned_malloc_32(size_t n) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  return (float*)_aligned_malloc(n, 32);
#else
  void* ptr = NULL;
  if (posix_memalign(&ptr, 32, n) != 0) return NULL;
  return (float*)ptr;
#endif
}

static inline void aligned_free(void* p) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  _aligned_free(p);
#else
  free(p);
#endif
}

extern "C" {
  // transpose matmul & parallel kernels
  void transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
  void hybrid_transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
}

#endif  //!__MATMUL__H__