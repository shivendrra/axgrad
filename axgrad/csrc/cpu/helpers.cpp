#include <stdlib.h>
#include <string.h>
#include "../inc/random.h"
#include "helpers.h"

static RNG global_rng;
static int rng_initialized = 0;

static inline void ensure_rng_initialized() {
  if (!rng_initialized) {
    rng_state(&global_rng, current_time_seed());
    rng_initialized = 1;
  }
}

// public function to set seed
void set_random_seed(uint64_t seed) {
  rng_state(&global_rng, seed);
  rng_initialized = 1;
}

// public function to reset RNG state
void reset_random_state() {
  rng_initialized = 0;
  ensure_rng_initialized();
}

void fill_randn(float* out, size_t size) {
  if (!out) return;
  ensure_rng_initialized();
  rng_randn(&global_rng, out, size);
}

void fill_uniform(float* out, float low, float high, size_t size) {
  if (!out || high <= low) return;
  ensure_rng_initialized();
  rng_rand_uniform(&global_rng, out, size, low, high);
}

void fill_randint(float* out, int low, int high, size_t size) {
  if (!out || high <= low) return;
  ensure_rng_initialized();

  // allocating temporary integer tensor
  int* temp = (int*)malloc(sizeof(int) * size);
  if (!temp) return;

  rng_randint(&global_rng, temp, size, low, high);  
  // converting integers to floats
  for (size_t i = 0; i < size; i++) {
    out[i] = (float)temp[i];
  }
  free(temp);
}

void ones_like_tensor_ops(float* out, size_t size) {
  if (!out) return;
  for (size_t i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_like_tensor_ops(float* out, size_t size) {
  if (!out) return;
  // using memset for better performance on large tensors
  if (size > 1000) {
    memset(out, 0, size * sizeof(float));
  } else {
    for (size_t i = 0; i < size; i++) {
      out[i] = 0.0f;
    }
  }
}

void ones_tensor_ops(float* out, size_t size) {
  if (!out) return;
  for (size_t i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_tensor_ops(float* out, size_t size) {
  if (!out) return;
  // using memset for better performance on large tensors
  if (size > 1000) {
    memset(out, 0, size * sizeof(float));
  } else {
    for (size_t i = 0; i < size; i++) {
      out[i] = 0.0f;
    }
  }
}

void fill_tensor_ops(float* out, float value, size_t size) {
  if (!out) return;

  // optimized filling for special values
  if (value == 0.0f) {
    zeros_tensor_ops(out, size);
    return;
  }
  if (value == 1.0f) {
    ones_tensor_ops(out, size);
    return;
  }

  // general case
  for (size_t i = 0; i < size; i++) {
    out[i] = value;
  }
}

void linspace_tensor_ops(float* out, float start, float step_size, size_t size) {
  if (!out) return;
  for (size_t i = 0; i < size; i++) {
    out[i] = start + (float)i * step_size;
  }
}