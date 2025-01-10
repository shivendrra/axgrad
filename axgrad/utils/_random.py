from typing import *
import math, time

class RNG:
  # simple random number generator fully deterministic, similar to python's random lib
  # used for building higher order random class
  def __init__(self, seed):
    self.state = seed

  def random_u32(self):
    # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
    # doing & 0xFFFFFFFF is the same as cast to uint32 in C
    self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
    self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
    self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
    return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

  def random(self):
    # random float32 in [0, 1)
    return (self.random_u32() >> 8) / 16777216.0

  def uniform(self, a=0.0, b=1.0):
    # random float32 in [a, b)
    return a + (b-a) * self.random()

  def sample(self, a: Sequence, k: int) -> list:
    # random sample of `k` elements from the sequence `a`
    if k > len(a):
      raise ValueError("Sample size cannot be greater than the population size.")
    selected = list(a)
    for i in range(k):
      j = int(self.random() * (len(a) - i)) + i
      selected[i], selected[j] = selected[j], selected[i]
    return selected[:k]

class random(RNG):
  def __init__(self, seed=None):
    if seed is None:
      # defaulting seed based on the current time
      seed = int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF
    super().__init__(seed)

  def seed(self, seed=None):
    if seed is None:
      seed = int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF
    self.state = seed

  def randint(self, low, high=None, size=None):
    # Return random integers from `low` (inclusive) to `high` (exclusive).
    # If high is None, return integers from 0 to `low`.
    if high is None:
      high = low
      low = 0
    return self._generate_random(lambda: low + int(self.random() * (high - low)), size)

  def rand(self, *size):
    # Generate random float numbers between 0 and 1. Works like np.random.rand.
    return self._generate_random(self.random, size)

  def uniform(self, low=0.0, high=1.0, size=None):
    # Return random floats in the half-open interval [low, high).
    return self._generate_random(lambda: super().uniform(low, high), size)

  def randn(self, *size):
    # Generate random numbers from a standard normal distribution using Box-Muller transform.
    def box_muller():
      u1 = self.random()
      u2 = self.random()
      z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
      return z0
    return self._generate_random(box_muller, size)

  def choice(self, a, size=None, replace=True):
    # Generate a random sample from a given 1D array `a`.
    # If replace is False, it generates without replacement.
    if not replace and size > len(a):
      raise ValueError("Cannot take a larger sample than population when 'replace=False'")
    
    if replace:
      return self._generate_random(lambda: a[int(self.random() * len(a))], size)
    else:
      return random.sample(a, size)

  def _generate_random(self, func, size):
    # Utility function to generate random numbers with or without shape.
    # If `size` is None, a single random value is returned.
    if size is None:
      return func()
    if isinstance(size, int):
      return [func() for _ in range(size)]
    elif isinstance(size, tuple):
      return self._nested_list(func, size)
    else:
      raise ValueError(f"Invalid size: {size}")

  def _nested_list(self, func, shape):
    # Recursively create a nested list with the given shape.
    if len(shape) == 1:
      return [func() for _ in range(shape[0])]
    else:
      return [self._nested_list(func, shape[1:]) for _ in range(shape[0])]