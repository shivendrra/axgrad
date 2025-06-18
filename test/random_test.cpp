#include <iostream>
#include <iomanip>
#include <chrono>
#include "helpers.h"

void print_tensor(const char* name, float* arr, size_t size, size_t max_print = 10) {
  std::cout << name << ": ";
  size_t print_size = (size < max_print) ? size : max_print;
  for (size_t i = 0; i < print_size; i++) {
    std::cout << std::fixed << std::setprecision(4) << arr[i] << " ";
  }
  if (size > max_print) {
    std::cout << "... (showing first " << max_print << " of " << size << ")";
  }
  std::cout << std::endl;
}

void test_performance(const char* test_name, void (*func)(float*, size_t), size_t size) {
  float* arr = new float[size];
  
  auto start = std::chrono::high_resolution_clock::now();
  func(arr, size);
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << test_name << " (" << size << " elements): " << duration.count() << " microseconds" << std::endl;
  delete[] arr;
}

void test_random_quality() {
  const size_t test_size = 10000;
  float* randn_arr = new float[test_size];
  float* uniform_arr = new float[test_size];
  float* randint_arr = new float[test_size];
  
  // Generate samples
  fill_randn(randn_arr, test_size);
  fill_uniform(uniform_arr, 0.0f, 1.0f, test_size);
  fill_randint(randint_arr, 1, 100, test_size);
  
  // Calculate basic statistics for randn
  double sum = 0.0, sum_sq = 0.0;
  for (size_t i = 0; i < test_size; i++) {
    sum += randn_arr[i];
    sum_sq += randn_arr[i] * randn_arr[i];
  }
  double mean = sum / test_size;
  double variance = (sum_sq / test_size) - (mean * mean);
  
  std::cout << "\nRandom Quality Test (n=" << test_size << "):" << std::endl;
  std::cout << "  randn mean: " << std::fixed << std::setprecision(4) << mean << " (should be ~0.0)" << std::endl;
  std::cout << "  randn variance: " << variance << " (should be ~1.0)" << std::endl;
  
  // Check uniform distribution bounds
  float min_uniform = uniform_arr[0], max_uniform = uniform_arr[0];
  for (size_t i = 1; i < test_size; i++) {
    if (uniform_arr[i] < min_uniform) min_uniform = uniform_arr[i];
    if (uniform_arr[i] > max_uniform) max_uniform = uniform_arr[i];
  }
  std::cout << "  uniform range: [" << min_uniform << ", " << max_uniform << "] (should be [0.0, 1.0))" << std::endl;
  
  // Check randint bounds
  int min_int = (int)randint_arr[0], max_int = (int)randint_arr[0];
  for (size_t i = 1; i < test_size; i++) {
    int val = (int)randint_arr[i];
    if (val < min_int) min_int = val;
    if (val > max_int) max_int = val;
  }
  std::cout << "  randint range: [" << min_int << ", " << max_int << "] (should be [1, 99])" << std::endl;
  delete[] randn_arr;
  delete[] uniform_arr;
  delete[] randint_arr;
}

int main() {
  const size_t test_size = 20;
  
  std::cout << "=== Array Operations Test ===" << std::endl;
  
  // Test basic array operations
  float* zeros_arr = new float[test_size];
  float* ones_arr = new float[test_size];
  float* fill_arr = new float[test_size];
  float* linspace_arr = new float[test_size];
  
  zeros_tensor_ops(zeros_arr, test_size);
  ones_tensor_ops(ones_arr, test_size);
  fill_tensor_ops(fill_arr, 3.14f, test_size);
  linspace_tensor_ops(linspace_arr, 0.0f, 0.5f, test_size);
  
  print_tensor("zeros", zeros_arr, test_size);
  print_tensor("ones", ones_arr, test_size);
  print_tensor("fill(3.14)", fill_arr, test_size);
  print_tensor("linspace(0, 0.5)", linspace_arr, test_size);
  
  std::cout << "\n=== Random Arrays Test ===" << std::endl;
  
  // Test random functions
  float* randn_arr = new float[test_size];
  float* uniform_arr = new float[test_size];
  float* randint_arr = new float[test_size];
  
  fill_randn(randn_arr, test_size);
  fill_uniform(uniform_arr, -5.0f, 5.0f, test_size);
  fill_randint(randint_arr, 10, 50, test_size);
  
  print_tensor("randn", randn_arr, test_size);
  print_tensor("uniform(-5, 5)", uniform_arr, test_size);
  print_tensor("randint(10, 50)", randint_arr, test_size);
  
  std::cout << "\n=== Testing Different Seeds ===" << std::endl;
  
  // Test with different seeds
  float* seed1_arr = new float[10];
  float* seed2_arr = new float[10];
  
  set_random_seed(12345);
  fill_randn(seed1_arr, 10);
  
  set_random_seed(54321);
  fill_randn(seed2_arr, 10);
  
  print_tensor("seed 12345", seed1_arr, 10);
  print_tensor("seed 54321", seed2_arr, 10);
  
  // Test reproducibility
  float* repro_arr = new float[10];
  set_random_seed(12345);
  fill_randn(repro_arr, 10);
  print_tensor("seed 12345 again", repro_arr, 10);
  
  std::cout << "\n=== Performance Tests ===" << std::endl;
  
  // Performance tests
  test_performance("zeros", zeros_tensor_ops, 1000000);
  test_performance("randn", fill_randn, 1000000);
  test_performance("uniform", [](float* out, size_t size) {
    fill_uniform(out, 0.0f, 1.0f, size);
  }, 1000000);
  
  // Quality test
  test_random_quality();
  
  // Cleanup
  delete[] zeros_arr;
  delete[] ones_arr;
  delete[] fill_arr;
  delete[] linspace_arr;
  delete[] randn_arr;
  delete[] uniform_arr;
  delete[] randint_arr;
  delete[] seed1_arr;
  delete[] seed2_arr;
  delete[] repro_arr;
  
  std::cout << "\nAll tests completed!" << std::endl;
  return 0;
}