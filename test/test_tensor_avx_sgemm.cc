
#define CATCH_CONFIG_MAIN
#include <algorithm>
#include <cassert>
#include <cmath>
#include "catch.hpp"
#include "tensor.h"

#define N 2048
#define EPSILON 0.001

using namespace tensor;
using Catch::Matchers::Floating::WithinAbsMatcher;

bool constexpr max_f32(float a, float b, float epsilon=EPSILON) {
	return (a - b) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool constexpr eql_f32(float a, float b, float epsilon=EPSILON) {
	return fabs(a-b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool constexpr aeql_f32(float a, float b, float epsilon=EPSILON) {
	return fabs(a-b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

TEST_CASE("AVX 8x8 sgemm", "[core]") {

  // RETARD RESHAPE RETURN BOOL BOOOO
  // RETARD ARANGE DOES NOT COMPARE FLOATS WITH EPSILON
  float* da = new alignas(32) float[N*N];
  float inc = 1/N;
  for(int i=0; i<N; i++) da[i] = inc*i;
  auto a = Tensor<float>(da, N*N, {N, N});
  std::cout << a << std::endl;

  float* db = new alignas(32) float[N*N];
  for(int i=0; i<N; i++) db[i] = inc*i;
  auto b = Tensor<float>(db, N*N, {N, N});
  std::cout << b << std::endl;
  
  SECTION("0-N arange dot") {
    auto c = Tensor<>::dot(a, b, N, N, N);
  }

}

