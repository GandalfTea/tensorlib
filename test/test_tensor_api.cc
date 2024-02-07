
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

TEST_CASE("Tensor API", "[core]") {
	std::unique_ptr<float[]> data = std::make_unique<float[]>(N*N);
	std::initializer_list<uint32_t> sp = {N, N};
	std::initializer_list<uint32_t> st = {N, 1};
	for(size_t i=0; i < N*N; i++) { data[i]=i; }
	Tensor<float> a(data, N*N, {N, N});
	uint32_t i = 0;

	SECTION("Getters") {
		SECTION("View") {
			CHECK_NOTHROW(a.view());
			auto view = a.view();
			i=0;
			SECTION("Correctness") { for(const auto& x : sp) { CHECK(view[i] == x); } }

			SECTION("Invalid Assignment") {
				// This does not throw
				// TODO: Find way to stop this
				// CHECK_THROWS(view[0] = 0);
			}
			SECTION("Iteration") {
				// No clue how yet	
			}
		}
	}

	SECTION("Copy Constructors and OPs") {
		SECTION("Shallow copy with ::like()") {
			std::unique_ptr<float[]> data = std::unique_ptr<float[]>(new float[N*N]());
			Tensor<float> a(data, N*N, {N,N});
			Tensor<float> b = Tensor<>::like(a);
			CHECK(!b.is_initialized);
			for(size_t i=0; i<b.ndim(); i++) CHECK(b.view()[i] == N);
		}	
		// TODO: Check for special tensors, also check all metadata
		SECTION("Hard copy with operator=") {
			std::unique_ptr<float[]> data = std::unique_ptr<float[]>(new float[N*N]());
			Tensor<float> a(data, N*N, {N,N});
			Tensor<float> b = a; 
			CHECK(b.is_initialized);
			for(size_t i=0; i<b.ndim(); i++) CHECK(b.view()[i] == N);
			auto nd = b.data();
			for(size_t i=0; i<b.size(); i++) CHECK(nd[i] == 0);
		}
	}

	SECTION("Random Number Generators") {
		SECTION("Uniform Distribution") {
			SECTION("0 - 1") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], -0.1f));
					CHECK(max_f32(1.3f, a[i]));
				}
			}
			SECTION("0 - 5") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500, 5.f);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], -0.1f));
					CHECK(max_f32(5.1f, a[i]));
				}
			}
			SECTION("-5 : -5") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500, 5.f, -5.f);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], -5.1f));
					CHECK(max_f32(5.1f, a[i]));
				}
			}
			SECTION("epsilon .5f") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500, 1.f, 0.f, 0, true, 0.5f);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], 0.4f));
					CHECK(max_f32(1.1f, a[i]));
				}
			}
		}
	}
}

TEST_CASE("Tensor OPs", "[core]") {

	// SECTION("Stride") { }
	
	SECTION("Reshape") {
		std::unique_ptr<float[]> data = std::make_unique<float[]>(N*N);
		std::initializer_list<uint32_t> sp = {N, N};
		std::initializer_list<uint32_t> st = {N, 1};
		for(size_t i=0; i < N*N; i++) { data[i]=i; }
		Tensor<float> a(data, N*N, {N, N});
		uint32_t i = 0;

		SECTION("Correct") {
			CHECK_NOTHROW(a.reshape({N/2, 2, N/2, 2})); 
			std::initializer_list<uint32_t> tsp = {N/2, 2, N/2, 2};
			std::initializer_list<uint32_t> tst = { (N/2)*2*2,(N/2)*2, 2, 1 };
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}

		CHECK_NOTHROW(a.reshape({N, N}));
		i=0;
		for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
		i=0;
		for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }

		SECTION("Invalid Product") {
			CHECK_THROWS(a.reshape({N, 1}));	
			i=0;
			for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("Argument 0") {
			CHECK_THROWS(a.reshape({N, N, 0}));	
			i=0;
			for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("One -1 arg") {
			CHECK_NOTHROW(a.reshape({N, -1}));
			std::initializer_list<uint32_t> tsp = {N, N};
			std::initializer_list<uint32_t> tst = {N, 1};
			i=0; for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0; for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("Multiple -1's") {
			CHECK_THROWS(a.reshape({N/2, -1, -1}));
			i=0; for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0; for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("1 dim -1") {
			CHECK_NOTHROW(a.reshape({-1}));
			std::initializer_list<uint32_t> tsp = {N*N};
			std::initializer_list<uint32_t> tst = {1};
			i=0; for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0; for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("1 Row") {
			CHECK_NOTHROW(a.reshape({N*N, 1}));
			std::initializer_list<uint32_t> tsp = {N*N, 1};
			std::initializer_list<uint32_t> tst = {1, 1};
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("1 Collumn") {
			CHECK_NOTHROW(a.reshape({1, N*N}));
			std::initializer_list<uint32_t> tsp = {1, N*N};
			std::initializer_list<uint32_t> tst = {N*N, 1};
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}
	}

	SECTION("Expand") {
		std::unique_ptr<float[]> data = std::unique_ptr<float[]>(new float[N]());
		std::initializer_list<uint32_t> sp = {N, 1};
		std::initializer_list<uint32_t> st = {1, 1};
		Tensor<> a(data, N, sp);
		uint32_t i = 0;

		SECTION("Correct") {
			CHECK_NOTHROW(a.expand({N, 5}));
			std::initializer_list<int32_t> tsp = {N, 5};
			std::initializer_list<int32_t> tst = {1, 0};
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i++] == x); }
      // 2048, 1
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i++] == x); }
      // 1, 1

			CHECK_NOTHROW(a(0, 0));
			CHECK_NOTHROW(a(0, 1));
			CHECK_NOTHROW(a(0, 2));
			CHECK_NOTHROW(a(0, 3));
			CHECK_NOTHROW(a(0, 4));
			CHECK(a(0,0).item()==a(0,1).item());
			CHECK(a(0,1).item()==a(0,2).item());
			CHECK(a(0,2).item()==a(0,3).item());
			CHECK(a(0,3).item()==a(0,4).item());
		}

		SECTION("Expansion on non-1 dimension") {
			a.reshape({N/2, 2});
			std::initializer_list<int32_t> tsp = {N/2, 2};
			std::initializer_list<int32_t> tst = {2, 1};
			CHECK_THROWS(a.expand({N/2, 5}));
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}

    a.strip();
		a.reshape({N, 1});

		SECTION("Wrong argument for other dimensions") {
			CHECK_THROWS(a.expand({N/2, 5}));
			i=0;
			for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
		}
	}

	SECTION("Permute") {
		std::unique_ptr<float[]> data = std::make_unique<float[]>(N*2);
		std::initializer_list<uint32_t> sp = {N, 2, 1};
		std::initializer_list<uint32_t> st = {2, 1, 1};
		for(size_t i=0; i < N*2; i++) { data[i]=i; }
		Tensor<float> a(data, N*2, sp);
		uint32_t i = 0;

		SECTION("Correct") {
			CHECK_NOTHROW(a.permute({1, 2, 0}));	
			std::initializer_list<uint32_t> tsp1 = {2, 1, N};
			std::initializer_list<uint32_t> tst1 = {1, 1, 2};
			i=0;
			for(const auto& x : tsp1) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst1) { CHECK(a.strides()[i] == x); i++; }

			CHECK_NOTHROW(a.permute({1, 2, 0}));	
			std::initializer_list<uint32_t> tsp2 = {1, N, 2};
			std::initializer_list<uint32_t> tst2 = {1, 2, 1};
			i=0;
			for(const auto& x : tsp2) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst2) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("Invalid dimension idx") {
			CHECK_THROWS(a.permute({3, 0, 1}));	
			i=0;
			for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("Repeting dimensions") {
			CHECK_THROWS(a.permute({0, 0, 1}));
			i=0;
			for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("Invalid number of dimensions") {
			CHECK_THROWS(a.permute({1, 0}));
			i=0;
			for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
		}
	}
}
