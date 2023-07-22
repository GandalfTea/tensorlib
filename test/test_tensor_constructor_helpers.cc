

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "tensor.h"

#include <iostream>

#define N 2048
#define EPSILON 0.001

using namespace tensor;
using Catch::Matchers::Floating::WithinAbsMatcher;

TEST_CASE("Tensor Constructor Helpers", "[core]") {
	SECTION("Construction Helpers") {
		SECTION("fill") {
			Tensor<float> a = Tensor<float>({N, N});
			CHECK_NOTHROW(a.fill(1.25));
			CHECK(a.is_initialized);
			CHECK(a.device == CPU);
			CHECK(a.size == N*N);
			CHECK(a.data());
			CHECK(a.strides());
			CHECK(a.view());
			CHECK(a.ndim() == 2);
			std::initializer_list<uint32_t> c = {N, N};
			std::shared_ptr<uint32_t[]> shp = a.view();
			uint32_t k=0;
			for(const auto &x : c) {
				CHECK(shp[k] == x);
				k++;
			}
			for(size_t i=0; i < a.view()[0]; i++) {
				for(size_t j=0; j < a.view()[1]; j++) {
					CHECK_THAT(a(i, j).data()[0], WithinAbsMatcher(1.25, EPSILON));
				}
			}
		}

		SECTION("static fill") {
			auto a = Tensor<>::fill({N, N}, 3.14);
			for(size_t i=0; i < a.view()[0]; i++) {
				for(size_t j=0; j < a.view()[1]; j++) {
					CHECK_THAT(a(i, j).data()[0], WithinAbsMatcher(3.14, EPSILON));
				}
			}
		}

		SECTION("arange") {
			CHECK_NOTHROW(Tensor<float>::arange(50));
			CHECK_NOTHROW(Tensor<float>::arange(50, 10));
			CHECK_NOTHROW(Tensor<float>::arange(50, 10, 5));

			CHECK_THROWS(Tensor<float>::arange(10, 20));
			CHECK_THROWS(Tensor<float>::arange(10, 0, 15));
			CHECK_THROWS(Tensor<float>::arange(10, 0, 10));

			SECTION("Top limit only") {
				Tensor<int8_t> a = Tensor<int8_t>::arange(50); 
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size == 50);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 1);
				std::initializer_list<uint32_t> c = {50};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t j=0;
				for(const auto &x : c) {
					CHECK(shp[j] == x);
					j++;
				}
				auto data = a.data();
				for(int8_t i=0; i < a.size; i++) {
					CHECK(data[i]==i);	
				}
			}

			SECTION("Top and negative bottom limit") {
				Tensor<int> a = Tensor<int>::arange(30, -20);
				CHECK(a.size == 50);
				auto data = a.data();
				for(int i=0; i < a.size; i++) {
					CHECK(data[i]==i-20);	
				}
			}

			SECTION("Top, botom and int step") {
				Tensor<int> a = Tensor<int>::arange(30, -20, 5);
				CHECK(a.size == 50/5);
				auto data = a.data();
				uint32_t inc = 0.f;
				for(int i=0; i < a.size; i++) {
					CHECK(data[i]==i-20+inc);	
					inc+=4;
				}
			}

			SECTION("Top and botom int, float step") {
				Tensor<float> a = Tensor<float>::arange(20, -30, 0.1f); 
				CHECK(a.size == 50/0.1f);
				auto data = a.data();
				float inc = -30.f;
				for(int i=0; i < a.size; i++) {
					CHECK_THAT(data[i], WithinAbsMatcher(inc, EPSILON));
					inc+=0.1f;
				}
			}
		}

		SECTION("randn") {
			SECTION("static") {
				Tensor<float> a = Tensor<float>::randn({N, N});	
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size == N*N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 2);
				std::initializer_list<uint32_t> c = {N,N};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t j=0;
				for(const auto &x : c) {
					CHECK(shp[j] == x);
					j++;
				}
				auto data = a.data();
				for(size_t i=0; i < a.size; i++) {
					CHECK(data[i] <= 1.f);
					CHECK(data[i] >= 0.f);
				}
			}
			SECTION("static : 1.f - 2.f") {
				Tensor<float> a = Tensor<float>::randn({N, N}, 2.f, 1.f);	
				auto data = a.data();
				for(size_t i=0; i < a.size; i++) {
					CHECK(data[i] <= 2.f);
					CHECK(data[i] >= 1.f);
				}
			}
			SECTION("static : -3.14 - 3.14") {
				Tensor<float> a = Tensor<float>::randn({N, N}, 3.14, -3.14);	
				auto data = a.data();
				for(size_t i=0; i < a.size; i++) {
					CHECK(data[i] < 3.15);
					CHECK(data[i] > -3.15);
				}
			}
			SECTION("non-static") {
				Tensor<float> a = Tensor<float>({N, N});
				CHECK_NOTHROW(a.randn(3.14, -3.14));
				auto data = a.data();
				for(size_t i=0; i < a.size; i++) {
					CHECK(data[i] < 3.15);
					CHECK(data[i] > -3.15);
				}
			}
		}
		/*
		SECTION("eye") {
			Tensor<float> a = Tensor<>::eye(N, 2);
			for(size_t i=0; i<N; i++) {
				std::cout << a << std::endl;
				CHECK(a(i, i).data()[0] == 1);
			}
		}
		*/
	}
}
