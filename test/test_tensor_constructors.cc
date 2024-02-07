
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "tensor.h"

#include <iostream>

#define N 2048
#define EPSILON 0.001

using namespace tensor;
using Catch::Matchers::Floating::WithinAbsMatcher;

TEST_CASE("Tensor Constructor", "[core]") {

	SECTION("Test virtual tensor constructor") {
		SECTION("initializer_list") {
			Tensor<float> a({N, N});
			CHECK(!a.data());
			CHECK(!a.is_initialized);
			CHECK(a.strides());
			CHECK(a.view());
			CHECK(a.size() == N*N);
			CHECK(a.ndim()==2);
			CHECK(a.device == CPU);
			std::initializer_list<uint32_t> shp = {N, N};
			std::shared_ptr<uint32_t[]> cshp = a.view();
			uint32_t i=0;
			for(const auto &x : shp) {
				CHECK(cshp[i] == x);
				i++;
			}
		}

		SECTION("sized_array") {
			sized_array<uint32_t> s;
			s.ptr = std::unique_ptr<uint32_t[]>(new uint32_t[2]());
			s.ptr[0] = N;
			s.ptr[1] = N;
			s.size = 2;
			Tensor<float> a(s);
			CHECK(!a.data());
			CHECK(!a.is_initialized);
			CHECK(a.strides());
			CHECK(a.view());
			CHECK(a.size() == N*N);
			CHECK(a.ndim()==2);
			CHECK(a.device == CPU);
			std::initializer_list<uint32_t> shp = {N, N};
			std::shared_ptr<uint32_t[]> cshp = a.view();
			uint32_t i=0;
			for(const auto &x : shp) {
				CHECK(cshp[i] == x);
				i++;
			}
		}
	}

	SECTION("Test unique_ptr constructor") { 
		SECTION("Correct") {
			SECTION("No template argument") {
				std::unique_ptr<float[]> data = std::unique_ptr<float[]>( new float[N]());
				CHECK_NOTHROW(Tensor<>(data, N, {2, 2, N/4}));
				auto a = Tensor<>(data, N, {2, 2, N/4});
				CHECK(typeid(a.data()[0]) == typeid(0.f));
			}

			SECTION("uint8_t") {
				std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint8_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("uint16_t") {
				std::unique_ptr<uint16_t[]> data = std::make_unique<uint16_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint16_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("uint32_t") {
				std::unique_ptr<uint32_t[]> data = std::make_unique<uint32_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint32_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("uint64_t") {
				std::unique_ptr<uint64_t[]> data = std::make_unique<uint64_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint64_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int8_t") {
				std::unique_ptr<int8_t[]> data = std::make_unique<int8_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int8_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int16_t") {
				std::unique_ptr<int16_t[]> data = std::make_unique<int16_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int16_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int32_t") {
				std::unique_ptr<int32_t[]> data = std::make_unique<int32_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int32_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int64_t") {
				std::unique_ptr<int64_t[]> data = std::make_unique<int64_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int64_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("float32") {
				std::unique_ptr<float[]> data = std::make_unique<float[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<float> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("float64") {
				std::unique_ptr<double[]> data = std::make_unique<double[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<double> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size() == N);
				CHECK(a.data());
				CHECK(a.strides());
				CHECK(a.view());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, N/4};
				std::shared_ptr<uint32_t[]> shp = a.view();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
		}
		SECTION("Incorrect shape") {
			std::unique_ptr<float[]> data = std::make_unique<float[]>(N);
			for(size_t i=0; i < N; i++) { data[i]=i; }
			CHECK_THROWS(Tensor<float>(data, N, {1, 2, N/4}));
		}
	}

	SECTION("Test sized_array shape constructor") {
		SECTION("unique_ptr") {
			sized_array<uint32_t> s;
			s.ptr = std::unique_ptr<uint32_t[]>(new uint32_t[3]());
			s.ptr[0] = 2;
			s.ptr[1] = 2;
			s.ptr[2] = N/4;
			s.size = 3;
			std::unique_ptr<float[]> data = std::make_unique<float[]>(N);
			for(size_t i=0; i < N; i++) { data[i]=i; }
			Tensor<float> a(data, N, s);
			CHECK(a.is_initialized);
			CHECK(a.device == CPU);
			CHECK(a.size() == N);
			CHECK(a.data());
			CHECK(a.strides());
			CHECK(a.view());
			CHECK(a.ndim() == 3);
			std::initializer_list<uint32_t> c = {2, 2, N/4};
			std::shared_ptr<uint32_t[]> shp = a.view();
			uint32_t i=0;
			for(const auto &x : c) {
				CHECK(shp[i] == x);
				i++;
			}
		}
	}
}


