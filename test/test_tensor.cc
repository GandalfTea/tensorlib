
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "tensor.h"

#define N 2048

using namespace tensor;

TEST_CASE("Tensor Constructor", "[core]") {
	SECTION("Test unique_ptr constructor") { 
		SECTION("Correct") {
			SECTION("uint8_t") {
				std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint8_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("uint16_t") {
				std::unique_ptr<uint16_t[]> data = std::make_unique<uint16_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint16_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("uint32_t") {
				std::unique_ptr<uint32_t[]> data = std::make_unique<uint32_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint32_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("uint64_t") {
				std::unique_ptr<uint64_t[]> data = std::make_unique<uint64_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint64_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int8_t") {
				std::unique_ptr<int8_t[]> data = std::make_unique<int8_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int8_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int16_t") {
				std::unique_ptr<int16_t[]> data = std::make_unique<int16_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int16_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int32_t") {
				std::unique_ptr<int32_t[]> data = std::make_unique<int32_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int32_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("int64_t") {
				std::unique_ptr<int64_t[]> data = std::make_unique<int64_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<int64_t> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
				uint32_t i=0;
				for(const auto &x : c) {
					CHECK(shp[i] == x);
					i++;
				}
			}
			SECTION("float") {
				std::unique_ptr<float[]> data = std::make_unique<float[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<float> a(data, N, {2, 2, 512});
				CHECK(a.device == GPU);
				CHECK(a.size == N);
				CHECK(a.bgrad == false);
				CHECK(a.data());
				CHECK(a.ndim() == 3);
				std::initializer_list<uint32_t> c = {2, 2, 512};
				std::shared_ptr<uint32_t[]> shp = a.get_shape();
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
			CHECK_THROWS(Tensor<float>(data, N, {1, 2, 512}));
		}

		// PASS, not implemented
		SECTION("Grad") {}
		SECTION("Test Storage GPU") {}
		SECTION("Test Storage CPU") {}
	}
	SECTION("Test initializer_list constructor") {
		SECTION("Correct") {}
		SECTION("Incorrect shape") {}
		SECTION("Grad") {}
		SECTION("Test Storage GPU") {}
		SECTION("Test Storage CPU") {}
	}
	SECTION("Test sized_array shape constructor") {
		SECTION("Correct") {}
	}
}

TEST_CASE("Tensor OPs", "[core]") {
	SECTION("Stride") {}
	SECTION("Reshape") {}
	SECTION("Expand") {}
	SECTION("Permute") {}
}

TEST_CASE("Single Tensor OPs", "[core]") {} 

TEST_CASE("Tensors OPs") {}


