
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "tensor.h"

using namespace tensor;

TEST_CASE("Test Tensor Metadata", "[core]") {
	SECTION("Null Float32 Tensor") {
		//auto t = Tensor<float>();	
	}

	SECTION("4096 Float32 Matrix") {
		std::unique_ptr<float[]> data = std::make_unique<float[]>(4096);
		for(size_t i=0; i < 4096; i++) { data[i]=i; }
		Tensor<float> a(data, 4096, {2, 2, 2, 512});

		CHECK(a.device == GPU);
		CHECK(a.size == 4096);
		CHECK(a.bgrad == false);
		CHECK(a.data());
		CHECK(a.ndim() == 4);

		std::initializer_list<uint32_t> c = {2, 2, 2, 512};
		std::shared_ptr<uint32_t[]> shp = a.get_shape();
		uint32_t i=0;
		for(const auto &x : c) {
			CHECK(shp[i] == x);
			i++;
		}
	}
}
