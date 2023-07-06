#define CATCH_CONFIG_MAIN 

#include <limits.h>
#include "catch.hpp"
#include "tensor.h"

using namespace tensor;

TEST_CASE("Test View Creation", "[core]") {
	SECTION("Null View") {
		View a = View();
		CHECK_FALSE(a.view);
		CHECK_FALSE(a.strides);
		CHECK(a.ndim() == 0);
		CHECK(a.telem() == 0);
		std::shared_ptr<uint32_t[]> dummy = std::make_unique<uint32_t[]>(1);
		dummy[0] = 1;
		size_t dim = 1;
		CHECK(a.reshape(dummy, dim) == VIEW_NOT_INITIALIZED);
		CHECK(a.permute(dummy, dim) == VIEW_NOT_INITIALIZED);
		CHECK(a.expand(dummy, dim)  == VIEW_NOT_INITIALIZED);
	}

	SECTION("One Element View {512, }") {
		std::initializer_list<uint32_t> cv = {512};
		std::initializer_list<uint32_t> cs = {1};
		View a = View({512}, 512);	

		CHECK(a.view);
		CHECK(a.strides);
		CHECK(a.ndim() == 1);
		CHECK(a.telem() == 512);

		uint32_t i=0;
		for(const auto &x: cv) {
			CHECK(a.view[i] == x);	
			i++;
		}
		i=0;
		for(const auto &x: cs) {
			CHECK(a.strides[i] == x);	
			i++;
		}
	}

	SECTION("Normal View {2, 2, 512}") {
		std::initializer_list<uint32_t> cv = {2, 2, 512};
		std::initializer_list<uint32_t> cs = {1024, 512, 1};
		View a = View({2, 2, 512}, 512*2*2);	

		CHECK(a.view);
		CHECK(a.strides);
		CHECK(a.ndim() == 3);
		CHECK(a.telem() == 512*2*2);

		uint32_t i=0;
		for(const auto &x: cv) {
			CHECK(a.view[i] == x);	
			i++;
		}
		i=0;
		for(const auto &x: cs) {
			CHECK(a.strides[i] == x);	
			i++;
		}
	}

	SECTION("TENSOR_MAX_STORAGE_SIZE Fail") {
		CHECK_THROWS(View({1}, UINT_MAX+10));	
	}

	SECTION("TENSOR_MAX_DIM Fail") {
		// No clue how to do this yet	
	}
}

TEST_CASE("Test Movement OPs", "[core]") {

}

