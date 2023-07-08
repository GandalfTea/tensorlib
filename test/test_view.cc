#define CATCH_CONFIG_MAIN 

#include <limits.h>
#include "catch.hpp"
#include "tensor.h"

using namespace tensor;

TEST_CASE("Test View Creation", "[core]") {

	SECTION("Null View") {
		CHECK_THROWS(View a = View());
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

