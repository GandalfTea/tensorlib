#define CATCH_CONFIG_MAIN 

#include <limits.h>
#include "catch.hpp"
#include "tensor.h"

using namespace tensor;

TEST_CASE("Test View Creation", "[core]") {

	SECTION("One Element View {512, }") {
		std::initializer_list<uint32_t> cv = {512};
		std::initializer_list<uint32_t> cs = {1};
		View a = View({512});	

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
		View a = View({2, 2, 512});	

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
		CHECK_THROWS(View({UINT_MAX, UINT_MAX}));	
	}

	SECTION("TENSOR_MAX_DIM Fail") {
		// No clue how to do this yet	
	}
}

TEST_CASE("Test Movement OPs", "[core]") {

	SECTION("Test reshape {2, 2, 512} --> {2, 1024}") {
		View a = View({2, 2, 512});	
		std::shared_ptr<uint32_t[]> newdim = std::make_unique<uint32_t[]>(2);
		newdim[0] = 2;
		newdim[1] = 1024;
		std::shared_ptr<uint32_t[]> newstrides = std::make_unique<uint32_t[]>(2);
		newstrides[0] = 1024;
		newstrides[1] = 1;

		size_t len = 2;
		CHECK(a.reshape(newdim, len) == SUCCESSFUL);
		for(size_t i=0; i<2; i++) {
			CHECK(a.view[i] == newdim[i]);
			CHECK(a.strides[i] == newstrides[i]);
		}
	}

	SECTION("Test Reshape Wrong Product") {
		View a = View({2, 2, 512});	
		std::shared_ptr<uint32_t[]> newdim = std::make_unique<uint32_t[]>(2);
		newdim[0] = 2;
		newdim[1] = 512;
		size_t len = 2;
		CHECK(a.reshape(newdim, len) == INVALID_ARGUMENTS);
		std::shared_ptr<uint32_t[]> newdim2 = std::make_unique<uint32_t[]>(2);
		newdim2[0] = 4096;
		newdim2[1] = 512;
		CHECK(a.reshape(newdim2, len) == INVALID_ARGUMENTS);
		std::shared_ptr<uint32_t[]> newdim3 = std::make_unique<uint32_t[]>(5);
		newdim3[0] = 2;
		newdim3[1] = 512;
		newdim3[2] = 512;
		newdim3[3] = 512;
		newdim3[4] = 512;
		len = 5;
		CHECK(a.reshape(newdim3, len) == INVALID_ARGUMENTS);
	}

	// Not sure what's the best way to fail this.
	SECTION("Test Reshape Wront Number of Dims") {
		View a = View({2, 2, 512});	
		std::shared_ptr<uint32_t[]> newdim = std::make_unique<uint32_t[]>(2);
		newdim[0] = 2;
		newdim[1] = 512;
		size_t len = 1;
		CHECK(a.reshape(newdim, len) == INVALID_ARGUMENTS);
	}

	SECTION("Test Permute {2, 2, 512} --> {2, 512, 2}") {
		View a = View({2, 2, 512});	
		std::shared_ptr<uint32_t[]> permarg = std::make_unique<uint32_t[]>(3);
		permarg[0] = 0;
		permarg[1] = 2;
		permarg[2] = 1;
		std::shared_ptr<uint32_t[]> newview = std::make_unique<uint32_t[]>(3);
		newview[0] = 2;
		newview[1] = 512;
		newview[2] = 2;
		std::shared_ptr<uint32_t[]> newstrides = std::make_unique<uint32_t[]>(3);
		newstrides[0] = 1024;
		newstrides[1] = 1;
		newstrides[2] = 512;
		size_t len = 3;

		CHECK(a.permute(permarg, len) == SUCCESSFUL);
		for(size_t i=0; i<3; i++) {
			CHECK(a.view[i] == newview[i]);
			CHECK(a.strides[i] == newstrides[i]);
		}
	}

	SECTION("Test Permute Invalid Dimentionality") {
		View a = View({2, 2, 512});	
		std::shared_ptr<uint32_t[]> permarg = std::make_unique<uint32_t[]>(2);
		permarg[0] = 0;
		permarg[1] = 2;
		size_t len = 2;
		CHECK(a.permute(permarg, len) == INVALID_DIMENSIONALITY);
		std::shared_ptr<uint32_t[]> permarg2 = std::make_unique<uint32_t[]>(4);
		permarg2[0] = 0;
		permarg2[1] = 2;
		permarg2[2] = 2;
		permarg2[3] = 2;
		len = 4;
		CHECK(a.permute(permarg2, len) == INVALID_DIMENSIONALITY);
	}

	SECTION("Test Permute Invalid Arguments") {
		View a = View({2, 2, 512});	
		std::shared_ptr<uint32_t[]> permarg = std::make_unique<uint32_t[]>(3);
		permarg[0] = 0;
		permarg[1] = 1;
		permarg[2] = 3;
		size_t len = 3;
		CHECK(a.permute(permarg, len) == INVALID_ARGUMENTS);
		std::shared_ptr<uint32_t[]> permarg2 = std::make_unique<uint32_t[]>(3);
		permarg2[0] = 0;
		permarg2[1] = 1;
		permarg2[2] = 4;
		CHECK(a.permute(permarg2, len) == INVALID_ARGUMENTS);
	}

	SECTION("Test Expand {1, 512} --> {5, 512}") {
		View a = View({1, 512});	
		std::shared_ptr<uint32_t[]> arg = std::make_unique<uint32_t[]>(2);
		arg[0] = 5;
		arg[1] = 512;
		std::shared_ptr<uint32_t[]> newstrides = std::make_unique<uint32_t[]>(2);
		newstrides[0] = 0;
		newstrides[1] = 1;
		size_t len = 2;
		CHECK(a.expand(arg, len) == SUCCESSFUL);
		for(size_t i=0; i<2; i++) {
			CHECK(a.view[i] == arg[i]);
			CHECK(a.strides[i] == newstrides[i]);
		}
	}

	SECTION("Test Expand INVALID_DIMENSIONALITY") {
		View a = View({1, 512});	
		std::shared_ptr<uint32_t[]> arg = std::make_unique<uint32_t[]>(3);
		arg[0] = 5;
		arg[1] = 512;
		arg[2] = 512;
		size_t len = 3;
		CHECK(a.expand(arg, len) == INVALID_DIMENSIONALITY);
		std::shared_ptr<uint32_t[]> arg2 = std::make_unique<uint32_t[]>(1);
		arg[0] = 5;
		len = 1;
		CHECK(a.expand(arg2, len) == INVALID_DIMENSIONALITY);
	}

	SECTION("Test Expand INVALID_ARGUMENTS") {
		View a = View({2, 512});	
		std::shared_ptr<uint32_t[]> arg = std::make_unique<uint32_t[]>(2);
		arg[0] = 5;
		arg[1] = 512;
		size_t len = 2;
		CHECK(a.expand(arg, len) == INVALID_ARGUMENTS);
	}
}

