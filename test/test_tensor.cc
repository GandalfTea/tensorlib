
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "tensor.h"

#define N 2048
#define EPSILON 0.001

using namespace tensor;
using Catch::Matchers::Floating::WithinAbsMatcher;

TEST_CASE("Tensor Constructor", "[core]") {

	SECTION("Test virtual tensor constructor") {
		Tensor<float> a({N, N});
		CHECK(!a.data());
		CHECK(!a.is_initialized);
		CHECK(a.strides());
		CHECK(a.view());
		CHECK(a.size == 0);
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

	SECTION("Test unique_ptr constructor") { 
		SECTION("Correct") {
			SECTION("uint8_t") {
				std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<uint8_t> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size == N);
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
				CHECK(a.size == N);
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
				CHECK(a.size == N);
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
				CHECK(a.size == N);
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
				CHECK(a.size == N);
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
				CHECK(a.size == N);
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
				CHECK(a.size == N);
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
				CHECK(a.size == N);
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
			SECTION("float") {
				std::unique_ptr<float[]> data = std::make_unique<float[]>(N);
				for(size_t i=0; i < N; i++) { data[i]=i; }
				Tensor<float> a(data, N, {2, 2, N/4});
				CHECK(a.is_initialized);
				CHECK(a.device == CPU);
				CHECK(a.size == N);
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

		SECTION("Test Storage CPU") {}
	}

	SECTION("Test initializer_list constructor") {
		SECTION("Correct") {
			Tensor<float> a({0, 1, 2, 3, 4, 5}, 6, {2, 3});
			CHECK(a.is_initialized);
			CHECK(a.device == CPU);
			CHECK(a.size == 6);
			CHECK(a.data());
			CHECK(a.strides());
			CHECK(a.view());
			CHECK(a.ndim() == 2);
			std::initializer_list<uint32_t> c = {2, 3};
			std::shared_ptr<uint32_t[]> shp = a.view();
			uint32_t i=0;
			for(const auto &x : c) {
				CHECK(shp[i] == x);
				i++;
			}
		}

		SECTION("Incorrect shape") {
			CHECK_THROWS(Tensor<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, 6, {2, 2, 2}));
		}

		SECTION("Test Storage CPU") {}
	}

	SECTION("Test sized_array shape constructor") {
		SECTION("Correct") {}
	}

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
					CHECK(data[i] <= 3.14);
					CHECK(data[i] >= -3.14);
				}
			}
		}
		SECTION("eye") {}
	}
}

TEST_CASE("API", "[core]") {
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

		SECTION("Stringify Row") {
			CHECK_NOTHROW(a.reshape({N*N, 1}));
			std::initializer_list<uint32_t> tsp = {N*N, 1};
			std::initializer_list<uint32_t> tst = {1, 1};
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}

		SECTION("Stringify Collumn") {
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
		std::unique_ptr<float[]> data = std::make_unique<float[]>(N);
		std::initializer_list<uint32_t> sp = {N, 1};
		std::initializer_list<uint32_t> st = {1, 1};
		for(size_t i=0; i < N; i++) { data[i]=i; }
		Tensor<float> a(data, N, sp);
		uint32_t i = 0;

		SECTION("Correct") {
			CHECK_NOTHROW(a.expand({N, 5}));
			std::initializer_list<uint32_t> tsp = {N, 5};
			std::initializer_list<uint32_t> tst = {1, 0};
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }

			CHECK_NOTHROW(a(0, 0));
			CHECK_NOTHROW(a(0, 1));
			CHECK_NOTHROW(a(0, 2));
			CHECK_NOTHROW(a(0, 3));
			CHECK_NOTHROW(a(0, 4));
			CHECK(a(0,0).data()[0]==a(0,1).data()[0]);
			CHECK(a(0,1).data()[0]==a(0,2).data()[0]);
			CHECK(a(0,2).data()[0]==a(0,3).data()[0]);
			CHECK(a(0,3).data()[0]==a(0,4).data()[0]);
		}

		SECTION("Expansion on non-1 dimension") {
			a.reshape({N/2, 2});
			std::initializer_list<uint32_t> tsp = {N/2, 2};
			std::initializer_list<uint32_t> tst = {2, 1};
			CHECK_THROWS(a.expand({N/2, 5}));
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i] == x); i++; }
		}

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
			/*
			CHECK_THROWS(a.permute({0, 0, 1}));
			i=0;
			for(const auto& x : sp) { CHECK(a.view()[i] == x); i++; }
			i=0;
			for(auto const& x : st) { CHECK(a.strides()[i] == x); i++; }
			*/
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

