

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "tensor.h"

using namespace tensor;

TEST_CASE("Benchmarks") {
	SECTION("Constructors and Destructors") {

		int N = 2048; 
		std::unique_ptr<float[]> data = std::make_unique<float[]>(N*N);
		for(size_t i=0; i < N*N; i++) { data[i]=i; }
		std::initializer_list<uint32_t> shape = {N, N};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float construction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::storage_for<Tensor<float>>> storage(meter.runs());
			meter.measure([&](int i) { storage[i].construct(data, N*N, shape); });
		};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float destruction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::destructable_object<Tensor<float>>> storage(meter.runs());
			for(auto&& o : storage) {
				o.construct(data, N*N, shape);
			}
			meter.measure( [&](int i) {storage[i].destruct(); });
		};

		N=4096;
		data = std::make_unique<float[]>(N*N);
		for(size_t i=0; i < N*N; i++) { data[i]=i; }
		shape = {N, N};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float construction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::storage_for<Tensor<float>>> storage(meter.runs());
			meter.measure([&](int i) { storage[i].construct(data, N*N, shape); });
		};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float destruction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::destructable_object<Tensor<float>>> storage(meter.runs());
			for(auto&& o : storage) {
				o.construct(data, N*N, shape);
			}
			meter.measure( [&](int i) {storage[i].destruct(); });
		};

		N=8192;
		data = std::make_unique<float[]>(N*N);
		for(size_t i=0; i < N*N; i++) { data[i]=i; }
		shape = {N, N};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float construction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::storage_for<Tensor<float>>> storage(meter.runs());
			std::initializer_list<uint32_t> shape = {N, N};
			meter.measure([&](int i) { storage[i].construct(data, N*N, shape); });
		};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float destruction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::destructable_object<Tensor<float>>> storage(meter.runs());
			for(auto&& o : storage) {
				o.construct(data, N*N, shape);
			}
			meter.measure( [&](int i) {storage[i].destruct(); });
		};
	}	
}

