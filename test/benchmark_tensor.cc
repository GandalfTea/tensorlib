

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "tensor.h"

using namespace tensor;

void create_tensor_2048x2048_float32_stack_initialized() {
	std::unique_ptr<float[]> data = std::make_unique<float[]>(2048*2048);
	for(size_t i=0; i < 2048*2048; i++) { data[i]=i; }
	Tensor<float> a(data, 2048*2048, {2048, 2048});
}

void create_tensor_2048x2048_float32_heap() {
	std::unique_ptr<float[]> data = std::unique_ptr<float[]>(new float[2048*2048]());
	Tensor<float> a(data, 2048*2048, {2048, 2048});
}

void create_tensor_4096x4096_float32_stack_initialized() {
	std::unique_ptr<float[]> data = std::make_unique<float[]>(4096*4096);
	for(size_t i=0; i < 4096*4096; i++) { data[i]=i; }
	Tensor<float> a(data, 4096*4096, {4096, 4096});
}

void create_tensor_4096x4096_float32_heap() {
	std::unique_ptr<float[]> data = std::unique_ptr<float[]>(new float[4096*4096]());
	Tensor<float> a(data, 4096*4096, {4096, 4096});
}

TEST_CASE("Benchmarks") {
	SECTION("Full initialization") {
		BENCHMARK(std::to_string(2048)+"x"+std::to_string(2048)+" float32 initialization") {
			create_tensor_2048x2048_float32_stack_initialized();
		};
		BENCHMARK(std::to_string(2048)+"x"+std::to_string(2048)+" float32 default init") {
			create_tensor_2048x2048_float32_heap();
		};
		BENCHMARK(std::to_string(4096)+"x"+std::to_string(4096)+" float32 initialization") {
			create_tensor_4096x4096_float32_stack_initialized();
		};
		BENCHMARK(std::to_string(4096)+"x"+std::to_string(4096)+" float32 default init") {
			create_tensor_4096x4096_float32_heap();
		};
	}

	SECTION("Constructors and Destructors") {

		uint32_t N = 2048; 
		std::unique_ptr<float[]> data = std::make_unique<float[]>(N*N);
		for(size_t i=0; i < N*N; i++) { data[i]=i; }
		std::initializer_list<uint32_t> shape = {N, N};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float32 construction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::storage_for<Tensor<float>>> storage(meter.runs());
			meter.measure([&](int i) { storage[i].construct(data, N*N, shape); });
		};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float32 destruction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::destructable_object<Tensor<float>>> storage(meter.runs());
			for(auto&& o : storage) {
				o.construct(data, N*N, shape);
			}
			meter.measure( [&](int i) {storage[i].destruct(); });
		};

		std::unique_ptr<double[]> data2 = std::make_unique<double[]>(N*N);
		for(size_t i=0; i < N*N; i++) { data2[i]=i; }
		shape = {N, N};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float64 construction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::storage_for<Tensor<double>>> storage(meter.runs());
			meter.measure([&](int i) { storage[i].construct(data2, N*N, shape); });
		};
		BENCHMARK_ADVANCED(std::to_string(N)+"x"+std::to_string(N)+" float64 destruction")(Catch::Benchmark::Chronometer meter) {
			std::vector<Catch::Benchmark::destructable_object<Tensor<double>>> storage(meter.runs());
			for(auto&& o : storage) {
				o.construct(data2, N*N, shape);
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

	SECTION("Random Number Generators") {
		int N = 4096;
		BENCHMARK("Uniform Dist 4096") {
			return Tensor<>::f32_generate_uniform_distribution(N);
		};
		BENCHMARK("Uniform Dist 4096x4096") {
			return Tensor<>::f32_generate_uniform_distribution(N*N);
		};
		BENCHMARK("Uniform Dist 4096x4096x4096") {
			return Tensor<>::f32_generate_uniform_distribution(N*N*N);
		};
	}

	SECTION("Movement OPs") {
		int N = 4096;
		std::unique_ptr<float[]> data = std::make_unique<float[]>(N*N);
		for(size_t i=0; i < N*N; i++) { data[i]=i; }
		Tensor<float> a(data, N*N, {N, N});
		BENCHMARK("Reshape "+std::to_string(N)+","+std::to_string(N)+" -> "+std::to_string(N/2)+",2,"+std::to_string(N/2)+",2") {
			return a.reshape({N/2, 2, N/2, 2});
		};
		BENCHMARK("Permute "+std::to_string(N/2)+",2,"+std::to_string(N/2)+",2"+" -> 2, 2,"+std::to_string(N/2)+","+std::to_string(N/2)) {
			return a.permute({1, 3, 0, 2});
		};
		a.reshape({N, N , 1});
		BENCHMARK("Expand "+std::to_string(N)+","+std::to_string(N)+",1"+" -> "+std::to_string(N)+","+std::to_string(N)+",5") {
			return a.expand({N, N, 5});
		};
		BENCHMARK("Stride 1 element") {
			return a(0, 0);
		};
		BENCHMARK("Stride "+std::to_string(N)+" elements") {
			return a(0);
		};
	}
}

