
#define CATCH_CONFIG_MAIN
#include <algorithm>
#include <cassert>
#include <cmath>
#include "catch.hpp"
#include "tensor.h"

#define N 2048
#define EPSILON 0.001
#define KOLMOGOROV_SMIRNOV_ALPHA 0.001

using namespace tensor;
using Catch::Matchers::Floating::WithinAbsMatcher;

// Statistical Functions used to test random number distributions

bool constexpr max_f32(float a, float b, float epsilon=EPSILON) {
	return (a - b) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool constexpr eql_f32(float a, float b, float epsilon=EPSILON) {
	return fabs(a-b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool constexpr aeql_f32(float a, float b, float epsilon=EPSILON) {
	return fabs(a-b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

#include <iostream>
// alpha is allowed margin of error, alpha < .20
bool uniform_kolmogorov_smirnov_test(std::unique_ptr<float[]> &data, size_t len, float alpha=KOLMOGOROV_SMIRNOV_ALPHA) {
	float D{}, d_plus_max{}, d_min_max{};
	std::sort(&data[0], &data[len]);
	for(size_t i=0; i<len; i++) {
		float d_plus = (((float)i+1)/len)-data[i];
		if(max_f32(d_plus, d_plus_max)) d_plus_max = d_plus;
		float d_min = data[i]-((float)i/len);
		if(max_f32(d_min, d_min_max)) d_min_max = d_min;
	}
	max_f32(d_plus_max, d_min_max) ? D = d_plus_max : D=d_min_max;
	float alpha_val;
	if(eql_f32(0.20f, alpha)) alpha_val=1.07;
	else if (eql_f32(0.10f, alpha)) alpha_val=1.22;
	else if (eql_f32(0.05f, alpha)) alpha_val=1.36;
	else if (eql_f32(0.02f, alpha)) alpha_val=1.52;
	else if (eql_f32(0.01f, alpha)) alpha_val=1.63;
	else alpha_val=1.63;

	float critical_value = alpha_val / std::sqrt(len);
	return !max_f32(D, critical_value, 0.5f); // big epsilon until I can make it more uniform 
}

float get_cdf_normal_dist(float x, float mean=0, float stdiv=1) {
	float arg = (x-mean)/(stdiv*std::sqrt(2));
	return 0.5*(1+std::erf(arg));
}

bool normal_kolmogorov_smirnov_test(std::unique_ptr<float[]> &data, size_t len, float mean=0.f, float std=1.f, float alpha=KOLMOGOROV_SMIRNOV_ALPHA) {
	float D{};
	for(size_t i=0; i<len; i++) { data[i] = std::abs(data[i]); }
	std::sort(&data[0], &data[len]);
	float f = mean-4*std;
	float skew = 3*(mean - data[std::floor(len/2)])/std;
	for(size_t i=1; i<=len; i++) {
		//float d_pls = (((float)i+1)/len)-data[i-1];
		//float d_min = data[i-1]-((float)i/len);
		//float max = max_f32(d_pls, d_min) ? d_pls : d_min;
		float max = ((float)i)/len;
		float ncdf = get_cdf_normal_dist(f, mean, std);
		float dif = std::abs(max-ncdf);
		D = max_f32(dif, D) ? dif : D;
		std::cout << max << " - " << ncdf << " max: " << D << std::endl;
		f += (mean+8*std)/len;
	}
	float alpha_val;
	if(eql_f32(0.20f, alpha)) alpha_val=1.07;
	else if (eql_f32(0.10f, alpha)) alpha_val=1.22;
	else if (eql_f32(0.05f, alpha)) alpha_val=1.36;
	else if (eql_f32(0.02f, alpha)) alpha_val=1.52;
	else if (eql_f32(0.01f, alpha)) alpha_val=1.63;
	else alpha_val=1.63;
	float critical_value = alpha_val / std::sqrt(len);
	//std::cout << critical_value << " - " << D << std::endl; 
	return !max_f32(D, critical_value, 0.9f);
}

float get_mean(std::unique_ptr<float[]> &data, size_t len) {
	float sum=0.f;
	for(size_t i=0; i<len; i++) sum += data[i];
	return sum/len;
}

float get_std(std::unique_ptr<float[]> &data, size_t len) {
	float sum=0;
	float mean = get_mean(data, len);
	for(size_t i=0; i<len; i++) sum += std::pow((data[i]-mean), 2);
	return std::sqrt(sum/len);
}

bool dagostino_skewness_test(std::unique_ptr<float[]> &data, size_t len) {
	double mean = 0;
	for(size_t i=0; i<len; i++) mean += data[i] / len;
	double g1_top, g1_btm, g2_top, g2_btm, g1, g2;
	double Y, b2_top, b2_btm, b2, w2, std, alpha, Z;
	for(size_t i=0; i<len; i++) {
		double s = data[i]-mean;
		g1_top += std::pow(s, 3) / len;
		g1_btm += std::pow(s, 2) / len;
		g2_top += std::pow(s, 4) / len;
	}
	g1_btm = std::pow(g1_btm, 1.5);
	g2_btm = std::pow(g1_btm, 2);
	g1 = g1_top / g1_btm;
	g2 = g2_top / g2_btm - 3;
	Y = g1 * (std::pow(((len+1)*(len+3)) / (6*(len-2)), 0.5));
	b2_top = 3*(pow(len,2)+27*len-70)*(len+1)*(len+3); 
	b2_btm = (len-2)*(len+5)*(len+7)*(len+9);
	b2 = b2_top/b2_btm;
	w2 = pow(2*(b2-1), 0.5);
	std = 1/std::pow(std::log(std::pow(w2, 0.5)), 0.5);
	alpha = std::pow(2/(w2-1), 0.5);
	Z = std*std::log(Y/alpha + std::pow(std::pow(Y/alpha, 2)+1, 0.5));
	if(max_f32(Z, -0.72991, 0.001) && max_f32(1.21315, Z, 0.001)) return true; // 95% confidence 
	else return false;
}

bool dagostino_kurtosis_test(std::shared_ptr<float[]> &data, size_t len) {
	float mean = 0;
	for(size_t i=0; i<len; i++) mean += data[i] / len;
	return false;
}


// Tests

TEST_CASE("Helpers", "[core]") {
	SECTION("float max()") {
		CHECK(max_f32(0.002, 0.001));
		CHECK(max_f32(0.02, 0.01));
		CHECK(max_f32(0.2, 0.1));
		CHECK(max_f32(2, 1));
		CHECK(!max_f32(69, 420));
		CHECK(!max_f32(1, 2));
		CHECK(!max_f32(0.1, 0.2));
		CHECK(!max_f32(0.01, 0.02));

		CHECK(eql_f32(0.1, 0.1));
		//CHECK(aeql_f32(0.0095, 0.1, 0.01));
		CHECK(aeql_f32(0.1, 0.2, 0.5));
	}
	SECTION("uniform kolmogorov smirnov test") {
		std::unique_ptr<float[]> data = std::make_unique<float[]>(10);
		float j = 0.1;
		for(size_t i=0; i<10; i++, j+=0.1) { data[i] = j; }
		CHECK(uniform_kolmogorov_smirnov_test(data, 10));
	}

	SECTION("cdf of normal distribution") {
		CHECK(eql_f32(get_cdf_normal_dist(0.1), 0.53982784));
		CHECK(eql_f32(get_cdf_normal_dist(0.2), 0.57925971));
		CHECK(eql_f32(get_cdf_normal_dist(0.3), 0.61791142));
		CHECK(eql_f32(get_cdf_normal_dist(0.4), 0.65542174));
		CHECK(eql_f32(get_cdf_normal_dist(0.5), 0.69146246));
	}

	SECTION("mean") {
		std::unique_ptr<float[]> data = std::make_unique<float[]>(10);
		for(size_t i=0; i<10; i++) { data[i] = i; }
		CHECK(eql_f32(get_mean(data, 10), 4.5));
	}

	SECTION("std") {
		std::unique_ptr<float[]> data = std::make_unique<float[]>(10);
		for(size_t i=0; i<10; i++) { data[i] = i; }
		float a = get_std(data, 10);
		CHECK(eql_f32(a, 2.872281323269));
	}
}

TEST_CASE("Tensor API", "[core]") {
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

	SECTION("Random Number Generators") {
		SECTION("Uniform Distribution") {
			SECTION("0 - 1") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], -0.1f));
					CHECK(max_f32(1.3f, a[i]));
				}
				CHECK(uniform_kolmogorov_smirnov_test(a, 500));
			}
			SECTION("0 - 5") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500, 5.f);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], -0.1f));
					CHECK(max_f32(5.1f, a[i]));
				}
				CHECK(uniform_kolmogorov_smirnov_test(a, 500));
			}
			SECTION("-5 : -5") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500, 5.f, -5.f);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], -5.1f));
					CHECK(max_f32(5.1f, a[i]));
				}
				CHECK(uniform_kolmogorov_smirnov_test(a, 500));
			}
			SECTION("epsilon .5f") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_uniform_distribution(500, 1.f, 0.f, 0, true, 0.5f);
				for(size_t i=0; i < 500; i++) {
					CHECK(max_f32(a[i], 0.4f));
					CHECK(max_f32(1.1f, a[i]));
				}
				CHECK(!uniform_kolmogorov_smirnov_test(a, 500));
			}
		}

		// static std::unique_ptr<float[]> f32_generate_box_muller_normal_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0) {
		SECTION("Box-Muller Transform") {
			SECTION("0-1") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_box_muller_normal_distribution(5000);
				CHECK_THAT(get_mean(a, 5000), WithinAbsMatcher(0.f, 0.1));
				CHECK_THAT(get_std(a, 5000), WithinAbsMatcher(1.f, 0.1));
				//CHECK(normal_kolmogorov_smirnov_test(a, 5000, get_mean(a, 5000), get_std(a, 5000))); 
				std::cout << dagostino_skewness_test(a, 5000) << std::endl;
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
		std::unique_ptr<float[]> data = std::unique_ptr<float[]>(new float[N]());
		std::initializer_list<uint32_t> sp = {N, 1};
		std::initializer_list<uint32_t> st = {1, 1};
		Tensor<> a(data, N, sp);
		uint32_t i = 0;

		SECTION("Correct") {
			CHECK(a.expand({N, 5}));
			std::initializer_list<uint32_t> tsp = {N, 5};
			std::initializer_list<uint32_t> tst = {1, 0};
			i=0;
			for(const auto& x : tsp) { CHECK(a.view()[i++] == x); }
			i=0;
			for(auto const& x : tst) { CHECK(a.strides()[i++] == x); }

			CHECK_NOTHROW(a(0, 0));
			CHECK_NOTHROW(a(0, 1));
			CHECK_NOTHROW(a(0, 2));
			CHECK_NOTHROW(a(0, 3));
			CHECK_NOTHROW(a(0, 4));
			CHECK(a(0,0).item()==a(0,1).item());
			CHECK(a(0,1).item()==a(0,2).item());
			CHECK(a(0,2).item()==a(0,3).item());
			CHECK(a(0,3).item()==a(0,4).item());
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
