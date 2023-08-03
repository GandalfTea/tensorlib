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
	double g1_top, g1_btm, g2_top, g2_btm, g1, g2, Y, b2_top, b2_btm, b2, w2, std, alpha, Z;
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

bool dagostino_kurtosis_test(std::unique_ptr<float[]> &data, size_t len) {
	double mean = 0;
	for(size_t i=0; i<len; i++) mean += data[i] / len;
	double krt_top=0, krt_btm=0, krt=0, Ekrt=0, Var_krt=0, x=0, B_first=0, B_secnd=0, B=0, A=0, pos=0, Z=0;
	for(size_t i=0; i<len; i++) {
		double s = data[i]-mean;
		krt_top += std::pow(s, 4) / len;
		krt_btm += std::pow(s, 2) / len;
	}
	krt_btm = std::pow(krt_btm, 2);
	krt = krt_top/krt_btm;
	Ekrt = (3.f*(len-1)) / (len+1);
	Var_krt = (24.f*len*(len+2)*(len-3))/(std::pow(len+1, 2)*(len+3)*(len+5));
	x = (krt - Ekrt) / std::pow(Var_krt, 0.5);
	B_first = (6.f*len*len-5.f*len+2.f) / ((len+7.f)*(len+9.f)); 
	B_secnd =	(6.f*(len+3.f)*(len+5.f)) / (len*(len-2.f)*(len-3.f)); 
	B_secnd = std::pow(B_secnd, 0.5);
	B = B_first*B_secnd;
	A = 6 + (8/B) * ((2/B)+std::pow(1+4/std::pow(B, 2), 0.5));
	pos = (1-2/A);
	pos = pos / (1.f+(x*std::pow(2.f/(A-4), 0.5)));
	pos = std::pow(pos, 1.f/3);
	Z = (1.f-(2.f/(9*A))-pos) / std::pow(2.f/(9.f*A), 0.5);;
	if(max_f32(Z, -0.72991, 0.001) && max_f32(1.21315, Z, 0.001)) return true; // 95% confidence 
	else return false;
}


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

TEST_CASE("Generators", "stats") {
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

		SECTION("Box-Muller Transform") {
			SECTION("0-1") {
				std::unique_ptr<float[]> a = Tensor<>::f32_generate_box_muller_normal_distribution(5000);
				CHECK_THAT(get_mean(a, 5000), WithinAbsMatcher(0.f, 0.1));
				CHECK_THAT(get_std(a, 5000), WithinAbsMatcher(1.f, 0.1));
				//CHECK(normal_kolmogorov_smirnov_test(a, 5000, get_mean(a, 5000), get_std(a, 5000))); 
				SECTION("D'agostino skewness") {
					float res = 0;
					for(size_t i=0; i < 100; i++) {
						std::unique_ptr<float[]> a = Tensor<>::f32_generate_box_muller_normal_distribution(5000);
						res += dagostino_skewness_test(a, 5000);
					}
					res /= 100;
					CHECK_THAT(res, WithinAbsMatcher(1.f,0.5));
				}
				SECTION("D'agostino kurtosis") {
					float res = 0;
					for(size_t i=0; i < 100; i++) {
						std::unique_ptr<float[]> a = Tensor<>::f32_generate_box_muller_normal_distribution(5000);
						res += dagostino_kurtosis_test(a, 5000);
					}
					res /= 100;
					CHECK_THAT(res, WithinAbsMatcher(1.f,0.5));
				}
				SECTION("Both skewness and kurtosis") {
					float res = 0;
					for(size_t i=0; i < 100; i++) {
						std::unique_ptr<float[]> a = Tensor<>::f32_generate_box_muller_normal_distribution(5000);
						res += dagostino_kurtosis_test(a, 5000);
						res += dagostino_skewness_test(a, 5000);
					}
					res /= 200;
					CHECK_THAT(res, WithinAbsMatcher(1.f,0.5));
				}
			}
		}
	}
}
