
#if defined(__AVX__) 
  #include<immintrin.h>
#endif

#if defined(__SSE__) || defined(__SSE4__)
  #include <xmmintrin.h>
#endif

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "ntensor.h"


#define N 1024
#define EPSILON 0.0001

using namespace tensorlib;

TEST_CASE("Virtual Tensors") {
  SECTION("empty") {
    SECTION("Tensor<float>{N}") {
      Tensor<float> a = Tensor<float>({N});
      CHECK(!a.is_initialized());
      CHECK(!a.is_allocated());
      CHECK(!a.has_grad());
      CHECK(!a.is_eye());
      CHECK(!a.is_sub());
      CHECK(a.device() == 1);
      CHECK(a.memsize() == 0);
      CHECK(a.numel() == N);
      CHECK(a.ndim() == 1);
      CHECK(a.storage() == nullptr);
      CHECK(a.shape()[0] == N);
      CHECK(a.strides()[0] == 1);
      CHECK_THROWS(a(0));
    }
    SECTION("Tensor<float>({N, N})") {
      Tensor<float> a = Tensor<float>({N, N});
      CHECK(!a.is_initialized());
      CHECK(!a.is_allocated());
      CHECK(!a.has_grad());
      CHECK(!a.is_eye());
      CHECK(!a.is_sub());
      CHECK(a.device() == 1);
      CHECK(a.memsize() == 0);
      CHECK(a.numel() == N*N);
      CHECK(a.ndim() == 2);
      CHECK(a.storage() == nullptr);
      const uint32_t* shp = a.shape();
      for(size_t i=0; i<a.ndim(); i++) CHECK(shp[i] == N);
      const uint32_t* strd = a.strides();
      CHECK(strd[0] == N);
      CHECK(strd[1] == 1);
      CHECK_THROWS(a(0));
    }
    SECTION("Tensor<float>(uint32_t* {N}, 1)") {
      uint32_t* shp = new uint32_t[1]; 
      shp[0] = N;
      Tensor<float> a = Tensor<float>(shp, 1);
      CHECK(!a.is_initialized());
      CHECK(!a.is_allocated());
      CHECK(!a.has_grad());
      CHECK(!a.is_eye());
      CHECK(!a.is_sub());
      CHECK(a.device() == 1);
      CHECK(a.memsize() == 0);
      CHECK(a.numel() == N);
      CHECK(a.ndim() == 1);
      CHECK(a.storage() == nullptr);
      CHECK(a.shape()[0] == N);
      CHECK(a.strides()[0] == 1);
      CHECK_THROWS(a(0));
    }
    SECTION("Tensor<float>(uint32_t* {N, N}, 2)") {
      uint32_t* shp = new uint32_t[2];
      shp[0] = N;
      shp[1] = N;
      Tensor<float> a = Tensor<float>(shp, 2);
      CHECK(!a.is_initialized());
      CHECK(!a.is_allocated());
      CHECK(!a.has_grad());
      CHECK(!a.is_eye());
      CHECK(!a.is_sub());
      CHECK(a.device() == 1);
      CHECK(a.memsize() == 0);
      CHECK(a.numel() == N*N);
      CHECK(a.ndim() == 2);
      CHECK(a.storage() == nullptr);
      const uint32_t* mshp = a.shape();
      for(size_t i=0; i<a.ndim(); i++) CHECK(mshp[i] == N);
      const uint32_t* strd = a.strides();
      CHECK(strd[0] == N);
      CHECK(strd[1] == 1);
      CHECK_THROWS(a(0));
    }
  }
  SECTION("allocation") {
    SECTION("Tensor<float>({N, N}).allocate()") {
      Tensor<float> a = Tensor<float>({N, N}).allocate();
      CHECK(a.is_initialized());
      CHECK(a.is_allocated());
      CHECK(!a.has_grad());
      CHECK(!a.is_eye());
      CHECK(!a.is_sub());
      CHECK(a.device() == 1);
      CHECK(a.memsize() == N*N);
      CHECK(a.numel() == N*N);
      CHECK(a.ndim() == 2);
      CHECK(a.storage() != nullptr);
      CHECK_NOTHROW(a.storage()[0]);
      CHECK_NOTHROW(a.storage()[N]);
      CHECK_NOTHROW(a.storage()[N*N-1]);
      CHECK(a.shape()[0] == N);
      CHECK(a.shape()[1] == N);
      CHECK(a.strides()[0] == N);
      CHECK(a.strides()[1] == 1);
      CHECK_NOTHROW(a(0));
      CHECK_NOTHROW(a(N-1));
      CHECK_NOTHROW(a(N-1, N-1));
    } 
#ifdef __AVX__
    SECTION("alignmet") {
      SECTION("AVX-256") {
        Tensor<float> a = Tensor<float>({N, N}).allocate();
        CHECK_NOTHROW(_mm256_load_ps(&(a.storage())[0])); // check if data is 32 byte aligned by default
      }
#endif
#ifdef __SSE__ 
      SECTION("SSE") {
        #undef DATA_ALIGNMENT
        #define DATA_ALIGNMENT 16
        Tensor<float> a = Tensor<float>({N, N}).allocate();
        CHECK_NOTHROW(_mm_load_ps(&(a.storage())[0]));
      }
#endif 
#ifdef __AVX512F__
      SECTION("AVX-512") {
        #undef DATA_ALIGNMENT
        #define DATA_ALIGNMENT 64; 
        Tensor<float> a = Tensor<float>({N, N}).allocate();
        CHECK_NOTHROW(_mm512_load_ps(&(a.storage())[0]));
      }
    }
#endif

    // Random number generation is tested in file ./test_random_distributions.cc
    SECTION("Tensor<float>({N, N}).randn()") {
      Tensor<float> a = Tensor<float>({N, N}).randn();
      CHECK(a.is_initialized());
      CHECK(a.is_allocated());
      CHECK(!a.has_grad());
      CHECK(!a.is_eye());
      CHECK(!a.is_sub());
      CHECK(a.device() == 1);
      CHECK(a.memsize() == N*N);
      CHECK(a.numel() == N*N);
      CHECK(a.ndim() == 2);
      CHECK(a.storage() != nullptr);
      CHECK_NOTHROW(a.storage()[0]);
      CHECK_NOTHROW(a.storage()[N]);
      CHECK_NOTHROW(a.storage()[N*N-1]);
      CHECK(a.shape()[0] == N);
      CHECK(a.shape()[1] == N);
      CHECK(a.strides()[0] == N);
      CHECK(a.strides()[1] == 1);
      CHECK_NOTHROW(a(0));
      CHECK_NOTHROW(a(N-1));
      CHECK_NOTHROW(a(N-1, N-1));
    }

    SECTION("Tensor<float>({N, N}).fill(69.f)") {
      Tensor<float> a = Tensor<float>({N, N}).fill(69.f);
      CHECK(a.is_initialized());
      CHECK(!a.is_allocated());
      CHECK(!a.has_grad());
      CHECK(!a.is_eye());
      CHECK(!a.is_sub());
      CHECK(a.device() == 1);
      CHECK(a.memsize() == 1);
      CHECK(a.numel() == N*N);
      CHECK(a.ndim() == 2);
      CHECK(a.storage() != nullptr);
      CHECK(a.shape()[0] == N);
      CHECK(a.shape()[1] == N);
      CHECK(a.strides()[0] == 0);
      CHECK(a.strides()[1] == 0);

      /* TODO: NOT IMPLEMENTED
      CHECK_NOTHROW(a(0));
      CHECK_NOTHROW(a(N-1));
      CHECK_NOTHROW(a(N-1, N-1));
      */
      const float* data = a.storage();
      for(size_t i=0; i<a.memsize(); i++) CHECK(eql_f32(data[i], 69.f));
    }
  }
}

TEST_CASE("Data initialization") {
  SECTION("empty malloc") {
    float* d = static_cast<float*>( aligned_alloc(32, N*N*sizeof(float)));
    Tensor<float> a = Tensor<float>(d, N*N, {N, N});
    CHECK(a.is_initialized());
    CHECK(a.is_allocated());
    CHECK(!a.has_grad());
    CHECK(!a.is_eye());
    CHECK(!a.is_sub());
    CHECK(a.device() == 1);
    CHECK(a.memsize() == N*N);
    CHECK(a.numel() == N*N);
    CHECK(a.ndim() == 2);
    CHECK(a.storage() != nullptr);
    CHECK(a.shape()[0] == N);
    CHECK(a.shape()[1] == N);
    CHECK(a.strides()[0] == N);
    CHECK(a.strides()[1] == 1);
    CHECK_NOTHROW(a(0));
    CHECK_NOTHROW(a(N-1));
    CHECK_NOTHROW(a(N-1, N-1));
  }
}

