
#if defined(__AVX__) 
  #include<immintrin.h>
#endif

#if defined(__SSE__) || defined(__SSE4__)
  #include <xmmintrin.h>
#endif

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "tensor.h"


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
    SECTION("alignmet") {
#ifdef __AVX__
      SECTION("AVX-256") {
        Tensor<float> a = Tensor<float>({N, N}).allocate();
        CHECK_NOTHROW(_mm256_load_ps(&(a.storage())[0])); // check if data is 32 byte aligned by default
      }
#endif
#ifdef __SSE__ 
      SECTION("SSE") {
        #undef MEMORY_ALIGNMENT
        #define MEMORY_ALIGNMENT 16
        Tensor<float> a = Tensor<float>({N, N}).allocate();
        CHECK_NOTHROW(_mm_load_ps(&(a.storage())[0]));
      }
#endif 
#ifdef __AVX512F__
      SECTION("AVX-512") {
        #undef MEMORY_ALIGNMENT
        #define MEMORY_ALIGNMENT 64; 
        Tensor<float> a = Tensor<float>({N, N}).allocate();
        CHECK_NOTHROW(_mm512_load_ps(&(a.storage())[0]));
      }
#endif
    }

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
#undef MEMORY_ALIGNMENT
#define MEMORY_ALIGNMENT 32

TEST_CASE("Data initialization") {
  SECTION("<alignment empty malloc") {
    float* d = alloc<float>(2);
    Tensor<float> a = Tensor<float>(d, 2, {2});
    CHECK(a.is_initialized());
    CHECK(a.is_allocated());
    CHECK(!a.has_grad());
    CHECK(!a.is_eye());
    CHECK(!a.is_sub());
    CHECK(a.device() == 1);
    CHECK(a.memsize() == 2);
    CHECK(a.numel() == 2);
    CHECK(a.ndim() == 1);
    CHECK(a.storage() != nullptr);
    CHECK(a.shape()[0] == 2);
    CHECK(a.strides()[0] == 1);
    CHECK_NOTHROW(a(0));
    CHECK_NOTHROW(a(1));
  }
  SECTION("empty malloc") {
    float* d = alloc<float>(N*N);
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
  SECTION("aligned_alloc malloc") {
    float* d = static_cast<float*>(aligned_alloc(MEMORY_ALIGNMENT, N*N*sizeof(float)));
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

  #undef N
  #define N 481
  SECTION("aligned_alloc random num malloc") {
    float* d = static_cast<float*>(aligned_alloc(MEMORY_ALIGNMENT, N*N*sizeof(float)));
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
  SECTION("shape mismatch small") {
    float* d = alloc<float>(N*N);
    CHECK_THROWS(Tensor<float>(d, N*N, {1, N}));
    free(d);
  }
  SECTION("shape mismatch big") {
    float* d = alloc<float>(N*N);
    CHECK_THROWS(Tensor<float>(d, N*N, {N, N, N}));
    free(d);
  }
  SECTION("wrong size small") {
    float* d = alloc<float>(N*N);
    CHECK_THROWS(Tensor<float>(d, N, {N, N}));
    free(d);
  }
  SECTION("wrong size big") {
    float* d = alloc<float>(N*N);
    CHECK_THROWS(Tensor<float>(d, N*N*2, {N, N}));
    free(d);
  }
  // TODO: this is technically valid
  /*
  SECTION("both wrong size and shape") {
    float* d = alloc<float>(N*N);
    CHECK_THROWS(Tensor<float>(d, N*N*2, {N, N, 2}));
    free(d);
  }
  */

  SECTION("special constructors") {
    SECTION("eye") {
      CHECK_THROWS(Tensor<int>({1}).eye());
      CHECK_THROWS(Tensor<int>({512}).eye());
      CHECK_THROWS(Tensor<int>({1, 1}).eye());
      CHECK_THROWS(Tensor<int>({1, 2}).eye());
      CHECK_THROWS(Tensor<int>({512, 513}).eye());
      CHECK_NOTHROW(Tensor<int>({2, 2}).eye());
      CHECK_NOTHROW(Tensor<int>({13, 13}).eye());
      CHECK_NOTHROW(Tensor<int>({2, 2, 2}).eye());
      CHECK_NOTHROW(Tensor<int>({512, 512}).eye());
      CHECK_NOTHROW(Tensor<int>({512, 512, 512}).eye());
      Tensor<int> a = Tensor<int>({512, 512}).eye();  
      for(size_t i=0; i<512; i++) {
        for(size_t j=0; j<512; j++) {
          CHECK(a(i,j).item() == ((i==j)?1:0));
        }
      }
      /* TODO: multi-dim eye is weird
      Tensor<int> b = Tensor<int>({128, 128, 128}).eye();  
      for(size_t i=0; i<512; i++) {
        for(size_t j=0; j<512; j++) {
          for(size_t k=0; k<512; k++) {
            CHECK(b(i,j,k).item() == ((i==j&&j==k)?1:0));
          }
        }
      }
      */
    }
    SECTION("eye allocated") {
      Tensor<int> a = Tensor<int>({128, 128}).eye().allocate();  
      const int* d = a.storage();
      for(size_t i=0; i<128; i++) {
        for(size_t j=0; j<128; j++) {
          CHECK(d[i*128+j] == ((i==j)?1:0));
        }
      }
      /* 
      Tensor<int> b = Tensor<int>({128, 128, 128}).eye().allocate();  
      d = b.storage();
      for(size_t i=0; i<128; i++) {
        for(size_t j=0; j<128; j++) {
          for(size_t k=0; k<128; k++) {
            CHECK(d[(i*128*128)+(j*128)+k] == ((i==j&&j==k)?1:0));
            if(d[(i*128*128)+(j*128)+k] == 1) std::cout << i << " " << j << " " << k << std::endl; 
          }
        }
      }
      */
    }
    SECTION("arange") {
      Tensor<int> a = Tensor<int>::arange(512);
      for(size_t i=0; i<512; i++) CHECK(a(i).item() == i);
      Tensor<int> b = Tensor<int>::arange(512, 128);
      for(size_t i=0, j=128; i<512-128; i++, j++) CHECK(b(i).item() == j);
      Tensor<int> c = Tensor<int>::arange(512, 128, 2);
      for(size_t i=0, j=128; i<(512-128)/2; i++, j+=2) CHECK(c(i).item() == j);
      float j = 0.f;
      Tensor<float> d = Tensor<float>::arange(1.f, 0.f, 0.1f);
      for(size_t i=0; i<(1.f-0.f)/0.1; i++, j+=0.1) CHECK( eql_f32(d(i).item(), j));
      j = -1.f;
      Tensor<float> e = Tensor<float>::arange(0.f, -1.f, 0.1f);
      for(size_t i=0; i<(1.f-0.f)/0.1; i++, j+=0.1) CHECK( eql_f32(e(i).item(), j));
      Tensor<int> f = Tensor<int>::arange(0, 100, -1);
      for(size_t i=0, j=100; i<100; i++, j-=1) CHECK(f(i).item() == j);
    }
  }
}

