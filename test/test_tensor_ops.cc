
#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "tensor.h"

using namespace tensorlib;
#define N 1024
#define EPSILON 0.0001

TEST_CASE("Unary OPs") {
  SECTION("sqrt") {
    float* org = alloc<float>(N*N);
    auto a = Tensor<float>({N, N}).randn(100, 1, UNIFORM);
    std::memcpy(org, a.storage(), N*N*sizeof(float));
    a.sqrt();
    auto aptr = a.storage();
    for(size_t i=0; i<N*N; i++) CHECK(eql_f32(aptr[i], std::sqrt(org[i])));
    CHECK(a.is_initialized());
    CHECK(a.is_allocated());
    CHECK(!a.has_grad());
    CHECK(!a.is_eye());
    CHECK(!a.is_sub());
    CHECK(a.device() == 1);
    CHECK(a.memsize() == N*N);
    CHECK(a.numel() == N*N);
    CHECK(a.ndim() == 2);
    free(org);
  }
  SECTION("exp2") {
    float* org = alloc<float>(N*N);
    auto a = Tensor<float>({N, N}).randn();
    std::memcpy(org, a.storage(), N*N*sizeof(float));
    a.exp2();
    auto aptr = a.storage();
    for(size_t i=0; i<N*N; i++) CHECK(eql_f32(aptr[i], std::exp2(org[i])));
    CHECK(a.is_initialized());
    CHECK(a.is_allocated());
    CHECK(!a.has_grad());
    CHECK(!a.is_eye());
    CHECK(!a.is_sub());
    CHECK(a.device() == 1);
    CHECK(a.memsize() == N*N);
    CHECK(a.numel() == N*N);
    CHECK(a.ndim() == 2);
    free(org);
  }
  SECTION("log2") {
    float* org = alloc<float>(N*N);
    auto a = Tensor<float>({N, N}).randn(100, 1, UNIFORM);
    std::memcpy(org, a.storage(), N*N*sizeof(float));
    a.log2();
    auto aptr = a.storage();
    for(size_t i=0; i<N*N; i++) CHECK(eql_f32(aptr[i], std::log2(org[i])));
    CHECK(a.is_initialized());
    CHECK(a.is_allocated());
    CHECK(!a.has_grad());
    CHECK(!a.is_eye());
    CHECK(!a.is_sub());
    CHECK(a.device() == 1);
    CHECK(a.memsize() == N*N);
    CHECK(a.numel() == N*N);
    CHECK(a.ndim() == 2);
    free(org);
  }
  SECTION("sin") {
    float* org = alloc<float>(N*N);
    auto a = Tensor<float>({N, N}).randn();
    std::memcpy(org, a.storage(), N*N*sizeof(float));
    a.sin();
    auto aptr = a.storage();
    for(size_t i=0; i<N*N; i++) CHECK(eql_f32(aptr[i], std::sin(org[i])));
    CHECK(a.is_initialized());
    CHECK(a.is_allocated());
    CHECK(!a.has_grad());
    CHECK(!a.is_eye());
    CHECK(!a.is_sub());
    CHECK(a.device() == 1);
    CHECK(a.memsize() == N*N);
    CHECK(a.numel() == N*N);
    CHECK(a.ndim() == 2);
    free(org);
  }
  SECTION("neg") {
    float* org = alloc<float>(N*N);
    auto a = Tensor<float>({N, N}).randn();
    std::memcpy(org, a.storage(), N*N*sizeof(float));
    a.neg();
    auto aptr = a.storage();
    for(size_t i=0; i<N*N; i++) CHECK(eql_f32(aptr[i], (-1*org[i])));
    CHECK(a.is_initialized());
    CHECK(a.is_allocated());
    CHECK(!a.has_grad());
    CHECK(!a.is_eye());
    CHECK(!a.is_sub());
    CHECK(a.device() == 1);
    CHECK(a.memsize() == N*N);
    CHECK(a.numel() == N*N);
    CHECK(a.ndim() == 2);
    free(org);
  }
}

TEST_CASE("Binary OPs") {
  SECTION("add") {
    SECTION("float") {
      auto a = Tensor<float>({N, N}).randn();
      auto b = Tensor<float>({N, N}).randn();
      auto c = a.add(b);
      auto aptr = a.storage();
      auto bptr = b.storage();
      auto cptr = c.storage();
      CHECK(c.storage() != nullptr);
      for(size_t i=0; i<N*N; i++) CHECK(cptr[i] == (aptr[i]+bptr[i]));
      CHECK(c.is_initialized());
      CHECK(c.is_allocated());
      CHECK(!c.has_grad());
      CHECK(!c.is_eye());
      CHECK(!c.is_sub());
      CHECK(c.device() == 1);
      CHECK(c.memsize() == N*N);
      CHECK(c.numel() == N*N);
      CHECK(c.ndim() == 2);
    }
    SECTION("int") {
      // TODO
    }
  }
  SECTION("sub") {
    SECTION("float") {
      auto a = Tensor<float>({N, N}).randn();
      auto b = Tensor<float>({N, N}).randn();
      auto c = a.sub(b);
      auto aptr = a.storage();
      auto bptr = b.storage();
      auto cptr = c.storage();
      CHECK(c.storage() != nullptr);
      for(size_t i=0; i<N*N; i++) CHECK(cptr[i] == (aptr[i]-bptr[i]));
      CHECK(c.is_initialized());
      CHECK(c.is_allocated());
      CHECK(!c.has_grad());
      CHECK(!c.is_eye());
      CHECK(!c.is_sub());
      CHECK(c.device() == 1);
      CHECK(c.memsize() == N*N);
      CHECK(c.numel() == N*N);
      CHECK(c.ndim() == 2);
    }
    SECTION("int") {
      // TODO
    }
  }
  SECTION("div") {
    SECTION("float") {
      auto a = Tensor<float>({N, N}).randn();
      auto b = Tensor<float>({N, N}).randn();
      auto c = a.div(b);
      auto aptr = a.storage();
      auto bptr = b.storage();
      auto cptr = c.storage();
      CHECK(c.storage() != nullptr);
      for(size_t i=0; i<N*N; i++) CHECK(cptr[i] == (aptr[i]/bptr[i]));
      CHECK(c.is_initialized());
      CHECK(c.is_allocated());
      CHECK(!c.has_grad());
      CHECK(!c.is_eye());
      CHECK(!c.is_sub());
      CHECK(c.device() == 1);
      CHECK(c.memsize() == N*N);
      CHECK(c.numel() == N*N);
      CHECK(c.ndim() == 2);
    }
    SECTION("int") {
      // TODO
    }
  }
}
