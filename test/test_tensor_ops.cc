
#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "ntensor.h"

using namespace tensorlib;
#define N 1024
#define EPSILON 0.0001


TEST_CASE("Unary OPs") {
  SECTION("sqrt") {}
  SECTION("exp2") {}
  SECTION("log2") {}
  SECTION("sin") {}
  SECTION("neg") {}
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
