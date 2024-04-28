
#include "tensor.h"
#include <zlib.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

using namespace tensorlib;

#define CHUNK 8000000  // 8 MB zlib
#define BATCH_SIZE 512
#define ALIGNMENT 32

typedef struct {
  float* x_train;
  float* y_train;
  float* x_test;
  float* y_test;
  size_t ltrain, ltest; 
  size_t sl = 28; // rows and cols length
} mnist_t;

typedef struct {
  float* weights;
  float* grad;
  uint32_t in_channels;
  uint32_t out_channels;
} Conv2d;


// https://zlib.net/manual.html
int gz_inflate(unsigned char** to, size_t* usize, FILE* src) {

  unsigned ql = *usize/4; // half length to increase buffer size
  unsigned ul = *usize; // track uncompressed length

  int ret, flush;
  unsigned have;
  z_stream strm;
  unsigned char* in = static_cast<unsigned char*>(calloc(sizeof(unsigned char), CHUNK));

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;
  ret = inflateInit2(&strm, (16+MAX_WBITS));
  if(ret != Z_OK) return ret; 

  do {
    strm.avail_in = fread(in, 1, CHUNK, src);
    if(ferror(src)) {
      (void)inflateEnd(&strm);
      return Z_ERRNO;
    }
    flush = feof(src) ? Z_FINISH : Z_NO_FLUSH;
    strm.next_in = in;
    do {
      if(strm.total_out >= ul) {
        unsigned char* to2 = static_cast<unsigned char*>(calloc(sizeof(unsigned char), ul+CHUNK));
        memcpy(to2, *to, ul);
        ul+=CHUNK;
        free(*to);
        *to=&to2[0];
      }
      strm.next_out = (unsigned char*)(*to+strm.total_out);
      strm.avail_out = ul - strm.total_out;

      ret = inflate(&strm, flush);
      assert(ret != Z_STREAM_ERROR);
      have = CHUNK - strm.avail_out;
    } while(strm.avail_out == 0);

    assert(strm.avail_in==0);
  } while(flush != Z_FINISH);
  //std::cout << ret << std::endl;
  assert(ret == Z_STREAM_END);
  ret = inflateEnd(&strm);
  *usize = ul;
  free(in);
  return ret;
}


// NOTE: MNIST is MSB first, high endian
mnist_t load_mnist() {
  mnist_t ret;

  FILE* x_train_fd = fopen("./datasets/train-images-idx3-ubyte.gz", "rb");  
  FILE* y_train_fd = fopen("./datasets/train-labels-idx1-ubyte.gz", "rb");  
  FILE* x_test_fd  = fopen("./datasets/t10k-images-idx3-ubyte.gz", "rb");  
  FILE* y_test_fd  = fopen("./datasets/t10k-labels-idx1-ubyte.gz", "rb");  

  // ./datasets/train-images-idx3-ubyte.gz
  fseek(x_train_fd, 0, SEEK_END);
  size_t x_train_s = ftell(x_train_fd);
  rewind(x_train_fd);
  unsigned char* x_train_b = static_cast<unsigned char*>(malloc(x_train_s*sizeof(unsigned char)));
  gz_inflate(&x_train_b, &x_train_s, x_train_fd); // updates x_train_s to inflated size
  ret.x_train = static_cast<float*>(malloc(sizeof(float)*(x_train_s-16))); // remove header
  assert(x_train_b[0] == 0x00);
  assert(x_train_b[1] == 0x00);
  assert(x_train_b[2] == 0x08); // unsigned byte dtype
  assert(x_train_b[3] == 0x03); // ?
  uint32_t n_imgs = (x_train_b[4]  << 24) + (x_train_b[5]  << 16) + (x_train_b[6]  << 8) + x_train_b[7];
  uint32_t n_rows = (x_train_b[8]  << 24) + (x_train_b[9]  << 16) + (x_train_b[10] << 8) + x_train_b[11];
  uint32_t n_cols = (x_train_b[12] << 24) + (x_train_b[13] << 16) + (x_train_b[14] << 8) + x_train_b[15];
  assert(n_imgs == 60000);
  assert(n_rows == 28 && n_cols == 28);
  for(size_t i=16, j=0; i<x_train_s-16; i++, j++) ret.x_train[j] = static_cast<float>(x_train_b[i]); // uchar -> float32
  free(x_train_b);

  // ./datasets/t10k-images-idx3-ubyte.gz
  fseek(x_test_fd, 0, SEEK_END);
  size_t x_test_s = ftell(x_train_fd);
  rewind(x_test_fd);
  unsigned char* x_test_b = static_cast<unsigned char*>(malloc(x_test_s*sizeof(unsigned char)));
  gz_inflate(&x_test_b, &x_test_s, x_test_fd); 
  ret.x_test = static_cast<float*>(malloc(sizeof(float)*(x_train_s-16))); // remove header
  assert(x_test_b[0] == 0x00);
  assert(x_test_b[1] == 0x00);
  assert(x_test_b[2] == 0x08);
  assert(x_test_b[3] == 0x03);
  n_imgs = (x_test_b[4]  << 24) + (x_test_b[5]  << 16) + (x_test_b[6]  << 8) + x_test_b[7];
  n_rows = (x_test_b[8]  << 24) + (x_test_b[9]  << 16) + (x_test_b[10] << 8) + x_test_b[11];
  n_cols = (x_test_b[12] << 24) + (x_test_b[13] << 16) + (x_test_b[14] << 8) + x_test_b[15];
  assert(n_imgs == 10000);
  assert(n_rows == 28 && n_cols == 28);
  for(size_t i=16, j=0; i<x_test_s-16; i++, j++) ret.x_test[j] = static_cast<float>(x_test_b[i]); // uchar -> float32
  free(x_test_b);

  // ./datasets/train-labels-idx1-ubyte.gz 
  fseek(y_train_fd, 0, SEEK_END);
  size_t y_train_s = ftell(y_train_fd);
  rewind(y_train_fd);
  unsigned char* y_train_b = static_cast<unsigned char*>(malloc(y_train_s*sizeof(unsigned char)));
  gz_inflate(&y_train_b, &y_train_s, y_train_fd); 
  ret.y_train = static_cast<float*>(malloc(sizeof(float)*(y_train_s-8))); // remove header
  assert(y_train_b[0] == 0x00);
  assert(y_train_b[1] == 0x00);
  assert(y_train_b[2] == 0x08); 
  assert(y_train_b[3] == 0x01);
  uint32_t n_lbs = (y_train_b[4]  << 24) + (y_train_b[5]  << 16) + (y_train_b[6]  << 8) + y_train_b[7];
  assert(n_lbs = 60000);
  for(size_t i=8, j=0; i<y_train_s-8; i++, j++) ret.y_train[j] = static_cast<float>(y_train_b[i]); // uchar -> float32
  free(y_train_b);

  // ./datasets/t10k-labels-idx1-ubyte.gz
  fseek(y_test_fd, 0, SEEK_END);
  size_t y_test_s = ftell(y_train_fd);
  rewind(y_test_fd);
  unsigned char* y_test_b = static_cast<unsigned char*>(malloc(y_test_s*sizeof(unsigned char)));
  gz_inflate(&y_test_b, &y_test_s, y_test_fd); 
  ret.y_test = static_cast<float*>(malloc(sizeof(float)*(y_test_s-8))); // remove header
  assert(y_test_b[0] == 0x00);
  assert(y_test_b[1] == 0x00);
  assert(y_test_b[2] == 0x08);
  assert(y_test_b[3] == 0x01);
  n_lbs = (y_test_b[4]  << 24) + (y_test_b[5]  << 16) + (y_test_b[6]  << 8) + y_test_b[7];
  assert(n_lbs = 60000);
  for(size_t i=8, j=0; i<y_test_s-8; i++, j++) ret.y_test[j] = static_cast<float>(y_test_b[i]); // uchar -> float32
  free(y_test_b);

  ret.ltrain = 60000;
  ret.ltest  = 10000; 
  ret.sl = 28;
  return ret;
}


inline void _mm256_conv2d_ps(const float* a, const float* b, float* c, int lda) {
  m256_t c0;
  c0.v = _mm256_setzero_ps();
  for(int i=0, j=0; i<25; i+=5, j+=lda) {
    c0.v = _mm256_fmadd_ps(_mm256_load_ps((float*)(a+j+0)), _mm256_broadcast_ss((float*)b+i  ), c0.v); 
    c0.v = _mm256_fmadd_ps(_mm256_load_ps((float*)(a+j+1)), _mm256_broadcast_ss((float*)b+i+1), c0.v); 
    c0.v = _mm256_fmadd_ps(_mm256_load_ps((float*)(a+j+2)), _mm256_broadcast_ss((float*)b+i+2), c0.v); 
    c0.v = _mm256_fmadd_ps(_mm256_load_ps((float*)(a+j+3)), _mm256_broadcast_ss((float*)b+i+3), c0.v); 
    c0.v = _mm256_fmadd_ps(_mm256_load_ps((float*)(a+j+4)), _mm256_broadcast_ss((float*)b+i+4), c0.v); 
  }
  _mm256_storeu_ps( &c[0],  c0.v);
  /* std::cout << "\n\nc0:";
  for(int i=0; i<8; i++) std::cout << std::setw(9) << c0.f[i] << ","; 
  std::cout << "\n"; */
}

// only 5x5
inline void conv2d(const float* a, const float* b, float* c, int lda) {
  for(int i=0; i<lda; i++) {
    _mm256_conv2d_ps(&a[0], &b[0], &c[0], lda);
    _mm256_conv2d_ps(&a[8], &b[0], &c[8], lda);
    _mm256_conv2d_ps(&a[16], &b[0], &c[16], lda);
    a+=lda;
    c+=3*8;
  }
}

/*
int pool(float* src, float** to, size_t img_count=BATCH_SIZE, size_t img_size=28, uint32_t kernel_size=5, uint32_t stride=1, uint32_t dilation=1) {
  uint64_t panels = (img_size-dilation*(kernel_size-1)-1)/stride; 
  *to = static_cast<float*>( aligned_alloc(ALIGNMENT, panels*panels*kernel_size*kernel_size*img_count*sizeof(float)) );

  for(size_t img=0; img<img_count; img++) {
    float* ws = &src[img*28*28];
    float* wt = &(*to)[img*panels*panels*kernel_size*kernel_size]; // std::pow is slow af
    uint32_t v_str=0, h_str=0; // vertical and horizontal accumulated stride idx

    #pragma omp parallel for private(ws, wt) shared(panels, img_count, img_size, kernel_size, stride, dilation) num_threads(omp_get_max_threads())
    for(size_t i=0; i<panels*panels; i++) {
      if(dilation==1) {
        for(size_t j=0; j<kernel_size; j++) 
          memcpy((void*)&wt[kernel_size*j], (void*)&ws[img_size*(v_str+j)+h_str], kernel_size*sizeof(float));
      } else { } // implement dilation cpy 
      if(i%panels==0 && i!=0) { 
        v_str+=stride; 
        h_str=0;
      }
      h_str+=stride;
      wt+=kernel_size*kernel_size;
    }
  }
}
*/

// randn
void f32_uniform(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0.f, float e=0.f) {
  std::mt19937 rng(std::random_device{}()); 
  if(!eql_f32(seed, 0.f)) rng.seed(seed); 
  std::uniform_real_distribution<float> dist(down, up); 
  if(!eql_f32(e, 0.f)) for(size_t i=0; i<count; i++) do to[i] = dist(rng); while (to[i] <= e); 
  else for(size_t i=0; i<count; i++) to[i] = dist(rng); 
}


#define I 10000

int main() {

  mnist_t mnist = load_mnist();

  Conv2d l1, l2, l4, l5;
  l1.weights = (float*)aligned_alloc(32, 32*5*5*sizeof(float));
  l2.weights = (float*)aligned_alloc(32, 32*5*5*sizeof(float));
  l4.weights = (float*)aligned_alloc(32, 64*5*5*sizeof(float));
  l5.weights = (float*)aligned_alloc(32, 64*5*5*sizeof(float));

  #pragma omp parallel num_threads(12)
  {
    // kaiming uniform -- https://arxiv.org/pdf/1502.01852.pdf
    float kbound32 = std::sqrt(3.f)*std::sqrt(2.f / (1.f + 5.f)) / std::sqrt(32*5*5);
    float kbound64 = std::sqrt(3.f)*std::sqrt(2.f / (1.f + 5.f)) / std::sqrt(64*5*5);
    f32_uniform(l1.weights, 32*5*5, kbound32, -kbound32);
    f32_uniform(l2.weights, 32*5*5, kbound32, -kbound32);
    f32_uniform(l4.weights, 64*5*5, kbound64, -kbound64);
    f32_uniform(l5.weights, 64*5*5, kbound64, -kbound64);
  }

  float* outs = (float*)aligned_alloc(32, 512*32*5*5*sizeof(float));

  #pragma omp parallel for shared(mnist, outs) collapse(3) num_threads(12)
  for(int i=0; i<512; i++) {
    for(int k=0; k<32; k++) {
      for(int j=0; j<24*24; j++) {
        conv2d(&mnist.x_train[i*28*28], &l1.weights[k*5*5], &outs[i*24*24+(24*24*k)], mnist.sl);
      }
    }
  }

  for(int i=0; i<24*24*3; i++) {
    if(i%24==0) std::cout << "\n";
    if(i%(24*24)==0) std::cout << "\n\n";
    std::cout << std::setw(10) << outs[i] << ", ";
  }
  std::cout << "\n\n";

/*
  long double sum1 = 0;
  for(int i=0; i<I; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int j=0; j<(512*5*5)/8; j++) {
      _5x5_conv_ps(&pa[0], &b[0], &c[0], 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end - start;
    sum1 += ms_double.count();
  }
  std::cout << "\n\nunaligned :" << sum1/I << " ms "<< std::endl;

  sum1 = 0;
  for(int i=0; i<I; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int j=0; j<(512*5*5)/8; j++) {
      _5x5_conv_ps_u(&a[0], &b[0], &c[0], 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end - start;
    sum1 += ms_double.count();
  }
  std::cout << "pad aligned :" << sum1/I << " ms \n\n"<< std::endl;

  std::cout << "\npa : ";
  for(int i=0; i<512*8*5; i++) { if(i%8==0) std::cout << "\n"; std::cout << std::setw(3) << pa[i] << ", "; }
  std::cout << "\n";
  */


/*
  free(a);
  free(pa);
  free(b);
  free(c);
  */
}
