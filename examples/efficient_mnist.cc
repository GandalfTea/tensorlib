
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

//bool constexpr lt_f32 (float a, float b)  { return (b - a) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPSILON); }

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
  //ret.x_train = static_cast<float*>(aligned_alloc(32, sizeof(float)*(x_train_s-16))); // remove header
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

// conv2d kernel
// ---------------------------------------------

// computes 8 output values at once
inline void _mm256_conv2d_ps(const float* a, const float* b, float* c, int lda) {
  int i, j=0;
  __m256 c0 = _mm256_setzero_ps();
  #pragma omp parallel for private(i, j) num_threads(omp_get_max_threads())
  for(i=0; i<5; i++) {
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+0)), _mm256_broadcast_ss((float*)b  ), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+1)), _mm256_broadcast_ss((float*)b+1), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+2)), _mm256_broadcast_ss((float*)b+2), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+3)), _mm256_broadcast_ss((float*)b+3), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+4)), _mm256_broadcast_ss((float*)b+4), c0); 
    a+=lda;
    b+=5;
  }
  _mm256_storeu_ps(c,  c0);
}

// iterates conv2d kernel over height ldc, loading in as many 
// ymm registers as necessary to fill all width ldc values.
inline void conv2d(float* a, float* b, float* c, int lda, int ldc) {
  uint32_t it = ceil((float)ldc/8);
  #pragma omp parallel for num_threads(omp_get_max_threads())
  for(int i=0; i<ldc; i++) {
    #pragma unroll
    for(int j=0; j<it; j++) {
      _mm256_conv2d_ps(&a[j*8], b, &c[j*8], lda);
    }
    a+=lda;
    c+=ldc;
  }
}

inline void batch_conv2d(float* in, float* ker, float* out, int lda=28, int lo=24, int batch=512, int channels=32) {
  #pragma omp parallel for ordered shared(in, ker, out) collapse(2) num_threads(omp_get_max_threads())
  for(int i=0; i<batch; i++) {
    for(int k=0; k<channels; k++) {
      #pragma omp ordered
      conv2d(&in[i*lda*lda], &ker[k*5*5], &out[i*lo*lo*channels + k*lo*lo], lda, lo);
    }
  }
}


// randn
// ---------------------------------------------
void f32_uniform(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0.f, float e=0.f) {
  std::mt19937 rng(std::random_device{}()); 
  if(!eql_f32(seed, 0.f)) rng.seed(seed); 
  std::uniform_real_distribution<float> dist(down, up); 
  if(!eql_f32(e, 0.f)) for(size_t i=0; i<count; i++) do to[i] = dist(rng); while (to[i] <= e); 
  else for(size_t i=0; i<count; i++) to[i] = dist(rng); 
}

// TODO: check for duplicates
void randn_batch(const float* src, float* to, uint32_t len, uint32_t count) {
  std::mt19937 rng(std::random_device{}()); 
  std::uniform_real_distribution<float> dist(0, len); // generate (count) randn int betweeen [0, len)
  for(int i=0; i<count; i++) {
    uint32_t ridx = (uint32_t)dist(rng);
    memcpy(&to[i*28*28], &src[ridx*28*28], 28*28*sizeof(float));
  }
}


// nn
// ---------------------------------------------
inline Conv2d allocate_conv2d_layer(uint32_t ks, uint32_t in_channels, uint32_t out_channels, uint32_t count) {
  Conv2d l;
  l.weights = (float*)aligned_alloc(32, out_channels*ks*ks*count*sizeof(float));
  l.grad = (float*)aligned_alloc(32, out_channels*ks*ks*count*sizeof(float));
  l.in_channels = in_channels;
  l.out_channels = out_channels;
  return l;
}

inline float relu(float* in) { return lt_f32(0.f, *in) ? *in : 0.f; }

void apply_relu(float* from, uint32_t count) { 
  #pragma omp parallel for num_threads(omp_get_max_threads())
  for(int i=0; i<count; i++) from[i] = relu(&from[i]); 
}

typedef struct {
  uint32_t size;
  uint32_t num_batches_tracked=0;
  bool affine;
  float* weight;
  float* bias;
  float* mean;
  float* var;
  float eps;
  float momentum;
} bnorm2d_state;

bnorm2d_state init_batchnorm2d(uint32_t size, bool affine=true) {
  bnorm2d_state r;
  r.size = size;
  r.affine = affine;
  r.eps = 1e-5;
  r.momentum = 0.1f;
  if(affine) {
    r.weight = (float*)aligned_alloc(32, size*sizeof(float));
    r.bias = (float*)aligned_alloc(32, size*sizeof(float));
    f32_uniform(r.weight, 32);
    f32_uniform(r.bias, 32);
  }
  return r;
}

/* NOTE: normalisation is done over the channel dimension because each channel represents
         one convolution kernel. This aids in feature extraction.

         https://stackoverflow.com/questions/45799926/why-batch-normalization-over-channels-only-in-cnn 
         related paper: https://arxiv.org/pdf/1803.08494                                                    */     

// https://arxiv.org/pdf/1502.03167
void inline run_batchnorm2d(bnorm2d_state* state, float* in, uint32_t count, bool training=false) {
  float* vars = (float*)calloc(32, sizeof(float));
  float* means = (float*)calloc(32, sizeof(float));
  float* invstd = (float*)calloc(32, sizeof(float));

  // batch mean 
  // TODO: this looks like shit code
  #pragma omp parallel for collapse(3) num_threads(omp_get_max_threads())
  for(int b=0; b<512; b++) {
    for(int c=0; c<32; c++) {
      for(int i=0; i<22*22; i++) {
        means[c] += in[b*22*22*32+c*22*22+i];
      }
    }
  }
  for(int i=0; i<32; i++) means[i] /= 22*22*512;

  // batch variance
  #pragma omp parallel for collapse(2) num_threads(omp_get_max_threads())
  for(int b=0; b<512; b++) {
    for(int c=0; c<32; c++) {
      uint32_t widx = b*22*22*32+c*22*22;
      for(int i=0; i<22*22; i++) {
        vars[c] += std::pow(in[widx+i]-means[c], 2);
      }
    }
  }

  for(int i=0; i<32; i++) vars[i] /= 22*22*512; 
  for(int i=0; i<32; i++) invstd[i] = 1 / std::sqrt(vars[i]+state->eps);

  // normalise
  #pragma omp parallel for collapse(2) num_threads(omp_get_max_threads())
  for(int i=0; i<512; i++) {
    for(int c=0; c<32; c++) {
      float* wptr = &in[i*22*22*32+c*22*22];
      for(int p=0; p<22*22; p++) {
        wptr[p] = (wptr[p]+state->bias[c]-means[c])*(invstd[c]*state->weight[c]);
        // pytorch first multiplies then adds bias-mean*invstd*weight;
        // https://github.com/pytorch/pytorch/blob/420b37f3c67950ed93cd8aa7a12e673fcfc5567b/aten/src/ATen/native/Normalization.cpp#L96 
      }
    }
  }
  // TODO: update running mean and std
}



inline void _max2d(const float* in, uint8_t ks, float* out, int lda) {
  uint16_t i, j;
  float max = in[0];
  #pragma omp parallel for num_threads(omp_get_max_threads())
  for(i=0; i<ks; i++) { // can prob be replaces with while(ks-- > 0)
    #pragma unroll
    for(j=0; j<ks; j++) {
      if(lt_f32(max, in[j])) max=in[j];
    }
    in+=lda;
  }
  *out = max;
}

inline void max_pool2d(float* a, float* b, uint32_t lda, uint32_t ldb, uint8_t kernel_size=2, uint8_t stride=0, uint8_t dilation=1) {
  #pragma omp parallel for collapse(2) num_threads(omp_get_max_threads())
  for(int i=0; i<ldb; i++) {
    for(int j=0; j<ldb; j++) {
      _max2d(&a[i*lda+j], kernel_size, &b[i*ldb+j], lda); 
    }
  }
}


constexpr uint32_t size_after_conv(uint32_t img_size, uint32_t kernel_size, uint32_t dilation=1, uint32_t stride=1) {
  return (img_size - kernel_size) / stride + 1;
}


#define I 10000

int main() {


  // load 512 batch
  mnist_t mnist = load_mnist();
  float* batch = (float*)aligned_alloc(32, 513*28*28*sizeof(float)); 
  randn_batch(mnist.x_train, batch, mnist.ltrain, 512);

  // init layers
  Conv2d l1 = allocate_conv2d_layer(5, 1,  32, 512);
  Conv2d l2 = allocate_conv2d_layer(5, 32, 32, 512);
  Conv2d l4 = allocate_conv2d_layer(3, 32, 64, 512);
  Conv2d l5 = allocate_conv2d_layer(3, 64, 64, 512);
  bnorm2d_state b1 = init_batchnorm2d(32);
  bnorm2d_state b2 = init_batchnorm2d(64);

  uint32_t ls1 = size_after_conv(28, 5);
  uint32_t ls2 = size_after_conv(ls1, 5);
  uint32_t ls3 = size_after_conv(ls2, 2);

  // kaiming uniform -- https://arxiv.org/pdf/1502.01852.pdf
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    float kbound32 = std::sqrt(3.f)*std::sqrt(2.f / (1.f + 5.f)) / std::sqrt(32*5*5);
    float kbound64 = std::sqrt(3.f)*std::sqrt(2.f / (1.f + 5.f)) / std::sqrt(64*5*5);
    f32_uniform(l1.weights, 32*5*5, kbound32, -kbound32);
    f32_uniform(l2.weights, 32*5*5, kbound32, -kbound32);
    f32_uniform(l4.weights, 64*5*5, kbound64, -kbound64);
    f32_uniform(l5.weights, 64*5*5, kbound64, -kbound64);
  }

  // intermidiary activations
  // compute resulting panels -- (img_size-dilation*(kernel_size-1)-1)/stride
  float* outs = (float*)aligned_alloc(32, 513*32*ls1*ls1*sizeof(float));
  float* outs2 = (float*)aligned_alloc(32, 513*32*ls2*ls2*sizeof(float));
  float* outs3 = (float*)aligned_alloc(32, 513*32*ls3*ls3*sizeof(float));

  batch_conv2d(batch, l1.weights, outs,  28, ls1, 512, 32);
  apply_relu(outs, 512*32*ls1*ls1);
  batch_conv2d(outs,  l2.weights, outs2, ls1, ls2, 512, 32);

  std::cout << "\n\n";
  for(int i=0; i<28*28; i++) {
    if(i%28==0) std::cout << "\n";
    std::cout << std::setw(3) << batch[i] << ", ";
  }

  apply_relu(outs2, 512*32*ls2*ls2);

  std::cout << "\n\n";
  for(int i=0; i<1*ls1*ls1; i++) {
    if(i%ls1==0) std::cout << "\n";
    if(i%(ls1*ls1)==0) std::cout << "\n";
    std::cout << std::setw(11) << outs[i] << ", ";
  }

  run_batchnorm2d(&b1, outs2, 512*32*ls2*ls2, true);

  std::cout << "\n\n";
  for(int i=0; i<ls2*ls2; i++) {
    if(i%ls2==0) std::cout << "\n";
    std::cout << std::setw(11) << outs2[i] << ", ";
  }

  max_pool2d(outs2, outs3, ls2, ls3);

  std::cout << "\n\n";
  for(int i=0; i<ls3*ls3; i++) {
    if(i%ls3==0) std::cout << "\n";
    std::cout << std::setw(11) << outs3[i] << ", ";
  }
  std::cout << "\n\n";


  free(l1.weights);
  free(l2.weights);
  free(l4.weights);
  free(l5.weights);
  free(l1.grad);
  free(l2.grad);
  free(l4.grad);
  free(l5.grad);
  free(outs);
  free(mnist.x_train);
  free(mnist.y_train);
  free(mnist.x_test);
  free(mnist.y_test);
}
