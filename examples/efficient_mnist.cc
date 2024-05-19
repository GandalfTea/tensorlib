
#include <omp.h>
#include <zlib.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>
#include <immintrin.h>


#define CHUNK 8000000  // 8 MB zlib
#define BATCH_SIZE 512
#define ALIGNMENT 32
#define EPSILON 1e-5 

// helpers

bool constexpr eql_f32 (float a, float b) { return fabs(a-b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * EPSILON); }
bool constexpr lt_f32 (float a, float b)  { return (b - a) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPSILON); }
constexpr uint32_t size_after_conv(uint32_t img_size, uint32_t kernel_size, uint32_t dilation=1, uint32_t stride=1) {
  return (img_size - kernel_size) / stride + 1;
}

typedef union {
  __m256 v;
  float f[8];
} m256_t;

// Load mnist and inflate

typedef struct {
  float* x_train;
  float* y_train;
  float* x_test;
  float* y_test;
  size_t ltrain, ltest; 
  size_t sl = 28; // rows and cols length
} mnist_t;

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
  assert(x_train_b[3] == 0x03); 
  uint32_t n_imgs = (x_train_b[4]  << 24) + (x_train_b[5]  << 16) + (x_train_b[6]  << 8) + x_train_b[7];
  uint32_t n_rows = (x_train_b[8]  << 24) + (x_train_b[9]  << 16) + (x_train_b[10] << 8) + x_train_b[11];
  uint32_t n_cols = (x_train_b[12] << 24) + (x_train_b[13] << 16) + (x_train_b[14] << 8) + x_train_b[15];
  assert(n_imgs == 60000);
  assert(n_rows == 28 && n_cols == 28);
  for(size_t i=16, j=0; i<x_train_s-16; i++, j++) ret.x_train[j] = static_cast<float>(x_train_b[i]); 
  free(x_train_b);

  // ./datasets/t10k-images-idx3-ubyte.gz
  fseek(x_test_fd, 0, SEEK_END);
  size_t x_test_s = ftell(x_train_fd);
  rewind(x_test_fd);
  unsigned char* x_test_b = static_cast<unsigned char*>(malloc(x_test_s*sizeof(unsigned char)));
  gz_inflate(&x_test_b, &x_test_s, x_test_fd); 
  ret.x_test = static_cast<float*>(malloc(sizeof(float)*(x_train_s-16))); 
  assert(x_test_b[0] == 0x00);
  assert(x_test_b[1] == 0x00);
  assert(x_test_b[2] == 0x08);
  assert(x_test_b[3] == 0x03);
  n_imgs = (x_test_b[4]  << 24) + (x_test_b[5]  << 16) + (x_test_b[6]  << 8) + x_test_b[7];
  n_rows = (x_test_b[8]  << 24) + (x_test_b[9]  << 16) + (x_test_b[10] << 8) + x_test_b[11];
  n_cols = (x_test_b[12] << 24) + (x_test_b[13] << 16) + (x_test_b[14] << 8) + x_test_b[15];
  assert(n_imgs == 10000);
  assert(n_rows == 28 && n_cols == 28);
  for(size_t i=16, j=0; i<x_test_s-16; i++, j++) ret.x_test[j] = static_cast<float>(x_test_b[i]);
  free(x_test_b);

  // ./datasets/train-labels-idx1-ubyte.gz 
  fseek(y_train_fd, 0, SEEK_END);
  size_t y_train_s = ftell(y_train_fd);
  rewind(y_train_fd);
  unsigned char* y_train_b = static_cast<unsigned char*>(malloc(y_train_s*sizeof(unsigned char)));
  gz_inflate(&y_train_b, &y_train_s, y_train_fd); 
  ret.y_train = static_cast<float*>(malloc(sizeof(float)*(y_train_s-8))); 
  assert(y_train_b[0] == 0x00);
  assert(y_train_b[1] == 0x00);
  assert(y_train_b[2] == 0x08); 
  assert(y_train_b[3] == 0x01);
  uint32_t n_lbs = (y_train_b[4]  << 24) + (y_train_b[5]  << 16) + (y_train_b[6]  << 8) + y_train_b[7];
  assert(n_lbs = 60000);
  for(size_t i=8, j=0; i<y_train_s-8; i++, j++) ret.y_train[j] = static_cast<float>(y_train_b[i]);
  free(y_train_b);

  // ./datasets/t10k-labels-idx1-ubyte.gz
  fseek(y_test_fd, 0, SEEK_END);
  size_t y_test_s = ftell(y_train_fd);
  rewind(y_test_fd);
  unsigned char* y_test_b = static_cast<unsigned char*>(malloc(y_test_s*sizeof(unsigned char)));
  gz_inflate(&y_test_b, &y_test_s, y_test_fd); 
  ret.y_test = static_cast<float*>(malloc(sizeof(float)*(y_test_s-8))); 
  assert(y_test_b[0] == 0x00);
  assert(y_test_b[1] == 0x00);
  assert(y_test_b[2] == 0x08);
  assert(y_test_b[3] == 0x01);
  n_lbs = (y_test_b[4]  << 24) + (y_test_b[5]  << 16) + (y_test_b[6]  << 8) + y_test_b[7];
  assert(n_lbs = 60000);
  for(size_t i=8, j=0; i<y_test_s-8; i++, j++) ret.y_test[j] = static_cast<float>(y_test_b[i]);
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
// computes 8 output values at once
// ---------------------------------------------

inline void _mm256_5x5_conv2d_ps(const float* a, const float* b, float* c, int lda) {
  uint8_t i;
  __m256 c0 = _mm256_setzero_ps();
  #pragma omp parallel for private(i) num_threads(omp_get_max_threads())
  for(i=0; i<5; i++) {
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a  )), _mm256_broadcast_ss((float*)b  ), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+1)), _mm256_broadcast_ss((float*)b+1), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+2)), _mm256_broadcast_ss((float*)b+2), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+3)), _mm256_broadcast_ss((float*)b+3), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+4)), _mm256_broadcast_ss((float*)b+4), c0); 
    a+=lda;
    b+=5;
  }
  _mm256_storeu_ps(c,  c0);
}

inline void _mm256_3x3_conv2d_ps(const float* a, const float* b, float* c, int lda) {
  uint8_t i;
  __m256 c0 = _mm256_setzero_ps();
  #pragma omp parallel for private(i) num_threads(omp_get_max_threads())
  for(i=0; i<3; i++) {
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a  )), _mm256_broadcast_ss((float*)b  ), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+1)), _mm256_broadcast_ss((float*)b+1), c0); 
    c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+2)), _mm256_broadcast_ss((float*)b+2), c0); 
    a+=lda;
    b+=3;
  }
  _mm256_storeu_ps(c,  c0);
}


inline void batch_conv2d_5(float* in, float* ker, float* out, int lda=28, int lo=24, int batch=512, int channels=32) {
  #pragma omp parallel for ordered shared(in, ker, out) num_threads(omp_get_max_threads())
  for(int i=0; i<batch; i++) {
    for(int k=0; k<channels; k++) {

      // iterates conv2d kernel over height ldc, loading in as many 
      // ymm registers as necessary to fill all width ldc values.

      float* wa = &in[i*lda*lda];
      float* wb = &ker[k*5*5];
      float* wc = &out[i*lo*lo*channels + k*lo*lo];
      uint32_t it = ceil((float)lo/8);
      #pragma omp parallel for num_threads(omp_get_max_threads())
      for(int i=0; i<lo; i++) {
        for(int j=0; j<it; j++) _mm256_5x5_conv2d_ps(&wa[j*8], wb, &wc[j*8], lda);
        wa+=lda;
        wc+=lo;
      }
    }
  }
}

inline void batch_conv2d_3(float* in, float* ker, float* out, int lda=28, int lo=24, int batch=512, int channels=32) {
  #pragma omp parallel for ordered shared(in, ker, out) num_threads(omp_get_max_threads())
  for(int i=0; i<batch; i++) {
    for(int k=0; k<channels; k++) {
      float* wa = &in[i*lda*lda];
      float* wb = &ker[k*3*3];
      float* wc = &out[i*lo*lo*channels + k*lo*lo];
      uint32_t it = ceil((float)lo/8);
      #pragma omp parallel for num_threads(omp_get_max_threads())
      for(int i=0; i<lo; i++) {
        for(int j=0; j<it; j++) _mm256_3x3_conv2d_ps(&wa[j*8], wb, &wc[j*8], lda);
        wa+=lda;
        wc+=lo;
      }
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
  std::uniform_real_distribution<float> dist(0, len); // [0, len) 
  #pragma omp parallel num_threads(omp_get_max_threads())
  while(count-- > 0) {
    memcpy(to, &src[(uint32_t)dist(rng)*28*28], 28*28*sizeof(float));
    to+=28*28;
  }
}


// nn
// ---------------------------------------------

inline void relu(float* in, uint32_t count) {
  #pragma omp parallel num_threads(omp_get_max_threads()) // TODO does this do anything?
  while(count-- > 0) if(lt_f32(in[count], 0.f)) in[count] = 0.f;
}


// Conv2D layer

typedef struct {
  float* weights;
  float* grad;
  uint32_t in_channels;
  uint32_t out_channels;
} conv2d;

inline conv2d init_conv2d_layer(uint32_t ks, uint32_t in_channels, uint32_t out_channels, uint32_t count, float up, float down) {
  conv2d l = (conv2d){ .in_channels=in_channels, .out_channels=out_channels };
  l.weights = (float*)aligned_alloc(ALIGNMENT, out_channels*ks*ks*count*sizeof(float));
  l.grad = (float*)aligned_alloc(ALIGNMENT, out_channels*ks*ks*count*sizeof(float));
  f32_uniform(l.weights, out_channels*ks*ks, up, down);
  return l;
}


// Batch Normalisation layer 
// https://arxiv.org/pdf/1502.03167

typedef struct {
  uint32_t size;
  uint32_t num_batches_tracked=0;
  bool affine;
  float* weights;
  float* bias;
  float* mean;
  float* var;
  float eps;
  float momentum;
} bnorm2d_state;

bnorm2d_state init_batchnorm2d(uint32_t size, bool affine=true) {
  bnorm2d_state r = (bnorm2d_state){ .size=size, .affine=affine, .eps=1e-5, .momentum=0.1f, };
  if(affine) {
    r.weights = (float*)aligned_alloc(32, size*sizeof(float));
    r.bias = (float*)aligned_alloc(32, size*sizeof(float));
    f32_uniform(r.weights, size);
    f32_uniform(r.bias, size);
  }
  return r;
}

// TODO: update running mean and std
/* NOTE: normalisation is done over the channel dimension because each channel represents
   one convolution kernel. This aids in feature extraction. */

void inline run_batchnorm2d(bnorm2d_state* state, float* in, uint32_t ch, uint32_t img_s, uint32_t batch=512, bool training=false) {
  float* vars = (float*)calloc(ch, sizeof(float));
  float* means = (float*)calloc(ch, sizeof(float));
  float* invstd = (float*)calloc(ch, sizeof(float));
  uint32_t s_ch  = img_s*img_s;
  uint32_t s_img = ch*s_ch;

  // batch mean 
  for(int b=0; b<batch; b++) {
    int c;
    #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:means[c])
    for(c=0; c<ch; c++) {
      uint32_t widx = b*s_img+c*s_ch;
      for(int i=0; i<s_ch; i++) {
        means[c] += in[widx+i];
      }
    }
  }
  for(int i=0; i<ch; i++) means[i] /= s_ch*batch;

  // batch variance and inverse standard deviation
  for(int b=0; b<batch; b++) {
    int c;
    #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:vars[c])
    for(c=0; c<ch; c++) {
      uint32_t widx = b*s_img+c*s_ch;
      for(int i=0; i<s_ch; i++) {
        vars[c] += std::pow(in[widx+i]-means[c], 2);
      }
    }
  }

  for(int i=0; i<ch; i++) vars[i] /= s_ch*batch; 
  for(int i=0; i<ch; i++) invstd[i] = 1 / std::sqrt(vars[i]+state->eps);

  // normalise, scale and shift
  #pragma omp parallel for collapse(2) num_threads(omp_get_max_threads())
  for(int i=0; i<batch; i++) {
    for(int c=0; c<ch; c++) {
      float* wptr = &in[i*s_img+c*s_ch];
      for(int p=0; p<s_ch; p++) {
        #pragma omp atomic write 
        wptr[p] = (wptr[p]+state->bias[c]-means[c])*(invstd[c]*state->weights[c]);
        // pytorch first multiplies then adds bias-mean*invstd*weights;
        // https://github.com/pytorch/pytorch/blob/420b37f3c67950ed93cd8aa7a12e673fcfc5567b/aten/src/ATen/native/Normalization.cpp#L96 
      }
    }
  }

/*
  std::cout << "\n\nmeans:\n";
  for(int i=0; i<ch; i++) {
    if(i%8==0) std::cout << "\n";
    std::cout << std::setw(11) << means[i] << ", ";
  }
  std::cout << "\n\nvariances:\n";
  for(int i=0; i<ch; i++) {
    if(i%8==0) std::cout << "\n";
    std::cout << std::setw(11) << vars[i] << ", ";
  }
  std::cout << "\n\ninv std:\n";
  for(int i=0; i<ch; i++) {
    if(i%8==0) std::cout << "\n";
    std::cout << std::setw(11) << invstd[i] << ", ";
  }
  std::cout << "\n\n";
  std::cout << "\n\n";
  */
}


// Max 2D Pooling

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

// default stride is kernel_size
inline void max_pool2d(float* a, float* b, uint32_t lda, uint32_t ldb, uint32_t ch, uint8_t kernel_size=2, uint8_t stride=2, uint8_t dilation=1) {
  uint32_t cnt = 0, i=0;
  #pragma omp parallel num_threads(omp_get_max_threads())
  while(cnt++ < ch*((lda*lda)/stride)) {
    if(i++==lda/stride) { a+=lda; i=0;}
    _max2d(a, kernel_size, b++, lda); 
    a+=stride;
  }
}


// Linear Layer

typedef struct {
  float* weights;
  float* grad;
  float* bias;
  uint32_t out_features;
} linear_state;

inline linear_state init_linear(uint32_t in_features, uint32_t out_features, bool bias=true) {
  linear_state r = (linear_state){ .grad=nullptr, .bias=nullptr, .out_features=out_features };  
  r.weights = (float*)aligned_alloc(ALIGNMENT, in_features*out_features*sizeof(float));
  float bound = std::sqrt(in_features);
  f32_uniform(r.weights, in_features*out_features, -1/bound, 1/bound);
  if(bias) {
    r.bias = (float*)malloc(out_features*sizeof(float));
    f32_uniform(r.bias, out_features, -1/bound, 1/bound);
  }
  return r;
}

inline void pack_a(int k, const float* a, int lda, float* to) {
  for(int j=0; j<k; j++) {
    const float *a_ij_ptr = &a[(j*lda)+0]; 
    *to = *a_ij_ptr;
    *(to+1) = *(a_ij_ptr+1); *(to+2) = *(a_ij_ptr+2);
    *(to+3) = *(a_ij_ptr+3); *(to+4) = *(a_ij_ptr+4);
    *(to+5) = *(a_ij_ptr+5); *(to+6) = *(a_ij_ptr+6);
    *(to+7) = *(a_ij_ptr+7);
    to += 8;
  }
}

// hardcoded matrix height of 10
inline void pack_b(int k, const float* b, int ldb, float* to) {
  int i;
  const float *b_i0_ptr = &b[0], *b_i1_ptr = &b[(1*ldb)],
  *b_i2_ptr = &b[(2*ldb)], *b_i3_ptr = &b[(3*ldb)],
  *b_i4_ptr = &b[(4*ldb)], *b_i5_ptr = &b[(5*ldb)],
  *b_i6_ptr = &b[(6*ldb)], *b_i7_ptr = &b[(7*ldb)];
  for(int i=0; i<k; i++) {
    *to = *b_i0_ptr; *(to+1) = *(b_i1_ptr); 
    *(to+2) = *(b_i2_ptr); *(to+3) = *(b_i3_ptr); 
    *(to+4) = *(b_i4_ptr); *(to+5) = *(b_i5_ptr); 
    *(to+6) = *(b_i6_ptr); *(to+7) = *(b_i7_ptr); 
    b_i0_ptr++; b_i1_ptr++; b_i2_ptr++; b_i3_ptr++; b_i4_ptr++; 
    b_i5_ptr++; b_i6_ptr++; b_i7_ptr++;
    to += 8;
  }
}

// TODO: segfault on c with aligned functions
inline void _sgemm(int k, float* a, float* b, float* c, int ldc) {
  m256_t c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, av, bv;
  __m256 w0, w1, w2, w3, w4, w5, w6, w7, w8, w9;
  c0.v = _mm256_setzero_ps(); c1.v = _mm256_setzero_ps();
  c2.v = _mm256_setzero_ps(); c3.v = _mm256_setzero_ps();
  c4.v = _mm256_setzero_ps(); c5.v = _mm256_setzero_ps();
  c6.v = _mm256_setzero_ps(); c7.v = _mm256_setzero_ps();
  c8.v = _mm256_setzero_ps(); c9.v = _mm256_setzero_ps();
  #pragma omp parallel for num_threads(omp_get_max_threads())
  for(int i=0; i<k; i++) {
    av.v = _mm256_load_ps((float*)a);
    bv.v = _mm256_broadcast_ss((float*)&b[0]); c0.v = _mm256_fmadd_ps(av.v, bv.v, c0.v);
    bv.v = _mm256_broadcast_ss((float*)&b[1]); c1.v = _mm256_fmadd_ps(av.v, bv.v, c1.v);
    bv.v = _mm256_broadcast_ss((float*)&b[2]); c2.v = _mm256_fmadd_ps(av.v, bv.v, c2.v);
    bv.v = _mm256_broadcast_ss((float*)&b[3]); c3.v = _mm256_fmadd_ps(av.v, bv.v, c3.v);
    bv.v = _mm256_broadcast_ss((float*)&b[4]); c4.v = _mm256_fmadd_ps(av.v, bv.v, c4.v);
    bv.v = _mm256_broadcast_ss((float*)&b[5]); c5.v = _mm256_fmadd_ps(av.v, bv.v, c5.v);
    bv.v = _mm256_broadcast_ss((float*)&b[6]); c6.v = _mm256_fmadd_ps(av.v, bv.v, c6.v);
    bv.v = _mm256_broadcast_ss((float*)&b[7]); c7.v = _mm256_fmadd_ps(av.v, bv.v, c7.v);
    bv.v = _mm256_broadcast_ss((float*)&b[8]); c8.v = _mm256_fmadd_ps(av.v, bv.v, c6.v);
    bv.v = _mm256_broadcast_ss((float*)&b[9]); c9.v = _mm256_fmadd_ps(av.v, bv.v, c7.v);
    a+=8; b+=10;

    std::cout << "\n";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << av.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << bv.f[i] << ", ";
    std::cout << "\n";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c0.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c1.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c2.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c3.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c4.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c5.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c6.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c7.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c8.f[i] << ", ";
    std::cout << "\n";
    for(int i=0; i<8; i++) std::cout << std::setw(13) << c9.f[i] << ", ";
  }
  w0 = _mm256_loadu_ps((float*)&c[0*ldc]); w1 = _mm256_loadu_ps((float*)&c[1*ldc]);
  w2 = _mm256_loadu_ps((float*)&c[2*ldc]); w3 = _mm256_loadu_ps((float*)&c[3*ldc]);
  w4 = _mm256_loadu_ps((float*)&c[4*ldc]); w5 = _mm256_loadu_ps((float*)&c[5*ldc]);
  w6 = _mm256_loadu_ps((float*)&c[6*ldc]); w7 = _mm256_loadu_ps((float*)&c[7*ldc]);
  w8 = _mm256_loadu_ps((float*)&c[8*ldc]); w9 = _mm256_loadu_ps((float*)&c[9*ldc]);
  c0.v = _mm256_add_ps(c0.v, w0); c1.v = _mm256_add_ps(c1.v, w1);
  c2.v = _mm256_add_ps(c2.v, w2); c3.v = _mm256_add_ps(c3.v, w3);
  c4.v = _mm256_add_ps(c4.v, w4); c5.v = _mm256_add_ps(c5.v, w5);
  c6.v = _mm256_add_ps(c6.v, w6); c7.v = _mm256_add_ps(c7.v, w7);
  c8.v = _mm256_add_ps(c8.v, w8); c9.v = _mm256_add_ps(c9.v, w9);
  _mm256_storeu_ps( &c[0*ldc], c0.v); _mm256_storeu_ps( &c[1*ldc], c1.v);
  _mm256_storeu_ps( &c[2*ldc], c2.v); _mm256_storeu_ps( &c[3*ldc], c3.v);
  _mm256_storeu_ps( &c[4*ldc], c4.v); _mm256_storeu_ps( &c[5*ldc], c5.v);
  _mm256_storeu_ps( &c[6*ldc], c6.v); _mm256_storeu_ps( &c[7*ldc], c7.v);
  _mm256_storeu_ps( &c[8*ldc], c8.v); _mm256_storeu_ps( &c[9*ldc], c9.v);
}

void sgemm(float* a, float* b, float* c, int m=512, int n=10, int k=576, int mb=8, int kb=8) {
  #pragma omp parallel for shared(a, b, c, m, n, k, mb, kb) default(none) num_threads(omp_get_max_threads())
  for(int i=0; i<k; i+=kb) {
    int ib = std::min(k-i, kb);
    float* pb = (float*)aligned_alloc(ALIGNMENT, ib*n*sizeof(float));
    for(int ii=0; ii<m; ii+=mb) {
      int iib = std::min(m-ii, mb);
      float* pa = (float*)aligned_alloc(ALIGNMENT, ib*iib*sizeof(float));
/*
      float* wa = &a[i*k+ii];
      float* wb = &b[i];
      for(int iii=0; iii<n; iii+=8) {
        if(ii==0) pack_b(iib, &wb[iii*n], n, &pb[iii*ib]);
        for(int iiii=0; iiii<iib; iiii+=8) {
          if(iii==0) pack_a(ib, &wa[iiii], k ,&pa[iiii*ib]);
          _sgemm(iib, &pa[iiii*ib], &pb[iii*ib], &c[ii+iii*n+iiii], n);
        }
      }
      */
      pack_b(iib, &b[i], n, pb);
      pack_a(ib, &a[i*k+ii], k, pa);
      _sgemm(iib, &pa[0*ib], &pb[0*ib], &c[ii], n);
    }
  }
}


int main() {

  // load 512 batch
  mnist_t mnist = load_mnist();
  float* batch = (float*)aligned_alloc(32, 513*28*28*sizeof(float)); 
  randn_batch(mnist.x_train, batch, mnist.ltrain, BATCH_SIZE);

  // initialize layers 
  conv2d l1, l2, l4, l5;
  bnorm2d_state l3, l6;
  linear_state l7;
  {
    // kaiming uniform -- https://arxiv.org/pdf/1502.01852.pdf
    float lhs = std::sqrt(3.f)*std::sqrt(2.f/(1.f+5.f)); // tinygrad does math.sqrt(5) as default a
    float kb325 = lhs / std::sqrt(32*5*5);
    float kb323 = lhs / std::sqrt(32*3*3);
    float kb643 = lhs / std::sqrt(64*3*3);

    l1 = init_conv2d_layer(5, 1,  32, 512, kb325, -kb325);
    l2 = init_conv2d_layer(5, 32, 32, 512, kb325, -kb325);
    l3 = init_batchnorm2d(32);
    l4 = init_conv2d_layer(3, 32, 64, 512, kb323, -kb323);
    l5 = init_conv2d_layer(3, 64, 64, 512, kb643, -kb643);
    l6 = init_batchnorm2d(64);
    l7 = init_linear(576, 10);
  }

  // img size compression per channel
  uint32_t ls0 = size_after_conv(28,  5);
  uint32_t ls1 = size_after_conv(ls0, 5);
  uint32_t ls2 = size_after_conv(ls1, 2, 1, 2);
  uint32_t ls3 = size_after_conv(ls2, 3);
  uint32_t ls4 = size_after_conv(ls3, 3);
  uint32_t ls5 = size_after_conv(ls4, 2, 1, 2);

  // intermediary activations
  float* outs0 = (float*)aligned_alloc(32, 513*32*ls0*ls0*sizeof(float));
  float* outs1 = (float*)aligned_alloc(32, 513*32*ls1*ls1*sizeof(float));
  float* outs2 = (float*)aligned_alloc(32, 513*32*ls2*ls2*sizeof(float));
  float* outs3 = (float*)aligned_alloc(32, 513*64*ls3*ls3*sizeof(float));
  float* outs4 = (float*)aligned_alloc(32, 513*64*ls4*ls4*sizeof(float));
  float* outs5 = (float*)aligned_alloc(32, 513*64*ls5*ls5*sizeof(float));
  float* outs6 = (float*)aligned_alloc(32, 544*10*sizeof(float));

  // run network
  batch_conv2d_5(batch, l1.weights, outs0, 28,  ls0, 512, 32);  relu(outs0, 512*32*ls1*ls1);
  batch_conv2d_5(outs0, l2.weights, outs1, ls0, ls1, 512, 32);  relu(outs1, 512*32*ls1*ls1);
  run_batchnorm2d(&l3, outs1, 32, ls1, 512, true);
  max_pool2d(outs1, outs2, ls1, ls2, 32);

  batch_conv2d_3(outs2, l4.weights, outs3, ls2, ls3, 512, 64);  relu(outs3, 512*64*ls3*ls3);
  batch_conv2d_3(outs3, l5.weights, outs4, ls3, ls4, 512, 64);  relu(outs4, 512*64*ls4*ls4);
  run_batchnorm2d(&l6, outs4, 64, ls4, 512, true);
  max_pool2d(outs4, outs5, ls4, ls5, 64);

  //sgemm(outs5, l7.weights, outs6, 512, 10, 576); 
  sgemm(outs5, l7.weights, outs6, 10, 10, 10); 

  /*
  std::cout << "\n\nl5:\n";
  for(int i=0; i<64*ls5*ls5; i++) {
    if(i%(ls5*ls5)==0) std::cout << "\n";
    if(i%ls5==0) std::cout << " - ";
    std::cout << std::setw(13) << outs5[i] << ", ";
  }

  std::cout << "\n\nl6:\n";
  for(int i=0; i<512*10; i++) {
    if(i%10==0) std::cout << "\n";
    std::cout << std::setw(13) << outs6[i] << ", ";
  }
  */

/*
  float* wo4 = outs4;

  std::cout << "\n\n";
  for(int i=0; i<64*ls4*ls4; i++) {
    if(i%ls4==0) std::cout << "\n";
    if(i%(ls4*ls4)==0) {
      wo4 += 63*ls4*ls4; 
      std::cout << "\n\n";
    }
    std::cout << std::setw(11) << wo4[i] << ", ";
  }
  */

/*
  std::cout << -kbound64 << " - " << kbound64;

  std::cout << "\n\n";
  for(int i=0; i<28*28; i++) {
    if(i%28==0) std::cout << "\n";
    std::cout << std::setw(3) << batch[i] << ", ";
  }

  std::cout << "\no:\n";
  for(int i=0; i<ls0*ls0; i++) {
    if(i%ls0==0) std::cout << "\n";
    std::cout << std::setw(11) << outs0[i] << ", ";
  }
  std::cout << "\nl1:\n";
  for(int i=0; i<ls1*ls1; i++) {
    if(i%ls1==0) std::cout << "\n";
    std::cout << std::setw(11) << outs1[i] << ", ";
  }
  std::cout << "\n\nl2:\n";
  for(int i=0; i<ls2*ls2; i++) {
    if(i%ls2==0) std::cout << "\n";
    std::cout << std::setw(11) << outs2[i] << ", ";
  }

  std::cout << "\n\nl3:\n";
  for(int i=0; i<ls3*ls3; i++) {
    if(i%ls3==0) std::cout << "\n";
    std::cout << std::setw(11) << outs3[i] << ", ";
  }

  std::cout << "\n\nl4:\n";
  for(int i=0; i<ls4*ls4; i++) {
    if(i%ls4==0) std::cout << "\n";
    std::cout << std::setw(11) << outs4[i] << ", ";
  }
  */





  std::cout << "\n\n";

/*
  free(l1.weights);
  free(l2.weights);
  free(l4.weights);
  free(l5.weights);
  free(l1.grad);
  free(l2.grad);
  free(l4.grad);
  free(l5.grad);
  free(outs0);
  free(mnist.x_train);
  free(mnist.y_train);
  free(mnist.x_test);
  free(mnist.y_test);
  */
}
