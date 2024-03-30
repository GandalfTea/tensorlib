
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

/*
class Linear {
  Linear() { }
  Tensor<float>& operator(Tensor<float>& x) { }
}

class Conv2d {
  uint32_t* kernel_dims;
  Tensor<float>* weight, bias;
  uint32_t  kernel_dim, stride, padding, dilation, groups;

  Conv2d(uint32_t in_channels, uint32_t out_channels, std::initializer_list<uint32_t> kernel, 
         uint32_t stride=1, uint32_t dilation=1, uint32_t groups=1, bool bias=true) 
    : kernel_dim(kernel.size()), stride(stride), padding(padding), dilation(dilation), groups(groups)
  {
    if(kernel.size() != 2) throw std::runtime_error("Invalid Conv2d kernel dimensions.");
    size_t i=0, sum=1;
    kernel_dims = new uint32_t[2];
    for(uint32_t& x : kernel) { kernel_dims[i++] = x; sum*=x; }
    float kbount = std::sqrt(3.f)*std::sqrt(2.f / (1.f + 5.f)) / std::sqrt(sum);
    this->weight = new Tensor<float>({out_channels, (uint32_t)in_channels/groups, kernel_dims[0], kernel_dims[1]}).randn(-kbound, kbound, UNIFORM); // kaiming uniform
    if(bias) {
      float bound = std::sqrt(x);
      this->bias = new Tensor<float>({out_channels}).randn(-bound, bound, UNIFORM);
    }
  }

  Tensor<float> operator(Tensor<float> x) {
    x.pool({5, 5}).reshape({512, 1, 1, 1, 24, 24, 5, 5}).expand({512, 1, 1, 32, 24, 24, 5, 5}).permute({0, 1, 3, 4, 5, 2, 6, 7});
    auto ret = weight.reshape({1, 1, 32, 1, 1, 1, 5, 5}).dot(x).sum({-1, -2, -3}).reshape({512, 32, 24, 24});
    if(bias) ret.add(bias.reshape({1, -1, 1, 1})); 
    return ret;
  }
}
*/

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
  std::cout << ret << std::endl;
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

/*
typedef union {
  __m256 v;
  float f[8];
} m256_t;

inline void _mm256_gemm_ps(const float* a, const float* b, float* c, int ids) {
  m256_t c0007, c1017, c2027, c3037, c4047, c5057, c6067, c7077, a_vreg, b_p0_vreg;
  c0007.v = _mm256_setzero_ps(); c1017.v = _mm256_setzero_ps();
  c2027.v = _mm256_setzero_ps(); c3037.v = _mm256_setzero_ps();
  c4047.v = _mm256_setzero_ps(); c5057.v = _mm256_setzero_ps();
  c6067.v = _mm256_setzero_ps(); c7077.v = _mm256_setzero_ps();
  for(int iiiii=0; iiiii<k; iiiii++) {
    __builtin_prefetch(a+8); __builtin_prefetch(b+8);
    a_vreg.v = _mm256_load_ps( (float*)a );
    b_p0_vreg.v = _mm256_load_ps( (float*)b );
    a += 8; b += 8;
    c0007.v += a_vreg.v * b_p0_vreg.f[0]; c1017.v += a_vreg.v * b_p0_vreg.f[1];
    c2027.v += a_vreg.v * b_p0_vreg.f[2]; c3037.v += a_vreg.v * b_p0_vreg.f[3];
    c4047.v += a_vreg.v * b_p0_vreg.f[4]; c5057.v += a_vreg.v * b_p0_vreg.f[5];
    c6067.v += a_vreg.v * b_p0_vreg.f[6]; c7077.v += a_vreg.v * b_p0_vreg.f[7];
  }
  __m256 w0, w1, w2, w3, w4, w5, w6, w7;
  w0 = _mm256_load_ps((float*)&c[0*ldc]); w1 = _mm256_load_ps((float*)&c[1*ldc]);
  w2 = _mm256_load_ps((float*)&c[2*ldc]); w3 = _mm256_load_ps((float*)&c[3*ldc]);
  w4 = _mm256_load_ps((float*)&c[4*ldc]); w5 = _mm256_load_ps((float*)&c[5*ldc]);
  w6 = _mm256_load_ps((float*)&c[6*ldc]); w7 = _mm256_load_ps((float*)&c[7*ldc]);
  c0007.v = _mm256_add_ps(c0007.v, w0); c1017.v = _mm256_add_ps(c1017.v, w1);
  c2027.v = _mm256_add_ps(c2027.v, w2); c3037.v = _mm256_add_ps(c3037.v, w3);
  c4047.v = _mm256_add_ps(c4047.v, w4); c5057.v = _mm256_add_ps(c5057.v, w5);
  c6067.v = _mm256_add_ps(c6067.v, w6); c7077.v = _mm256_add_ps(c7077.v, w7);
  _mm256_store_ps( &c[0*ldc], c0007.v); _mm256_store_ps( &c[1*ldc], c1017.v);
  _mm256_store_ps( &c[2*ldc], c2027.v); _mm256_store_ps( &c[3*ldc], c3037.v);
  _mm256_store_ps( &c[4*ldc], c4047.v); _mm256_store_ps( &c[5*ldc], c5057.v);
  _mm256_store_ps( &c[6*ldc], c6067.v); _mm256_store_ps( &c[7*ldc], c7077.v);
}
*/

inline void pack_pad(int c, float* a, float* to) {
  for(int i=0; i<c; i++) {
    const float* t = &a[0];
    *(to+0) = *(a+0);
    *(to+1) = *(a+1);
    *(to+2) = *(a+2);
    *(to+3) = *(a+3);
    *(to+4) = *(a+4);
    *(to+5) = 0.f;
    *(to+6) = 0.f;
    *(to+7) = 0.f;
    to+=8; a+=8;
  }
}

// NOTE: handles 3 more values then needed
inline void _5x5_conv_ps(const float* a, const float* b, float* c, int ldc) {
  m256_t c0, c1, c2, c3, b0, b1, b2, b3, av;
  c0.v = _mm256_setzero_ps(); 
  c1.v = _mm256_setzero_ps();
  c2.v = _mm256_setzero_ps(); 
  c3.v = _mm256_setzero_ps();

  for(int i=0; i<5; i++) {
    std::cout << i << std::endl;
    av.v = _mm256_load_ps((float*)a);
    b0.v = _mm256_broadcast_ss((float*)b); 
    b1.v = _mm256_broadcast_ss((float*)b+1);
    b2.v = _mm256_broadcast_ss((float*)b+2); 
    b3.v = _mm256_broadcast_ss((float*)b+3);

    std::cout << "\n av : ";
    for(int i=0; i<8; i++) std::cout <<  av.f[i] << " ";
    std::cout << "\n b0 : ";
    for(int i=0; i<8; i++) std::cout <<  b0.f[i] << " ";
    c0.v += _mm256_mul_ps(av.v, b0.v); 
    c1.v += _mm256_mul_ps(av.v, b1.v);
    c2.v += _mm256_mul_ps(av.v, b2.v); 
    c3.v += _mm256_mul_ps(av.v, b3.v);

    std::cout << "\n c0 : ";
    for(int i=0; i<8; i++) std::cout <<  c0.f[i] << " ";
    std::cout << "\n c1 : ";
    for(int i=0; i<8; i++) std::cout <<  c1.f[i] << " ";
    a+=5; b+=5;
  } 
  _mm256_store_ps( &c[8],  c0.v); _mm256_store_ps( &c[16],  c1.v);
  _mm256_store_ps( &c[24], c2.v); _mm256_store_ps( &c[32], c3.v);
*/

/*
 [ ] work with 4x8 __m256
 [ ] Pack across dimensions to fill num_threads * L1/core ~8000 f32/core
 [ ] Branch into different threads and multiply from pack
*/

/*

inline size_t get_idx(uint32_t* strides, int sl, uint32_t* idxs, int il) {
  assert(sl == il); // for now
  uint64_t ret;
  for(size_t i=0; i<sl; i++) ret += strides[i]*idxs[i]; 
  return ret;
}

void bmm256( const float* a, const float* b, float* c, int lds, int bs, int bn) {
  for(int b=0; b<bn; b+=bs) {
    m256_t c04, c14, c24, c34, av, bv; 
  }

}

template<int kb=128, int ib=128, int th=1>
void bmm256(const float* a, const float* b, float* c, int lds, int bs, int bn) {
  for(int b=0; b<bn; b++) {
    int bstr=bs*b;
    a+=bstr; b+=bstr; c+=bstr;

    //#pragma omp parallel for shared(a, b, c, m, n, k) default(none) collapse(1) num_threads(th)
    for(int i=0; i<k; i+=kb) {
      int ib = std::min(k-i, kb);
      float* pb = new alignas(32) float[ib*n];
      for(int ii=0; ii<m; ii+=mb) {
        int iib = std::min(m-ii, mb);
        float* pa = new alignas(32) float[ib*iib];
        float* wa = &a[i*k+ii];
        float* wb = &b[i];
        for(int iii=0; iii<n; iii+=8) {
          if(ii==0) pack_b(ib, &wb[iii*n], n, &pb[iii*ib]);
          for(int iiii=0; iiii<iib; iii+=8) {
            if(iii==0) pack_a(ib, &wa[iiii], k ,&pa[iiii*ib]);
            _8x8_m256_gemm(iib, &pa[iiii*ib], &pb[iii*ib], &c[ii+iii*n+iiii], n);
          }
        }
      }
    }
  }
}
*/

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


#define I 100

int main() {
  //mnist_t ret;

  //mnist_t mnist = load_mnist();
  //float* pooling;
  //float* expand;

  /*
  long double sum1 = 0;
  for(int i=0; i<I; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    pool(mnist.x_train, &to, 512, 28, 5, 1, 1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end - start;
    sum1 += ms_double.count();
  }
  std::cout << "pool :" << sum1/I << " ms "<< std::endl;
  std::cout << 60000*24*24*5*5*sizeof(float) << " bytes" << std::endl;
  */

/*
  pool(mnist.x_train, &pooling, 512); // {60000, 1, 24, 24, 5, 5}

  expand = static_cast<float*>( aligned_alloc(ALIGNMENT, 512*32*24*24*5*5*sizeof(float)) ); //  {60000, 1, 32, 24, 24, 5, 5}
  for(size_t i=0; i<512; i++) { 
    for(size_t j=0; j<32; j++) {
      memcpy(&expand[i*24*24*5*5*j], &pooling[i*24*24*5*5], 24*24*5*5*sizeof(float)); 
    }
  }
  //free(pooling);

  auto x = Tensor(expand, 512*32*24*24*5*5, {512, 32, 24, 24, 1, 5, 5});
  //x.permute({0, 2, 3, 4, 1, 5, 6});
  float kbound = std::sqrt(3.f)*std::sqrt(2.f / (1.f + 5.f)) / std::sqrt(5*5);
  //auto weight = Tensor<float>({1, 32, 1, 1, 1, 5, 5}).randn(-kbound, kbound, UNIFORM);
  auto weight = Tensor<float>({32, 1, 5, 5}).randn(-kbound, kbound, UNIFORM);
  weight.reshape({1, 32, 1, 1, 1, 5, 5});
  // TODO expand fails

  // bmm


  std::cout << weight << std::endl;
  std::cout << x << std::endl;
  */

  float* a =  (float*)aligned_alloc(32, 512*5*5*sizeof(float));
  float* pa = (float*)aligned_alloc(32, 512*5*5*sizeof(float));
  for(int i=0; i<512*5*5; i++) a[i] = (float)i;
  pack_pad((512*5*5)/8, &a[0], &pa[0]);

  float* b = (float*)aligned_alloc(32, 512*5*5*sizeof(float));
  for(int i=0; i<5*5; i++) b[i] = 0.5f; 

  float* c = (float*)aligned_alloc(32, 512*5*5*sizeof(float));

  _5x5_conv_ps(&a[0], &b[0], &c[0], 0);

  std::cout << "\na : ";
  for(int i=0; i<5*5; i++) { if(i%5==0) std::cout << "\n"; std::cout << std::setw(3) << a[i] << ", "; }
  std::cout << "\n";

  for(int i=0; i<5*5; i++) { if(i%5==0) std::cout << "\n"; std::cout << std::setw(3) << b[i] << ", "; }
  std::cout << "\n";

  for(int i=0; i<5*5; i++) { if(i%5==0) std::cout << "\n"; std::cout << std::setw(3) << c[i] << ", "; }
  std::cout << "\n";

  free(a);
  free(b);
  free(c);

/*

  for(size_t i=0; i<28*28; i++) {
    if(i%28==0) std::cout << "\n";
    if(i%(28*28)==0) std::cout << "\n\n";
    std::cout << std::setw(3) << mnist.x_train[i] << ", ";
  } 

  std::cout << "\n\n";

  for(size_t i=0; i<24*24*5*5; i+=5) {
    if(i%(5*5)==0 && i!=0) { 
      i+=25*3;
      std::cout << "\n\n"; 
    }
    if(i%(24*24*5*5)==0) std::cout << "\nconvolutions:\n";
    for(size_t j=0; j<5; j++) std::cout << std::setw(4) << to[j+i] << ", ";
    std::cout <<  "  |  ";
    for(size_t j=0; j<5; j++) std::cout << std::setw(4) << to[j+25+i] << ", ";
    std::cout <<  "  |  ";
    for(size_t j=0; j<5; j++) std::cout << std::setw(4) << to[j+25*2+i] << ", ";
    std::cout <<  "  |  ";
    for(size_t j=0; j<5; j++) std::cout << std::setw(4) << to[j+25*3+i] << ", ";
    std::cout << "\n";
  }
  */
  
  /*
  for(size_t i=0; i<mnist.ltrain; i++) {
    if(i%28==0) std::cout << "\n";
    if(i%(28*28)==0) std::cout << "\n\n";
    std::cout << std::setw(3) << mnist.x_train[i] << ", ";
  } 
  auto x_train_t = Tensor<float>(mnist.x_train, mnist.ltrain*mnist.sl*mnist.sl, {mnist.ltrain, 1, 28, 28});
  x_train_t.repreat({1, 1, 6, 6});
  std::cout << x_train_t << std::endl;
  */
}
