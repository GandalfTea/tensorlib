#include <zlib.h>
#include <stdio.h>
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <immintrin.h>

#ifdef linux
  #include <linux/perf_event.h>
  #include <linux/hw_breakpoint.h>
  #include <sys/syscall.h>
  #include <sys/ioctl.h>
  #include <pthread.h>
  #include <stdlib.h>
#endif

#define DEBUG 1

#define CHUNK 8*(2<<19) // 8 MiB - zlib
#define BATCH 512
#define CLASSES 10
#define EPSILON 1e-5 
#define ALIGNMENT 32
#define LR 0.1 

#define EQL_F32(a, b) fabs(a-b) <= ((fabs(a)>fabs(b) ? fabs(b) : fabs(a))*EPSILON)
#define LT_F32(a, b) (b-a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPSILON)
#define SIZE_AFTER_CONV(is, ks, d, s) (is-ks)/s+1

typedef uint8_t u8;

#ifdef linux
  static long perf_event_open(struct perf_event_attr* hw_event, pid_t pid, 
                              int cpu, int group_fd, unsigned long flags) 
  {
    return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  }
#endif

// Load mnist and inflate
typedef struct {
  u8* x_train;
  u8* x_test;
  u8* y_train;
  u8* y_test;
  size_t ltrain, ltest; 
  size_t sl = 28; // rows and cols length
} mnist_t;

// https://zlib.net/manual.html
int gz_inflate(u8** to, size_t* usize, FILE* src) {

  unsigned have;
  int ret, flush;
  unsigned ul = *usize; // track uncompressed length
  unsigned rl = *usize; // track real length 
  u8* in = (u8*)malloc(sizeof(u8)*CHUNK);

  z_stream strm;
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
        u8* to2 = (u8*)realloc(*to, sizeof(u8)*(ul+CHUNK));
        if(to2 == NULL) {
          free(*to);
          return -1;
        }
        *to = to2;
        ul+=CHUNK;
      }
      strm.next_out = (u8*)(*to+strm.total_out);
      strm.avail_out = ul - strm.total_out;
      ret = inflate(&strm, flush);
      assert(ret != Z_STREAM_ERROR);
      have = CHUNK - strm.avail_out;
    } while(strm.avail_out == 0);
    assert(strm.avail_in==0);
  } while(flush != Z_FINISH);
  assert(ret == Z_STREAM_END);
  ret = inflateEnd(&strm);
  rl = strm.total_out;
  u8* to3 = (u8*)realloc(*to, sizeof(u8)*rl);
  if(to3 != NULL) {
    *usize = rl;
    *to = to3;
  } else *usize = ul;    
  free(in);
  return ret;
}


// this works on bytes 
// conversion to float happens in randn_batch()
#define INFLATE(src, pmu_fd, num, ndim, disp, des) \
  FILE* file = fopen(src, "rb"); \
  fseek(file, 0, SEEK_END); \
  size_t size = ftell(file); rewind(file); \
  u8* tmp_ptr = (u8*)malloc(size); \
  ioctl(pmu_fd, PERF_EVENT_IOC_ENABLE, 0); \
  if(gz_inflate(&tmp_ptr, &size, file) == -1) exit(EXIT_FAILURE); \
  ioctl(pmu_fd, PERF_EVENT_IOC_DISABLE, 0); \
  assert(tmp_ptr[0] == 0x00); \
  assert(tmp_ptr[1] == 0x00); \
  assert(tmp_ptr[2] == 0x08); \
  assert(tmp_ptr[3] == ndim); \
  uint32_t n = (tmp_ptr[4]  << 24) + (tmp_ptr[5]  << 16) + (tmp_ptr[6]  << 8) + tmp_ptr[7]; \
  assert(n == num); \
  if(!read(pmu_fd, &c, sizeof(c))) exit(EXIT_FAILURE); \
  if(tmp_ptr) des = &tmp_ptr[disp]; \
  else { printf("\nFailed to inflate file %s\n", src); free(tmp_ptr); exit(EXIT_FAILURE); } \
  printf("\n [ %12lld ] inflating %-26s %6.2f MB", c, "train-images-idx3-ubyte.gz", size*1e-6); \
  lc=c; 

static mnist_t load_mnist() {
  int pmu_fd;
  long long lc=0, c;
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe = (perf_event_attr){.type=PERF_TYPE_HARDWARE, .size=sizeof(pe), 
       .config=PERF_COUNT_HW_CPU_CYCLES, .disabled=1, .exclude_kernel=1, .exclude_hv=1};
  pmu_fd = perf_event_open(&pe, 0, -1, -1, 0);
  if(pmu_fd == -1) {
    fprintf(stderr, "Error opening PMU event leader %llx\n", pe.config);
    exit(EXIT_FAILURE);
  }
  ioctl(pmu_fd, PERF_EVENT_IOC_RESET, 0);

  mnist_t ret;
  { INFLATE("./datasets/train-images-idx3-ubyte.gz", pmu_fd, 60000, 0x03, 15, ret.x_train); }
  { INFLATE("./datasets/train-labels-idx1-ubyte.gz", pmu_fd, 60000, 0x01, 7,  ret.y_train); }
  { INFLATE("./datasets/t10k-images-idx3-ubyte.gz",  pmu_fd, 10000, 0x03, 15, ret.x_test); }
  { INFLATE("./datasets/t10k-labels-idx1-ubyte.gz",  pmu_fd, 10000, 0x01, 7,  ret.y_test); }

  printf("\n\n");
  close(pmu_fd);

  ret.ltrain = 60000;
  ret.ltest  = 10000; 
  ret.sl = 28;
  return ret;
}

#undef INFLATE


// conv2d kernel
// ---------------------------------------------

inline void _mm256_conv2d_ps(const float* a, const float* b, float* c, int lda, int ks) {
  int i=0, j=0;
  __m256 c0 = _mm256_setzero_ps();
  for(; i<ks; i++) {
    #pragma unroll
    for(j=0; j<ks; j++) c0 = _mm256_fmadd_ps(_mm256_loadu_ps((float*)(a+j)), _mm256_broadcast_ss((float*)b+j), c0); 
    a+=lda;
    b+=ks;
  }
  _mm256_storeu_ps(c,  c0);
}

// run kernel on a batch of images 
// TODO: only run 8 kernel when possible
static void batch_conv2d(const float* in, const float* ker, float* out, int ks, 
                         int lda=28, int lo=24, int batch=512, int channels=32) 
{
  int b, c, x, y, i, j;
  const int sco = lo*lo;            // stride channel out
  const int sci = lda*lda;          // strinde channel in
  const int sio = lo*lo*channels;   // stride image out
  const int it = ceil((float)lo/8); 
  if(true) {
    for(b=0; b<batch; b++) {
      for(c=0; c<channels; c++) {
        for(i=0; i<lo; i++) {
          for(j=0; j<it*8; j+=8)
            _mm256_conv2d_ps(&in[b*sci+i*lda+j], &ker[c*ks*ks], &out[b*sio+c*sco+i*lo+j], lda, ks);
        }
      }
    }
  } else {
    for(b=0; b<batch; b++) {
      for(c=0; c<channels; c++) {
        for(x=0; x<lo; x++) {
          for(y=0; y<lo; y++) { 
            for(i=0; i<ks; i++) {
              for(j=0; j<ks; j++) 
                out[b*sio+c*sco+x*lo+y] += in[b*sci+x*lda+y+i*ks+j] * ker[c*ks*ks+i*ks+j];
            }
          }
        }
      }
    }
  }
}

// randn ----------------------------------------------

static void f32_uniform(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0.f, float e=0.f) {
  std::mt19937 rng(std::random_device{}()); 
  if(!EQL_F32(seed, 0.f)) rng.seed(seed); 
  std::uniform_real_distribution<float> dist(down, up); 
  if(!EQL_F32(e, 0.f)) for(size_t i=0; i<count; i++) do to[i] = dist(rng); while (to[i] <= e); 
  else for(size_t i=0; i<count; i++) to[i] = dist(rng); 
}

// TODO: check for duplicates
static void randn_batch(const u8* x_src, const u8* y_src, 
                        float* x_out, u8* y_out, uint32_t len, uint32_t count) 
{
  int i;
  const int is = 28*28;
  std::mt19937 rng(std::random_device{}()); 
  std::uniform_real_distribution<float> dist(0, len); // [0, len) 
  while(count-- > 0) {
    uint32_t rn = (uint32_t)dist(rng);
    for(i=0; i<is; i++) x_out[i] = (float)x_src[rn*is+i];
    *y_out = y_src[rn]; y_out++;
    x_out+=is;
  }
}

// activation ----------------------------------------

inline void relu(float* in, uint32_t count) {
  while(count-- > 0) if(LT_F32(in[count], 0.f)) in[count] = 0.f;
}

static void backprop_relu(float* x, float* dy, uint32_t count) {
  while(count-- > 0) if(LT_F32(x[count], 0.f)) dy[count] = 0.f; 
}


// Conv2D layer --------------------------------------

typedef struct {
  float* weights;
  float* grad;
  uint32_t in_channels;
  uint32_t out_channels;
} conv2d;

inline conv2d init_conv2d_layer(uint32_t ks, uint32_t in_channels, uint32_t out_channels, 
                                uint32_t count, float up, float down) 
{
  conv2d l = (conv2d){ .in_channels=in_channels, .out_channels=out_channels };
  l.weights = (float*)aligned_alloc(ALIGNMENT, out_channels*ks*ks*count*sizeof(float));
  l.grad = (float*)calloc(out_channels*ks*ks*count, sizeof(float));
  f32_uniform(l.weights, out_channels*ks*ks, up, down);
  return l;
}


// Batch Normalisation layer ------------------------- 
// https://arxiv.org/pdf/1502.03167

typedef struct {
  uint32_t size;
  uint32_t num_batches_tracked=0;
  bool affine;
  float* weights;
  float* bias;
  float* means;
  float* invstd;
  float* var;
  float eps;
  float momentum;
} bnorm2d_state;

bnorm2d_state init_batchnorm2d(uint32_t size, bool affine=true) {
  bnorm2d_state r = (bnorm2d_state){ .size=size, .affine=affine, .eps=EPSILON, .momentum=0.1f, };
  if(affine) {
    r.weights = (float*)aligned_alloc(32, size*sizeof(float));
    r.bias    = (float*)aligned_alloc(32, size*sizeof(float));
    r.means  = (float*)calloc(size, sizeof(float));
    r.invstd = (float*)calloc(size, sizeof(float));
    f32_uniform(r.weights, size);
    f32_uniform(r.bias, size);
  }
  return r;
}

// TODO: update running mean and std
/* NOTE: normalisation is done over the channel dimension because each channel represents
   one convolution kernel. This aids in feature extraction. */

void inline run_batchnorm2d(bnorm2d_state* s, float* in, uint32_t ch, uint32_t img_s, 
                            uint32_t batch=512, bool training=false) 
{
  memset(s->means, 0, ch*sizeof(float));
  memset(s->invstd, 0, ch*sizeof(float));
  float* vars = (float*)calloc(ch, sizeof(float));
  const uint32_t s_ch  = img_s*img_s;
  const uint32_t s_img = ch*s_ch;

  for(int b=0; b<batch; b++) {    // batch mean 
    int c;
    for(c=0; c<ch; c++) {
      uint32_t widx = b*s_img+c*s_ch;
      for(int i=0; i<s_ch; i++) s->means[c] += in[widx+i];
    }
  }

  for(int i=0; i<ch; i++) s->means[i] /= s_ch*batch;

  // batch variance and inverse standard deviation
  for(int b=0; b<batch; b++) {
    int c;
    for(c=0; c<ch; c++) {
      const uint32_t widx = b*s_img+c*s_ch;
      for(int i=0; i<s_ch; i++) vars[c] += std::pow(in[widx+i]-s->means[c], 2);
    }
  }

  for(int i=0; i<ch; i++) vars[i] /= s_ch*batch; 
  for(int i=0; i<ch; i++) s->invstd[i] = 1 / std::sqrt(vars[i]+s->eps);

  // normalise, scale and shift
  // pytorch first multiplies then adds bias-mean*invstd*weights;
  // https://github.com/pytorch/pytorch/blob/420b37f3c67950ed93cd8aa7a12e673fcfc5567b/aten/src/ATen/native/Normalization.cpp#L96 
  for(int i=0; i<batch; i++) {
    for(int c=0; c<ch; c++) {
      float* wptr = &in[i*s_img+c*s_ch];
      for(int p=0; p<s_ch; p++) 
        wptr[p] = (wptr[p]+s->bias[c]-s->means[c])*(s->invstd[c]*s->weights[c]);
    }
  }
  free(vars);
}


// Max 2D Pooling ------------------------------------

static void maxpool2d(uint32_t* idxs, float* in, float* out, uint32_t img_s, uint32_t oimg_s,
                       uint32_t ch, uint32_t batch=512, uint32_t ks=2, uint32_t s=2, uint32_t dilation=1) 
{
  float max;
  int b, c, x, y, i, j, ridx, iidx=0;
  const int s_img = ch*img_s*img_s;
  const int s_ch  = img_s*img_s;
  const int s_img_out = ch*oimg_s*oimg_s;
  const int s_ch_out  = oimg_s*oimg_s;
  for(b=0; b<batch; b++) {
    for(c=0; c<ch; c++) {
      for(x=0; x<img_s/s; x++) {
        for(y=0; y<img_s/s; y++) {
          const int t0 = b*s_img+c*s_ch+x*img_s*s+y*s;
          max=in[t0];
          ridx=t0;
          for(i=0; i<ks; i++) {
            for(j=0; j<ks; j++) {
              if(LT_F32(max, in[t0+i*img_s+j])) {
                max=in[t0+i*img_s+j];
                ridx=t0+i*img_s+j;
              }
            }
          }
          out[b*s_img_out+c*s_ch_out+x*oimg_s+y] = max;
          idxs[iidx++] = ridx;
        }
      }
    }
  }
}

static void backprop_maxpool(uint32_t* idxs, const float* y, float* dx, uint32_t count) {
  int i, j;
  for(i=0; i<count; i++) dx[idxs[i]] = y[i]; 
}


// Linear layer --------------------------------------

typedef struct {
  float* weights;
  float* grad;
  float* bias;
  uint32_t out_features;
} linear_state;

static inline linear_state init_linear(uint32_t in_features, uint32_t out_features, bool bias=true) {
  linear_state r = (linear_state){ .grad=nullptr, .bias=nullptr, .out_features=out_features };  
  r.grad = (float*)calloc(in_features*out_features, sizeof(float));
  r.weights = (float*)aligned_alloc(ALIGNMENT, in_features*out_features*sizeof(float));
  float bound = std::sqrt(in_features);
  f32_uniform(r.weights, in_features*out_features, -1/bound, 1/bound);
  if(bias) {
    r.bias = (float*)malloc(out_features*sizeof(float));
    f32_uniform(r.bias, out_features, -1/bound, 1/bound);
  }
  return r;
}

static void sgemm(const float* a, const float* b, float* c, const int m=512, 
                         const int n=10, const int k=576) 
{
  int mi, ki, ni;
  for(mi=0; mi<m; mi++)
    for(ki=0; ki<k; ki++) 
      for(ni=0; ni<n; ni++) c[mi*n+ni] += a[mi*k+ki] * b[ki*n+ni];
}

static void transpose(const float* src, float* to, const uint32_t m, const uint32_t n) {
  int mi, ni;
  for(mi=0; mi<m; mi++) 
    for(ni=0; ni<n; ni++) to[ni*m+mi] = src[mi*n+ni];
}


// Sparse Categorical Cross-Entropy Loss --------------

static float max(float* in, uint32_t count) {
  float b = in[0];
  while(count-- > 0) if(LT_F32(b, in[count])) b = in[count]; 
  return b;
}

// logsoftmax vs softmax : https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
static void log_softmax(float* in, uint32_t count) {
  float sum = 0.f;
  const float maxf = max(in, count); 
  float* m = (float*)malloc(count*sizeof(float));

  for(int i=0; i<count; i++) { 
    const float bf = in[i] - maxf; // normalising induces numerical stability 
    m[i] = bf;
    sum += exp(bf);
  }
  const float slog = log2(sum);
  for(int i=0; i<count; i++) in[i] = m[i] - slog; 
  free(m);
} 

// skip one-hot by plucking out indexes directly
static float sparse_categorical_crossentropy(float* in, uint32_t count, uint8_t* y, 
                uint32_t batch=512, uint32_t num_classes=10, float smoothing=0.f) 
{
  float sum=0.f;
  int i=0, j=0;
  for(; j<batch*num_classes; j+=num_classes) { 
    log_softmax(&in[j], num_classes);
    sum += in[j+(int)y[i++]];
  }
  return -sum/batch + smoothing;
}

// Adam optimiser ------------------------------------

void init_adam() {}


// Model ---------------------------------------------

typedef struct {
  conv2d l1;
  conv2d l2;
  bnorm2d_state l3;
  conv2d l4;
  conv2d l5;
  bnorm2d_state l6;
  linear_state l7;
} model;

// reuse the same memory allocations for each epoch
typedef struct {
  float* outs0; // intermediary activations
  float* outs1; 
  float* outs2; 
  float* outs3; 
  float* outs4; 
  float* outs5; 
  float* outs6; 
  uint32_t ls0; // img size compression per channel
  uint32_t ls1;
  uint32_t ls2;
  uint32_t ls3;
  uint32_t ls4;
  uint32_t ls5;
  uint32_t* idxs0; // store indexes for maxpool backprop
  uint32_t* idxs1;
} activations;

static void init_activations(activations* a) {
  a->ls0 = SIZE_AFTER_CONV(28,  5, 1, 1);
  a->ls1 = SIZE_AFTER_CONV(a->ls0, 5, 1, 1);
  a->ls2 = SIZE_AFTER_CONV(a->ls1, 2, 1, 2);
  a->ls3 = SIZE_AFTER_CONV(a->ls2, 3, 1, 1);
  a->ls4 = SIZE_AFTER_CONV(a->ls3, 3, 1, 1);
  a->ls5 = SIZE_AFTER_CONV(a->ls4, 2, 1, 2);
  a->outs0 = (float*)aligned_alloc(ALIGNMENT, BATCH*32*a->ls0*a->ls0*sizeof(float));
  a->outs1 = (float*)aligned_alloc(ALIGNMENT, BATCH*32*a->ls1*a->ls1*sizeof(float));
  a->outs2 = (float*)aligned_alloc(ALIGNMENT, BATCH*32*a->ls2*a->ls2*sizeof(float));
  a->outs3 = (float*)aligned_alloc(ALIGNMENT, BATCH*64*a->ls3*a->ls3*sizeof(float));
  a->outs4 = (float*)aligned_alloc(ALIGNMENT, BATCH*64*a->ls4*a->ls4*sizeof(float));
  a->outs5 = (float*)aligned_alloc(ALIGNMENT, BATCH*64*a->ls5*a->ls5*sizeof(float));
  a->outs6 = (float*)aligned_alloc(ALIGNMENT, BATCH*10*sizeof(float));
  a->idxs0 = (uint32_t*)malloc(BATCH*32*a->ls2*a->ls2*sizeof(uint32_t));
  a->idxs1 = (uint32_t*)malloc(BATCH*64*a->ls5*a->ls5*sizeof(uint32_t));
}

inline void zero_activations(activations* a) {
  memset(a->outs0, 0, 512*32*a->ls0*a->ls0*sizeof(float));
  memset(a->outs1, 0, 512*32*a->ls1*a->ls1*sizeof(float));
  memset(a->outs2, 0, 512*32*a->ls2*a->ls2*sizeof(float));
  memset(a->outs3, 0, 512*64*a->ls3*a->ls3*sizeof(float));
  memset(a->outs4, 0, 512*64*a->ls4*a->ls4*sizeof(float));
  memset(a->outs5, 0, 512*64*a->ls5*a->ls5*sizeof(float));
  memset(a->outs6, 0, 512*10*sizeof(float));
}

inline void zero_grad(model* m, activations* a) {
  memset(m->l7.grad, 0, 576*10*sizeof(float));
  //memset(m->l6.grad, 0, 512*64*a->ls4*a->ls4*sizeof(float));
}

// kaiming uniform -- https://arxiv.org/pdf/1502.01852.pdf
static model init_model(void) {
  model r;
  const float lhs = std::sqrt(3.f)*std::sqrt(2.f/(1.f+5.f)); // tinygrad does math.sqrt(5)**2 as default a
  const float kb325 = lhs / std::sqrt(32*5*5);
  const float kb323 = lhs / std::sqrt(32*3*3);
  const float kb643 = lhs / std::sqrt(64*3*3);
  r.l1 = init_conv2d_layer(5, 1,  32, 512, kb325, -kb325);
  r.l2 = init_conv2d_layer(5, 32, 32, 512, kb325, -kb325);
  r.l3 = init_batchnorm2d(32);
  r.l4 = init_conv2d_layer(3, 32, 64, 512, kb323, -kb323);
  r.l5 = init_conv2d_layer(3, 64, 64, 512, kb643, -kb643);
  r.l6 = init_batchnorm2d(64);
  r.l7 = init_linear(576, 10);
  return r;
}


// Backprop ------------------------------------------

static void backprop_sccl(const float* loss, float* out, float* probs, 
                          const uint8_t* y, const uint32_t classes=10) 
{
  int i=0, j=0, k=0, yidx;
  for(i=0; i<BATCH; i++) {
    yidx = y[k++]; 
    for(j=0; j<classes; j++) out[i*10+j] = -probs[i+j]*probs[yidx];
    out[i*10+yidx] = probs[yidx]*(1-probs[yidx]);
  }
}

inline void update_params(const float lr, float* params, const float* grad, uint32_t count) {
  while(count-- > 0) params[count] -= lr*grad[count];
}

// TODO: transpose
static void backprop_loss_l7(const float* loss, linear_state* l, float* probs, 
                             const uint8_t* labels, float* xgrad) 
{
  float* wt = (float*)calloc(10*576, sizeof(float));
  float* xt = (float*)calloc(576*512, sizeof(float));
  float* dY = (float*)calloc(BATCH*10, sizeof(float));
  transpose(probs, xt, 512, 576);    // 512x576 -> 576x512
  transpose(l->weights, wt, 576, 10); // 576x10 -> 10x576
  backprop_sccl(loss, dY, probs, labels, 10);     // dY - 512x10 (loss + softmax)
  sgemm(xt, dY, l->grad, 576, 10, 512); // Xt @ dY = dW - 576x512 @ 512x10 = 576x10 
  sgemm(dY, wt, xgrad, 512, 576, 10);   // dY @ Wt = dX - 512x10  @ 10x576 = 512x576
  update_params(LR, l->weights, l->grad, 576*10);
  free(dY); 
  free(wt);
  free(xt);
}

static void backprop_batchnorm(bnorm2d_state s, const float* dwin, const float* xi, float* dx, 
                               const uint32_t img_s, const uint32_t ch, const float lr) 
{
  float* dx_hat = (float*)calloc(BATCH*ch*img_s*img_s, sizeof(float));
  float* sum_dx_mul_dx = (float*)calloc(BATCH, sizeof(float)); 
  float* sum_dx_hat = (float*)calloc(BATCH, sizeof(float)); 
  float* db = (float*)calloc(ch, sizeof(float));
  float* dw = (float*)calloc(ch, sizeof(float));
  const uint32_t s_ch  = img_s*img_s; 
  const uint32_t s_img = ch*s_ch;
  int i, j, c, b, wi;

  // ∇x_hat = dy * gamma - 512x64x3x3 * 64 = 512x64x3x3 
  for(b=0; b<BATCH; b++) { 
    for(c=0; c<ch; c++) {
      wi = b*s_img+c*s_ch;
      for(i=0; i<img_s; i++) { 
        for(j=0; j<img_s; j++) {
          dx_hat[wi+i*img_s+j] += dwin[wi+i*img_s+j] * s.weights[c];
          sum_dx_hat[b]    += dx_hat[wi+i*img_s+j];
          sum_dx_mul_dx[b] += dx_hat[wi+i*img_s+j]*dwin[wi+i*img_s+j];
        }
      }
    }
  }
  // ∇y, ∇B, ∇x
  for(b=0; b<BATCH; b++) {
    for(c=0; c<ch; c++) { 
      wi = b*s_img+c*s_ch;
      for(i=0; i<img_s; i++) {
        for(j=0; j<img_s; j++) {
          db[c] += dwin[wi+i*img_s+j]; 
          dw[c] += dwin[wi+i*img_s+j] * xi[wi+i*img_s+j]; 
          dx[wi+i*img_s+j] = 1.f/BATCH * 1.f/pow(s.invstd[c], 2) * (BATCH*dx_hat[wi+i*img_s+j] 
                             - sum_dx_hat[b] - dwin[wi+i*img_s+j]*sum_dx_mul_dx[b]); 
        }
      }
    }
  }
  // update gamma and beta
  for(i=0; i<ch; i++) {
    s.weights[i] += lr*dw[i];
    s.bias[i]    += lr*db[i];
  }

  free(sum_dx_mul_dx);
  free(sum_dx_hat);
  free(dx_hat);
  free(db);
  free(dw);
}

static void backprop(const float* loss, model* m, activations* a, const uint8_t* labels, const uint32_t count) {

  float* l7dw = (float*)calloc(BATCH*576, sizeof(float));
  float* l6dw = (float*)calloc(BATCH*64*a->ls4*a->ls4, sizeof(float));
  float* l5dw = (float*)calloc(BATCH*64*a->ls4*a->ls4, sizeof(float));

  backprop_loss_l7(loss, &m->l7, a->outs6, labels, l7dw);
  backprop_maxpool(a->idxs1, l7dw, l6dw, BATCH*64*a->ls5*a->ls5);
  backprop_batchnorm(m->l6, l6dw, a->outs4, l5dw, a->ls4, 64, LR);
  //backprop_relu(a->outs4, l5dw, 512*64*a->ls4*a->ls4);

/*
  for(int i=0; i<80; i++) {
    for(int j=0; j<15; j++) printf("%10.2f, ", l7dw[i*15+j]);
    printf("\n");
  }
*/

  free(l7dw);
}


// perf_events_open PMU helpers ----------------------

struct pmu_vars {
  int fd_cycles;
  int fd_inst;
  int fd_clock;
  long long c_cycles;
  long long c_inst;
  long long c_clock;
};

static void init_pmu_trackers(pmu_vars* pmu) {
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));  

  pe = (perf_event_attr){.type=PERF_TYPE_HARDWARE, .size=sizeof(pe), 
       .config=PERF_COUNT_HW_CPU_CYCLES, .disabled=1, .exclude_kernel=1, .exclude_hv=1};
  pmu->fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);

  pe = (perf_event_attr){.type=PERF_TYPE_HARDWARE, .size=sizeof(pe),
       .config=PERF_COUNT_HW_INSTRUCTIONS, .exclude_kernel=1, .exclude_hv=1};
  pmu->fd_inst = perf_event_open(&pe, 0, -1, pmu->fd_cycles, 0);

  pe = (perf_event_attr){.type=PERF_TYPE_SOFTWARE, .size=sizeof(pe),
       .config=PERF_COUNT_SW_TASK_CLOCK, .exclude_kernel=1, .exclude_hv=1};
  pmu->fd_clock = perf_event_open(&pe, 0, -1, pmu->fd_cycles, 0);

  if(pmu->fd_cycles == -1 || pmu->fd_inst == -1 || pmu->fd_clock == -1){
    fprintf(stderr, "Error initializing PMU trackers %llx\n", pe.config);
    exit(EXIT_FAILURE);
  }
}


#define GET_PMU_VALUE(fd, c) if(read(fd, &c, sizeof(c))==-1) exit(EXIT_FAILURE);

#define END_PMU_TRACKING() \
  ioctl(pmu.fd_cycles, PERF_EVENT_IOC_DISABLE, 0); \
  ioctl(pmu.fd_clock,  PERF_EVENT_IOC_DISABLE, 0); \
  ioctl(pmu.fd_inst,   PERF_EVENT_IOC_DISABLE, 0);

#define BEGIN_PMU_TRACKING() \
  ioctl(pmu.fd_cycles, PERF_EVENT_IOC_RESET, 0); \
  ioctl(pmu.fd_inst,   PERF_EVENT_IOC_RESET, 0); \
  ioctl(pmu.fd_clock,  PERF_EVENT_IOC_RESET, 0); \
  ioctl(pmu.fd_cycles, PERF_EVENT_IOC_ENABLE, 0); \
  ioctl(pmu.fd_inst,   PERF_EVENT_IOC_ENABLE, 0); \
  ioctl(pmu.fd_clock,  PERF_EVENT_IOC_ENABLE, 0);

#define PRINT_PROFILE(name, fan_in, fan_out) \
  GET_PMU_VALUE(pmu.fd_cycles, pmu.c_cycles) \
  GET_PMU_VALUE(pmu.fd_inst, pmu.c_inst) \
  GET_PMU_VALUE(pmu.fd_clock, pmu.c_clock) \
  printf("\n[%8d : %-8d]  %-12s   %9lld cycles . %4.2f IPS .  %5.2f msec", \
          fan_in, fan_out, name, pmu.c_cycles, \
          (float)pmu.c_inst/c_cycles, (float)pmu.c_clock*1e-6);

#define RUN_PROFILE(run, loss, tc) \
  GET_PMU_VALUE(pmu.fd_cycles, pmu.c_cycles) \
  GET_PMU_VALUE(pmu.fd_inst, pmu.c_inst) \
  GET_PMU_VALUE(pmu.fd_clock, pmu.c_clock) \
  printf("\r [ %12lld ]  epoch %-3d    %8.3f loss . %9lld cycles . %4.2f IPS . %5.2f msec", \
          tc, run, loss, pmu.c_cycles, (float)pmu.c_inst/pmu.c_cycles, (float)pmu.c_clock*1e-6); \
  if(run % 10 == 0) printf("\n"); \
  fflush(stdout);


static float forward(model* m, activations* a, float* input, u8* labels) {
  batch_conv2d(input, m->l1.weights, a->outs0, 5, 28, a->ls0, 512, 32); 
  relu(a->outs0, 512*32*a->ls0*a->ls0);
  batch_conv2d(a->outs0, m->l2.weights, a->outs1, 5, a->ls0, a->ls1, 512, 32);  
  relu(a->outs1, 512*32*a->ls1*a->ls1);
  run_batchnorm2d(&m->l3, a->outs1, 32, a->ls1, 512, true);
  maxpool2d(a->idxs0, a->outs1, a->outs2, a->ls1, a->ls2, 32);

  batch_conv2d(a->outs2, m->l4.weights, a->outs3, 3, a->ls2, a->ls3, 512, 64);  
  relu(a->outs3, 512*64*a->ls3*a->ls3);
  batch_conv2d(a->outs3, m->l5.weights, a->outs4, 3, a->ls3, a->ls4, 512, 64);  
  relu(a->outs4, 512*64*a->ls4*a->ls4);
  run_batchnorm2d(&m->l6, a->outs4, 64, a->ls4, 512, true);
  maxpool2d(a->idxs1, a->outs4, a->outs5, a->ls4, a->ls5, 64);

  sgemm(a->outs5, m->l7.weights, a->outs6, 512, 10, 576);  // 512x576 @ 576x10 -> 512x10  

/*
  for(int i=0; i<80; i++) {
    for(int j=0; j<15; j++) printf("%10.5f, ", a->outs6[i*15+j]);
    printf("\n");
  }
  */

  return sparse_categorical_crossentropy(a->outs0, 512, labels);
}


int main(void) {

  mnist_t mnist = load_mnist();

  model m = init_model();
  activations a;
  init_activations(&a);

#if DEBUG > 1

/*
  BEGIN_PMU_TRACKING();
  batch_conv2d(x_batch, m.l1.weights, outs0, 5, 28,  ls0, 512, 32);  relu(outs0, 512*32*ls0*ls0);
  END_PMU_TRACKING();
  PRINT_PROFILE("batchconv 1", 512*28*28, 512*32*ls0*ls0);

  BEGIN_PMU_TRACKING();
  batch_conv2d(outs0, m.l2.weights, outs1, ls0, ls1, 5, 512, 32);  relu(outs1, 512*32*ls1*ls1);
  END_PMU_TRACKING();
  PRINT_PROFILE("batchconv 2", 512*32*ls0*ls0, 512*32*ls1*ls1);

  BEGIN_PMU_TRACKING();
  run_batchnorm2d(&m.l3, outs1, 32, ls1, 512, true);
  END_PMU_TRACKING();
  PRINT_PROFILE("batchnorm 1", 512*32*ls1*ls1, 512*32*ls1*ls1);

  BEGIN_PMU_TRACKING();
  max_pool2d(outs1, outs2, ls1, ls2, 32);
  END_PMU_TRACKING();
  PRINT_PROFILE("maxpool2d 1", 512*32*ls1*ls1, 512*32*ls2*ls2);

  BEGIN_PMU_TRACKING();
  batch_conv2d(outs2, m.l4.weights, outs3, 3, ls2, ls3, 512, 64);  relu(outs3, 512*64*ls3*ls3);
  END_PMU_TRACKING();
  PRINT_PROFILE("batchconv 3", 512*32*ls2*ls2, 512*64*ls3*ls3);

  BEGIN_PMU_TRACKING();
  batch_conv2d(outs3, m.l5.weights, outs4, 3, ls3, ls4, 512, 64);  relu(outs4, 512*64*ls4*ls4);
  END_PMU_TRACKING();
  PRINT_PROFILE("batchconv 4", 512*64*ls3*ls3, 512*64*ls4*ls4);

  BEGIN_PMU_TRACKING();
  run_batchnorm2d(&m.l6, outs4, 64, ls4, 512, true);
  END_PMU_TRACKING();
  PRINT_PROFILE("batchnorm 2", 512*64*ls4*ls4, 512*64*ls4*ls4);

  BEGIN_PMU_TRACKING();
  max_pool2d(outs4, outs5, ls4, ls5, 64);
  END_PMU_TRACKING();
  PRINT_PROFILE("maxpool2d 2", 512*64*ls4*ls4, 512*64*ls5*ls5);

  BEGIN_PMU_TRACKING();
  sgemm(outs5, m.l7.weights, outs6, 512, 10, 576);
  END_PMU_TRACKING();
  PRINT_PROFILE("linear 1", 512*64*ls5*ls5, 512*10);

  BEGIN_PMU_TRACKING();
  float loss = sparse_categorical_crossentropy(outs6, 512, y_batch);
  END_PMU_TRACKING();
  PRINT_PROFILE("loss", 512*10, 512);
  */

#elif DEBUG > 0

  static long long tc=0;
  static int epoch=0;

  struct pmu_vars pmu;
  init_pmu_trackers(&pmu);

  float loss=0;
  float* x_batch = (float*)aligned_alloc(ALIGNMENT, 512*28*28*sizeof(float)); 
  u8* y_batch = (u8*)aligned_alloc(ALIGNMENT, 512*sizeof(u8));

  for(int i=0; i<2; i++) {
    BEGIN_PMU_TRACKING();

    zero_activations(&a);
    zero_grad(&m, &a);
    randn_batch(mnist.x_train, mnist.y_train, x_batch, y_batch, mnist.ltrain, BATCH);
    loss = forward(&m, &a, x_batch, y_batch);
    backprop(&loss, &m, &a, y_batch, 512*10);

    END_PMU_TRACKING();
    RUN_PROFILE(epoch, loss, tc);
    tc += pmu.c_cycles;
    ++epoch;
  }

  free(x_batch);
  free(y_batch);

#else 

#endif

  printf("\n\n");
}
