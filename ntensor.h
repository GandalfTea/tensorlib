
#pragma once

#include <limits.h> // only used here
#define TENSOR_MAX_DIM (2 << 15)
#define TENSOR_MAX_STORAGE UINT_MAX 

#define DEBUG 3
#define EPSILON 0.0001
#define MEMORY_ALIGNMENT 32 // avx 32 bytes 

#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <stdlib.h>
#include <cstdlib>
#include <stdexcept>
#include <initializer_list>
#include <type_traits> // arange std::is_floating_point

using std::size_t;

// must compile with appropriate flags for this to work
#if defined(__AVX__) && (defined(__FMA__) || defined(__FMA4__))
  #define AVX_CPU_GEMMS
  #include "cpu_gemms.h"
#endif

// test for general memory alignment 
#define is_aligned(ptr, bytes) (((uintptr_t)(const void*)(ptr)) % bytes == 0)

// kernels
namespace intrin {
  static inline void _t_f(const float* from, float* to, int lda, int ldb);
  static inline void _t_i(const int* from, int* to, int lda, int ldb);
  void _f_gemm(float* a, float* b, float* c, int m, int n, int k);
}

#if DEBUG > 2
#include <iomanip>
#include <iostream>
#include <typeinfo> 
#include <chrono>

template<typename T>
std::string to_string_with_precision(const T val, const uint32_t n=6) {
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << val;
	return std::move(out).str();
}
std::string bytes_to_str(size_t size) {
  if(size >= 1e9) return to_string_with_precision((float)size/1e9, 2)+" GB";
  else if (size >= 1e6) return to_string_with_precision((float)size/1e6, 2)+" MB";
  else if (size >= 1e3) return to_string_with_precision((float)size/1e3, 2)+" kB";
  else return std::to_string(size)+" B";
}
#endif

// float comparison
bool constexpr eql_f32 (float a, float b) { return fabs(a-b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * EPSILON); }
bool constexpr lt_f32 (float a, float b)  { return (b - a) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPSILON); }

namespace tensorlib {

typedef enum {
  GPU,
  CPU,
} Device;

typedef enum {
  UNIFORM,
  NORMAL,
  CHI_SQUARED,
  KAIMING_UNIFORM,
  kAIMING_NORMAL,
} randn_dist;

struct View {
  uint32_t* view= nullptr;
  uint32_t* strides = nullptr;
  uint32_t numdim;
  uint64_t elem=1;
  bool bcontiguous;

  View(std::initializer_list<uint32_t> argview, bool contig=true) 
    :numdim(argview.size()), bcontiguous(contig)
  {
    if(numdim >= TENSOR_MAX_DIM) throw std::length_error("TENSOR_MAX_DIM Exceeded.");
    view = new uint32_t[numdim];
    size_t i=0;
    for(const auto& x : argview) {
      if(x == 0) throw std::invalid_argument("View cannot contain values smaller then or equal to 0.");
      view[i++] = x;
      elem *= x;
    }
    if(elem > TENSOR_MAX_STORAGE) throw std::length_error("TENSOR_MAX_STORAGE Exceeded.");
    this->restride();
  }

  View(uint32_t* argview, size_t& len, bool contig=true) 
    :numdim(len), view(&argview[0]), bcontiguous(contig) 
  {
    if(numdim >= TENSOR_MAX_DIM) throw std::length_error("TENSOR_MAX_DIM Exceeded.");
    for(size_t i=0; i<len; i++) {
      if(argview[i] == 0) throw std::invalid_argument("View cannot contain values smaller then or equal to 0.");
      elem*=argview[i];
    }
    if(elem > TENSOR_MAX_STORAGE) throw std::length_error("TENSOR_MAX_STORAGE Exceeded.");
    this->restride();
  }
  ~View() {
    if(view) delete[] view;
    if(strides) delete[] strides;
  }
  inline void restride() {
    if(strides) delete[] strides;
    strides = new uint32_t[numdim];
    strides[numdim-1] = 1;
    for(size_t i=numdim-1; i>0; --i) strides[i-1] = strides[i]*view[i];
  }
};

// float* data = tensorlib::allocate<float>(N*N);
template<typename T>
T* alloc(size_t size) {
  if(sizeof(T)*size > MEMORY_ALIGNMENT) return static_cast<T*>( aligned_alloc(MEMORY_ALIGNMENT, size*sizeof(T)) );
  else return static_cast<T*>( malloc(size*sizeof(T)) );
}

template<typename T=float>
class Tensor {
  private:
    View* mview;
    T* mstorage = nullptr;
    float* grad = nullptr;
    void* endptr = nullptr; //sub-tensor ending address
    bool bsub=false; // subtensor?
    bool bgrad=false;
    bool beye=false; // special indexing 
    bool ballocated=false;
    bool binitialized=false;
    uint32_t disklen=0;
    Device mdevice;

  public:
    // Virtual tensors, no underlying data
    Tensor(std::initializer_list<uint32_t> shp, Device dev=CPU)
      : mview(new View(shp, false)), mdevice(dev) {}

    Tensor(uint32_t* shp, size_t slen, Device dev=CPU)
      : mview(new View(&shp[0], slen, false)), mdevice(dev) {}

    // Data initialization 
    Tensor(T* data, uint64_t size, std::initializer_list<uint32_t> shp, bool grad=false, Device dev=CPU)
      : mview(new View(shp)), disklen(size), bgrad(grad), mdevice(dev), binitialized(true), ballocated(true)
    {
      if(size != mview->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside the given shape.");
      #ifdef FORCE_ALIGNMENT
        mstorage = alloc<T>(size); std::memcpy(&data[0], &mstorage[0], size*sizeof(T));
      #else
        mstorage = &data[0];
      #endif
    }

    Tensor(T* data, uint64_t size, uint32_t* shape, size_t slen, bool grad=false, Device dev=CPU, bool sub=false)
      : mview(new View(&shape[0], slen)), disklen(size), bgrad(grad), mdevice(dev), binitialized(true), ballocated(true), bsub(sub) 
    {
      if(size != mview->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape.");
      #ifdef FORCE_ALIGNMENT
        mstorage = alloc<T>(size); std::memcpy(&mstorage[0], &data[0], size*sizeof(T));
      #else
        mstorage = &data[0];
      #endif
    }

    // sub-tensor, shares data ownership with main tensor
    Tensor(T* data, void* endptr, uint64_t size, uint32_t* shape, size_t slen, bool grad=false, Device dev=CPU)
      : mstorage(data), endptr(endptr), bsub(true), mview(new View(&shape[0], slen)), 
        disklen(size), bgrad(grad), mdevice(dev), binitialized(true), ballocated(true) 
    { if(size != mview->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape."); }

    // if shpmismatch stride correction is left to the user
    Tensor(std::initializer_list<T> data, std::initializer_list<uint32_t> shp, bool shpmismatch=false, bool grad=false, Device dev=CPU)
      : mview(new View(shp)), disklen(data.size()), bgrad(grad), mdevice(dev), binitialized(true)
    {
      if(!shpmismatch && data.size() != mview->elem) throw std::length_error("Elements do not fit the given shape");
      else if (data.size() == mview->elem) ballocated=true; // otherwise virtual
      size_t i=0;
      mstorage = alloc<T>(data.size());
      for(const T& x : data) mstorage[i++] = x;
    }

    constexpr Tensor(Tensor<T>& t) 
      : binitialized(t.binitialized), ballocated(t.ballocated), bgrad(t.bgrad), beye(t.beye), bsub(t.bsub),
        mdevice(t.mdevice), disklen(t.disklen), mview(t.mview), mstorage(t.mstorage), grad(t.grad)
    { t.mview = nullptr; t.mstorage = nullptr; t.grad = nullptr; }

    ~Tensor() {
      if(mstorage && !bsub) free(mstorage);
      if(mview) delete mview;
      if(grad) delete[] grad;
    }

    // Getters /*-------------------------------------------------------------------------------------------------*/
    const T* storage() { return (mstorage) ? &mstorage[0] : static_cast<T*>(nullptr); }
    const uint32_t ndim() { return mview->numdim; }
    const uint64_t numel() { return mview->elem; }
    const uint32_t memsize() { return disklen; }
    const uint32_t* shape() { return &(mview->view)[0]; }
    const uint32_t* strides() { return &(mview->strides)[0]; }
    const uint32_t device() { return mdevice; }
    bool is_initialized() { return binitialized; }
    bool is_allocated() { return ballocated; }
    bool is_contiguous() { return mview->bcontiguous; }
    bool is_eye() { return beye; }
    bool is_sub() { return bsub; }
    bool has_grad() { return bgrad; }
    T item() { return (disklen == 1 && mview->elem == 1) ? mstorage[0] : throw std::runtime_error("Invalid call to .item()"); }

		// Constructor helpers /*------------------------------------------------------------------------------------*/
    Tensor<T>& allocate() {
      if(binitialized && !beye) throw std::runtime_error("Attempted to allocate() initialized tensor.");
      if(mstorage) free(mstorage);
      disklen = mview->elem;
      mstorage = alloc<T>(mview->elem);
      if(bsub) for(size_t i=0; i<disklen; i++) mstorage[i] = mstorage[i]; // if subtensor, move data
      else if(beye) {
        memset(mstorage, (T)0, mview->elem*sizeof(T)); // set all to 0
        for(size_t i=0; i<mview->view[0]; i++) mstorage[i*(mview->view[0]*(mview->numdim-1))+i]=(T)1; 
        beye=false;
      }
      ballocated=binitialized=mview->bcontiguous=true;
      return *this;
    }

    Tensor<T>& fill(T v) {
      if(mstorage) throw std::runtime_error("Cannot fill() initialized tensor."); 
      for(size_t i=0; i<this->mview->numdim; i++) this->mview->strides[i] = 0;
      mstorage = alloc<T>(1);
      mstorage[0] = v;
      disklen=binitialized=mview->bcontiguous=1;
      ballocated=false;
      return *this;
    }

    Tensor<T>& randn(T up=(T)1, T down=(T)0, randn_dist dist=NORMAL, uint32_t seed=0, bool grad=false, Device mdevice=CPU) {
      if(mstorage) throw std::runtime_error("Cannot randn() initialized tensor."); 
      size_t count = (mview->elem % 2 == 0) ? mview->elem : mview->elem+1; // box muller requires %2
      mstorage = alloc<T>(count);
      switch(dist){
        case NORMAL: f32_randn_box_muller_normal(&mstorage[0], this->mview->elem, up, down, seed); break;
        case KAIMING_UNIFORM: f32_randn_kaiming_uniform(&mstorage[0], this->mview->elem);
        case UNIFORM: f32_randn_uniform(&mstorage[0], this->mview->elem, up, down, seed); break;
        case CHI_SQUARED: f32_randn_chi_squared(&mstorage[0], this->mview->elem, up, down, seed); break;
      }
      binitialized=ballocated=mview->bcontiguous=true;
      disklen=mview->elem;
      return *this;
    }

    Tensor<T>& eye() {
      if(mview->numdim < 2) throw std::runtime_error("Cannot create an identity tensor of less then 2 dimensions.");
      size_t nlx = mview->numdim;
      while(--nlx>0 && mview->view[nlx]==mview->view[0]);
      if(nlx != 0) throw std::runtime_error("All tensor dimensions must be equal to create eye().");
      #ifdef FORCE_ALLOCATE
        mstorage = alloc<T>(mview->elem);
        memset(&mstorage[0], 0, mview->elem*sizeof(T));
        for(size_t i=0; i<view->view[0]; i++) mstorage[i*(mview->view[0]*(mview->numdim-1))+i]=1; 
        ballocated=binitialized=true;
      #else
        mstorage = alloc<T>(1);
        beye=binitialized=true;
        ballocated=mview->bcontiguous=false;
      #endif
      return *this;
    }

    static Tensor<T> arange(T stop, T start=(T)0, T step=(T)1, bool grad=false, Device mdevice=CPU) {
			uint32_t size = (std::abs(stop)+std::abs(start))/std::abs(step);
      T* nd = alloc<T>(size);
			for(size_t i=0; i < size; i++) nd[i] = start+(i*step);
      return Tensor<T>(nd, size, {size}, grad, mdevice);
    }

    // Move semantics /*---------------------------------------------------------------------------------------*/
    static Tensor<T> like(Tensor<T>& from) {
      uint32_t* shp = new uint32_t[from.mview->numdim];
      std::memcpy(&shp[0], &(mview->view[0]), mview->numdim*sizeof(uint32_t));
      return Tensor<T>(shp, from.mview->numdim, from.mdevice);
    }

    Tensor<T> copy() {  // shares data ownership 
      uint32_t* shp = new uint32_t[mview->numdim];
      std::memcpy(&shp[0], &(mview->view[0]), mview->numdim*sizeof(uint32_t));
      return Tensor<T>(&mstorage[0], disklen, &shp[0], mview->numdim, bgrad, mdevice, true);
    }
    
    Tensor<T> clone() { // copies data
      uint32_t* shp = new uint32_t[mview->numdim];
      T* nd = alloc<T>(mview->elem);
      std::memcpy(&shp[0], &(mview->view[0]), mview->numdim*sizeof(uint32_t));
      std::memcpy(&nd[0], &mstorage[0], disklen*sizeof(T));
      return Tensor<T>(&nd[0], disklen, &shp[0], mview->numdim, bgrad, mdevice);
    }

    // Indexing /*---------------------------------------------------------------------------------------------*/

    // sub-matrices share data ownership, there's no data movement
    // TODO: add sub-indexing with {}
    template<typename... Args>
    Tensor<T> operator()(uint32_t idx, Args... args) {
      if(!binitialized || !mstorage || !mview) throw std::runtime_error("Cannot index into uninitialized tensor.");
      uint32_t idx_len = sizeof...(args)+1;
      if(idx_len > mview->numdim) throw std::invalid_argument("Too many indices in operator().");
      uint32_t idxs[idx_len] = { idx, args... }; // should not compile if type is wrong
      if(!ballocated) {
        if(beye) {
          if(idx_len < mview->numdim) {} // TODO, use calloc
          uint32_t nlx = idx_len;
          while(--nlx>0 && idxs[nlx]==idxs[0]);
          return Tensor<T>({ ((nlx==0) ? (T)1 : (T)0) }, {1}, false, false, mdevice);
        }
      }
      uint64_t start_idx=0;
      for(size_t i=0; i<idx_len; i++) {
        if(idxs[i] >= mview->view[i]) throw std::invalid_argument("Invalid index for dimension " + std::to_string(mview->view[i])+".");
        start_idx += mview->strides[i]*idxs[i];
      }
      if(idx_len < mview->numdim) {
        if(bsub) {} // TODO: already a sub-matrix
        uint64_t end_idx = start_idx + mview->strides[idx_len-1];
        uint32_t* nshp = new uint32_t[mview->numdim-idx_len];
        for(size_t i=0, ii=idx_len; ii<mview->numdim; i++, ii++) nshp[i] = mview->view[ii];
        return Tensor<T>(&mstorage[start_idx], &mstorage[end_idx], end_idx-start_idx, nshp, mview->numdim-idx_len, grad, mdevice);
      }
      return Tensor<T>({mstorage[start_idx]}, {1}, false, grad, mdevice);
    }

    // Movement OPs /*------------------------------------------------------------------------------------------*/

    //TODO: does not keep track of previous changes to the tensor 
    Tensor<T>& reshape(std::initializer_list<int32_t> nshp) {
      size_t i=0, ac=1;
      int infer_idx=-1;
      uint32_t* nv = new uint32_t[nshp.size()];
      for(const int32_t& x : nshp) {
        if(x<-1) throw std::invalid_argument("Invalid arguments in reshape().");
        if(x==-1) {
          if(infer_idx!=-1) throw std::invalid_argument("Can only infer one dimension in reshape()");
          infer_idx=i; continue;
        }
        nv[i++]=x; ac*=x;
      }
      if(infer_idx!=-1) {
        uint32_t inf = mview->elem/ac;
        nv[infer_idx]=inf;
        ac*=inf;
      }
      if(mview->elem!=ac) throw std::invalid_argument("Dimensions in reshape() do not match the data.");
      if(mview->view) delete[] mview->view;
      mview->view = nv;
      mview->numdim=nshp.size();
      mview->restride();
      return *this;
    }

    Tensor<T>& permute(std::initializer_list<uint32_t> idxs) {
      if(idxs.size() != mview->numdim) throw std::invalid_argument("Invalid number of dimensions in permute()");
      uint32_t consum = ((idxs.size()-1)*idxs.size())/2; // using consecutive number sum to not allow repeated dimensions
      uint32_t* nv = new uint32_t[idxs.size()];
      uint32_t* ns = new uint32_t[idxs.size()];
      uint32_t consum_c=0, i=0;
      for(const uint32_t& v : idxs) {
        if(v>=mview->numdim) throw std::invalid_argument("Invalid value in permute()"); 
        nv[i]=mview->view[v];
        ns[i++]=mview->strides[v];
        consum_c+=v;
      }
      if(consum!=consum_c) throw std::invalid_argument("Repeating dimensions in permute()");
      delete[] mview->view;
      delete[] mview->strides;
      mview->view=nv;
      mview->strides=ns;
      mview->bcontiguous=false;
      return *this;
    }

    // TODO: does not behave like Torch.expand()
    Tensor<T>& expand(std::initializer_list<int32_t> nshp) {
      if(nshp.size() < mview->numdim) throw std::invalid_argument("Invalid number of dimensions.");
      uint32_t hits=0, i=0;
      uint64_t ac=1;
      uint32_t* nv = new uint32_t[nshp.size()];
      uint32_t* ns = new uint32_t[nshp.size()];
      for(const int32_t& x : nshp) {
        if(x<-1) throw std::invalid_argument("Invalid arguments in expand()");
        if(x==-1) {
          if(hits<mview->numdim) {
            nv[i]=mview->view[hits];
            ns[i]=mview->strides[hits++];
            continue;
          } else throw std::invalid_argument("Too many -1's in expand()");
          nv[i]=x;
          ac*=x;
          if(mview->view[hits] == 1 && x!=1) {
            ns[i]=0;
            hits++;
            continue;
          } 
          if(x==mview->view[hits]) ns[i]=mview->strides[hits++];
          else ns[i]=0;
        }
      }
      if(hits!=mview->numdim) throw std::invalid_argument("Invalid arguments in expand()");
      delete[] mview->view;
      delete[] mview->strides;
      mview->view=nv;
      mview->strides=ns;
      mview->elem=ac;
      mview->numdim=nshp.size();
      return *this;
    }

    Tensor<T>& strip() {
      uint32_t idx=0, nndim=0;
      uint64_t ac=1;
      for(size_t i=0; i<mview->numdim; i++) if(mview->strides[i]!=0) nndim++;
      uint32_t* nv = new uint32_t[nndim];
      uint32_t* ns = new uint32_t[nndim];
      for(size_t i=0; i<mview->numdim; i++) {
        if(mview->strides[i]!=0) {
          nv[idx] = mview->view[i];
          ns[idx++] = mview->strides[i];
          ac*=mview->view[i];
        }
      }
      if(ac!=disklen) throw std::runtime_error("Failed to strip() tensor.");
      delete[] mview->view;
      delete[] mview->strides;
      mview->view=nv;
      mview->strides=ns;
      mview->elem = disklen;
      mview->numdim = nndim;
    }

    // TODO
    Tensor<T>& shrink(std::initializer_list<uint32_t> argview) {}
    Tensor<T>& flip(std::initializer_list<uint32_t> argview) {}
    Tensor<T>& pad(std::initializer_list<uint32_t> argview) {}

    // Data movement OPs /*-----------------------------------------------------------------------------------*/

    // NOTE: we already have permute?
    // auto a = Tensor<float>::arange(512*512).reshape({512, 512}).transpose();
    Tensor<T>& transpose() {
      if(!ballocated || !binitialized) throw std::runtime_error("Cannot transpose uninitialised Tensor.");
      if(mview->numdim != 2) throw std::runtime_error("Transposition can only be done on 2D Tensors. Use .permute().");
      T* nd = alloc<T>(mview->elem);
      if(std::is_floating_point<T>::value) intrin::_t_f((const T*)&mstorage[0], &nd[0, mview->strides[0], mview->strides[0]]);
      else intrin::_t_i((const T*)&mstorage[0], &nd[0, mview->strides[0], mview->strides[0]]);
    }

    // Static data generation /*------------------------------------------------------------------------------*/
		static void f32_randn_uniform(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0.f, float e=0.f) {
 			static std::mt19937 rng(std::random_device{}());
			if(!eql_f32(seed, 0.f)) rng.seed(seed);
			static std::uniform_real_distribution<float> dist(down, up);
			if(!eql_f32(e, 0.f)) for(size_t i=0; i<count; i++) do to[i] = dist(rng); while (to[i] <= e);
			else for(size_t i=0; i<count; i++) to[i] = dist(rng);
		}

		static void f32_randn_chi_squared(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0) {
 			static std::mt19937 rng(std::random_device{}());
			if(!eql_f32(seed, 0.f)) rng.seed(seed);
			static std::chi_squared_distribution<float> dist(2);
			for(size_t i=0; i<count; i++) { 
				float n = dist(rng);
				if(n >= down && n <= up) to[i] = n;  
			}
		}

    // https://arxiv.org/pdf/1502.01852.pdf
    static void f32_randn_kaiming_uniform(float* to, uint32_t count, float a=0.01) {
      double bound = std::sqrt(3.f)*std::sqrt(2.f/(1+std::pow(a, 2))) / std::sqrt(count);
      return f32_randn_uniform(to, count, bound, -bound);
    }

		static void f32_randn_box_muller_normal(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0.f) {
			if(count % 2 != 0) count++;  // NOTE: If count is odd, it adds an extra element
			constexpr float epsilon = std::numeric_limits<float>::epsilon();
			constexpr float two_pi = 2.0 * M_PI;
      float* u1 = alloc<float>(count/2);
      float* u2 = alloc<float>(count/2);
			Tensor<>::f32_randn_uniform(&u1[0], count/2, up, down, seed, epsilon);
			Tensor<>::f32_randn_uniform(&u2[0], count/2, up, down, seed);
			for(size_t i=0, j=0; i<count/2; i++, j+=2) {
				auto mag = std::sqrt(-2.0 * std::log(u1[i]));
				to[j]   = mag * std::sin(two_pi * u2[i]);
				to[j+1] = mag * std::cos(two_pi * u2[i]);
			}
      free(u1);
      free(u2);
		}
    
    // Helpers
    static inline uint64_t count_elem(std::initializer_list<uint32_t> argview) {
      uint64_t sum=1;
      for(const uint32_t& x : argview) sum *= x;
      return sum;
    }
};
  

// Vector Kernels 
namespace intrin {

// transpose float kernel
#if defined(__AVX512_F__)
  // TODO
#elif defined(__AVX__)
static inline void _t_f(const float* from, float* to, int lda, int ldb) {
  __m256 t0, t1, t2, t3, t4,t5, t6, t7,
         r0, r1, r2, r3, r4, r5, r6, r7;
  r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[0*lda+0])), _mm_load_ps(&from[4*lda+0]), 1);
  r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[1*lda+0])), _mm_load_ps(&from[5*lda+0]), 1);
  r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[2*lda+0])), _mm_load_ps(&from[6*lda+0]), 1);
  r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[3*lda+0])), _mm_load_ps(&from[7*lda+0]), 1);
  r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[0*lda+4])), _mm_load_ps(&from[4*lda+4]), 1);
  r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[1*lda+4])), _mm_load_ps(&from[5*lda+4]), 1);
  r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[2*lda+4])), _mm_load_ps(&from[6*lda+4]), 1);
  r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&from[3*lda+4])), _mm_load_ps(&from[7*lda+4]), 1);
  t0 = _mm256_unpacklo_ps(r0, r1); t1 = _mm256_unpackhi_ps(r0, r1);
  t2 = _mm256_unpacklo_ps(r2, r3); t3 = _mm256_unpackhi_ps(r2, r3);
  t4 = _mm256_unpacklo_ps(r4, r5); t5 = _mm256_unpackhi_ps(r4, r5);
  t6 = _mm256_unpacklo_ps(r6, r7); t7 = _mm256_unpackhi_ps(r6, r7);
  r0 = _mm256_shuffle_ps(t0, t2, 0x44); r1 = _mm256_shuffle_ps(t0, t2, 0xee);
  r2 = _mm256_shuffle_ps(t1, t3, 0x44); r3 = _mm256_shuffle_ps(t1, t3, 0xee);
  r4 = _mm256_shuffle_ps(t4, t6, 0x44); r5 = _mm256_shuffle_ps(t4, t6, 0xee);
  r6 = _mm256_shuffle_ps(t5, t7, 0x44); r7 = _mm256_shuffle_ps(t5, t7, 0xee);
  _mm256_store_ps( &to[0*ldb], r0); _mm256_store_ps( &to[1*ldb], r1);
  _mm256_store_ps( &to[2*ldb], r2); _mm256_store_ps( &to[3*ldb], r3);
  _mm256_store_ps( &to[4*ldb], r4); _mm256_store_ps( &to[5*ldb], r5);
  _mm256_store_ps( &to[6*ldb], r6); _mm256_store_ps( &to[7*ldb], r7);
}
#elif defined (__SSE__)
static inline void _t_f(const float* from, float* to, int lda, int ldb) {
  // TODO: SSE transpose
}

#else
static inline void _t_f(const float* from, float* to, int lda, int ldb) {
  // TODO: RAW transpose
}
#endif


//transpose int
#if defined(__AVX2__)
#elif defined(__SSE__)
#else
#endif

// sgemm
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

inline void pack_b(int k, const float* b, int ldb, float* to) {
  int i;
  const float *b_i0_ptr = &b[0], *b_i1_ptr = &b[(1*ldb)],
              *b_i2_ptr = &b[(2*ldb)], *b_i3_ptr = &b[(3*ldb)],
              *b_i4_ptr = &b[(4*ldb)], *b_i5_ptr = &b[(5*ldb)],
              *b_i6_ptr = &b[(6*ldb)], *b_i7_ptr = &b[(7*ldb)];
  for(i=0; i<k; i++) {
    *to = *b_i0_ptr;
    *(to+1) = *(b_i1_ptr); *(to+2) = *(b_i2_ptr);
    *(to+3) = *(b_i3_ptr); *(to+4) = *(b_i4_ptr);
    *(to+5) = *(b_i5_ptr); *(to+6) = *(b_i6_ptr);
    *(to+7) = *(b_i7_ptr);
    to += 8;
    b_i0_ptr++; b_i1_ptr++; b_i2_ptr++; b_i3_ptr++; b_i4_ptr++; b_i5_ptr++; b_i6_ptr++; b_i7_ptr++;
  }
}

// sgemm float
#ifdef __AVX__
typedef union {
  __m256 v;
  float f[8];
} m256_t;

inline void _8x8_m256_gemm(int k, const float* a, const float* b, float* c, int ldc) {
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

template<int mb=128, int kb=128, int th=1>
void _f_gemm(float* a, float* b, float* c, int m, int n, int k) {
  #pragma omp parallel for shared(a, b, c, m, n, k) default(none) collapse(1) num_threads(th)
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
#elif defined(__SSE__)
  // TODO

#else
template<int block, int tile, int th=1>
void _f_gemm(float* a, float* b, float* c, int m, int n, int k) {
  #pragma omp parallel for shared(c, a, b, m, n, k) default(none) collapse(4) num_threads(th)
  for(size_t row_tile=0; row_tile < m; row_tile += block) {
    for(size_t column_tile=0; column_tile < n; column_tile += block) {
      for(size_t inner_tile=0; inner_tile < k; inner_tile += tile) {
        for(size_t row=row_tile; row < row_tile+block; row++) {
          int i_tile_end = std::min<float>(k, inner_tile+tile);
          for(size_t in=inner_tile; in<i_tile_end; in++) {
            for(size_t col=column_tile; col<column_tile+block; col++) {
              c[row*n+col] += a[row*k+in] * b[in*n+col];
            }
          }
        }
      }
    }
  }
}
#endif

//sgemm int
#ifdef __AVX2__
#elif defined(__SSE__)
#else
#endif

} // intrin namespace
} // tensorlib namespace


// std::cout repr

template<typename T>
inline std::ostream& operator<<(std::ostream& outs, tensorlib::Tensor<T>& tensor) {
	std::string repr = "<Tensor view([";
	auto shape = tensor.shape();
  auto strides = tensor.strides();
  std::string str="";
	for(size_t i=0; i < tensor.ndim(); i++) {
		repr += std::to_string(shape[i]);
		repr += (i == tensor.ndim()-1) ? "" : ", ";
    str += std::to_string(strides[i]);
		str += (i == tensor.ndim()-1) ? "" : ", ";
	}
	repr += "]";
  repr += "(" + str + ")), on ";
	repr += (tensor.device() == 1) ? "CPU" : "GPU"; 
  if(tensor.is_initialized()) {
    repr += ", type=";
    repr += typeid(tensor.storage()[0]).name();
    repr += std::to_string(sizeof(tensor.storage()[0])*8);
  }
  repr += ", grad="+std::to_string(tensor.has_grad());
  if(tensor.is_eye() || !tensor.is_allocated()) {
    repr += ", disk=" + std::to_string(sizeof(tensor)) + " B";
  } else if (!tensor.is_initialized()) {
    repr += ", disk=" + std::to_string(sizeof(tensor)) + " B";
  } else {
    repr += ", disk=" + bytes_to_str(tensor.memsize()*sizeof(tensor.storage()[0]) + sizeof(tensor));
  }
  repr += (tensor.is_initialized()) ? "" : ", is_initialized=false";
  repr += (!tensor.is_initialized()) ? "" : (tensor.is_allocated()) ? "" : ", is_allocated=false";
	repr += ">";
	return outs << repr;
}
