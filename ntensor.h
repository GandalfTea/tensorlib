
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

#if DEBUG > 2
#include <omp.h>
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
} randn_dist;

struct View {
  uint32_t* view= nullptr;
  uint32_t* strides = nullptr;
  uint32_t numdim;
  uint64_t elem=1;

  View(std::initializer_list<uint32_t> argview) 
    :numdim(argview.size())
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

  View(uint32_t* argview, size_t& len) 
    :numdim(len), view(&argview[0])
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
    const View* mview;
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
      : mview(new View(shp)), mdevice(dev) {}

    Tensor(uint32_t* shp, size_t slen, Device dev=CPU)
      : mview(new View(&shp[0], slen)), mdevice(dev) {}

    // Data initialization
    Tensor(T* data, uint64_t size, std::initializer_list<uint32_t> shp, bool grad=false, Device dev=CPU)
      : mview(new View(shp)), disklen(size), bgrad(grad), mdevice(dev), binitialized(true), ballocated(true)
    {
      if(size != mview->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape.");
      #ifdef FORCE_ALIGNMENT
        mstorage = alloc<T>(size); for(size_t i=0; i<size; i++) mstorage[i] = data[i]; 
      #else
        mstorage = &data[0];
      #endif
    }

    Tensor(T* data, uint64_t size, uint32_t* shape, size_t slen, bool grad=false, Device dev=CPU, bool sub=false)
      : mview(new View(&shape[0], slen)), disklen(size), bgrad(grad), mdevice(dev), binitialized(true), ballocated(true), bsub(sub) 
    {
      if(size != mview->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape.");
      #ifdef FORCE_ALIGNMENT
        mstorage = alloc<T>(size); for(size_t i=0; i<size; i++) mstorage[i] = data[i]; 
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

    // Getters /*-------------------------------------------------------------*/
    const T* storage() { return (mstorage) ? &mstorage[0] : static_cast<T*>(nullptr); }
    const uint32_t ndim() { return mview->numdim; }
    const uint64_t numel() { return mview->elem; }
    const uint32_t memsize() { return disklen; }
    const uint32_t* shape() { return &(mview->view)[0]; }
    const uint32_t* strides() { return &(mview->strides)[0]; }
    const uint32_t device() { return mdevice; }
    bool is_initialized() { return binitialized; }
    bool is_allocated() { return ballocated; }
    bool is_eye() { return beye; }
    bool is_sub() { return bsub; }
    bool has_grad() { return bgrad; }
    T item() { return (disklen == 1 && mview->elem == 1) ? mstorage[0] : throw std::runtime_error("Invalid call to .item()"); }

		// Constructor helpers /*-------------------------------------------------*/
    Tensor<T>& allocate() {
      if(binitialized) throw std::runtime_error("Attempted to allocate() initialized tensor.");
      T* nd = alloc<T>(mview->elem);
      if(bsub) for(size_t i=0; i<disklen; i++) nd[i] = mstorage[i]; // if subtensor, move data
      disklen = mview->elem;
      ballocated = binitialized = true;
      mstorage = &nd[0];
      return *this;
    }

    // auto a = Tensor<float>({1024, 1024}).fill(0.f);
    Tensor<T>& fill(T v) {
      if(mstorage) throw std::runtime_error("Cannot fill() initialized tensor."); 
      mstorage = alloc<T>(1);
      mstorage[0] = v;
      for(size_t i=0; i<this->mview->numdim; i++) this->mview->strides[i] = 0;
      binitialized = true;
      ballocated = false;
      disklen=1;
      return *this;
    }

    // auto a = Tensor<float>({1024, 1024}).randn();
    Tensor<T>& randn(T up=(T)1, T down=(T)0, randn_dist dist=NORMAL, uint32_t seed=0, bool grad=false, Device mdevice=CPU) {
      if(mstorage) throw std::runtime_error("Cannot randn() initialized tensor."); 
      size_t count = (mview->elem % 2 == 0) ? mview->elem : mview->elem+1; // box muller requires %2
      mstorage = alloc<T>(count);
      switch(dist){
        case NORMAL: f32_generate_box_muller_normal_distribution(&mstorage[0], this->mview->elem, up, down, seed); break;
        case UNIFORM: f32_generate_uniform_distribution(&mstorage[0], this->mview->elem, up, down, seed); break;
        case CHI_SQUARED: f32_generate_chi_squared_distribution(&mstorage[0], this->mview->elem, up, down, seed); break;
      }
      binitialized = ballocated = true;
      disklen=mview->elem;
      return *this;
    }

    // auto a = Tensor<int>({1024, 1024}).eye();
    Tensor<T>& eye() {
      if(mview->numdim < 2 || mview->elem < 4) throw std::runtime_error("Cannot create an identity tensor of less then 2 dimensions or elements.");
      size_t nlx = mview->numdim;
      while(--nlx>0 && mview->view[nlx]==mview->view[0]);
      if(nlx != 0) throw std::runtime_error("Tensor dimensions must be equal to create eye().");
      #ifdef FORCE_ALLOCATE
        mstorage = calloc(mview->elem, sizeof(T)); // initialized allocated mem to 0
        for(size_t i=0; i<view->view[0]; i++) mstorage[i*(mview->view[0]*(mview->numdim-1))+i]=1; 
        ballocated=binitialized=true;
      #else
        T* nd = alloc<T>(1);
        beye=binitialized=true;
        ballocated=false;
      #endif
      return *this;
    }

    // auto a = Tensor<float>::arange(1024*1024).reshape({1024, 1024});
    static Tensor<T> arange(T stop, T start=(T)0, T step=(T)1, bool grad=false, Device mdevice=CPU) {
      if(std::is_floating_point<T>::value) {
			  if(lt_f32((T)stop, (T)start) || lt_f32((T)step, (T)0) 
           || eql_f32((T)step, (T)0) || lt_f32((T)stop, (T)step) 
           || eql_f32((T)step, (T)stop)) throw std::runtime_error("Invalid arguments in arange().");
      } else if(stop < start || step <= 0 || step >= stop) throw std::runtime_error("Invalid arguments in arange().");
			uint32_t size = (std::abs(stop)+std::abs(start))/step;
      T* nd = static_cast<T*>( aligned_alloc(MEMORY_ALIGNMENT, size*sizeof(T)) );
			for(size_t i=0; i < size; i++) nd[i] = start + i*step;
      return Tensor<T>(nd, size, {size}, grad, mdevice);
    }

    // Move semantics /*-------------------------------------------------*/
    static Tensor<T> like(Tensor<T>& from) {
      uint32_t* shp = new uint32_t[from.mview->numdim];
      for(size_t i=0; i<from.mview->numdim; i++) shp[i] = from.mview->view[i];
      return Tensor<T>(shp, from.mview->numdim, from.mdevice);
    }

    Tensor<T> copy() {  // shares data ownership 
      uint32_t* shp = new uint32_t[mview->numdim];
      for(size_t i=0; i<mview->numdim; i++) shp[i] = mview->view[i];
      return Tensor<T>(&mstorage[0], disklen, &shp[0], mview->numdim, bgrad, mdevice, true);
    }
    
    Tensor<T> clone() { // copies data
      uint32_t* shp = new uint32_t[mview->numdim];
      T* nd = alloc<T>(mview->elem);
      for(size_t i=0; i<mview->numdim; i++) shp[i] = mview->view[i];
      for(size_t i=0; i<disklen; i++) nd[i] = mstorage[i];
      return Tensor<T>(&nd[0], disklen, &shp[0], mview->numdim, bgrad, mdevice);
    }

    // Indexing /*-------------------------------------------------*/

    // sub-matrices share data ownership, there's no data movement
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
          return Tensor<>({ ((nlx==0) ? (T)1 : (T)0) }, {1}, false, false, mdevice);
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

    // Static data generation /*-------------------------------------------------*/

		static void f32_generate_uniform_distribution(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0, 
										                                bool bepsilon=false, float epsilon=0) 
		{
 			static std::mt19937 rng(std::random_device{}());
			if(seed!=0) rng.seed(seed);
			static std::uniform_real_distribution<> dist(down, up);
			if(bepsilon) {
				for(size_t i=0; i<count; i++) {
					do {
						to[i] = dist(rng);
					} while (to[i] <= epsilon);
				}
			} else for(size_t i=0; i<count; i++) to[i] = dist(rng);
		}

		static void f32_generate_chi_squared_distribution(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0) {
 			static std::mt19937 rng(std::random_device{}());
			if(seed!=0) rng.seed(seed);
			static std::chi_squared_distribution<float> dist(2);
			for(size_t i=0; i<count; i++) { 
				float n = dist(rng);
				if(n >= down && n <= up) to[i] = n;  
			}
		}

		// If count is odd, it adds an extra element
		static void f32_generate_box_muller_normal_distribution(float* to, uint32_t count, float up=1.f, float down=0.f, double seed=0) {
			if(count % 2 != 0) count++; 
			constexpr float epsilon = std::numeric_limits<float>::epsilon();
			constexpr float two_pi = 2.0 * M_PI;
      float* u1 = alloc<float>(count/2);
      float* u2 = alloc<float>(count/2);
			Tensor<>::f32_generate_uniform_distribution(&u1[0], count/2, up, down, seed, true, epsilon);
			Tensor<>::f32_generate_uniform_distribution(&u2[0], count/2, up, down, seed);
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
  
}

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
