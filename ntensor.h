
#pragma once

#include <limits.h> // only used here
#define TENSOR_MAX_DIM (2 << 15)
#define TENSOR_MAX_STORAGE UINT_MAX 

#define DEBUG 3
#define EPSILON 0.0001
#define DATA_ALIGNMENT 32 // avx 32 bytes 

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
    delete strides;
    uint32_t* nstrd = new uint32_t[numdim];
    nstrd[numdim-1] = 1;
    for(size_t i=numdim-1; i>0; --i) nstrd[i-1] = nstrd[i]*view[i];
    strides = nstrd;
  }
};


// float* data = tensorlib::allocate<float>(N*N);
template<typename T>
T* alloc(size_t size) {
  if(sizeof(T)*size > DATA_ALIGNMENT) return static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*sizeof(T)) );
  else return static_cast<T*>( malloc(size*sizeof(T)) );
}


template<typename T=float>
class Tensor {

  private:
    const View* mview;

    T* mstorage = nullptr;
    bool bsub=false; // subtensor?
    void* endptr = nullptr; //sub-tensor ending address

    float* grad = nullptr;

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
      // There doesn't seem to be a way to check memory block alignment
      // of argument data, so this will run every time
      #ifdef FORCE_ALIGN
        if(sizeof(T)*size > DATA_ALIGNMENT) mstorage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*sizeof(T)) );
        else mstorage = static_cast<T*>( malloc(size*sizeof(T)) );
        for(size_t i=0; i<size; i++) mstorage[i] = data[i]; 
      #else
        mstorage = &data[0];
      #endif
    }

    Tensor(T* data, uint64_t size, uint32_t* shape, size_t slen, bool grad=false, Device dev=CPU, bool sub=false)
      : mview(new View(&shape[0], slen)), disklen(size), bgrad(grad), mdevice(dev), binitialized(true), ballocated(true), bsub(sub) 
    {
      if(size != mview->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape.");
      #ifdef FORCE_ALIGN
        if(sizeof(T)*size > DATA_ALIGNMENT) mstorage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*sizeof(T)) );
        else mstorage = static_cast<T*>( malloc(size*sizeof(T)) );
        for(size_t i=0; i<size; i++) mstorage[i] = data[i]; 
      #else
        mstorage = &data[0];
      #endif
    }

    // sub-tensor, shares data ownership with main tensor
    Tensor(T* data, void* endptr, uint64_t size, uint32_t* shape, size_t slen, bool grad=false, Device dev=CPU)
      : mstorage(data), endptr(endptr), bsub(true), mview(new View(&shape[0], slen)), 
        disklen(size), bgrad(grad), mdevice(dev), binitialized(true), ballocated(true) 
    {
      if(size != mview->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape.");
    }

    // if shpmismatch stride correction is left to the user
    Tensor(std::initializer_list<T> data, std::initializer_list<uint32_t> shp, bool shpmismatch=false, bool grad=false, Device dev=CPU)
      : mview(new View(shp)), disklen(data.size()), bgrad(grad), mdevice(dev), binitialized(true)
    {
      if(!shpmismatch && data.size() != mview->elem)
        throw std::length_error("Elements do not fit the given shape. To mismatch shape, give shpmismatch=true.");
      else if (data.size() == mview->elem) ballocated=true; // otherwise virtual
      size_t i=0;
      if(sizeof(T)*data.size() > DATA_ALIGNMENT) mstorage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, data.size()*sizeof(T)) );
      else mstorage = static_cast<T*>( malloc(data.size()*sizeof(T)) );
      for(const T& x : data) mstorage[i++] = x;
    }

    constexpr Tensor(Tensor<T>& t) 
      : binitialized(t.binitialized), ballocated(t.ballocated), bgrad(t.bgrad), beye(t.beye), bsub(t.bsub),
        mdevice(t.mdevice), disklen(t.disklen), mview(t.mview), mstorage(t.mstorage), grad(t.grad)
    {
      // prevent deallocation
      t.mview = nullptr;
      t.mstorage = nullptr;
      t.grad = nullptr;
    }

    // Destructors
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
    const uint32_t* view() { return &(mview->view)[0]; }
    const uint32_t* strides() { return &(mview->strides)[0]; }
    const uint32_t device() { return mdevice; }
    bool is_initialized() { return binitialized; }
    bool is_allocated() { return ballocated; }
    bool is_eye() { return beye; }
    bool is_sub() { return bsub; }
    bool has_grad() { return bgrad; }

		// Constructor helpers /*-------------------------------------------------*/

    // for virtual tensors and sub-tensors
    Tensor<T>& allocate() {
      if(binitialized) throw std::runtime_error("Attempted to allocate() initialized tensor.");
      T* nd = alloc<T>(mview->elem);
      if(bsub) for(size_t i=0; i<disklen; i++) nd[i] = mstorage[i];
      ballocated = true;
      binitialized = true;
      mstorage = &nd[0];
      return *this;
    }

    // auto a = Tensor<>::fill({2048, 2048}, 69.f);
    static Tensor<T> fill(std::initializer_list<uint32_t> shp, T& v, bool grad=false, Device device=CPU) {
      Tensor<T> ret = Tensor<T>({v}, shp, true, grad, device);
      ret.mview->strides[0] = 0;
      return ret;
    }

/*
    static Tensor<T> fill(T* shp, uint64_t len, T& v, bool grad=false, Device device=CPU) {
      Tensor<T> ret = Tensor<T>({v}, shp, len, true, grad, device);
      ret.mview->strides[0] = 0;
      return ret;
    }
*/

    // auto b = Tensor<>::like(a).fill(69.f);
    void fill(T v) {
      if(mstorage) throw std::runtime_error("Cannot fill initialized tensor."); 
      mstorage = new T[1];
      mstorage[0] = v;
      for(size_t i=0; i<this->mview->numdim; i++) this->mview->strides[i] = 0;
      this->binitialized = true;
      this->ballocated = false;
    }

    // auto a = Tensor<float>::arange(1024*1024, 0).reshape({1024, 1024});
    static Tensor<T> arange(T stop, T start=(T)0, T step=(T)1, bool grad=false, Device mdevice=CPU) {
      if(std::is_floating_point<T>::value) {
			  if(lt_f32((T)stop, (T)start) || lt_f32((T)step, (T)0) 
           || eql_f32((T)step, (T)0) || lt_f32((T)stop, (T)step) 
           || eql_f32((T)step, (T)stop)) throw std::runtime_error("Invalid arguments in arange().");
      } else if(stop < start || step <= 0 || step >= stop) throw std::runtime_error("Invalid arguments in arange().");
			uint32_t size = (std::abs(stop)+std::abs(start))/step;
      T* nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*sizeof(T)) );
			for(size_t i=0; i < size; i++) nd[i] = start + i*step;
      return Tensor<T>(nd, size, {size}, grad, mdevice);
    }

    // auto a = Tensor<float>::randn({1024, 1024}, 3.14, -3.14, CHI_SQUARED, 177013);
    static Tensor<T> randn(std::initializer_list<uint32_t> shp, T up=(T)1, T down=(T)0, randn_dist dist=NORMAL,
                           uint32_t seed=0, bool grad=false, Device mdevice=CPU) 
    {
      uint64_t elem = count_elem(shp);
      T* nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, elem*sizeof(T)) );
      switch(dist){
        case NORMAL: nd = f32_generate_box_muller_normal_dist(elem, up, down, seed); break;
        case UNIFORM: nd = f32_generate_uniform_dist(elem, up, down, seed); break;
        case CHI_SQUARED: nd = f32_generate_chi_squared_dist(elem, up, down, seed); break;
      }
      return Tensor<T>(nd, elem, shp, grad, mdevice);
    }

    // auto b = Tensor<float>::like(a).randn();
    void randn(T up=(T)1, T down=(T)0, randn_dist dist=NORMAL, uint32_t seed=0, bool grad=false, Device mdevice=CPU) {
      mstorage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, this->mview->elem*sizeof(T)) );
      switch(dist){
        case NORMAL: mstorage = f32_generate_box_muller_normal_dist(this->mview->elem, up, down, seed); break;
        case UNIFORM: mstorage = f32_generate_uniform_dist(this->mview->elem, up, down, seed); break;
        case CHI_SQUARED: mstorage = f32_generate_chi_squared_dist(this->mview->elem, up, down, seed); break;
      }
    }

    // by default not allocated into memory. value is decided when indexing
    // TODO: does this work for dims>2?
    static Tensor<T> eye(uint32_t size, uint32_t dims=2, bool resolved=false, Device mdevice=CPU) {
      if(size < 2 || dims < 2) throw std::runtime_error("Cannot create an identity tensor of less then 2 dimensions or elements.");
      #ifdef FORCE_ALLOCATE
        T* nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*dims*sizeof(T)) );
        for(size_t i=0; i<size; i++) nd[i*size+i]=size; 
        uint32_t* shp = new uint32_t[dims];
        for(size_t i=0; i<dims; i++) shp[i]=size;
        Tensor<T> ret = Tensor<T>(nd, size*dims, shp, dims, false, mdevice);
        ret.ballocated=true;
        ret.binitialized=true;
      #else
        T* nd = new T[1];
        uint32_t* shp = new uint32_t[dims];
        for(size_t i=0; i<dims; i++) shp[i]=size;
        Tensor<T> ret = Tensor<T>(nd, 1, shp, dims, false, mdevice);
        ret.beye=true;
        ret.ballocated=false;
        ret.binitialized=true;
        return ret;
      #endif
    }


    // Move semantics /*-------------------------------------------------*/

    // new mview, no data ptr 
    static Tensor<T> like(Tensor<T>& from) {
      uint32_t* shp = new uint32_t[from.mview->numdim];
      for(size_t i=0; i<from.mview->numdim; i++) shp[i] = from.mview->view[i];
      return Tensor<T>(shp, from.mview->numdim, from.mdevice);
    }

    // new mview, same data ptr
    Tensor<T> copy() {
      uint32_t* shp = new uint32_t[mview->numdim];
      for(size_t i=0; i<mview->numdim; i++) shp[i] = mview->view[i];
      return Tensor<T>(&mstorage[0], disklen, &shp[0], mview->numdim, bgrad, mdevice, true);
    }

    // new mview, new data ptr 
    Tensor<T> clone() {
      uint32_t* shp = new uint32_t[mview->numdim];
      T* nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, mview->elem*sizeof(T)) );
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
          if(idx_len < mview->numdim) {} // TODO
          uint32_t nlx = idx_len;
          while(--nlx>0 && idxs[nlx]==idxs[0]);
          return Tensor<>({ ((nlx==0) ? (float)1 : (float)0) }, {1}, false, false, mdevice);
        }
      }
      uint64_t start_idx=0;
      for(size_t i=0; i<idx_len; i++) {
        if(idxs[i] >= mview->view[i]) throw std::invalid_argument("Invalid index for dimension " + std::to_string(i)+".");
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

    // TODO merge this func with allocate on virtual tensors
    // allocate new data for sub-tensor
    // auto b = a(1).detatch()
    void detatch() {
      if(!bsub) std::runtime_error("Call of .detatch() on root tensor.");
      T* nd;
      if(sizeof(T)*view->elem > DATA_ALIGNMENT) nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, view->elem*sizeof(T)) );
      else nd = static_cast<T*>( malloc(view->elem*sizeof(T)) );
      for(size_t i=0; i<view->elem; i++) nd[i] = mstorage[i];
      mstorage = &nd[0];
      bsub = false;
      endptr = nullptr;
    }

    // Static data generation /*-------------------------------------------------*/

		static float* f32_generate_uniform_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0, 
										                                                  bool bepsilon=false, float epsilon=0) 
		{
 			static std::mt19937 rng(std::random_device{}());
			if(seed!=0) rng.seed(seed);
			static std::uniform_real_distribution<> dist(down, up);
      float* ret = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, count*sizeof(float)) );
			if(bepsilon) {
				for(size_t i=0; i<count; i++) {
					do {
						ret[i] = dist(rng);
					} while (ret[i] <= epsilon);
				}
			} else for(size_t i=0; i<count; i++) ret[i] = dist(rng);
			return ret;
		}

		static float* f32_generate_chi_squared_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0) {
 			static std::mt19937 rng(std::random_device{}());
			if(seed!=0) rng.seed(seed);
			static std::chi_squared_distribution<float> dist(2);
      float* ret = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, count*sizeof(float)) );
			for(size_t i=0; i<count; i++) { 
				float n = dist(rng);
				if(n >= down && n <= up) ret[i] = n;  
			}
			return ret;
		}

		// If count is odd, it adds an extra element
		static float* f32_generate_box_muller_normal_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0) {
			if(count % 2 != 0) count++; 
			constexpr float epsilon = std::numeric_limits<float>::epsilon();
			constexpr float two_pi = 2.0 * M_PI;
      float* ret = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, count*sizeof(float)) );
			auto u1 = Tensor<>::f32_generate_uniform_distribution(count/2, up, down, seed, true, epsilon);
			auto u2 = Tensor<>::f32_generate_uniform_distribution(count/2, up, down, seed);
			for(size_t i=0, j=0; i<count/2; i++, j+=2) {
				auto mag = std::sqrt(-2.0 * std::log(u1[i]));
				ret[j]   = mag * std::sin(two_pi * u2[i]);
				ret[j+1] = mag * std::cos(two_pi * u2[i]);
			}
			return ret;
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
	auto shape = tensor.view();
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
