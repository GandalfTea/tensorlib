
#pragma once

#include <limits.h> // only used here
#define TENSOR_MAX_DIM (2 << 15)
#define TENSOR_MAX_STORAGE_SIZE UINT_MAX 

#define DEBUG 3
#define EPSILON 0.0001
#define DATA_ALIGNMENT 32 // avx 256-bit

#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <stdlib.h>
#include <stdexcept>
#include <initializer_list>
#include <type_traits> // arange std::is_floating_point

using std::size_t;

// must compile with appropriate flags for this to work
#if defined(__AVX__) && (defined(__FMA__) || defined(__FMA4__))
  #define AVX_CPU_GEMMS
  #include "cpu_gemms.h"
#endif

// test for general alignment 
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

  View(uint32_t* argview, uint32_t& len) 
    :numdim(len), view(&argview)
  {
    if(numdim >= TENSOR_MAX_DIM) throw std::length_error("TENSOR_MAX_DIM Exceeded.");
    for(size_t i=0; i<len; i++) {
      if(argview[i] == 0) throw std::invalid_argument("View cannot contain values smaller then or equal to 0.");
      elem*=argview[i];
    }
    if(elem > TENSOR_MAX_STORAGE) throw std::length_error("TENSOR_MAX_STORAGE Exceeded.");
    this->restride();
  }

  inline void restride() {
    strides[numdim-1] = 1;
    for(size_t i=numdim-1; i>0; --i) strides[i-1] = strides[i]*view[i];
  }
};

template<typename T=float>
class Tensor {

  private:
    T* storage = nullptr;
    T* grad = nullptr;
    View* view;
    bool bgrad=false;
    bool beye=false; // special indexing 
    bool is_allocated=false;
    bool is_initialized=false;
    uint32_t disklen=0;
    Device device;

  public:
    
    // Virtual tensors, no underlying data
    Tensor(std::initializer_list<uint32_t> shp, Device dev=CPU)
      : view(&View(shp)), device(dev) {}

    Tensor(uint32_t* shp, size_t s_len, Device dev=CPU)
      : view(&View(shp, s_len)), device(dev) {}

    // Data initialization
    Tensor(T* data, uint64_t size, std::initializer_list<uint32_t> shp, bool grad=false, Device dev=CPU)
      : view(&View(shp)), disklen(size), grad(grad), device(dev), is_initialized(true), is_allocated(true)
    {
      if(size != view->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape.");
      // There doesn't seem to be a way to check memory block alignment
      // of argument data, so this will run every time
      #ifdef FORCE_ALIGN
        storage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*sizeof(T)) );
        for(size_t i=0; i<size; i++) storage[i] = data[i]; 
      #else
        storage = &data[0];
      #endif
    }

    Tensor(T* data, uint64_t size, uint32_t* shape, size_t s_len, bool grad=false, Device dev=CPU)
      : view(&View(shape, s_len)), disklen(size), grad(grad), device(dev), is_initialized(true), is_allocated(true) 
    {
      if(size != view->elem) throw std::length_error(std::to_string(size)+" elements do not fit inside of given shape.");
      #ifdef FORCE_ALIGN
        storage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*sizeof(T)) );
        for(size_t i=0; i<size; i++) storage[i] = data[i]; 
      #else
        storage = &data[0];
      #endif
    }

    // if nalloc, stride correction is left to the user
    Tensor(std::initializer_list<T> data, std::initializer_list<uint32_t> shp, bool shpmismatch=false, bool grad=false, Device dev=CPU)
      : view(&View(shp)), disklen(data.size()), grad(grad), device(dev), is_initialized(true)
    {
      if(!shpmismatch && data.size() != view->elem)
        throw std::length_error("Elements do not fit the given shape. To mismatch shape, give shpmismatch=true.");
      else if (data.size() == view->elem) is_allocated=true; // otherwise virtual
      size_t i=0;
      storage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, data.size()*sizeof(T)) );
      for(const T& x : data) storage[i++] = x;
    }

		// Constructor helpers
    /*-------------------------------------------------*/

    // auto a = Tensor<>::fill({2048, 2048}, 69.f);
    static Tensor<T> fill(std::initializer_list<uint32_t> shp, T& v, bool grad=false, Device device=CPU) {
      Tensor<T> ret = Tensor<T>({v}, shp, true, grad, device);
      ret.view->strides[0] = 0;
      return ret;
    }
    static Tensor<T> fill(T* data, uint64_t len, T& v, bool grad=false, Device device=CPU) {
      Tensor<T> ret = Tensor<T>({v}, shp, true, grad, device);
      ret.view->strides[0] = 0;
      return ret;
    }

    // auto b = Tensor<>::like(a).fill(69.f);
    void fill(T v) {
      if(storage) throw std::runtime_error("Cannot fill initialized tensor."); 
      storage = new T[1];
      stogare[0] = v;
      for(size_t i=0; i<this->view->numdim; i++) this->view->strides[i] = 0;
      this->is_initialized = true;
      this->is_allocated = false;
    }

    // auto a = Tensor<float>::arange(1024*1024, 0).reshape({1024, 1024});
    static Tensor<T> arange(T stop, T start=(T)0, T step=(T)1, bool grad=false, Device device=CPU) {
      if(std::is_floating_point<T>::value) {
			  if(lt_f32((T)stop, (T)start) || lt_f32((T)step, (T)0) 
           || eql_f32((T)step, (T)0) || lt_f32((T)stop, (T)step) 
           || eql_f32((T)step, (T)stop)) throw std::runtime_error("Invalid arguments in arange().");
      } else if(stop < start || step <= 0 || step >= stop) throw std::runtime_error("Invalid arguments in arange().");
			uint32_t size = (std::abs(stop)+std::abs(start))/step;
      T* nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*sizeof(T)) );
			for(size_t i=0; i < size; i++) nd[i] = start + i*step;
      return Tensor<T>(nd, size, {size}, grad, device);
    }

    // auto a = Tensor<float>::randn({1024, 1024}, 3.14, -3.14, CHI_SQUARED, 177013);
    static Tensor<T> randn(std::initializer_list<uint32_t> shp, T up=(T)1, T down=(T)0, Distribution dist=NORMAL,
                           uint32_t seed=0, bool grad=false, Device device=CPU) 
    {
      uint64_t elem = count_elem(shp);
      T* nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, elem*sizeof(T)) );
      switch(dist){
        case NORMAL: nd = f32_generate_box_muller_normal_dist(elem, up, down, seed); break;
        case UNIFORM: nd = f32_generate_uniform_dist(elem, up, down, seed); break;
        case CHI_SQUARED: nd = f32_generate_chi_squared_dist(elem, up, down, seed); break;
      }
      return Tensor<T>(nd, elem, shp, grad, device);
    }

    // auto b = Tensor<float>::like(a).randn();
    void randn(T up=(T)1, T down=(T)0, Distribution dist=NORMAL, uint32_t seed=0, bool grad=false, Device device=CPU) {
      storage = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, this->view->elem*sizeof(T)) );
      switch(dist){
        case NORMAL: storage = f32_generate_box_muller_normal_dist(this->view->elem, up, down, seed); break;
        case UNIFORM: storage = f32_generate_uniform_dist(this->view->elem, up, down, seed); break;
        case CHI_SQUARED: storage = f32_generate_chi_squared_dist(this->view->elem, up, down, seed); break;
      }
    }

    // by default not allocated into memory. value is decided when indexing
    // TODO: does this work for dims>2?
    static Tensor<T> eye(uint32_t size, uint32_t dims=2, bool resolved=false, Device device=CPU) {
      if(size < 2 || dims < 2) throw std::runtime_error("Cannot create an identity tensor of less then 2 dimensions or elements.");
      #ifdef FORCE_ALLOCATE
        T* nd = static_cast<T*>( aligned_alloc(DATA_ALIGNMENT, size*dims*sizeof(T)) );
        for(size_t i=0; i<size; i++) nd[i*size+i]=size; 
        uint32_t* shp = new uint32_t[dims];
        for(size_t i=0; i<dims; i++) shp[i]=size;
        Tensor<T> ret = Tensor<T>(nd, size*dims, shp, dims, false, device);
        ret.is_allocated=true;
        ret.is_initialized=true;
      #else
        T* nd = new T[1];
        uint32_t* shp = new uint32_t[dims];
        for(size_t i=0; i<dims; i++) shp[i]=size;
        Tensor<T> ret = Tensor<T>(nd, 1, shp, dims, false, device);
        ret.beye=true;
        ret.is_allocated=false;
        ret.is_initialized=true;
        return ret;
      #endif
    }

    // Move semantics 
    /*-------------------------------------------------*/

    // Shallow copy, no data pointer
    // auto b = Tensor<float>::like(a);
    static Tensor<T> like(Tensor<T>& from) {
      uint32_t shp* = new uint32_t[from->view->numdim];
      for(size_t i=0; i<from->view->numdim; i++) shp[i] = from->view->view[i];
      return Tensor<T>(shp, from->view->numdim, from->device);
    }

    // new view, same data
    Tensor<T> clone() {
      uint32_t shp* = new uint32_t[view->numdim];
      for(size_t i=0; i<view->numdim; i++) shp[i] = view->view[i];
      return Tensor<T>(&storage[0], disklen, &shp[0], view->numdim, bgrad, device);
    }

    // TODO maybe?
    // new view, new data
    Tensor<T> deepclone() {}


    // Indexing 
    /*-------------------------------------------------*/



    // Static data generation
    /*-------------------------------------------------*/

		static std::unique_ptr<float[]> f32_generate_uniform_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0, 
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

		static std::unique_ptr<float[]> f32_generate_chi_squared_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0) {
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
		static std::unique_ptr<float[]> f32_generate_box_muller_normal_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0) {
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

