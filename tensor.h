#ifndef TENSOR
#define TENSOR

#include <limits.h>
#define TENSOR_MAX_DIM (2 << 15)
#define TENSOR_MAX_STORAGE_SIZE UINT_MAX 

#if defined(__AVX__) && (defined(__FMA__) || defined(__FMA4__))
  #define AVX_CPU_GEMMS
  #include "cpu_gemms.h"
#endif

#define DEBUG 3
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

#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <stdlib.h>
#include <stdexcept>
#include <initializer_list>

using std::size_t;


// TODO: Rename to tensorlib
namespace tensor {

typedef enum {
	SUCCESSFUL,
	INVALID_ARGUMENTS,
	MEMORY_ALLOCATION_ERROR,
	INVALID_NUMBER_OF_DIMENSIONS,
	GLOBAL_LIMIT_EXCEDED,
	UNEXPECTED_ERROR,
	NOT_IMPLEMENTED,
} op_ret;

typedef enum {
	RESHAPE,
	PERMUTE,
	EXPAND,
	SHRINK,
	FLIP,
	PAD
} MovementOPs;

template<typename T>
struct sized_array {
	std::shared_ptr<T[]> ptr = nullptr;
	size_t size = 0;
};


// VIEW
/*-------------------------------------------------------------------------------------------------------*/

struct View {

	std::shared_ptr<uint32_t[]> view = nullptr;
	std::shared_ptr<uint32_t[]> strides = nullptr;

	uint32_t ndim()  { return this->numdim; }
	uint64_t numel() { return this->elements; }
  uint64_t disksize() { return this->diskelem; }

	private:
		uint32_t numdim = 0;
		uint64_t elements = 0;
    uint64_t diskelem = 0;

		void restride() {
      uint32_t* nstr = new alignas(32) uint32_t[this->numdim];
      nstr[this->numdim-1] = 1;
			for(size_t i=this->numdim-1; i > 0; --i) nstr[i-1] = nstr[i]*view[i];
      this->strides = std::unique_ptr<uint32_t[]>(nstr);
		}

  public: 

	View(std::initializer_list<uint32_t> argview) 
		: numdim(argview.size())
	{
		if(argview.size() >= TENSOR_MAX_DIM) throw std::length_error("TENSOR_MAX_DIM Exceeded.");
    uint32_t* nv = new alignas(32) uint32_t[argview.size()];
		uint32_t i = 0; 
		uint64_t acc = 1;
		for(const auto& x : argview) {
			if(x <= 0) throw std::invalid_argument("View arguments cannot be equal or smaller than 0."); 
      nv[i++] = x;
			acc *= x;
		}
		if(acc > TENSOR_MAX_STORAGE_SIZE) throw std::length_error("TENSOR_MAX_STORAGE_SIZE Exceeded.");
    this->view = std::unique_ptr<uint32_t[]>(nv);
		this->elements = acc;
		this->diskelem = acc;
		this->restride();
	}

	View(std::shared_ptr<uint32_t[]>& argview, size_t &dim) 
		: numdim(dim)
	{
		if(dim >= TENSOR_MAX_DIM) throw std::length_error("TENSOR_MAX_DIM Exceeded.");
    uint32_t* nv = new alignas(32) uint32_t[dim];
		uint64_t acc = 1;
		for(size_t i=0; i<dim; i++) {
			if(argview[i] <= 0) throw std::invalid_argument("View arguments cannot be equal or smaller than 0."); 
			nv[i] = argview[i];
			acc *= argview[i];
		}
		if(acc > TENSOR_MAX_STORAGE_SIZE) throw std::length_error("TENSOR_MAX_STORAGE_SIZE Exceeded.");
    this->view = std::unique_ptr<uint32_t[]>(nv);
		this->elements = acc;
		this->diskelem = acc;
		this->restride();
	}

  // MOVEMENT OPS
	// NOTE: No changes are made if movement OPs fail.
/*-------------------------------------------------*/
	
	// NOTE: Allows for one inferred dim
  // NOTE: Because array is signed int, dimension number is half
	op_ret reshape(std::shared_ptr<int32_t[]> &argview, size_t &dim) {
		if(dim >= TENSOR_MAX_DIM) return GLOBAL_LIMIT_EXCEDED;
		uint64_t acc = 1;
		bool b_infer = false;
		size_t infer_idx = 0;
    uint32_t* ptr = new alignas(32) uint32_t[dim];
		for(size_t i=0; i < dim; i++) { 
      if(argview[i] < -1) return INVALID_ARGUMENTS;
      if(argview[i] == -1) {
        if(b_infer) return INVALID_ARGUMENTS;
        b_infer = true;
        infer_idx = i;
        //ptr[i]=0;
        continue;
      }
			ptr[i] = argview[i];
			acc *= argview[i]; 
		}
		if(b_infer) { 
			uint32_t inf = this->elements/acc;
			ptr[infer_idx] = inf;
			acc *= inf; 
		}
		if(acc != this->elements) return INVALID_ARGUMENTS;
		this->view = std::unique_ptr<uint32_t[]>(ptr); 
		this->numdim=dim;
		this->restride();
		return SUCCESSFUL;
	}


  // NOTE: using consecutive sum to not allow repeat dims
	op_ret permute(std::shared_ptr<uint32_t[]> &idxs, size_t &len) {
    uint32_t consum = ((len-1)*(len))/2;
    uint32_t consum_c = 0;
		if(len != this->numdim) return INVALID_NUMBER_OF_DIMENSIONS;
    uint32_t* newview = new alignas(32) uint32_t[len];
    uint32_t* newstrd = new alignas(32) uint32_t[len];
    uint32_t* oview = this->view.get();
    uint32_t* ostrd = this->strides.get();
		for(size_t i=0; i < len; i++) {
			if(idxs[i] >= this->numdim || idxs[i] < 0) return INVALID_ARGUMENTS;
			newview[i] = oview[idxs[i]];	
			newstrd[i] = ostrd[idxs[i]];
      consum_c += idxs[i];
		}
    if(consum_c != consum) return INVALID_ARGUMENTS;
    this->view = std::unique_ptr<uint32_t[]>(newview);
    this->strides = std::unique_ptr<uint32_t[]>(newstrd);
		return SUCCESSFUL;
	}

  // TODO: signed int32 restricts input values 
  // NOTE: -1 is a relative index to the next existing dimension in line, it's position is irrelevant
  // {128, 128} -> {5, 128, 69, 128, 5}
  // {128, 128} -> {5,  -1, 69, 128, 5}
  // {128, 128} -> {5, 128, 69,  -1, 5}  
  op_ret expand(std::shared_ptr<int32_t[]> &argview, size_t &len) {
    if(len < this->numdim) return INVALID_NUMBER_OF_DIMENSIONS;
    uint32_t hits = 0;
    uint64_t acc = 1;
    uint32_t* newview = new alignas(32) uint32_t[len];
    uint32_t* newstrd = new alignas(32) uint32_t[len];
    uint32_t* oview = this->view.get();
    uint32_t* ostrd = this->strides.get();
    for(size_t i=0; i<len; i++) {
      if(argview[i] < -1) return INVALID_ARGUMENTS;
      if(argview[i] == -1) {
        if(hits < this->numdim) {
          newview[i] = oview[hits];
          newstrd[i] = ostrd[hits];
          hits++;
          continue;
        } else return INVALID_ARGUMENTS;
      }
      newview[i] = argview[i];
      acc *= argview[i];
      if(oview[hits] == 1) hits++;
      if(argview[i] == oview[hits]) {
        newstrd[i] = (oview[hits] == 1) ? 0 : ostrd[hits];
        hits++;
      } else newstrd[i] = 0;
    }
    if(hits != this->numdim) return INVALID_ARGUMENTS;
    this->view = std::unique_ptr<uint32_t[]>(newview);
    this->strides = std::unique_ptr<uint32_t[]>(newstrd);
    this->elements = acc;
    this->numdim = len;
    return SUCCESSFUL;
  }

  op_ret strip() {
    uint32_t idx = 0;
    uint64_t acc = 1;
    uint32_t nndim = 0;
    for(size_t i=0; i<this->numdim; i++) if(this->strides[i] != 0) nndim++; // Shit code
    uint32_t* nview = new alignas(32) uint32_t[nndim];
    uint32_t* nstrd = new alignas(32) uint32_t[nndim];
    uint32_t* oview = this->view.get();
    uint32_t* ostrd = this->strides.get();
    for(size_t i=0; i<this->numdim; i++) {
      if(this->strides[i] != 0) {
        nview[idx] = oview[i]; 
        nstrd[idx] = ostrd[i];
        acc *= oview[i];
        idx++;
      }
    } 
    if(acc != this->diskelem) return INVALID_ARGUMENTS;
    this->view = std::unique_ptr<uint32_t[]>(nview);
    this->strides = std::unique_ptr<uint32_t[]>(nstrd);
    this->numdim = idx;
    this->elements = this->diskelem;
    return SUCCESSFUL;
  }

	// TODO: Implement
	op_ret shrink(std::shared_ptr<uint32_t[]> &argview, size_t &len) { return NOT_IMPLEMENTED; }
	op_ret flip(std::shared_ptr<uint32_t[]> &argview, size_t &len) { return NOT_IMPLEMENTED; }
	op_ret pad(std::shared_ptr<uint32_t[]> &argview, size_t &len) { return NOT_IMPLEMENTED; }

};


// TENSOR
/*-------------------------------------------------------------------------------------------------------*/

typedef enum {
	GPU,
	CPU
} Device;

typedef enum {
	UNIFORM, 
	NORMAL, 
	CHI_SQUARED,
} Distribution;

template<typename T = float>
class Tensor {

	std::shared_ptr<T[]> storage = nullptr;
  std::shared_ptr<float[]> grad = nullptr;
	std::shared_ptr<View> shape = nullptr;

	public : 
    bool bgrad = false;
		bool beye = false;
		bool bresolved = false;
		bool is_initialized=false;
  	Device device;

	public:

		// Virtual Tensors, contains no data
		Tensor(std::initializer_list<uint32_t> shp, Device device=CPU)
			: shape(std::make_unique<View>(View(shp))), device(device) {}

		Tensor(sized_array<uint32_t> shp, Device device=CPU)
			: device(device), shape(std::make_unique<View>(View(shp.ptr, shp.size))) {}


		// Data init
		Tensor(std::unique_ptr<T[]> &arr, size_t size, std::initializer_list<uint32_t> shape, Device device=CPU) 
			: storage(std::move(arr)), shape(std::make_unique<View>(View(shape))),
				device(device), is_initialized(true), bresolved(true)
		{
			if(size != this->shape->numel()) throw std::length_error("Shape does not match data.");
		};

		Tensor(std::initializer_list<T> arr, std::initializer_list<uint32_t> shape, Device device=CPU, bool is_fill=false)
			: shape(std::make_unique<View>(View(shape))), device(device), is_initialized(true), bresolved(true)
		{
			if(!is_fill && arr.size() != this->shape->numel()) throw std::length_error("Shape does not match data.");
			std::unique_ptr<T[]> narr = std::unique_ptr<T[]>(new T[arr.size()]);
			uint32_t i = 0;
			for(const auto& x : arr) { narr[i] = x; i++; }
			this->storage = std::move(narr);
		};


		// These are mostly for internal use
		Tensor(std::unique_ptr<T[]> &arr, uint64_t size, sized_array<uint32_t> shape, Device device=CPU)
			: storage(std::move(arr)), device(device), is_initialized(true), bresolved(true),
				shape(std::make_unique<View>(View(shape.ptr, shape.size))) {}


		// Constructor helpers
/*-------------------------------------------------*/

		
		// auto a = Tensor<>::eye(4096, 4);
		static Tensor<T> eye(uint32_t size, uint32_t dims=2, Device device=CPU) {
			if(size < 2 || dims < 2) throw std::runtime_error("Cannot create a 1 dim identity tensor.");
			auto ptr = std::make_unique<uint32_t[]>(dims);
			sized_array<uint32_t> shp { std::move(ptr), dims };
			for(size_t i=0; i<dims; i++) shp.ptr[i] = size; 
			auto ret = Tensor<T>(shp, device); 
			ret.beye = true;
			ret.is_initialized = true;
			return ret; 
		}
		
		// auto a = Tensor<>::fill({2048, 2048}, 69.f);
		static Tensor<T> fill(std::initializer_list<uint32_t> shp, T v, Device device=CPU) {
			auto ret = Tensor<T>({v}, shp, device, true);
			for(size_t i=0; i < ret.shape->ndim(); i++) { ret.shape->strides[i] = 0;	}
			return ret;
		}

		// internal
		static Tensor<T> fill(sized_array<uint32_t> shp, T v, Device device=CPU) {
			auto ret = Tensor<T>({v}, shp, device, true);
			for(size_t i=0; i < ret.shape->ndim(); i++) { ret.shape->strides[i] = 0;	}
			return ret;
		}

		// auto b = Tensor<>::like(a).fill(69.f);
		void fill(T v) {
			if(this->storage) throw std::runtime_error("Cannot fill initialized Tensor.");
			this->storage = std::unique_ptr<T[]>( new T[1]() );
			if(v!=0) this->storage[0]=v;
			std::shared_ptr<uint32_t[]> strd = std::make_unique<uint32_t[]>(this->shape->ndim());
			for(size_t i=0; i < this->shape->ndim(); i++) { strd[i] = 0;	}
			this->shape->strides = strd;
			this->is_initialized = true;
		}

		// auto a = Tensor<>::randn({4096, 4096}, 3.14, -3.14);
		static Tensor<T> randn(std::initializer_list<uint32_t> shp, T up=1.f, T down=0.f, 
										       uint32_t seed=0, Device device=CPU, Distribution dist=NORMAL) 
		{
			auto ret = Tensor<float>(shp, device);
			std::unique_ptr<float[]> data;
			switch(dist) {
				case UNIFORM:
					data = Tensor<>::f32_generate_uniform_distribution(ret.size(), up, down, seed);
					break;
				case NORMAL: 
					data = Tensor<>::f32_generate_box_muller_normal_distribution(ret.size(), up, down, seed);
				case CHI_SQUARED:
					data = Tensor<>::f32_generate_chi_squared_distribution(ret.size(), up, down, seed);
					break;
			}
			ret.set_data(data, ret.size());
			return ret;
		}

		// auto b = Tensor<>::like(a).randn(69.f, -69.f);
		void randn(T up=1.f, T down=0.f, uint32_t seed=0, Distribution dist=NORMAL) {
			std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[this->size()]);
			T range = std::abs(up)+std::abs(down);
			if(seed!=0) std::srand(seed);
			for(size_t i=0; i < this->size(); i++) { data[i] = down + (std::rand()/(RAND_MAX/(up-down)));	}
			this->set_data(data, this->size());
		}

		// auto a = Tensor<float>::arange(50);
		// TODO: Allow going backwards
		static Tensor<T> arange(T stop, T start=0, T step=1, Device device=CPU) {
			if(stop < start || step <= 0 || step >= stop) throw std::runtime_error("Invalid Arguments.");
			uint32_t size = (std::abs(stop)+std::abs(start))/step;
			std::unique_ptr<T[]> data = std::make_unique<T[]>(size);
			for(size_t i=0; i < size; i++) {
				data[i] = start + i*step;
			}
			return Tensor<T>(data, size, {size}, device);
		}

		// Move semantics
/*-------------------------------------------------*/
		
		// Shallow copy
		// Returns a virtual tensor of same shape and type as rhs, no data.
		// > auto b = Tensor<>::like(a).randn(); 
		// > auto b = Tensor<>::like(a).fill(0);
		static Tensor<T> like(Tensor<T> &rhs) {
			if(rhs.beye || !rhs.is_initialized || !rhs.bresolved) throw std::runtime_error("Invalid Tensor Argument in Tensor::like."	);
			sized_array<uint32_t> nshp;
			nshp.ptr = std::unique_ptr<uint32_t[]>(new uint32_t[rhs.ndim()]);
			nshp.size = rhs.ndim();
			std::shared_ptr<uint32_t[]> shp = rhs.view();
			for(size_t i=0; i<rhs.ndim(); i++) nshp.ptr[i] = shp[i];
			return Tensor<T>(nshp);
		}

		// Returns a tensor of same shape and a pointer to the rhs data.
		void operator=(Tensor<T> &rhs) {}

		// Apply arg function to each element in storage
		void exec(void (*f)(T)) {
			for(size_t i=0; i<this->disklen(); i++) this->storage[i] = f(this->storage[i]);
		}

    // Element and sub-tensor selection
/*-------------------------------------------------*/
		// TODO: Move no data 
		template<typename... Args>
		Tensor<T> operator()(Args... args) {
			if(!this->shape->ndim() || !this->is_initialized) throw std::runtime_error("Tensor has not been initialised");
			if(sizeof...(args) > this->shape->ndim() || sizeof...(args) < 0) throw std::runtime_error("Invalid arguments.");
			const std::initializer_list<uint32_t> tmp {args...}; 
			std::unique_ptr<uint32_t[]> idxs = std::make_unique<uint32_t[]>(tmp.size());
			uint32_t i=0;
			for(const auto& x : tmp) { idxs[i] = x; i++; }

			if(this->beye && !this->bresolved) {
				auto str = this->stride(idxs, tmp.size());
				uint32_t accum = 0;
				for(size_t i=0; i<this->shape->ndim(); i++) accum += this->shape->strides[i];
				if(str[0]==str[1]) {
					if(str[0]%accum==0) {
						return Tensor<T>({1}, {1}, this->device);
					} else {
						return Tensor<T>({0}, {1}, this->device);
					}
				} else {
					// Do modulo until you find the first match, then just iterate every accum elements.
					std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[str[1]-str[0]]());
					uint32_t idx=0;
					bool bfound = false;
					for(size_t i=str[0]; i=str[1]; i++) {
						if(i%accum==0) {
							bfound=true;
							idx = i;
							break;
						}
					}		
					if(bfound && idx < str[1]) {
						for(size_t i=idx; i>=str[1]; i+=accum){
							data[i-str[0]] = 1;
						}	
					}
					sized_array<uint32_t> shape;
					shape.ptr = std::make_unique<uint32_t[]>(this->shape->ndim()-tmp.size());
					shape.size = this->shape->ndim()-tmp.size();
					for(size_t i=tmp.size(); i < this->shape->ndim(); i++) {
						shape.ptr[i-tmp.size()] = this->shape->view[i];	
					}
					return Tensor<T>(data, str[i]-str[0], shape, this->device);
				}
			}
			return this->stride_and_create(idxs, tmp.size());
		}

		// Movement OPs
		// TODO: These might allow for unwanted changes to the data. Maybe clone?
/*-------------------------------------------------*/
		
		bool reshape(std::initializer_list<int32_t> nview) { return this->execute_movement_op(nview, RESHAPE); }
		bool permute(std::initializer_list<uint32_t> nview) { return this->execute_movement_op(nview, PERMUTE); }
		bool expand(std::initializer_list<uint32_t> nview) { return this->execute_movement_op(nview, EXPAND); }

		std::unique_ptr<uint32_t[]> stride(std::unique_ptr<uint32_t[]>& idxs, uint32_t len) {
			const uint64_t startidx = this->accumulate_strides(idxs, len);
			uint64_t endidx = startidx;
			if(len < this->shape->ndim()) endidx += this->shape->strides[len-1]; 
			auto ret = std::make_unique<uint32_t[]>(2);
			ret[0] = startidx;
			ret[1] = endidx;
			return ret;
		}


    // SGEMMS
/*-------------------------------------------------*/

    // naive mul for now 
    template<uint32_t rows, uint32_t cols, uint32_t in>
    static Tensor<T> dot(Tensor<T> &lhs, Tensor<T> &rhs) {
      if(lhs.ndim() == 2 && rhs.ndim() == 2) {
        sized_array<uint32_t> s {std::unique_ptr<uint32_t[]>(new uint32_t[2]), 2};
        s.ptr[0] = rows;
        s.ptr[1] = cols;
        std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[s.ptr[0]*s.ptr[1]]);
        Tensor<T> ret = Tensor<T>(data, s.ptr[0]*s.ptr[1], s);

#if DEBUG
        auto start = std::chrono::high_resolution_clock::now();
#endif
#ifdef AVX_CPU_GEMMS
        _m256_gemm<128, 128, rows, cols, in>(lhs.data().get(), rhs.data().get(), ret.data().get());
#else
        lhs.tgemm<64, 16, rows, cols, in>(lhs.data().get(), rhs.data().get(), ret.data().get());
#endif
        //lhs.tgemm<64, rows, cols, in>(lhs.data().get(), rhs.data().get(), ret.data().get());
#if DEBUG
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = end - start;
				std::cout << "<sgemm GFLOP=" << ((long)2*rows*cols*in)/1e9 << " runtime=" << (float)ms_double.count() << "ms  ";
				std::cout << ((long double)((long)2*rows*cols*in)/(ms_double.count()/1000))/1e9 << " GFLOPS load=" << 
								     bytes_to_str((rows*in+in*cols)*sizeof(float)) << ">" << std::endl;
#endif
        return ret;
      }
    }

    // auto a = Tensor<>::get_processor_information();
#ifdef AVX_CPU_GEMMS
    static uint32_t get_cpu_info( uint32_t op ) {
      uint32_t eax, ebx, ecx, edx; 
      _cpuid(op, eax, ebx, ecx, edx);
      return eax;
    }
#endif
    

    // Variables and member functions
/*-------------------------------------------------*/
		public: 

    // TODO: size and numel are the same
		std::shared_ptr<T[]> data() { return this->storage; }
		uint32_t ndim() { return this->shape->ndim(); }
		uint64_t size() { return this->shape->numel(); }
		uint64_t numel() { return this->shape->numel(); }
    uint64_t disklen() { return this->shape->disksize(); }
    op_ret strip() { return this->shape->strip(); }

		T item() {
			if(this->size() == 1) {
				return this->storage[0];
			} else {
				throw std::runtime_error("Call of .item() on Tensor with multiple elements.");
			}
		}

		void set_data(std::unique_ptr<T[]>& d, uint64_t len) {
			if(this->size() != len) throw std::runtime_error("New data does not match existing shape.");
			this->storage = std::move(d);
			this->is_initialized = true;
		}

		// TODO: Somehow protect this from changes
		// Maybe make shape a unique_ptr or smth

		std::shared_ptr<uint32_t[]> view() { 
			if(!this->shape->view) throw std::runtime_error("Tensor has not been initialized.");
			std::shared_ptr<uint32_t[]> ret = this->shape->view;
			return ret; 
		}

		std::shared_ptr<uint32_t[]> strides() {
			if(!this->shape->view) throw std::runtime_error("Tensor has not been initialized.");
			std::shared_ptr<uint32_t[]> ret = this->shape->strides;
			return ret;
		}



    // Random number generators
/*-------------------------------------------------*/

		// Random number generators
		static std::unique_ptr<float[]> f32_generate_uniform_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0, 
										                                                  bool bepsilon=false, float epsilon=0) 
		{
 			static std::mt19937 rng(std::random_device{}());
			if(seed!=0) rng.seed(seed);
			static std::uniform_real_distribution<> dist(down, up);
			std::unique_ptr<float[]> ret = std::unique_ptr<float[]>(new float[count]);
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
			std::unique_ptr<float[]> ret = std::unique_ptr<float[]>(new float[count]);
			for(size_t i=0; i<count; i++) { 
				float n = dist(rng);
				if(n >= down && n <= up) ret[i] = n;  
			}
			return ret;
		}

		// NOTE: If count is odd, it adds an extra element
		static std::unique_ptr<float[]> f32_generate_box_muller_normal_distribution(uint32_t count, float up=1.f, float down=0.f, double seed=0) {
			if(count % 2 != 0) count++; 
			constexpr float epsilon = std::numeric_limits<float>::epsilon();
			constexpr float two_pi = 2.0 * M_PI;
			std::unique_ptr<float[]> ret = std::unique_ptr<float[]>(new float[count]);
			auto u1 = Tensor<>::f32_generate_uniform_distribution(count/2, up, down, seed, true, epsilon);
			auto u2 = Tensor<>::f32_generate_uniform_distribution(count/2, up, down, seed);
			for(size_t i=0, j=0; i<count/2; i++, j+=2) {
				auto mag = std::sqrt(-2.0 * std::log(u1[i]));
				ret[j]   = mag * std::sin(two_pi * u2[i]);
				ret[j+1] = mag * std::cos(two_pi * u2[i]);
			}
			return ret;
		}


    // Memory OPs
/*-------------------------------------------------*/

    // NOTE: This requires tensor data to be alignas(32) in memory
    void dt() {
      if(this->ndim() == 2) {
        float* t = this->storage.get();
        float* tt = new alignas(32) float[this->disklen()];
        uint32_t* shp = this->view().get();
        this-> _8x8_transpose_ps<128>(t, tt, shp[0], shp[1]);
        this->storage = std::shared_ptr<T[]>(tt);
      } else {
        throw std::logic_error("multidimentional tensor data transpose not implemented. bad call to <tensor>.dt().");
      }
    }


/*-------------------------------------------------*/
	protected:

		Tensor<T> stride_and_create(std::unique_ptr<uint32_t[]>& idxs, uint32_t len) {
			const uint64_t startidx = this->accumulate_strides(idxs, len);
			if(len < this->shape->ndim()) {
				const uint64_t endidx = startidx + this->shape->strides[len-1]; 
				std::unique_ptr<T[]> data = std::make_unique<T[]>(endidx-startidx);
				for(size_t i=startidx; i<=endidx; i++) {
					data[i-startidx] = this->storage[i];
				}
				sized_array<uint32_t> shape;
				shape.ptr = std::make_unique<uint32_t[]>(this->shape->ndim()-len);
				shape.size = this->shape->ndim()-len;
				for(size_t i=len; i < this->shape->ndim(); i++) {
					shape.ptr[i-len] = this->shape->view[i];	
				}
				return Tensor<T>(data, endidx-startidx, shape);
			} else {
				return Tensor<T>({this->storage[startidx]}, {1});
			}
		}

		bool return_from_err(op_ret ret) {
			switch(ret) {
				case SUCCESSFUL:
					return 1;
				case INVALID_ARGUMENTS: throw std::invalid_argument("Invalid Arguments."); 
				case INVALID_NUMBER_OF_DIMENSIONS: throw std::invalid_argument("Invalid Dimensions."); 
				case GLOBAL_LIMIT_EXCEDED: throw std::length_error("Global tensor size restriction exceeded.");
				default:
					return 0;
			}
		}

		bool execute_movement_op(std::initializer_list<int32_t> nview, MovementOPs op) {
			if(op != RESHAPE && op != EXPAND) return 0;
			sized_array<int32_t> s { std::make_unique<int32_t[]>(nview.size()), nview.size() };
			uint32_t i = 0;
			for(const auto& x : nview) s.ptr[i++] = x;

			op_ret ret;
			switch(op) {
        case RESHAPE:
			    return this->return_from_err(this->shape->reshape(s.ptr, s.size));
          break;
        case EXPAND:
					return this->return_from_err(this->shape->expand(s.ptr, s.size));
					break;
      }
		}

		bool execute_movement_op(std::initializer_list<uint32_t> nview, MovementOPs op) {
			sized_array<uint32_t> shape { std::make_unique<uint32_t[]>(nview.size()), nview.size() };
			uint32_t i = 0;
			for(const auto& x : nview) shape.ptr[i++] = x;
			op_ret ret;
			switch(op) {
				case PERMUTE:
					ret = this->shape->permute(shape.ptr, shape.size);
					break;
				case SHRINK:
				case FLIP:
				case PAD:
					return 0; // Not implemented yet
			}
			return this->return_from_err(ret);
		}

		inline uint64_t accumulate_strides (std::unique_ptr<uint32_t[]>& arr, size_t len) {
			uint64_t acc = 0; 
			for(size_t i=0; i < len; i++) {
				if(arr[i] >= this->shape->view[i]) throw std::invalid_argument("index out of bounds.");
				acc += this->shape->strides[i]*arr[i];
			}
			return acc;
		}


// NON-AVX SGEMMS 
    template<int rows, int columns, int inners>
    inline void gemm(T* const lhs, T* const rhs, T* const result) {
      #pragma omp parallel for shared(result, lhs, rhs) default(none) collapse(2) num_threads(24)
      for(size_t row=0; row < rows; row++) {
        for(size_t in=0; in < inners; in++) {
          for(size_t col=0; col < columns; col++) {
            result[row*columns+col] += lhs[row*columns+in] * rhs[in*columns+col];
          }
        }
      }
    }

    template<int block, int tile, int rows, int columns, int inner>
    inline void tgemm(const float* lhs, const float* rhs, float* out) {
      #pragma omp parallel for shared(out, lhs, rhs) default(none) collapse(4) num_threads(24)
      for(size_t row_tile=0; row_tile < rows; row_tile += block) {
        for(size_t column_tile=0; column_tile < columns; column_tile += block) {
          for(size_t inner_tile=0; inner_tile < inner; inner_tile += tile) {
            for(size_t row=row_tile; row < row_tile+block; row++) {
              int i_tile_end = std::min<float>(inner, inner_tile+tile);
              for(size_t in=inner_tile; in<i_tile_end; in++) {
                for(size_t col=column_tile; col<column_tile+block; col++) {
                  out[row*columns+col] += lhs[row*inner+in] * rhs[in*columns+col];
                }
              }
            }
          }
        }
      }
    }


// MEMORY TRANSPOSE
    template <int block>
    inline void _8x8_transpose_ps(const float* from, float* to, int lda, int ldb) {
      #pragma omp parallel for shared(from, to, lda, ldb) default(none) collapse(2) num_threads(24)
      for(int i=0; i<lda; i+=block) {
        for(int j=0; j<ldb; j+=block) {
          int mk = std::min(i+block, lda);
          int ml = std::min(j+block, ldb);
          for(int k=i; k<mk; k+=8) {
            for(int l=j; l<ml; l+=8) {

#if defined(__AVX__) && defined(__FMA__)
              this->_t_load_8x8_ps(&from[k*lda+l], &to[l*ldb+k], lda, ldb);
#else
              throw std::logic_error("non-AVX data transpose not yet implemented. bad call to <tensor>.dt().");
#endif
            }
          }
        }
      }
    }

#if defined(__AVX__) && defined(__FMA__)
    static inline void _t_load_8x8_ps(const float* from, float* to, int lda, int ldb) {
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

      t0 = _mm256_unpacklo_ps(r0, r1);
      t1 = _mm256_unpackhi_ps(r0, r1);
      t2 = _mm256_unpacklo_ps(r2, r3);
      t3 = _mm256_unpackhi_ps(r2, r3);
      t4 = _mm256_unpacklo_ps(r4, r5);
      t5 = _mm256_unpackhi_ps(r4, r5);
      t6 = _mm256_unpacklo_ps(r6, r7);
      t7 = _mm256_unpackhi_ps(r6, r7);

      r0 = _mm256_shuffle_ps(t0, t2, 0x44);
      r1 = _mm256_shuffle_ps(t0, t2, 0xee);
      r2 = _mm256_shuffle_ps(t1, t3, 0x44);
      r3 = _mm256_shuffle_ps(t1, t3, 0xee);
      r4 = _mm256_shuffle_ps(t4, t6, 0x44);
      r5 = _mm256_shuffle_ps(t4, t6, 0xee);
      r6 = _mm256_shuffle_ps(t5, t7, 0x44);
      r7 = _mm256_shuffle_ps(t5, t7, 0xee);

      _mm256_store_ps( &to[0*ldb], r0);
      _mm256_store_ps( &to[1*ldb], r1);
      _mm256_store_ps( &to[2*ldb], r2);
      _mm256_store_ps( &to[3*ldb], r3);
      _mm256_store_ps( &to[4*ldb], r4);
      _mm256_store_ps( &to[5*ldb], r5);
      _mm256_store_ps( &to[6*ldb], r6);
      _mm256_store_ps( &to[7*ldb], r7);
    }
#endif 

};



// OUTPUT REPR 
/*-------------------------------------------------*/

template<typename T>
inline std::ostream& operator<<(std::ostream& outs, Tensor<T>& tensor) {
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
	repr += (tensor.device == 1) ? "CPU" : "GPU"; 
  if(tensor.is_initialized) {
    repr += ", type=";
    repr += typeid(tensor.data()[0]).name();
    repr += std::to_string(sizeof(tensor.data()[0])*8);
  }
  repr += ", grad="+std::to_string(tensor.bgrad);
  if(tensor.beye || !tensor.bresolved) {
    repr += ", disk=" + std::to_string(sizeof(tensor)) + " B";
  } else if (!tensor.is_initialized) {
    repr += ", disk=" + std::to_string(sizeof(tensor)) + " B";
  } else {
    repr += ", disk=" + bytes_to_str(tensor.disklen()*sizeof(tensor.data()[0]) + sizeof(tensor));
  }
  repr += (tensor.is_initialized) ? "" : ", is_initialized=false";
  repr += (!tensor.is_initialized) ? "" : (tensor.bresolved) ? "" : ", resolved=false";
	repr += ">";
	return outs << repr;
}

inline std::ostream& operator<<(std::ostream& outs, View& view) {
	std::string repr = "View[(";
	for(size_t i=0; i <= view.ndim()-1; i++) {
		repr += std::to_string(view.view[i]);
		repr += ", ";
	}
	repr += "), (";
	for(size_t i=0; i < view.ndim()-1; i++) {
		repr += std::to_string(view.strides[i]);
		repr += ", ";
	}
	uint64_t size = view.strides[0] * view.view[0] * 32;
	repr += "), disk: ";
	repr += std::to_string( size*1.25e-7);
	repr += " MB ]";

	return outs << repr;
}


} // namespace
#endif
