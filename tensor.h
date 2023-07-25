
#ifndef TENSOR
#define TENSOR

#include <string>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <limits.h>
#include <stdexcept>
#include <initializer_list>

#define TENSOR_MAX_DIM (2 << 15)
#define TENSOR_MAX_STORAGE_SIZE UINT_MAX 
#define DEBUG 0

using std::size_t;

namespace tensor {


template<typename T>
struct sized_array {
	std::shared_ptr<T[]> ptr = nullptr;
	size_t size = 0;
};

typedef enum {
	SUCCESSFUL,
	INVALID_ARGUMENTS,
	MEMORY_ALLOCATION_ERROR,
	INVALID_DIMENSIONALITY,
	GLOBAL_LIMIT_EXCEDED,
	UNEXPECTED_ERROR,
} OPRet;

typedef enum {
	RESHAPE,
	PERMUTE,
	EXPAND,
} MovementOPs;

struct View {
	std::shared_ptr<uint32_t[]> view = nullptr;
	std::shared_ptr<uint32_t[]> strides = nullptr;

	View(std::initializer_list<uint32_t> argview) {
		if(argview.size() > TENSOR_MAX_DIM) throw std::runtime_error("GLOBAL_LIMIT_EXCEDED");
		this->numdim = argview.size();

		uint8_t i = 0;
		uint64_t sum = 1;
		this->view = std::make_unique<uint32_t[]>(argview.size());
		for(const auto& x : argview) {
			this->view[i] = x;
			sum *= x;
			i++;
		}
		if(sum > TENSOR_MAX_STORAGE_SIZE) {
			throw std::runtime_error("Number of elements in Tensor exceeds TENSOR_MAX_STORAGE_SIZE.");
		}
		this->total = sum;
		this->restride();
	}

	OPRet reshape(std::shared_ptr<uint32_t[]> &argview, size_t &newdim) {
		if(newdim >= TENSOR_MAX_DIM) return GLOBAL_LIMIT_EXCEDED;
		uint64_t product = 1;
		for(size_t i=0; i < newdim; i++) { 
			if(argview[i] == 0) return INVALID_ARGUMENTS;
			product *= argview[i]; 
		}
		if(product != this->total) return INVALID_ARGUMENTS;

		this->numdim = newdim;
		this->view = std::make_unique<uint32_t[]>(newdim);
		for(size_t i=0; i < newdim; i++) {
			this->view[i] = argview[i];
		}
		this->restride();
		return SUCCESSFUL;
	}

	OPRet permute(std::shared_ptr<uint32_t[]> &idxs, size_t &len) {
		if(len != this->numdim) return INVALID_DIMENSIONALITY;
		std::shared_ptr<uint32_t[]> newview = std::make_unique<uint32_t[]>(len);
		std::shared_ptr<uint32_t[]> newstrides = std::make_unique<uint32_t[]>(len);
		//TODO: Do not allow repeat dims
		for(size_t i=0; i < len; i++) {
			if(idxs[i] >= this->numdim) return INVALID_ARGUMENTS;
			newview[i] = this->view[idxs[i]];	
			newstrides[i] = this->strides[idxs[i]];
		}
		this->view = newview;
		this->strides = newstrides;
		return SUCCESSFUL;
	}

	// For now we don't support new dimensions.
	OPRet expand(std::shared_ptr<uint32_t[]> &argview, size_t &len) {
		if(len != this->numdim) return INVALID_DIMENSIONALITY;
		for(size_t i=0; i < len; i++) {
			if(argview[i] != this->view[i]) {
				if(this->view[i] == 1) {
					this->view[i] = argview[i];
					this->strides[i] = 0;
				} else {
					return INVALID_ARGUMENTS;
				}
			}
		}
		return SUCCESSFUL;
	}

	// TODO:  Implement SHRINK, FLIP and PAD

	uint32_t ndim()  { return this->numdim; }
	uint64_t numel() { return this->total; }

	private:
		uint32_t numdim = 0;
		uint64_t total = 0;

		OPRet restride() {
			this->strides = std::make_unique<uint32_t[]>(this->numdim);
			for(size_t i=0; i <= numdim; i++) { this->strides[i]=1; }
			for(size_t i=this->numdim; i > 0; i--) {
				if(i==this->numdim) continue;	
				this->strides[i-1] = this->strides[i] * view[i];
			}	
			return SUCCESSFUL;
		}
};

typedef enum {
	GPU,
	CPU
} Device;

typedef enum {
	INVALID_SHAPE_OPERATION,
	MEMORY_ALLOCATION_FAILURE,
} TensorError;

class TensorException : public std::exception {
	public:
		TensorError err;
		const char* msg;
		TensorException(TensorError err, const char* msg) : err(err), msg(msg) {}
};

template<typename T = float>
class Tensor {

	std::shared_ptr<T[]> storage = nullptr;
	std::shared_ptr<View> shape = nullptr;

	public : 
		bool beye = false;
		bool bresolved = false;
		bool is_initialized=false;
  	Device device;

	public:

		// Virtual Tensors, contains no data
		
		Tensor(std::initializer_list<uint32_t> shp, Device device=CPU)
			: shape(std::make_unique<View>(View(shp))), device(device) {}

		Tensor(sized_array<uint32_t> shp, Device device=CPU)
			: device(device) 
		{
				uint64_t numel=1;
				for(size_t i=0; i<shp.size; i++) numel *= shp.ptr[i];
				this->shape = std::make_unique<View>(View({numel}));
				this->shape->reshape(shp.ptr, shp.size);
		}

		// Data init

		Tensor(std::unique_ptr<T[]> &arr, size_t size, std::initializer_list<uint32_t> shape, Device device=CPU) 
			: storage(std::move(arr)), shape(std::make_unique<View>(View(shape))),
				device(device), is_initialized(true), bresolved(true)
		{
			if(size != this->shape->numel()) throw std::runtime_error("Shape does not match data.");
		};

		Tensor(std::initializer_list<T> arr, std::initializer_list<uint32_t> shape, Device device=CPU, bool is_fill=false)
			: shape(std::make_unique<View>(View(shape))), device(device), is_initialized(true), bresolved(true)
		{
			if(!is_fill) if(arr.size() != this->shape->numel()) throw std::runtime_error("Shape does not match data.");
			std::unique_ptr<T[]> narr = std::unique_ptr<T[]>(new T[arr.size()]);
			uint32_t i = 0;
			for(const auto& x : arr) { narr[i] = x; i++; }
			this->storage = std::move(narr);
		};

		// These are mostly for internal use

		Tensor(std::initializer_list<T> arr, sized_array<uint32_t> shp, Device device=CPU, bool is_fill=false)
			: device(device), is_initialized(true)
		{
			if(is_fill) {
				uint64_t numel=1;
				for(size_t i=0; i<shp.size; i++) numel *= shp.ptr[i];
				this->shape = std::make_unique<View>(View({numel}));
				switch(this->shape->reshape(shp.ptr, shp.size)) {
					case SUCCESSFUL: break;
					default: throw std::runtime_error("Error reshaping tensor.");
				};
				for(size_t i=0; i < this->shape->ndim(); i++) this->shape->strides[i] = 0;
			} else {
				this->shape = std::make_unique<View>(View({arr.size()}));
				this->shape->reshape(shp.ptr, shp.size);
				if(arr.size() != this->shape->numel()) throw std::runtime_error("Shape does not match data.");
			}
			std::unique_ptr<T[]> narr = std::unique_ptr<T[]>(new T[arr.size()]);
			uint32_t i = 0;
			for(const auto& x : arr) { narr[i] = x; i++; }
			this->storage = std::move(narr);
		};

		Tensor(std::unique_ptr<T[]> &arr, uint64_t size, sized_array<uint32_t> shape, Device device=CPU)
			: storage(std::move(arr)), device(device), is_initialized(true)
		{
			this->shape = std::make_unique<View>(View({size}));
			this->shape->reshape(shape.ptr, shape.size);
		}

		// Constructor helpers
		
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
										       uint32_t seed=0, Device device=CPU) 
		{
			auto ret = Tensor<T>(shp, device);
			std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[ret.size()]);
			T range = std::abs(up)+std::abs(down);
			if(seed!=0) std::srand(seed);
			for(size_t i=0; i < ret.size(); i++) { data[i] = down + (std::rand()/(RAND_MAX/(up-down)));	}
			ret.set_data(data, ret.size());
			return ret;
		}

		// auto b = Tensor<>::like(a).randn(69.f, -69.f);
		void randn(T up=1.f, T down=0.f, uint32_t seed=0) {
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
		
		static void like(Tensor rhs) {}
		void operator=(Tensor<T>& rhs) {}

		// Lambda OPs
		void exec() {}


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
		
		bool reshape(std::initializer_list<uint32_t> nview) { return this->execute_movement_op(nview, RESHAPE); }
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

		// TODO: These might allow for unwanted changes to the data. Maybe clone?

		public: 

		std::shared_ptr<T[]> data() { return this->storage; }
		uint32_t ndim() { return this->shape->ndim(); }
		uint64_t size() { return this->shape->numel(); }

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

		bool execute_movement_op(std::initializer_list<uint32_t> nview, MovementOPs op) {
			sized_array<uint32_t> shape;
			shape.size = nview.size();
			shape.ptr = std::make_unique<uint32_t[]>(nview.size());
			uint32_t i = 0;
			for(const auto& x : nview) {
				shape.ptr[i] = x;	
				i++;
			}
			OPRet ret;
			switch(op) {
				case RESHAPE:
					ret = this->shape->reshape(shape.ptr, shape.size);
					break;
				case PERMUTE:
					ret = this->shape->permute(shape.ptr, shape.size);
					break;
				case EXPAND:
					ret = this->shape->expand(shape.ptr, shape.size);
			}
			switch(ret) {
				case SUCCESSFUL:
					return 1;
				case INVALID_ARGUMENTS:
					throw TensorException(INVALID_SHAPE_OPERATION, "Invalid Arguments in function permute().");
					return 0;
				case INVALID_DIMENSIONALITY:
					throw TensorException(INVALID_SHAPE_OPERATION, "Invalid Dimensions given for function permute().");
					return 0;
				case GLOBAL_LIMIT_EXCEDED:
					throw TensorException(INVALID_SHAPE_OPERATION, "Global tensor size restriction exceeded in function reshape().");
					return 0;
				default:
					return 0;
			}
		}

		uint64_t accumulate_strides (std::unique_ptr<uint32_t[]>& arr, size_t len) {
			uint64_t acc = 0; 
			for(size_t i=0; i < len; i++) {
				if(arr[i] >= this->shape->view[i]) throw TensorException(INVALID_SHAPE_OPERATION, "Index out of bounds.");
				acc += this->shape->strides[i]*arr[i];
			}
			return acc;
		}
};



// OUTPUT REPR 

template<typename T>
inline std::ostream& operator<<(std::ostream& outs, Tensor<T>& tensor) {
	std::string repr = "<Tensor (";
	auto shape = tensor.view();
	for(size_t i=0; i < tensor.ndim(); i++) {
		repr += std::to_string(shape[i]);
		repr += ", ";
	}
	repr += ") on ";
	repr += (tensor.device == 1) ? "CPU" : "GPU"; 
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
