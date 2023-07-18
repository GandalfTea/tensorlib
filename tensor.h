
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


	uint32_t ndim() { return this->numdim; }

	uint64_t telem() { return this->total; }

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

template<typename T>
class Tensor {

	std::shared_ptr<T[]> storage = nullptr;
	std::shared_ptr<View> shape = nullptr;

	public : 
		bool is_initialized=false;
  	Device device;
		uint64_t size = 0;

	public:
		// Virtual Tensor, contains no data
		Tensor(std::initializer_list<uint32_t> shp, Device device=CPU)
			: shape(std::make_unique<View>(View(shape))), device(device) {}

		Tensor(std::unique_ptr<T[]> &arr, uint32_t size, std::initializer_list<uint32_t> shape, Device device=CPU) 
			: size(size), storage(std::move(arr)), shape(std::make_unique<View>(View(shape))),
				device(device), is_initialized(true)
		{
			if(this->shape->telem() != size) {
				throw std::runtime_error("Invalid Tensor Shape.");
			}
		};

		Tensor(std::initializer_list<T> arr, uint32_t size, std::initializer_list<uint32_t> shape, Device device=CPU)
			: size(size), shape(std::make_unique<View>(View(shape))), device(device), is_initialized(true)
		{
			std::unique_ptr<T[]> narr = std::make_unique<T[]>(arr.size());
			uint32_t i = 0;
			for(const auto& x : arr) { narr[i] = x; i++; }
			this->storage = std::move(narr);
			if(this->shape->telem() != size) {
				throw std::runtime_error("Invalid Tensor Shape.");
			}
		};

		// Mostly for internal use
		Tensor(std::unique_ptr<T[]> &arr, uint32_t size, sized_array<uint32_t> shape, Device device=CPU)
			: size(size), storage(std::move(arr)), device(device), is_initialized(true)
		{
			this->shape = std::make_unique<View>(View({size}));
			this->shape->reshape(shape.ptr, shape.size);
			if(this->shape->telem() != size) {
				throw std::runtime_error("Invalid Tensor Shape.");
			}
		}

		// Constructor helpers

		// Tensor<float> a({40, 40});
		// a.fill(0.f);
		void fill(T v) {
			if(this->storage) { throw std::runtime_error("Cannot fill initialized Tensor."); }
			std::unique_ptr<T[]> strg = std::make_unique<T[]>(1);
			strg[0]=v;
			this->storage = std::move(strg);
			std::shared_ptr<uint32_t[]> strd = std::make_unique<uint32_t[]>(this->shape->ndim());
			for(size_t i=0; i < this->shape->ndim(); i++) {
				strd[i] = 0;	
			}
			this->shape->strides = strd;
		}

		void eye() {}

		// Tensor<float> a = Tensor.randn({40, 40});
		static Tensor<T> randn(std::initializer_list<uint32_t> shp, T up=1.f, T down=0.f, 
										       uint32_t seed=0, Device device=CPU) 
		{
			uint64_t numel = 1;
			for(const auto& x : shp) numel *= x;
			std::unique_ptr<T[]> data = std::make_unique<T[]>(numel);
			uint32_t range = std::abs(up)+std::abs(down);
			if(seed!=0) std::srand(seed);
			for(size_t i=0; i < numel; i++) { data[i] = rand() % range - down;	}
			return Tensor<T>(data, numel, shp, device);
		}

		// Tensor<float> a = Tensor({40, 40});
		// a.randn();
		void randn(T up=1.f, T down=0.f, uint32_t seed=0) {
			std::unique_ptr<T[]> data = std::make_unique<T[]>(this->shape->telem());
			uint32_t range = std::abs(up)+std::abs(down);
			if(seed!=0) std::srand(seed);
			for(size_t i=0; i < this->shape->telem(); i++) { data[i] = rand() % range - down;	}
			this->storge = std::move(data);
		}

		// Tensor<float> a = Tensor.arange(50).reshape({25, 2});
		static Tensor<T> arange(T stop, T start=0, size_t step=1, Device device=CPU) {
			if(stop < start || step <= 0 || step >= stop) throw std::runtime_error("Invalid Arguments.");
			int32_t i=0;
			std::unique_ptr<T[]> data = std::make_unique<T[]>(stop/step);
			for(size_t f=start; f<stop/step; f+step) { data[i] = f; i++; }
			return Tensor<T>(data, i, {i}, device);
		}

		// Move semantics
		
		static void like() {}
		void operator=(Tensor<T>& rhs) {}

		// Movement OPs
		template<typename... Args>
		Tensor<T> operator()(Args... args) {
			if(!this->shape->ndim()) throw std::runtime_error("Tensor has not been initialised");
			if(sizeof...(args) > this->shape->ndim() || sizeof...(args) < 0) throw std::runtime_error("Invalid arguments.");
			const std::initializer_list<uint32_t> tmp {args...}; 

			const uint64_t startidx = this->accumulate(tmp);

			if(tmp.size() < this->shape->ndim()) {
				const uint64_t endidx = startidx + this->shape->strides[tmp.size()-1]; 
				std::unique_ptr<T[]> data = std::make_unique<T[]>(endidx-startidx);
				for(size_t i=startidx; i<=endidx; i++) {
					data[i-startidx] = this->storage[i];
				}

				sized_array<uint32_t> shape;
				shape.ptr = std::make_unique<uint32_t[]>(this->shape->ndim()-tmp.size());
				shape.size = this->shape->ndim()-tmp.size();

				for(size_t i=tmp.size(); i < this->shape->ndim(); i++) {
					shape.ptr[i-tmp.size()] = this->shape->view[i];	
				}
				Tensor<T> ret(data, endidx-startidx, shape);
				return ret;

			} else {
				std::unique_ptr<T[]> data = std::make_unique<T[]>(1);
				data[0] = this->storage[startidx];
				Tensor<T> ret(data, 1, {1});
				return ret;
			}
		}

		bool reshape(std::initializer_list<uint32_t> nview) { return this->execute_movement_op(nview, RESHAPE); }
		bool permute(std::initializer_list<uint32_t> nview) { return this->execute_movement_op(nview, PERMUTE); }
		bool expand(std::initializer_list<uint32_t> nview) { return this->execute_movement_op(nview, EXPAND); }


		// TODO: These might allow for unwanted changes to the data. Maybe clone?

		std::shared_ptr<T[]> data() { return this->storage; }
		uint32_t ndim() { return this->shape->ndim(); }

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

		uint64_t accumulate(const std::initializer_list<uint32_t> arr) {
			uint32_t i = 0;
			uint64_t acc = 0; 

			for(const auto x : arr) {
				if(x >= this->shape->view[i]) throw TensorException(INVALID_SHAPE_OPERATION, "Index out of bounds.");
				acc += this->shape->strides[i]*x;
				i++;
			}
			return acc;
		}
};



// OUTPUT REPR 

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

template<typename T>
inline std::ostream& operator<<(std::ostream& outs, Tensor<T>& tensor) {
	std::string repr = "<Tensor (";
	auto shape = tensor.get_shape();
	for(size_t i=0; i < tensor.ndim(); i++) {
		repr += std::to_string(shape[i]);
		repr += ", ";
	}
	repr += ") on ";
	repr += (tensor.get_device() == 1) ? "CPU" : "GPU"; 
	repr += ">";
	return outs << repr;
}




} // namespace
#endif
