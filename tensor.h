
#ifndef TENSOR
#define TENSOR

#include <array>
#include <limits.h>
#include <memory>
#include <numeric>
#include <cassert>
#include <initializer_list>

#define TENSOR_MAX_DIM (2 << 15)
#define TENSOR_MAX_STORAGE_SIZE UINT_MAX 
#define DEBUG 0

// debug
#include <iostream>

using std::size_t;

namespace tensor {

enum MovementOps {
	RESHAPE,
	PERMUTE,
	EXPAND,
	PAD,
	SHRINK,
	STRIDE
};

enum Device {
	GPU,
	CPU
};

struct tensor_shape_array {
	std::shared_ptr<uint32_t[]> ptr = nullptr;
	size_t size = 0;
};


struct View {
	std::shared_ptr<uint32_t[]> view = nullptr;
	std::shared_ptr<uint32_t[]> strides = nullptr;

	View(std::initializer_list<uint32_t> argview, uint64_t t=0) {
		assert(argview.size() <= TENSOR_MAX_DIM);
		assert(t <= TENSOR_MAX_STORAGE_SIZE);
		this->numdim = argview.size();
		this->total = t;

		uint8_t i = 0;
		this->view = std::make_unique<uint32_t[]>(argview.size());
		for(const auto& x : argview) {
			this->view[i] = x;
			i++;
		}
		this->restride();
	}

	void reshape(std::initializer_list<uint32_t> argview) {
		assert(!(argview.size() > TENSOR_MAX_DIM)); 
		uint64_t product = 1;
		for(const auto& x : argview) { product *= x; }
		assert(product == this->total);

		this->view = std::make_unique<uint32_t[]>(argview.size());
		uint32_t i = 0;
		for(const auto& x : argview) {
			this->view[i] = x;
			i++;
		}
		this->restride();
	}

	void reshape(std::shared_ptr<uint32_t[]> &argview, size_t &newdim) {
		assert(newdim < TENSOR_MAX_DIM);
		uint64_t product = 1;
		for(size_t i=0; i < newdim; i++) { product *= argview[i]; }
		assert(product == this->total);

		this->view = std::make_unique<uint32_t[]>(newdim);
		for(size_t i=0; i < newdim; i++) {
			this->view[i] = argview[i];
		}
		this->restride();
	}

	uint32_t ndim() {
		return this->numdim;
	}

	private:
		uint32_t numdim = 0;
		uint64_t total = 0;

		void restride() {
			this->strides = std::make_unique<uint32_t[]>(this->numdim);
			for(size_t i=0; i <= numdim; i++) { this->strides[i]=1; }
			for(size_t i=this->numdim; i > 0; i--) {
				if(i==this->numdim) continue;	
				this->strides[i-1] = this->strides[i] * view[i];
			}	
		}
};

inline std::ostream& operator<<(std::ostream& outs, View& view) {
	std::string repr = "View[(";
	for(size_t i=0; i <= view.ndim()-1; i++) {
		repr += std::to_string(view.view[i]);
		repr += ", ";
	}
	repr += "), (";
	for(size_t i=0; i <= view.ndim()-1; i++) {
		repr += std::to_string(view.strides[i]);
		repr += ", ";
	}
	uint64_t size = view.strides[0] * view.view[0] * 32;
	repr += "), disk: ";
	repr += std::to_string( size*1.25e-7);
	repr += " MB ]";

	return outs << repr;
}


template<uint32_t M>
class ShapeTracker {

	//friend class Tensor;
	/*
		[ ] store multiple views
		[ ] store original shape
		[ ] broadcast and permute
	*/
	public:
		struct View view;
		size_t size;

	private:
		bool is_valid_view(std::initializer_list<uint32_t> shape) {
			uint32_t p = 1;
			for(const uint32_t& i : shape) { p *= i; }
			if(this->storage_size % p == 0) return 1;	
			else return 0;
		}
};


template<typename T>
class Tensor {

	std::shared_ptr<T[]> storage = nullptr;
	std::shared_ptr<View> shape = nullptr;
	std::unique_ptr<Tensor<T>> grad = nullptr;

  Device device;
	uint64_t size = 0;
	bool bgrad;

	public:
		Tensor(std::unique_ptr<T[]> &arr, uint32_t size, std::initializer_list<uint32_t> shape, bool grad=false, Device device=GPU) 
			: size(size), storage(std::move(arr)), shape(std::make_unique<View>(View(shape, size))),
				bgrad(grad), device(device)
		{};

		Tensor(std::initializer_list<T> &arr, uint32_t size, std::initializer_list<uint32_t> shape, bool grad=false, Device device=GPU)
			: size(size), shape(std::make_unique<View>(View(shape, size))), bgrad(grad), device(device)
		{
			std::unique_ptr<T> narr = std::make_unique<T>(arr.size());
			uint32_t i = 0;
			for(const auto& x : arr) {
				narr[i] = x;
				i++;
			}
			this->storage = std::move(narr);
		};
			

		// This is mostly used internally, as there is no way to transform 
		// a C array into an initializer_list with variable number of elements
		Tensor(std::unique_ptr<T[]> &arr, uint32_t size, tensor_shape_array shape, bool grad=false, Device device=GPU)
			: size(size), storage(std::move(arr)), bgrad(grad), device(device)
		{
			this->shape = std::make_unique<View>(View({1, size}, size));
			this->shape->reshape(shape.ptr, shape.size);
		}


		// Indexing into a Tensor always returns another Tensor
		// TODO: shape.ndim could be nullptr
		template<typename... Args>
		Tensor<T> operator()(Args... args) {
			assert(this->shape->ndim());
			assert(sizeof...(args) <= this->shape->ndim() && sizeof...(args) > 0);
			const std::initializer_list<uint32_t> tmp {args...}; 

			const uint64_t startidx = this->accumulate(tmp);

			if(tmp.size() < this->shape->ndim()) {
				const uint64_t endidx = startidx + this->shape->strides[tmp.size()-1]; 
				std::unique_ptr<T[]> data = std::make_unique<T[]>(endidx-startidx);
				for(size_t i=startidx; i<=endidx; i++) {
					data[i-startidx] = this->storage[i];
				}

				tensor_shape_array shape;
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


		// TODO: This might allow for unwanted changes to the data. Maybe clone?
		std::unique_ptr<T[]> data() { return this->storage; }
		std::shared_ptr<uint32_t[]> get_shape() { 
			assert(!!this->shape->view);
			std::shared_ptr<uint32_t[]> ret = this->shape->view;
			return ret; 
		}
		uint32_t ndim() {
			return this->shape->ndim();
		}

		Device get_device() { return this->device; }

		bool reshape(std::initializer_list<uint32_t> nview) {
			tensor_shape_array shape;
			shape.size = nview.size();
			shape.ptr = std::make_unique<uint32_t[]>(nview.size());
			uint32_t i = 0;
			for(const auto& x : nview) {
				shape.ptr[i] = x;	
				i++;
			}
			this->shape->reshape(shape.ptr, shape.size);
			return 0;
		}

	protected:

		uint64_t accumulate(const std::initializer_list<uint32_t> arr) {
			uint32_t i = 0;
			uint64_t acc = 0; 

			for(const auto x : arr) {
				assert(x <= this->shape->view[i]);
				acc += this->shape->strides[i]*x;
				i++;
			}
			return acc;
		}
};



// OUTPUT REPR 

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
	repr += " with grad (0)>";
	return outs << repr;
}




} // namespace
#endif
