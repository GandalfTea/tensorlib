
#ifndef TENSOR
#define TENSOR

#define TENSOR_MAX_DIM 2 << 15
#define TENSOR_MAX_STORAGE_SIZE 2 << 63 

#include <array>
#include <memory>
#include <numeric>
#include <cassert>
#include <initializer_list>

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


template<typename T>
struct tensor_indexing_return {
	uint64_t size;
	std::unique_ptr<T[]> data;
};


template<uint32_t N>
struct View {
	std::array<uint32_t, N> view;
	std::array<uint32_t, N> strides;

	View() {
		this->view.fill(0);
		this->strides.fill(0);
	};

	View(std::initializer_list<uint32_t> argview) {
		assert(argview.size() == N && !(argview.size() > TENSOR_MAX_DIM));
		uint8_t i = 0;
		for(const auto& x : argview) {
			this->view[i] = x;
			i++;
		}
		this->restride(this->view);
	}

	void reshape(std::unique_ptr<uint32_t[]> &argview) {
		assert(sizeof(argview)/4 == N && !(sizeof(argview)/4 > TENSOR_MAX_DIM));
		for(size_t i=0; i < sizeof(argview)/4; i++) {
			this->view[i] = argview[i];
		}
		this->restride(this->view);
	}

	private:
		 void restride(std::array<uint32_t, N> view) {
			std::array<uint32_t, N> tmp;
			tmp.fill(1);
			for(size_t i=N; i > 0; i--) {
				if(i==N) continue;	
				tmp[i-1] = tmp[i] * view[i];
			}	
			this-> strides = tmp;
		}
};


template<uint32_t M>
class ShapeTracker {

	//friend class Tensor;
	/*
		[ ] store multiple views
		[ ] store original shape
		[ ] broadcast and permute
	*/
	public:
		struct View<M> view;
		size_t size;

	private:
		bool is_valid_view(std::initializer_list<uint32_t> shape) {
			uint32_t p = 1;
			for(const uint32_t& i : shape) { p *= i; }
			if(this->storage_size % p == 0) return 1;	
			else return 0;
		}
};


template<typename T, size_t N, size_t M>
class Tensor {

	std::array<T, N> storage;
	uint64_t storage_size;
  Device device = GPU;
	View<M> shape;
	bool bgrad;

	public:
		// This is for users for simpler Tensor initialization
		Tensor(T arr[N], const std::initializer_list<uint32_t> shape, bool grad = false) 
			: storage_size(N), bgrad(grad)
		{ 
			for(size_t i=0; i < N; i++) { this->storage[i] = arr[i]; }

			uint32_t i = 0;
			std::unique_ptr<uint32_t[]> newshp = std::make_unique<uint32_t[]>(shape.size());
			for(const auto x : shape) { newshp[i] = x; i++; }
			this->shape.reshape(newshp);
		};

		// This is used internally, as there is no way to transform a C array
		// into an initializer_list with variable number of elements
		Tensor(std::unique_ptr<T[]> arr, std::unique_ptr<uint32_t[]> shape, bool grad=false)
			: storage_size(N), bgrad(grad)
		{
			for(size_t i=0; i < N; i++) { this->storage[i] = arr[i]; }
			this->shape.reshape(shape);
		}

		// Indexing into a Tensor always returns another Tensor
		template<typename... Args>
		Tensor operator()(Args... args) {
			assert( sizeof...(args) <= M && sizeof...(args) > 0);
			const uint64_t nsize = shape.strides[sizeof...(args)-1]; 
			const std::initializer_list<uint32_t> tmp {args...}; 

			const uint64_t startidx = this->accumulate(tmp);

			if(tmp.size() < M) {
				const uint64_t endidx = startidx + this->shape.strides[tmp.size()-1]; 
				std::unique_ptr<T[]> data = std::make_unique<T[]>(endidx-startidx);
				for(size_t i=startidx; i<=endidx; i++) {
					data[i-startidx] = this->storage[i];
				}
				std::unique_ptr<uint32_t[]> new_dimm = std::make_unique<uint32_t[]>(M-tmp.size());
				for(size_t i=tmp.size(); i<M; i++) {
					new_dimm[i-tmp.size()] = this->shape.view[i];	
				}
				Tensor<T, nsize, sizeof(new_dimm)/sizeof(new_dimm[0])> ret(data, new_dimm);
				return ret;

			} else {
				std::unique_ptr<T[]> data = std::make_unique<T[]>(1);
				data[0] = this->storage[startidx];
				const Tensor<T, 1, 1> ret(data, {1});
				return ret;
			}
		}

		std::array<T, N> data() {
			return this->storage;
		}

		std::array<uint32_t, M> get_shape() { return this->shape.view; }
		Device get_device() { return this->device; }

		bool reshape(std::unique_ptr<uint32_t[]> &nview) {
			this->shape.reshape(nview);
			return 0;
		}

	private:
		constexpr uint64_t accumulate(const std::initializer_list<uint32_t> arr) {
			uint32_t i = 0;
			uint64_t acc = 0; 
			for(const auto& x : arr) {
				assert(x <= this->shape.view[i]-1);
				acc += this->shape.strides[i]*x;
				i++;
			}
			return acc;
		}
};



// OUTPUT REPR 

template<typename T, size_t N, size_t M>
inline std::ostream& operator<<(std::ostream& outs, Tensor<T, N, M>& tensor) {
	std::string repr = "<Tensor (";
	for(const auto x : tensor.get_shape()) {
		repr += std::to_string(x);
		repr += ", ";
	}
	repr += ") on ";
	repr += (tensor.get_device() == 1) ? "CPU" : "GPU"; 
	repr += " with grad (0)>";
	return outs << repr;
}


template<typename T, size_t N>
inline std::ostream& operator<<(std::ostream& outs, std::array<T, N> arr) {
	std::string repr = "";
	for(const auto x : arr) {
		repr += std::to_string(x);
		repr += " ";
	}
	return outs << repr;
}


template<uint32_t M>
inline std::ostream& operator<<(std::ostream& outs, View<M>& view) {
	std::string repr = "View[(";
	for(const auto x : view.view) {
		repr += std::to_string(x);
		repr += ", ";
	}
	repr += "), (";
	for(const auto x : view.strides) {
		repr += std::to_string(x);
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
