
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
		this->calculate_strides(this->view);
	}

	void reshape(std::initializer_list<uint32_t> argview) {
		assert(argview.size() == N && !(argview.size() > TENSOR_MAX_DIM));
		uint8_t i = 0;
		for(const auto& x : argview) {
			this->view[i] = x;
			i++;
		}
		this->calculate_strides(this->view);
	}

	private:
		 void calculate_strides(std::array<uint32_t, N> view) {
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
		Tensor(T arr[N], const std::initializer_list<uint32_t> shape, bool grad = false) 
			: storage_size(N), bgrad(grad)
		{ 
			for(size_t i=0; i < N; i++) { this->storage[i] = arr[i]; }
			this->shape.reshape(shape);
		};



		template<typename... Args>
		tensor_indexing_return<T> operator()(Args... args) {
			assert( sizeof...(args) <= M && sizeof...(args) > 0);
			std::initializer_list<uint32_t> tmp = std::initializer_list<uint32_t>{args...}; 

			uint32_t i=0;
			std::array<uint32_t, M> idxs;
			idxs.fill(0);
			for(const auto& x : tmp) {
				assert(x <= this->shape.view[i]-1);
				idxs[i] = this->shape.strides[i]*x;
				i++;
			}

			if(tmp.size() < M) {
				tensor_indexing_return<T> ret;
				const uint64_t startidx = std::accumulate(std::begin(idxs), std::end(idxs), 0);
				const uint64_t endidx = startidx + this->shape.strides[tmp.size()-1]; 

				ret.data = std::make_unique<T[]>(endidx-startidx);
				for(size_t i=startidx; i<=endidx; i++) {
					ret.data[i-startidx] = this->storage[i];
				}
				ret.size = endidx-startidx+1; 
				return ret;

			} else {
				tensor_indexing_return<T> ret;
				ret.size = 1; 
				std::cout << "Returning one value." << std::endl;
				const uint64_t idx = std::accumulate(std::begin(idxs), std::end(idxs), 0);
				ret.data = std::make_unique<T[]>(1);
				ret.data[0] = this->storage[idx];

				return ret;
			}
		}



		std::array<T, N> data() {
			return this->storage;
		}

		std::array<uint32_t, M> get_shape() { return this->shape.view; }
		Device get_device() { return this->device; }

		bool reshape(std::initializer_list<uint32_t> nview) {
			// TODO: Check if work lol
			this->shape.reshape(nview);
			return 0;
		}

	private:
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
