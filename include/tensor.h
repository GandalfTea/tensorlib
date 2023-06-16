
#ifndef TENSOR
#define TENSOR

#define TENSOR_MAX_DIM 2 << 15
#define TENSOR_MAX_STORAGE_SIZE 2 << 63 

#include <array>
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

template<uint32_t N>
struct View {
	std::array<uint32_t, N> view;
	std::array<uint32_t, N> strides;

	View() {};

	View(std::initializer_list<uint32_t> argview) {
		assert(argview.size() == N && !(argview.size() > TENSOR_MAX_DIM));
		uint8_t i = 0;
		for(const auto& x : argview) {
			this->view[i] = x;
			i++;
		}
		this->calculate_strides(this->view);
	}

	View(View<N>& v) {
		this->view = v.view;
		this->strides = v.strides;
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

	public:
  	Device device = GPU;

	public:
		Tensor(T arr[], const std::initializer_list<uint32_t> shape) 
			: storage_size(sizeof(arr)/sizeof(arr[0]))
		{ 
			for(size_t i=0; i < sizeof(arr)/sizeof(arr[0]); i++) {
				this->storage[i] = arr[i];
			}
			this->shape.reshape(shape);
		};

		std::array<uint32_t, M> get_shape() {
			return this->shape.view;
		}

		bool reshape(std::initializer_list<uint32_t> nview) {
			return 0;
		}

	private:
		uint64_t storage_size;
		std::array<T, N> storage;
		View<M> shape;
};


// uint32_t arr[] = { . . . };
// auto a = Tensor<uint32_t, arr.size()>( arr, {2, 2, -1} );
// auto a = Tensor<typeof arr, arr.size()>(arr, {2, 2, -1});


// OUTPUT REPR 

template<typename T, size_t N, size_t M>
inline std::ostream& operator<<(std::ostream& outs, Tensor<T, N, M>& tensor) {
	std::string repr = "<Tensor (";
	for(const auto x : tensor.get_shape()) {
		repr += std::to_string(x);
		repr += ", ";
	}
	repr += ") on ";
	repr += (tensor.device == 1) ? "CPU" : "GPU"; 
	repr += " with grad (0)>";
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
