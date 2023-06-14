
#ifndef TENSOR
#define TENSOR
#endif

#include <array>

using std::size_t;

struct View {
	uint32_t* view;
};

template<size_t N>
class ShapeTracker {
	public:
		struct View<N> view;
		size_t size;
};

template<typename T, size_t N>
class Tensor {
	public:
		Tensor(T* arr, std::initialiser_list<uint32_t> shape) 
			: storage(*arr)
		{
			int acc = 0;
			for(const auto& s: *shape) { acc *= *s; }
			if(N % acc == 0) this->shape(*shape)
			else {
				// RAISE ERROR
			}
		};

		//std::array<std::size_t, this->shape.view.size()> shape() {
		//	return this->shape.view;
		//}

	private:
		std::array<T, N> storage;
		View shape;
};


// uint32_t arr[] = { . . . };
// auto a = Tensor<uint32_t, arr.size()>( arr, {2, 2, 1} );
