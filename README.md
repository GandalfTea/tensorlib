
Lightweight header-only tensor library for C++17 and later.

&nbsp;

### Initialisation
Create a virtual tensor and allocate the memory:
```c++
Tensor<float> a = Tensor<float>({1024, 1024}).allocate();
```
or fill with special values:
```c++
Tensor<float> b = Tensor<float>({1024, 1024}).randn(); // 0-1, box-muller normal
Tensor<float> c = Tensor<float>({1024, 1024}).fill(0.f);
Tensor<float> d = Tensor<float>({1024, 1024}).eye();
```
Copy existing tensors:
```c++
Tensor<float> e = Tensor<>::like(a); // returns virtual tensor with no memory
Tensor<float> f = a.copy();          // new tensor that shares memory with a
Tensor<float> g = a.clone();         // new tensor that copied the memory of a
```
```c++
Tensor<float> h = Tensor<>::like(a).eye();    // identity tensor identical to a
Tensor<float> i = Tensor<>::like(a).fill(0.f) // 0-full tensor identical to a
```
All the memory allocated by the library is aligned according to the macro `MEMORY_ALIGNMENT` defaulted to 32 bytes for AVX-256, unless the size of the block is smaller then the alignment requirement.

<!--

&nbsp;

### Neural networks
```c++
#include "tensor.h"

using tensorlib::Tensor;
using std::sqrt;

class LinearNet {
  Tensor<float> w1, b1, w2, b2;

  LinearNet() {
    w1 = Tensor<float>({784, 128}, true).randn();
    b1 = Tensor<float>({128}).randn(sqrt(-1/128), sqrt(1/128), UNIFORM);
    w2 = Tensor<float>({128, 10}, true).randn();
    b2 = Tensor<float>({10}).randn(sqrt(-1/128), sqrt(1/128), UNIFORM);
  }

  operator(Tensor<float> x) {
    x.dot(w1).add(b1).relu().dot(w2).add(b2);
  }
}
```
-->

&nbsp;

### Data initialisation

Allocate the data using the included `alloc<T>` function. This ensures the memory block is aligned correctly:
```c++
float* data = alloc<float>(1024*1024);
Tensor<float> a = Tensor<float>(data, 1024*1024, {1024, 1024});
```
If you must allocate the memory yourself, use `aligned_alloc` or `malloc` if the number of items is small. Never use `new[]`, because garbage collection is handled with `free()` and the memory will leak.
```c++
float* data = static_cast<float*>( aligned_alloc(MEMORY_ALIGNMENT, 1024*1024*sizeof(float)) );
Tensor<float> b = Tensor<float>(data, 1024*1024, {1024/2, 2, 1024});
```
If the macro `FORCE_ALIGNMENT` is defined, all input arrays will be copied into new aligned memory blocks. You are free to use whatever array allocation you want:
```c++
#define FORCE_ALIGNMENT
float* data[1024*1024];
Tensor<float> c = Tensor<float>(data, 1024*1024, {1024, 1024});
```

&nbsp;

Elements can also be entered manually:
```c++
Tensor<float> a = Tensor<float>({5}, {1});
Tensor<float> b = Tensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
```

&nbsp;

### Helpers and Getters
```c++
Tensor<float> a = Tensor<float>({2, 1024, 2048});
```
* `a.ndim()` returns number of dimensions.
* `a.numel()` returns the number of elements in the tensor.
* `a.device()` returns the device on which the memory is stored.
* `a.memsize()` returns the number of elements allocated in memory.      
* `a.has_grad()` returns a `bool` that tracks if the tensor keeps a gradient.
* `a.is_initialized()` returns a `bool` that tracks if underlying tensor data exists.       
* `a.is_allocated()` returns a `bool` that tracks if all the tensor values exist in memory.
* `a.is_eye()` returns a `bool` that tracks if the tensor is an indentity tensor.
* `a.is_sub()` returns a `bool` that tracks if the tensor is a subtensor.   
* `a.storage()` returns a `const T*` to the underlying tensor data.
* `a.shape()` returns a `const uint32_t*` to the tensor view data.
* `a.strides()` returns a `const uint32_t*` to the tensor stride data.

&nbsp;

### Construction helpers
Help construct special tensors. Throw if tensor data already exists.

&nbsp;

* `fill` : fill a tensor with one value.
```c++
auto a = Tensor<float>::fill({2048, 2048}, 1.0);
auto b = Tensor<float>({2048, 2048}).fill(0.0);
auto c = Tensor<>::like(a).fill(1.f); // Shallow copy and fill
```
This only stores one value in memory and redirects to it given all correct indices. Because the view does not correspond to contiguous memory, data can only be accessed using `operator()` and `bresolved` will return `false`. Manually accessing the `data()` array will yield random values or errors.

&nbsp;

* `arange` : fill a tensor with evenly spaced values in a given interval.
```c++
static Tensor<T> arange(T stop, T start=0, T step=1, Device device=CPU) {
```
```c++
auto a = Tensor<float>::arange(50);
auto b = Tensor<float>::arange(50, 10);
auto c = Tensor<float>::arange(50, 10, 5);
```
Stores all resulting values in memory, `bresolved` is `true`.

&nbsp;


* `randn` : fill a tensor with randomly generated values that fit a given distribution.
      
Distributions can be either `NORMAL`, `UNIFORM` or `CHI_SQUARED`.

```c++
void randn(T up=1.f, T down=0.f, uint32_t seed=0, Distribution dist=NORMAL);
static Tensor<T> randn(std::initializer_list<uint32_t> shp, T up=1.f, T down=0.f,
                       uint32_t seed=0, Device device=CPU, Distribution dist=NORMAL);
```
```c++
auto a = Tensor<float>::randn({2048, 2048}, 3.14, -3.14);
auto b = Tensor<>::like(a).randn(3.14); // Shallow copy and fill with randn()
```
Stores all resulting values in memory, `bresolved` is `true`.     
       
You can also randomly generate arrays of the same distributions:
```c++
static std::unique_ptr<float[]> f32_generate_uniform_distribution(uint32_t count, float up=1.f,
                           float down=0.f, double seed=0, bool bepsilon=false, float epsilon=0);

static std::unique_ptr<float[]> f32_generate_chi_squared_distribution(uint32_t count,
                            float up=1.f, float down=0.f, double seed=0);

static std::unique_ptr<float[]> f32_generate_box_muller_normal_distribution(uint32_t count,
                           float up=1.f, float down=0.f, double seed=0);
```
```c++
auto a = Tensor<>::f32_generate_uniform_distribution(2048*2048, 3.14, -3.14);
auto b = Tensor<>::f32_generate_chi_squared_distribution(2048*2048, 3.14, -3.14);
auto c = Tensor<>::f32_generate_box_muller_normal_distribution(2048*2048, 3.14, -3.14);
```

&nbsp;


* `eye`: creates an identity tensor.
```c++
static Tensor<T> eye(uint32_t size, uint32_t dims=2, Device device=CPU);
```
```c++
auto a = Tensor<>::eye(4096, 4);
auto b = Tensor<>::like(a).eye(); // Not yet implemented
```
Values are generated when indexing, `bresolved` is `false`.

&nbsp;

### Virtual tensors     
Tensors that have no underlying data.
```c++
auto a = Tensor<float>({2048, 2048});
```
```c++
sized_array<uint32_t> s { std::unique_ptr<uint32_t[]>(new uint32_t[2]()), 2};
s.ptr[0] = 2;
s.ptr[1] = 3;
Tensor<float> b(s);
```

&nbsp;


### Movement Operations
They perform changes to the `view` and `stride` of the tensor, but make no changes to the underlying data.


&nbsp;

* `reshape`: change the `view` of the tensor without changing the number of elements inside.
```c++
auto a = Tensor<float>(data, 2048*2048, {2048, 2048});
a.reshape({2, 1024, 2048});
a.reshape({1024, 2, 1024, 2});
```

&nbsp;

* `permute`: change the order of the dimensions inside of the tensor.
```c++
auto a = Tensor<float>(data, 2048*2048, {1024, 2, 1024, 2});
a.permute({1, 3, 0, 2}) // {2, 2, 1024, 1024}
```
