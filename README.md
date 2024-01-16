
Lightweight header-only tensor library for C++17 and later.

&nbsp;

### Data initialization

&nbsp;

####  `std::unique_ptr<T[]>`      
Array constructor with elements of type `T`. Default template type is `float`. This requires the total number of elements in the tensor as second argument `size`. Shape is either a `std::initializer_list<uint32_t>` or a `tensor::sized_array<uint32_t>`. Data allocation is left to the user.
```c++
Tensor( std::unique_ptr<T[]> &arr, size_t size, std::initializer_list<uint32_t> shape, Device device=CPU );
Tensor( std::unique_ptr<T[]> &arr, size_t size, sized_array<uint32_t> shape, Device device=CPU );
```
&nbsp;

* `std::initializer_list<uint32_t>`
```c++
std::unique_ptr<float[]> data = std::unique_ptr<float[]>( new float[2048*2048]() );
auto a = Tensor<float>(data, 2048*2048, {2, 1024, 2048});

// NOTE: template type can be inferred. This is equivalent:     
auto a = Tensor<>(data, 2048*2048, {2, 1024, 2048});
```

&nbsp;

* `tensor::sized_array<T>` is a proprietary data type for programatically sizing a tensor of known dimention number but unknown dimension size. It explicitly tracks the number of dimensions in `size`.
```c++
template<typename T>
struct tensor::sized_array {
  std::shared_ptr<T[]> ptr = nullptr;
  size_t size = 0;
};
```
```c++
sized_array<uint32_t> s;
s.ptr = std::unique_ptr<uint32_t[]>(new uint32_t[2]());
s.ptr[0] = 2;
s.ptr[1] = 3;
s.size = 2;
std::unique_ptr<float[]> data = std::unique_ptr<float[]>( new float[2*3]() );
auto b = Tensor<b>(data, 2*3, s);
// OR 
auto a = Tensor<float>({1, 2, 3, 4, 5}, s);
```

&nbsp;

#### `std::initializer_list<T>` manual element entry.       
```c++
Tensor(std::initializer_list<T> arr, std::initializer_list<uint32_t> shape, Device device=CPU, bool is_fill=false)
```
```c++
auto a = Tensor<float>({1, 2, 3, 4, 5}, {2, 3});
```
&nbsp;

### Construction helpers

&nbsp;

* `fill` : fill a tensor with one value. Throws if tensor data already exists.
```c++
static Tensor<T> fill(std::initializer_list<uint32_t> shp, T value, Device device=CPU);
void fill(T value);
```
```c++
auto a = Tensor<float>::fill({2048, 2048}, 1.f);
auto b = Tensor<float>({2048, 2048}).fill(0.0);
auto c = Tensor<>::like(a).fill(1.f); // Shallow copy and fill
```
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
You can also randomly generate arrays of the same distributions:
```c++
static std::unique_ptr<float[]> f32_generate_uniform_distribution(uint32_t count, float up=1.f,
                           float down=0.f, double seed=0, bool bepsilon=false, float epsilon=0);

static std::unique_ptr<float[]> f32_generate_chi_squared_distribution(uint32_t count, float up=1.f,
                           float down=0.f, double seed=0);

static std::unique_ptr<float[]> f32_generate_box_muller_normal_distribution(uint32_t count, float up=1.f,
                           float down=0.f, double seed=0);
```
```c++
auto a = Tensor<>::f32_generate_uniform_distribution(2048*2048, 3.14, -3.14);
auto b = Tensor<>::f32_generate_chi_squared_distribution(2048*2048, 3.14, -3.14);
auto c = Tensor<>::f32_generate_box_muller_normal_distribution(2048*2048, 3.14, -3.14);
```
&nbsp;


* `eye`: create identity tensor.
```c++
static Tensor<T> eye(uint32_t size, uint32_t dims=2, Device device=CPU);
```
```c++
auto a = Tensor<>::eye(4096, 4);
auto b = Tensor<>::like(a).eye(); // Not yet implemented
```

&nbsp;

### Virtual tensors (no data)

`std::initializer_list`     
```c++
auto a = Tensor<float>({2048, 2048});
```
`tensor::sized_array`     
```c++
sized_array<uint32_t> s;
s.ptr = std::unique_ptr<uint32_t[]>(new uint32_t[2]());
s.ptr[0] = 2048;
s.ptr[1] = 2048;
s.size = 2;
Tensor<float> a(s);
```
