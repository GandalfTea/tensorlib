#include "tensor.h"

class Conv2d {
  uint32_t* kernel_dims;
  Tensor<float>* weight, bias;
  uint32_t  kernel_dim, stride, padding, dilation, groups;

  Conv2d(uint32_t in_channels, uint32_t out_channels, std::initializer_list<uint32_t> kernel, 
         uint32_t stride=1, uint32_t dilation=1, uint32_t groups=1, bool bias=true) 
    : kernel_dim(kernel.size()), stride(stride), padding(padding), dilation(dilation), groups(groups)
  {
    if(kernel.size() != 2) throw std::runtime_error("Invalid Conv2d kernel dimensions.");
    size_t i=0, sum=1;
    kernel_dims = new uint32_t[2];
    for(uint32_t& x : kernel) { kernel_dims[i++] = x; sum*=x; }
    float kbount = std::sqrt(3.f)*std::sqrt(2.f / (1.f + 5.f)) / std::sqrt(sum);
    this->weight = new Tensor<float>({out_channels, (uint32_t)in_channels/groups, kernel_dims[0], kernel_dims[1]}).randn(-kbound, kbound, UNIFORM); // kaiming uniform
    if(bias) {
      float bound = std::sqrt(x);
      this->bias = new Tensor<float>({out_channels}).randn(-bound, bound, UNIFORM);
    }
  }

  Tensor<float> operator(Tensor<float> x) {
    x.pool({5, 5}).reshape({512, 1, 1, 1, 24, 24, 5, 5}).expand({512, 1, 1, 32, 24, 24, 5, 5}).permute({0, 1, 3, 4, 5, 2, 6, 7});
    auto ret = weight.reshape({1, 1, 32, 1, 1, 1, 5, 5}).dot(x).sum({-1, -2, -3}).reshape({512, 32, 24, 24});
    if(bias) ret.add(bias.reshape({1, -1, 1, 1})); 
    return ret;
  }
}

class Linear {
  Linear() { }
  Tensor<float>& operator(Tensor<float>& x) { }
}


class Model {
  Tensor<float>* layers; 

  Model() {
    layers = new Tensor<float>[14];
    layers[0]  = Conv2d(1, 32, 5);  layers[1] = Tensor<>::relu;
    layers[2]  = Conv2d(32, 32, 5); layers[3] = Tensor<>::relu;
    layers[4]  = BatchNorm2d(1, 32, 5); layers[5] = Tensor<>::relu;
    layers[6]  = Conv2d(32, 64, 3); layers[7] = Tensor<>::relu;
    layers[8]  = Conv2d(64, 64, 3); layers[9] = Tensor<>::relu;
    layers[10] = Tensor<>::flatten(1); layers[11] = Linear(576, 10); 
  }
}

int main(
  auto samples = x_train(Tensor<>::randint(512, x_train.shape()[0]));
)
