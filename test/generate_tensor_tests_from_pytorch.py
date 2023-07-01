
import torch

N = 4096
DIM = 4

def check_each_value_against_pytorch():
    f = open("test_tensor_indexing_with_pytorch.cc", "w")
    rep = """
#include <iostream>
#include <cassert>
#include "tensor.h"

using namespace tensor;

bool test_indexing_from_pytorch() {
"""
    rep += f"\tstd::unique_ptr<float[]> data = std::make_unique<float[]>({N});\n"
    rep += f"\tfor(size_t i=0; i < {N}; i++) {{ data[i]=i; }}\n"
    rep += f"\tTensor<float> a(data, {N}, {{2, 2, 2, 512}});\n\n"

    t = torch.Tensor(torch.arange(0, N))
    t = t.reshape(2, 2, 2, 512)

    for i in range(0, len(t)):
        for j in range(0, len(t[0])):
            for k in range(0, len(t[0, 0])):
                for l in range(0, len(t[0, 0, 0])):
                    rep += (f"\tassert( a({i}, {j}, {k}, {l}).data()[0] == {t[i, j, k, l]});  ")
                    rep += "\tstd::cout << '.';\n"
    rep += "};"
    rep += """
int main() {
    test_indexing_from_pytorch();
    return 0;
}
"""
    f.write(rep);



if __name__ == "__main__":
    check_each_value_against_pytorch()

