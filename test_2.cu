#include <thrust/detail/tuple_cat.h>
#include <iostream>
#include <cassert>

int main() {
    int x = 1;
    int y = 2;

    thrust::tuple<int&, int&> a = thrust::tie(x, y);

    int z = 3;
    int w = 4;
    thrust::tuple<int&, int&> b = thrust::tie(z, w);

    std::cout << x << " " << y << " " << z << " " << w << std::endl;

    thrust::get<1>(a) = 5;

    assert(y == 5);

    std::cout << x << " " << y << " " << z << " " << w << std::endl;

    thrust::tuple<int&, int&, int&, int&> c = thrust::tuple_cat(a, b);

    thrust::get<2>(c) = 6;

    assert(z == 6);

    std::cout << x << " " << y << " " << z << " " << w << std::endl;

}
