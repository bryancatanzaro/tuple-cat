#include <thrust/tuple.h>

#include "tuple_cat.h"
#include <iostream>

struct A {
    int val;
    __host__ __device__ A(int v) : val(v) {}
};
struct B {
    int val;
    __host__ __device__ B(int v) : val(v) {}
};
struct C {
    int val;
    __host__ __device__ C(int v) : val(v) {}
};
struct D {
    int val;
    __host__ __device__ D(int v) : val(v) {}
};


int main() {
    A a(4);
    B b(2);
    C c(3);
    D d(1);
    
    typedef thrust::tuple<A, B> AB;
    AB ab = thrust::make_tuple(a,
                               b);
    typedef thrust::tuple<C, D> CD;
    CD cd = thrust::make_tuple(c,
                               d);
    
    
    typedef typename thrust::tuple_cat_type<
        AB, CD >::type concat_type;

    concat_type abcd = thrust::tuple_cat(ab, cd);

    std::cout << "Should be: 4 2 3 1. Result: ";
    
    std::cout <<
        thrust::get<0>(abcd).val << " " <<
        thrust::get<1>(abcd).val << " " <<
        thrust::get<2>(abcd).val << " " <<
        thrust::get<3>(abcd).val << std::endl;
}
