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

    //Test concatenating empty tuples.
    thrust::tuple<> x;
    //Empty with empty
    x = tuple_cat(x, x);
    //Empty in front
    concat_type y = tuple_cat(x, abcd);
    //Empty in back
    y = tuple_cat(abcd, x);

    //Test concatenating up to maximum tuple size
    typedef thrust::tuple<int, int, int, int, int, int> iiiiii;
    iiiiii six_i = thrust::make_tuple(1,2,3,4,5,6);
    thrust::tuple_cat(abcd, six_i);
    
#ifdef ERROR
    //This should static assert because the resulting tuple is too large
    thrust::tuple_cat(six_i, six_i);
#endif
}
