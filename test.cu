#include <thrust/tuple.h>
#include "tuple_cat.h"
#include <iostream>
#include <typeinfo>
#include <cassert>

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
    
    
    typedef typename thrust::tuple_cat_result<
        AB, CD >::type concat_type;

    assert(typeid(concat_type) == typeid(thrust::tuple<A,B,C,D>));

    concat_type abcd = thrust::tuple_cat(ab, cd);
    std::cout << "Should be: 4 2 3 1. Result: ";
    
    std::cout <<
        thrust::get<0>(abcd).val << " " <<
        thrust::get<1>(abcd).val << " " <<
        thrust::get<2>(abcd).val << " " <<
        thrust::get<3>(abcd).val << std::endl;

    assert(4 == thrust::get<0>(abcd).val);
    assert(2 == thrust::get<1>(abcd).val);
    assert(3 == thrust::get<2>(abcd).val);
    assert(1 == thrust::get<3>(abcd).val);
        
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
    typedef thrust::tuple<A, B, C, D, int, int, int, int, int, int> maximum_tuple;
    maximum_tuple m = thrust::tuple_cat(abcd, six_i);

    //Concat empties with maximum
    m = tuple_cat(x, m);
    m = tuple_cat(m, x);

    thrust::tuple_cat(thrust::tie(a.val, b.val), thrust::tie(c.val, d.val)) =
        thrust::make_tuple(5, 2, 3, 6);

    std::cout << "Should be: 5 2 3 6. Result: ";
    std::cout << a.val << " " << b.val << " " << c.val << " " << d.val << std::endl;

    assert(a.val == 5);
    assert(b.val == 2);
    assert(c.val == 3);
    assert(d.val == 6);

    abcd = thrust::tuple_cat(thrust::tuple<>(), ab, thrust::tuple<>(), thrust::tuple<>(), cd);

    six_i = thrust::tuple_cat(thrust::make_pair(3, 2),
                              thrust::make_pair(4, 5),
                              thrust::make_pair(6, 7));
#ifdef ERROR
    //This should static assert because the resulting tuple is too large
    thrust::tuple_cat(six_i, six_i);
#endif
}
