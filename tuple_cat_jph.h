#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace detail
{


template<int i, typename Tuple>
  struct tuple_element_or_null
    : eval_if<
        i < tuple_size<Tuple>::value,
        tuple_element<i,Tuple>,
        identity_<null_type>
      >
{};


template<typename Tuple, typename T>
struct tuple_prepend_result
{
  typedef thrust::tuple<
    T,
    typename tuple_element_or_null<0,Tuple>::type,
    typename tuple_element_or_null<1,Tuple>::type,
    typename tuple_element_or_null<2,Tuple>::type,
    typename tuple_element_or_null<3,Tuple>::type,
    typename tuple_element_or_null<4,Tuple>::type,
    typename tuple_element_or_null<5,Tuple>::type,
    typename tuple_element_or_null<6,Tuple>::type,
    typename tuple_element_or_null<7,Tuple>::type,
    typename tuple_element_or_null<8,Tuple>::type
  > type;
};

template<typename Tuple>
struct tuple_prepend_result<Tuple,thrust::null_type>
{
  typedef Tuple type;
};

template<int i, typename Tuple>
inline __host__ __device__
typename lazy_enable_if<
  i < tuple_size<Tuple>::value,
  tuple_element<i,Tuple>
>::type
get_or_null(const Tuple &t)
{
  return thrust::get<i>(t);
}

template<int i, typename Tuple>
inline __host__ __device__
typename enable_if<
  i >= tuple_size<Tuple>::value,
  thrust::null_type
>::type
get_or_null(const Tuple &)
{
  return thrust::null_type();
}

template<typename Tuple, typename T>
inline __host__ __device__
typename tuple_prepend_result<Tuple,T>::type
tuple_prepend(const Tuple &t, const T &x)
{
  typedef typename tuple_prepend_result<Tuple,T>::type result_type;
  return result_type(x, get_or_null<0>(t), get_or_null<1>(t), get_or_null<2>(t), get_or_null<3>(t), get_or_null<4>(t), get_or_null<5>(t), get_or_null<6>(t), get_or_null<7>(t), get_or_null<8>(t));
}

template<typename Tuple>
inline __host__ __device__
typename tuple_prepend_result<Tuple,thrust::null_type>::type
tuple_prepend(const Tuple &t, thrust::null_type)
{
  return t;
}

} // end detail

template<typename Tuple1,                      typename Tuple2 = thrust::null_type, typename Tuple3 = thrust::null_type,
         typename Tuple4  = thrust::null_type, typename Tuple5 = thrust::null_type, typename Tuple6 = thrust::null_type,
         typename Tuple7  = thrust::null_type, typename Tuple8 = thrust::null_type, typename Tuple9 = thrust::null_type,
         typename Tuple10 = thrust::null_type>
class tuple_cat_result;

template<typename Tuple1, typename Tuple2>
class tuple_cat_result<Tuple1,Tuple2>
{
  private:
    typedef typename detail::tuple_prepend_result<
      Tuple2,
      typename detail::tuple_element_or_null<9,Tuple1>::type
    >::type type9;

    typedef typename detail::tuple_prepend_result<
      type9,
      typename detail::tuple_element_or_null<8,Tuple1>::type
    >::type type8;

    typedef typename detail::tuple_prepend_result<
      type8,
      typename detail::tuple_element_or_null<7,Tuple1>::type
    >::type type7;

    typedef typename detail::tuple_prepend_result<
      type7,
      typename detail::tuple_element_or_null<6,Tuple1>::type
    >::type type6;

    typedef typename detail::tuple_prepend_result<
      type6,
      typename detail::tuple_element_or_null<5,Tuple1>::type
    >::type type5;

    typedef typename detail::tuple_prepend_result<
      type5,
      typename detail::tuple_element_or_null<4,Tuple1>::type
    >::type type4;

    typedef typename detail::tuple_prepend_result<
      type4,
      typename detail::tuple_element_or_null<3,Tuple1>::type
    >::type type3;

    typedef typename detail::tuple_prepend_result<
      type3,
      typename detail::tuple_element_or_null<2,Tuple1>::type
    >::type type2;
    
    typedef typename detail::tuple_prepend_result<
      type2,
      typename detail::tuple_element_or_null<1,Tuple1>::type
    >::type type1;

  public:
    typedef typename detail::tuple_prepend_result<
      type1,
      typename detail::tuple_element_or_null<0,Tuple1>::type
    >::type type;
};

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4,
         typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8,
         typename Tuple9, typename Tuple10>
struct tuple_cat_result
{
  private:
    typedef typename tuple_cat_result<
      Tuple9, Tuple10
    >::type tuple9_10;
  
    typedef typename tuple_cat_result<
      Tuple8, tuple9_10
    >::type tuple8_10;
  
    typedef typename tuple_cat_result<
      Tuple7, tuple8_10
    >::type tuple7_10;
  
    typedef typename tuple_cat_result<
      Tuple6, tuple7_10
    >::type tuple6_10;
  
    typedef typename tuple_cat_result<
      Tuple5, tuple6_10
    >::type tuple5_10;
  
    typedef typename tuple_cat_result<
      Tuple4, tuple5_10
    >::type tuple4_10;
  
    typedef typename tuple_cat_result<
      Tuple3, tuple4_10
    >::type tuple3_10;
  
    typedef typename tuple_cat_result<
      Tuple2, tuple3_10
    >::type tuple2_10;

  public:
    typedef typename tuple_cat_result<
      Tuple1, tuple2_10
    >::type type;
};

template<typename Tuple1, typename Tuple2>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2)
{
  using detail::tuple_prepend;
  using detail::get_or_null;

  // take t2 and prepend t1's elements
  return tuple_prepend(
      tuple_prepend(
        tuple_prepend(
          tuple_prepend(
            tuple_prepend(
              tuple_prepend(
                tuple_prepend(
                  tuple_prepend(
                    tuple_prepend(
                      tuple_prepend(
                        t2,
                        get_or_null<9>(t1)
                      ),
                      get_or_null<8>(t1)
                    ),
                    get_or_null<7>(t1)
                  ),
                  get_or_null<6>(t1)
                ),
                get_or_null<5>(t1)
              ),
              get_or_null<4>(t1)
            ),
            get_or_null<3>(t1)
          ),
          get_or_null<2>(t1)
        ),
        get_or_null<1>(t1)
      ),
      get_or_null<0>(t1)
  );
}

// XXX perhaps there's a smarter way to accumulate
template<typename Tuple1, typename Tuple2, typename Tuple3>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3)
{
  return tuple_cat(t1, tuple_cat(t2,t3));
}

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4)
{
  return tuple_cat(t1, t2, tuple_cat(t3,t4));
}

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5)
{
  return tuple_cat(t1, t2, t3, tuple_cat(t4,t5));
}

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6)
{
  return tuple_cat(t1, t2, t3, t4, tuple_cat(t5,t6));
}

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7)
{
  return tuple_cat(t1, t2, t3, t4, t5, tuple_cat(t6,t7));
}

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7, const Tuple8 &t8)
{
  return tuple_cat(t1, t2, t3, t4, t5, t6, tuple_cat(t7,t8));
}

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8, typename Tuple9>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8,Tuple9>::type
tuple_cat(const Tuple2 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7, const Tuple8 &t8, const Tuple9 &t9)
{
  return tuple_cat(t1, t2, t3, t4, t5, t6, t7, tuple_cat(t8,t9));
}

template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8, typename Tuple9, typename Tuple10>
inline __host__ __device__
typename tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8,Tuple9,Tuple10>::type
tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7, const Tuple8 &t8, const Tuple9 &t9, const Tuple10 &t10)
{
  return tuple_cat(t1, t2, t3, t4, t5, t6, t7, t8, tuple_cat(t9,t10));
}

} // end thrust

