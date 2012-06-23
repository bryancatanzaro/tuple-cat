tuple-cat
=========
Provides `thrust::tuple_cat`, which concatenates two `thrust::tuple`s.
Since `thrust::tuple`s are limited to 10 elements, the resulting tuple type
must not have more than 10 elements.