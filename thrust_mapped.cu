#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include "mapped_allocator/mapped_allocator.hpp"

int main(void) {
    // generate random data on the host
    const int N = 100;
    thrust::host_vector<int> h_vec(N);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer to device and compute sum
    thrust::device_vector<int> d_vec = h_vec;
    int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

    std::vector<int, mapped_allocator<int> > h_mapped_vec(N); // host mapped memory vector
    int *d_ptr; // pointer to mapped device memory
    thrust::generate(h_mapped_vec.begin(), h_mapped_vec.end(), rand);
    cudaHostGetDevicePointer((void **)&d_ptr, (void *)&h_mapped_vec[0], 0);
    thrust::device_ptr<int> d_vec_ptr(d_ptr);
    thrust::sort(d_ptr, d_ptr + N);

    for(int i = 0; i < N; i++) {
        std::cout << d_ptr[i] << std::endl;
    }

    return 0;
}
