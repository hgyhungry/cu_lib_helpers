#ifndef CU_LIB_HELPERS_CUDA_HELPER
#define CU_LIB_HELPERS_CUDA_HELPER

#include <stddef.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <iostream>
#include <math.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"


#ifndef __NVCC__
#define __syncthreads()

#endif



#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__                             \
                << " of file " << __FILE__ << std::endl;                \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }



// I feel the logic about randomization is wrong....need to think about random seed carefully.

template<typename DType>
struct InitGen {
    uint seed;
    // DType mean;
    InitGen():seed(time(0)) {}
    InitGen(uint seed):seed(seed) {}
    DType gen_next() 
    {
        std::cerr << "Random generator in abstract type is not implemented. Use <int> or <float> specialization. \n";
        return (DType)0;
    }
};

template<>
struct InitGen<int>
{
    uint seed;
    InitGen() { seed = time(0); srand(seed); }
    InitGen(uint seed): seed(seed) { srand(seed); }
    int gen_next()
    {
        return ((rand() % 100) - 50);
    }
};

template<>
struct InitGen<float>
{
    uint seed; 
    InitGen() { seed = time(0); srand(seed); }
    InitGen(uint seed): seed(seed) { srand(seed); }
    float gen_next()
    {
        return (float)(rand() %1000 - 500)/1000;
    }
};

template<typename DType>
class CUArray {

private:

std::vector<DType> host_array;
DType *device_data = NULL;
int size_=0;

public:
void free_() 
{
    if (device_data != NULL)
    {
        CUDA_CHECK(cudaFree(device_data));
        device_data=NULL;
    }
}

CUArray(): size_(0), device_data(NULL) {}
CUArray(std::vector<DType>& data)
{
    size_ = data.size();
    host_array = data;
    device_data = NULL ;
}

~CUArray() { free_(); }

void init_random(int size)
{
    host_array = std::vector<DType>(size);

    InitGen<DType> generator;
    for (int i=0; i<size; i++)
        host_array[i] = generator.gen_next();

    size_ = host_array.size();
}

void init_zeros(int size)
{
    host_array = std::vector<DType>(size, (DType)0);
    size_ = host_array.size();
}
void assign(std::vector<DType>& data)
{
    size_ = data.size();
    host_array = data;
    
    if (device_data != NULL) {
        // already allocated
        free_();
        device_data = NULL;
    }
}

void sync_device() 
{
    free_();
    CUDA_CHECK(cudaMalloc((void**)&device_data, size_ * sizeof(DType)));
    CUDA_CHECK(cudaMemcpy(device_data, host_array.data(), size_ * sizeof(DType), cudaMemcpyHostToDevice));
}

std::vector<DType> sync_host()
{
    assert((size_ == host_array.size()));
    CUDA_CHECK(cudaMemcpy(host_array.data(), device_data, size_ * sizeof(DType), cudaMemcpyDeviceToHost));
    return host_array;
}

DType* get_device_data() { return device_data; }
int get_size() { return size_; }
DType* get_host_data() { return host_array.data(); }

std::vector<DType> &get_host_array() { return host_array; }
};

// credit to cub library.
struct GpuTimer
{
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GpuTimer() 
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start()
    {
        cudaEventRecord(startEvent, 0);
    }

    void stop()
    {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
    }

    float elapsed_msecs()
    {
        float elapsed;
        cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
        return elapsed;
    }
};

#endif // CU_LIB_HELPERS_CUDA_HELPER