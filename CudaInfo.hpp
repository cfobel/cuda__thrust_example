/**
   @file CudaInfo.hpp
   @brief Header file containing definitions for CudaInfo class.
  
   System:         Star+ CUDA (C++)
   Language: C++
  
   (c) Copyright Christian Fobel 2010
  
   Author: Christian Fobel
   E-Mail: cfobel@uoguelph.ca
  
   Compiler Options: -lboost_regex -lboost_filesystem

   Requires: Boost C++ Libraries >= 1.40.0 (http://www.boost.org/doc/)
*/  

#ifndef ___CUDA_INFO_HPP___
#define ___CUDA_INFO_HPP___

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <stdint.h>


namespace cuda_info {

using namespace std;

typedef void(*kernel_ptr)();

class CudaInfo {
    int deviceCount;
    int driverVersion;
    int runtimeVersion;

    void get_driver_info();
public:
    CudaInfo() {
        get_driver_info();
    }

    string get_info();
    int get_device_count() const { return deviceCount; }
    int get_driver_version() const { return driverVersion; }
    int get_runtime_version() const { return runtimeVersion; }
};


class CudaKernel {
    kernel_ptr kernel;
    cudaFuncAttributes func_attr;

    void checked_get_function_attributes();
public:
    CudaKernel(kernel_ptr kernel) : kernel(kernel) {
        checked_get_function_attributes();
    }
    string get_kernel_info();
    int get_kernel_shared_mem() const { return func_attr.sharedSizeBytes; }
    int get_kernel_reg_count() const { return func_attr.numRegs; }
    int get_kernel_max_threads_per_block() const { return func_attr.maxThreadsPerBlock; }
};


class CudaDevice {
    int id;
    string device_name;
    int major;
    int minor;

    size_t totalGlobalMem;
    int multiProcessorCount;
    int totalConstantMemory;
    int sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int blockDim[3];
    int gridDim[3];
    int memPitch;
    int textureAlign;
    int clockRate;
    int gpuOverlap;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;

    void get_device_attributes();
public:
    CudaDevice(int id) : id(id) {
        get_device_attributes();
    }

    void set_device();
    string get_device_info();

    int get_multi_processor_count() const { return multiProcessorCount; }
    int get_max_shared_memory() const { return sharedMemPerBlock; }
    int get_max_shared_memory(cudaFuncCache pref) const { 
        if(pref == cudaFuncCachePreferL1 && major >= 2) {
            // If L1 cache is preferred, return 16K
            return 16 * (1 << 10);
        }
        return sharedMemPerBlock; 
    }
    int get_major() const { return major; }
    int get_minor() const { return minor; }
    int get_max_threads_per_block() const { return maxThreadsPerBlock; }
    int get_max_regs_per_block() const { return regsPerBlock; }
};

}

#endif
