/**
   @file CudaInfo.cpp
   @brief Code for gathering device attributes.
  
   System:         Star+ CUDA (C++)
   Language: C++
  
   (c) Copyright Christian Fobel 2010
  
   Author: Christian Fobel
   E-Mail: cfobel@uoguelph.ca
  
   Compiler Options: -lboost_regex -lboost_filesystem

   Requires: Boost C++ Libraries >= 1.40.0 (http://www.boost.org/doc/)
*/  
#include <iostream>
#include <cassert>
#include <boost/format.hpp>
#include <boost/algorithm/string/join.hpp>
#include "CudaInfo.hpp"
#ifdef LOGGING
#include "Logging.hpp"
extern Logger logger;
#endif

using namespace std;

//! A convenience shortcut to the Boost.Format function
/*! @see http://www.boost.org/doc/libs/1_40_0/libs/format/doc/format.html */
#define _ boost::format

namespace cuda_info {

void CudaInfo::get_driver_info() {
	deviceCount = 0;

	CUresult err = cuInit(0);
    cuDeviceGetCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	assert(deviceCount != 0);

    #if CUDART_VERSION >= 2020
        // Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
    #endif
}


string CudaInfo::get_info() {
    vector<string> summary; 

    summary.push_back("CUDA info:");
    summary.push_back((_("  Device Count:          %d") 
                        % deviceCount).str());
    summary.push_back((_("  CUDA Driver Version:   %d.%d") 
                            % (driverVersion / 1000)
                            % (driverVersion % 100)).str()); 
    summary.push_back((_("  CUDA Runtime Version:  %d.%d") 
                            % (runtimeVersion / 1000)
                            % (runtimeVersion % 100)).str()); 

    return boost::algorithm::join(summary, "\n");
}


string CudaKernel::get_kernel_info() {
    vector<string> summary; 

    summary.push_back("Kernel attributes:");
    summary.push_back((_("  constSizeBytes:         %d") % func_attr.constSizeBytes).str());
    summary.push_back((_("  localSizeBytes:         %d") % func_attr.localSizeBytes).str());
    summary.push_back((_("  maxThreadsPerBlock:     %d") % func_attr.maxThreadsPerBlock).str());
    summary.push_back((_("  numRegs:                %d") % func_attr.numRegs).str());
    summary.push_back((_("  sharedSizeBytes:        %d") % func_attr.sharedSizeBytes).str());;

    return boost::algorithm::join(summary, "\n");
}


string CudaDevice::get_device_info() {
    vector<string> summary; 

    summary.push_back((_("Device name:    %s") % device_name).str());
    summary.push_back((_("  CUDA Cap. Major revision #:   %d") % major).str());
    summary.push_back((_("  CUDA Cap. Minor revision #:   %d") % minor).str());
    summary.push_back((_("  totalGlobalMem                %d bytes") % totalGlobalMem).str());
    summary.push_back((_("  multiProcessorCount           %d") % multiProcessorCount).str());
    summary.push_back((_("  totalConstantMemory           %d") % totalConstantMemory).str());
    summary.push_back((_("  sharedMemPerBlock             %d") % sharedMemPerBlock).str());
    summary.push_back((_("  regsPerBlock                  %d") % regsPerBlock).str());
    summary.push_back((_("  warpSize                      %d") % warpSize).str());
    summary.push_back((_("  maxThreadsPerBlock            %d") % maxThreadsPerBlock).str());
    summary.push_back((_("  blockDim[3]                   %d x %d x %d") 
            % blockDim[0] % blockDim[1] % blockDim[2]).str());
    summary.push_back((_("  gridDim[3]                    %d x %d x %d") 
            % gridDim[0] % gridDim[1] % gridDim[2]).str());
    summary.push_back((_("  memPitch                      %d bytes") % memPitch).str());
    summary.push_back((_("  textureAlign                  %d bytes") % textureAlign).str());
    summary.push_back((_("  clockRate                     %.2f GHz") % (clockRate * 1e-6)).str());
    summary.push_back((_("  gpuOverlap                    %d") % gpuOverlap).str());
    summary.push_back((_("  integrated                    %d") % integrated).str());
    summary.push_back((_("  canMapHostMemory              %d") % canMapHostMemory).str());
    summary.push_back((_("  computeMode                   %d") % computeMode).str());
    summary.push_back((_("  kernelExecTimeoutEnabled      %d") % kernelExecTimeoutEnabled).str());

    return boost::algorithm::join(summary, "\n");
}


void CudaKernel::checked_get_function_attributes() {
    cudaError_t error = cudaFuncGetAttributes(&func_attr,
                            (char*)kernel);

    if(error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + std::string(cudaGetErrorString(error)));
    }
} // end checked_get_function_attributes()


void CudaDevice::get_device_attributes() {
	char deviceName[256];

	CUresult err = cuInit(0);
    cuDeviceGetName(deviceName, 256, id);
    device_name = string(deviceName);

    // This function call returns 9999 for both major & minor fields, 
    // if no CUDA capable devices are present
    cuDeviceComputeCapability(&major, &minor, id);

    cuDeviceTotalMem(&totalGlobalMem, id);
#if CUDA_VERSION >= 2000
    cuDeviceGetAttribute( &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, id );
#endif
    cuDeviceGetAttribute( &totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, id );
    cuDeviceGetAttribute( &sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, id );
    cuDeviceGetAttribute( &regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, id );
    cuDeviceGetAttribute( &warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, id );
    cuDeviceGetAttribute( &maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, id );
    cuDeviceGetAttribute( &blockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, id );
    cuDeviceGetAttribute( &blockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, id );
    cuDeviceGetAttribute( &blockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, id );
    cuDeviceGetAttribute( &gridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, id );
    cuDeviceGetAttribute( &gridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, id );
    cuDeviceGetAttribute( &gridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, id );
    cuDeviceGetAttribute( &memPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, id );
    cuDeviceGetAttribute( &textureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, id );
    cuDeviceGetAttribute( &clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, id );
#if CUDA_VERSION >= 2000
    cuDeviceGetAttribute( &gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, id );
#endif

#if CUDA_VERSION >= 2020
    cuDeviceGetAttribute( &kernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, id );
    cuDeviceGetAttribute( &integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, id );
    cuDeviceGetAttribute( &canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, id );
    cuDeviceGetAttribute( &computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, id);
#endif
}


void CudaDevice::set_device() {
    // Call cudaThreadExit() explicitly since CUDA docs say:
    // "It is a known issue that cudaThreadExit() may not be called implicitly on
    // host thread exit. Due to this, developers are recommended to explicitly
    // call cudaThreadExit() while the issue is being resolved"
#ifdef LOGGING
    logger.notice(_("Setting CUDA device: cudaSetDevice(%d)") % id);
#endif
    cudaThreadExit();
    cudaSetDevice(id);
}

}
