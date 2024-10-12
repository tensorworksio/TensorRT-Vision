#pragma once

#include "fmt/format.h"
#include <cuda_runtime_api.h>

namespace cuda
{
    inline void checkCudaErrorCode(cudaError_t code)
    {
        if (code == cudaSuccess)
            return;
        std::string errMsg = fmt::format("CUDA operation failed with error code {}: {}", code, cudaGetErrorString(code));
        throw std::runtime_error(errMsg);
    }

    inline void getDeviceNames(std::vector<std::string> &deviceNames)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);

        for (int device = 0; device < numGPUs; device++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);

            deviceNames.push_back(std::string(prop.name));
        }
    }
} // namespace cuda
