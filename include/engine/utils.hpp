#pragma once

#include "fmt/format.h"
#include <opencv2/opencv.hpp>
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

namespace trt
{
    inline cv::Mat blobFromMats(const std::vector<cv::Mat> &batchInput)
    {
        cv::Mat dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_32FC3);
        size_t width = batchInput[0].cols * batchInput[0].rows;
        for (size_t img = 0; img < batchInput.size(); img++)
        {
            std::vector<cv::Mat> input_channels{
                cv::Mat(batchInput[0].rows, batchInput[0].cols, CV_32F, &(dst.ptr()[0 + width * 3 * sizeof(float) * img])),
                cv::Mat(batchInput[0].rows, batchInput[0].cols, CV_32F, &(dst.ptr()[width * sizeof(float) + width * 3 * sizeof(float) * img])),
                cv::Mat(batchInput[0].rows, batchInput[0].cols, CV_32F, &(dst.ptr()[width * 2 * sizeof(float) + width * 3 * sizeof(float) * img]))};
            cv::split(batchInput[img], input_channels); // HWC -> CHW
        }
        return dst;
    }
} // namespace trt
