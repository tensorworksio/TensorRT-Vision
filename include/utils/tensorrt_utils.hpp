#pragma once

#include <opencv2/opencv.hpp>

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
