#pragma once

#include <types/detection.hpp>

namespace trt
{
    class DetectionProcessor
    {
    public:
        virtual ~DetectionProcessor() = default;

        // Process a single frame to get detections
        virtual std::vector<Detection> process(const cv::Mat &frame) = 0;

        // Process multiple frames to get batched detections
        virtual std::vector<std::vector<Detection>> process(const std::vector<cv::Mat> &frames) = 0;
    };

    class ClassificationProcessor
    {
    public:
        virtual ~ClassificationProcessor() = default;

        // Process a single frame to get classifications
        virtual Detection process(const cv::Mat &frame) = 0;

        // Process multiple frames to get batched classifications
        virtual std::vector<Detection> process(const std::vector<cv::Mat> &frames) = 0;
    };
}