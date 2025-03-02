#pragma once

#include <types/detection.hpp>
#include <engine/processor.hpp>

namespace seg
{

    class SegmenterInterface
    {
    public:
        virtual ~SegmenterInterface() = default;
        virtual std::vector<Detection> process(const cv::Mat &frame) = 0;
        virtual std::vector<std::vector<Detection>> process(const std::vector<cv::Mat> &frames) = 0;
    };

    template <typename EngineOutput>
    class Segmenter : public SegmenterInterface, public trt::ModelProcessor<std::vector<Detection>, EngineOutput>
    {
    public:
        Segmenter(const trt::EngineConfig &config)
            : trt::ModelProcessor<std::vector<Detection>, EngineOutput>(config) {}

        std::vector<Detection> process(const cv::Mat &frame) override
        {
            return trt::ModelProcessor<std::vector<Detection>, EngineOutput>::process(frame);
        }

        std::vector<std::vector<Detection>> process(const std::vector<cv::Mat> &frames) override
        {
            return trt::ModelProcessor<std::vector<Detection>, EngineOutput>::process(frames);
        }
    };

} // seg