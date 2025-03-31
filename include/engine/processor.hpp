#pragma once

#include "engine.hpp"

namespace trt
{
    template <typename OutputType, typename EngineOutput>
    class ModelProcessor
    {
    public:
        ModelProcessor(const EngineConfig &config);
        virtual ~ModelProcessor() = default;

        // Image & batch inference
        OutputType process(const cv::Mat &image);
        std::vector<OutputType> process(const std::vector<cv::Mat> &imageBatch);

    private:
        // Image & batch preprocessing
        virtual bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) = 0;
        bool preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch);

        // Image & batch postprocessing
        virtual OutputType postprocess(const EngineOutput &featureVector) = 0;
        std::vector<OutputType> postprocess(const std::vector<EngineOutput> &featureBatch);

    protected:
        std::unique_ptr<Engine> engine = nullptr;
    };
} // namespace trt

#include "processor.impl.hpp"

namespace trt
{
    using SingleOutput = std::vector<float>;
    using MultiOutput = std::vector<std::vector<float>>;

    template <typename OutputType>
    using SISOProcessor = ModelProcessor<OutputType, SingleOutput>;

    template <typename OutputType>
    using SIMOProcessor = ModelProcessor<OutputType, MultiOutput>;
}; // namespace trt