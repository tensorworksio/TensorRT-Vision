#pragma once

#include "engine.hpp"
#include "interface.hpp"
#include <types/detection.hpp>

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

    template <typename EngineOutput>
    class Detector : public trt::DetectionProcessor, public trt::ModelProcessor<std::vector<Detection>, EngineOutput>
    {
    public:
        Detector(const trt::EngineConfig &config)
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

    template <typename EngineOutput>
    class Classifier : public trt::ClassificationProcessor, public trt::ModelProcessor<Detection, EngineOutput>
    {
    public:
        Classifier(const trt::EngineConfig &config)
            : trt::ModelProcessor<Detection, EngineOutput>(config) {}

        Detection process(const cv::Mat &frame) override
        {
            return trt::ModelProcessor<Detection, EngineOutput>::process(frame);
        }

        std::vector<Detection> process(const std::vector<cv::Mat> &frames) override
        {
            return trt::ModelProcessor<Detection, EngineOutput>::process(frames);
        }
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

    using SISODetector = Detector<SingleOutput>;
    using SIMODetector = Detector<MultiOutput>;

    using SISOClassifier = Classifier<SingleOutput>;
    using SIMOClassifier = Classifier<MultiOutput>;
}; // namespace trt
