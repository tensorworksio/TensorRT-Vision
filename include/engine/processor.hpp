#pragma once

#include "engine.hpp"

namespace trt
{
    template <typename OutputType>
    class ModelProcessor
    {
    public:
        ModelProcessor(const EngineConfig &config);
        virtual ~ModelProcessor() = default;

        // Image inference
        OutputType process(const cv::Mat &image);

        // Batch inference
        std::vector<OutputType> process(const std::vector<cv::Mat> &imageBatch);

    private:
        // Image preprocessing
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg);
        virtual bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) = 0;

        // Batch preprocessing
        bool preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch);
        bool preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch, cv::Size size);

        // Image postprocessing
        virtual OutputType postprocess(const std::vector<float> &featureVector) = 0;

        // Batch postprocessing
        std::vector<OutputType> postprocess(const std::vector<std::vector<float>> &featureBatch);

    protected:
        std::unique_ptr<Engine> engine = nullptr;
    };

} // namespace trt

#include "processor.impl.hpp"