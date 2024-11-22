#pragma once
#include <utils/detection_utils.hpp>

#include "engine.hpp"

namespace trt
{
    class ModelProcessor
    {
    public:
        ModelProcessor(const EngineConfig &config);
        virtual ~ModelProcessor() = default;

        // inference
        bool process(const cv::Mat &image, std::vector<Detection> &detections);
        bool process(const std::vector<cv::Mat> &imageBatch, std::vector<std::vector<Detection>> &detectionBatch);

    private:
        // preprocessing
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg);
        virtual bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) = 0;
        bool preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch);
        bool preprocess(const std::vector<cv::Mat> &inputBatch, std::vector<cv::Mat> &outputBatch, cv::Size size);

        // postprocessing
        virtual bool postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections) = 0;
        bool postprocess(std::vector<std::vector<float>> &features, std::vector<std::vector<Detection>> &detections);

    protected:
        std::unique_ptr<Engine> engine = nullptr;
    };

} // namespace trt