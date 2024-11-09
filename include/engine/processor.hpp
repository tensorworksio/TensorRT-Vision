#pragma once
#include "engine/engine.hpp"
#include "common/detection_utils.hpp"

namespace trt
{

    class ModelProcessor
    {

    public:
        ModelProcessor(const EngineConfig &config);
        virtual ~ModelProcessor() = default;
        // inference
        bool process(const cv::Mat &image, std::vector<Detection> &detections);

    private:
        // preprocessing
        virtual bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) = 0;
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg);
        bool preprocess(std::vector<cv::Mat> &inputBatch, cv::Size size);
        bool preprocess(std::vector<cv::Mat> &inputBatch);
        bool preprocess(std::vector<std::vector<cv::Mat>> &inputs);

        // postprocessing
        virtual bool postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections) = 0;

    protected:
        std::unique_ptr<Engine> engine = nullptr;
    };

} // namespace trt