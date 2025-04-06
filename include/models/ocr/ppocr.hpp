#pragma once

#include <engine/processor.hpp>
#include <types/detection.hpp>
#include <utils/json_utils.hpp>

namespace ocr
{
    struct PPOCRConfig : JsonConfig
    {
        trt::EngineConfig engine{};
        int topK = 1000;
        float maskThreshold = 0.5f;
        float confidenceThreshold = 0.5f;

        void loadFromJson(const nlohmann::json &data) override
        {
            if (data.contains("engine"))
                engine.loadFromJson(data["engine"]);
            if (data.contains("top_k"))
                topK = data["top_k"].get<int>();
            if (data.contains("mask_threshold"))
                maskThreshold = data["mask_threshold"].get<float>();
        }

        std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<PPOCRConfig>(*this); }
    };

    class PPOCRV3Detector : public trt::SISODetector
    {
    public:
        PPOCRV3Detector(const PPOCRConfig &t_config) : trt::SISODetector(t_config.engine), config(t_config) {};
        virtual ~PPOCRV3Detector() = default;
        const PPOCRConfig &getConfig() const { return config; };

    protected:
        const PPOCRConfig config;

    private:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        std::vector<Detection> postprocess(const trt::SingleOutput &engineOutputs) override;
    };

    class PPOCRV3Recognizer : public trt::SIMOClassifier
    {
    public:
        PPOCRV3Recognizer(const PPOCRConfig &t_config) : trt::SIMOClassifier(t_config.engine), config(t_config) {};
        virtual ~PPOCRV3Recognizer() = default;
        const PPOCRConfig &getConfig() const { return config; };

    protected:
        const PPOCRConfig config;

    private:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        Detection postprocess(const trt::MultiOutput &engineOutputs) override;
    };

} // namespace ocr