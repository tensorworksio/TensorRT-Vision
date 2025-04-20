#pragma once

#include <engine/processor.hpp>
#include <types/detection.hpp>
#include <utils/json_utils.hpp>

namespace ocr
{
    struct PPOCRConfig : JsonConfig
    {
        trt::EngineConfig detector{};
        trt::EngineConfig recognizer{};
        int topK = 1000;
        float maskThreshold = 0.5f;
        int minArea = 100;
        float confidenceThreshold = 0.5f;
        std::vector<std::string> vocabulary{};
        void loadFromJson(const nlohmann::json &data) override
        {
            if (data.contains("detector"))
                detector.loadFromJson(data["detector"]);
            if (data.contains("recognizer"))
                recognizer.loadFromJson(data["recognizer"]);
            if (data.contains("top_k"))
                topK = data["top_k"].get<int>();
            if (data.contains("mask_threshold"))
                maskThreshold = data["mask_threshold"].get<float>();
            if (data.contains("min_area"))
                minArea = data["min_area"].get<int>();
            if (data.contains("confidence_threshold"))
                confidenceThreshold = data["confidence_threshold"].get<float>();
            if (data.contains("vocabulary"))
            {
                vocabulary = data["vocabulary"].get<std::vector<std::string>>();
                vocabulary.insert(vocabulary.begin(), ""); // Add blank label
                vocabulary.insert(vocabulary.end(), "?");  // Unknown label
            }
        }

        std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<PPOCRConfig>(*this); }
    };

    class PPOCRV3Detector : public trt::SISODetector
    {
    public:
        PPOCRV3Detector(const PPOCRConfig &t_config) : trt::SISODetector(t_config.detector), config(t_config) {};
        virtual ~PPOCRV3Detector() = default;
        const PPOCRConfig &getConfig() const { return config; };

    protected:
        const PPOCRConfig config;

    private:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        std::vector<Detection> postprocess(const trt::SingleOutput &engineOutputs) override;
    };

    class PPOCRV3Recognizer : public trt::SISOProcessor<std::string>
    {
    public:
        PPOCRV3Recognizer(const PPOCRConfig &t_config) : trt::SISOProcessor<std::string>(t_config.recognizer), config(t_config) {};
        virtual ~PPOCRV3Recognizer() = default;
        const PPOCRConfig &getConfig() const { return config; };

    protected:
        const PPOCRConfig config;

    private:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        std::string postprocess(const trt::SingleOutput &featureVector) override;
    };

} // namespace ocr