#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include "engine/processor.hpp"

namespace trt
{

    struct ClassifierConfig : EngineConfig
    {
        std::string activation{};
        float confidenceThreshold{0.9f};
        std::vector<std::string> classNames{};

        static ClassifierConfig loadFromJson(const std::string &filename)
        {
            std::ifstream file(filename);
            auto data = nlohmann::json::parse(file);

            ClassifierConfig config;
            config.loadEngineConfig(data);

            if (data.contains("activation"))
            {
                config.activation = data["activation"].get<std::string>();
            }
            if (data.contains("confidenceThreshold"))
            {
                config.confidenceThreshold = data["confidenceThreshold"].get<float>();
            }
            if (data.contains("classNames"))
            {
                config.classNames = data["classNames"].get<std::vector<std::string>>();
            }

            return config;
        }

        std::shared_ptr<const EngineConfig> clone() const override { return std::make_shared<ClassifierConfig>(*this); }
    };

    class Classifier : public ModelProcessor
    {
    public:
        Classifier(const ClassifierConfig &config) : ModelProcessor(config)
        {
            // Ensure the passed config is of the correct type
            if (dynamic_cast<const ClassifierConfig *>(m_config.get()) == nullptr)
            {
                throw std::invalid_argument("Invalid config type: expected ClassifierConfig");
            }
        }
        Detection process(const cv::Mat &image);
        const std::string getClassName(int class_id) const;
        const ClassifierConfig &getClassifierConfig() const { return static_cast<const ClassifierConfig &>(*m_config); };

    protected:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
        bool postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections) override;
    };

} // namespace trt