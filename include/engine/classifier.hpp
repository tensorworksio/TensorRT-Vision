#pragma once

#include <fstream>
#include "utils/json_utils.hpp"
#include "engine/processor.hpp"

namespace trt
{

    struct ClassifierConfig : JsonConfig
    {
        EngineConfig engine;
        float confidenceThreshold{0.9f};
        std::vector<std::string> classNames{};

        void loadFromJson(const nlohmann::json &data) override
        {
            if (data.contains("engine"))
                engine.loadFromJson(data["engine"]);
            if (data.contains("confidence_threshold"))
                confidenceThreshold = data["confidence_threshold"].get<float>();
            if (data.contains("class_names"))
                classNames = data["class_names"].get<std::vector<std::string>>();
        }

        static ClassifierConfig load(const std::string &filename)
        {
            std::ifstream file(filename);
            auto data = nlohmann::json::parse(file);
            ClassifierConfig config;
            config.loadFromJson(data);
            return config;
        }

        std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<ClassifierConfig>(*this); }
    };

    class Classifier : public ModelProcessor
    {
    public:
        Classifier(const ClassifierConfig &t_config) : ModelProcessor(t_config.engine), config(t_config) {}
        Detection process(const cv::Mat &image);
        const std::string getClassName(int class_id) const;
        const ClassifierConfig &getConfig() const { return config; };

    protected:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
        bool postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections) override;

    private:
        const ClassifierConfig config;
    };

} // namespace trt