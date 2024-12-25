#pragma once

#include <types/detection.hpp>
#include <utils/json_utils.hpp>
#include <engine/processor.hpp>

struct DetectorConfig : JsonConfig
{
    trt::EngineConfig engine{};
    float confidenceThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    float nmsEta = 1.f;
    int topK = 100;
    std::vector<std::string> classNames{};

    void loadFromJson(const nlohmann::json &data) override
    {
        if (data.contains("engine"))
            engine.loadFromJson(data["engine"]);
        if (data.contains("confidence_threshold"))
            confidenceThreshold = data["confidence_threshold"].get<float>();
        if (data.contains("nms_threshold"))
            nmsThreshold = data["nms_threshold"].get<float>();
        if (data.contains("nms_eta"))
            nmsEta = data["nms_eta"].get<float>();
        if (data.contains("top_k"))
            topK = data["top_k"].get<int>();
        if (data.contains("class_names"))
            classNames = data["class_names"].get<std::vector<std::string>>();
    }

    std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<DetectorConfig>(*this); }
};

class Detector : public trt::ModelProcessor<std::vector<Detection>>
{
public:
    Detector(const DetectorConfig &t_config) : ModelProcessor(t_config.engine), config(t_config) {};
    virtual ~Detector() = default;
    const DetectorConfig &getConfig() const { return config; };
    const std::string getClassName(int class_id) const
    {
        return (static_cast<size_t>(class_id) < config.classNames.size()) ? config.classNames[class_id] : std::to_string(class_id);
    };

protected:
    const DetectorConfig config;
};
