#pragma once

#include <types/detection.hpp>
#include <engine/processor.hpp>
#include <utils/json_utils.hpp>

struct YoloConfig : JsonConfig
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

    std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<YoloConfig>(*this); }
};

class Yolo : public trt::SIMOProcessor<std::vector<Detection>>
{
public:
    Yolo(const YoloConfig &t_config)
        : trt::SIMOProcessor<std::vector<Detection>>(t_config.engine), config(t_config) {};
    virtual ~Yolo() = default;
    const YoloConfig &getConfig() const { return config; };
    const std::string getClassName(int class_id) const
    {
        return (static_cast<size_t>(class_id) < config.classNames.size()) ? config.classNames[class_id] : std::to_string(class_id);
    };

protected:
    const YoloConfig config;
    float m_ratioWidth = 1.f;
    float m_ratioHeight = 1.f;
    float m_imgWidth = 0.f;
    float m_imgHeight = 0.f;

private:
    bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
    std::vector<Detection> postprocess(const trt::MultiOutput &engineOutputs) override;
};