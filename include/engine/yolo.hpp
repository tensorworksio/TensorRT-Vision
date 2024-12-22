#pragma once
#include <fstream>
#include <utils/json_utils.hpp>
#include <types/detection.hpp>
#include <engine/processor.hpp>

enum class YoloVersion : int
{
    v7 = 7,
    v8 = 8,
    v11 = 11,
    UNKNOWN = -1
};

struct YoloConfig : JsonConfig
{
    // Engine config
    trt::EngineConfig engine{};
    // Yolo version
    YoloVersion version = YoloVersion::UNKNOWN;
    // confidence threshold used to filter detected objects
    float confidenceThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.45f;
    // Adaptive threshold coefficient NMS
    float nmsEta = 1.f;
    // Max number of detected objects to return
    int topK = 100;
    // Class names
    std::vector<std::string> classNames{};

    void loadFromJson(const nlohmann::json &data) override
    {
        if (data.contains("engine"))
            engine.loadFromJson(data["engine"]);
        if (data.contains("version"))
            version = static_cast<YoloVersion>(data["version"].get<int>());
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

    static YoloConfig load(const std::string &filename)
    {
        std::ifstream file(filename);
        auto data = nlohmann::json::parse(file);

        YoloConfig config;
        config.loadFromJson(data["yolo"]);
        return config;
    }

    std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<YoloConfig>(*this); }
};

class Yolo : public trt::ModelProcessor<std::vector<Detection>>
{
public:
    Yolo(const YoloConfig &t_config) : ModelProcessor(t_config.engine), config(t_config) {};

    virtual ~Yolo() = default;
    const std::string getClassName(int class_id) const;
    const YoloConfig &getConfig() const { return config; };

private:
    bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
    virtual std::vector<Detection> postprocess(const std::vector<float> &featureVector);

protected:
    const YoloConfig config;
    float m_ratioWidth = 1.f;
    float m_ratioHeight = 1.f;
    float m_imgWidth = 0.f;
    float m_imgHeight = 0.f;
};

class Yolov7 : public Yolo
{
public:
    Yolov7(const YoloConfig &t_config) : Yolo(t_config) {};

private:
    std::vector<Detection> postprocess(const std::vector<float> &featureVector) override;
};

using Yolov8 = Yolo;
using Yolov11 = Yolo;

class YoloFactory
{
public:
    static std::unique_ptr<Yolo> create(const YoloConfig &t_config)
    {
        switch (t_config.version)
        {
        case YoloVersion::v7:
            return std::make_unique<Yolov7>(t_config);
        case YoloVersion::v8:
            return std::make_unique<Yolov8>(t_config);
        case YoloVersion::v11:
            return std::make_unique<Yolov11>(t_config);
        default:
            throw std::invalid_argument(fmt::format("Unsupported YOLO version {}", static_cast<int>(t_config.version)));
        }
    }
};