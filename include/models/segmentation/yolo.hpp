#pragma once

#include <types/detection.hpp>
#include <utils/json_utils.hpp>
#include "segmenter.hpp"

namespace seg
{
    enum class YoloVersion
    {
        YOLOv8,
        YOLOv11,
        UNKNOWN
    };

    inline std::string getYoloVersionString(YoloVersion version)
    {
        switch (version)
        {
        case YoloVersion::YOLOv8:
            return "yolov8";
        case YoloVersion::YOLOv11:
            return "yolov11";
        default:
            throw std::runtime_error("Unkown yolo version");
        }
    };

    inline auto &getYoloModels()
    {
        static std::array<YoloVersion, 2> models{
            YoloVersion::YOLOv8,
            YoloVersion::YOLOv11};

        return models;
    };

    inline YoloVersion getYoloVersion(const std::string &name)
    {
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        for (const auto &version : getYoloModels())
        {
            if (lower_name == getYoloVersionString(version))
            {
                return version;
            }
        }
        return YoloVersion::UNKNOWN;
    };

    struct YoloConfig : JsonConfig
    {
        trt::EngineConfig engine{};
        float confidenceThreshold = 0.25f;
        float nmsThreshold = 0.45f;
        float maskThreshold = 0.5f;
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

    class Yolo : public Segmenter<trt::MultiOutput>
    {
    public:
        Yolo(const YoloConfig &t_config)
            : Segmenter<trt::MultiOutput>(t_config.engine), config(t_config) {};
        virtual ~Yolo() = default;
        const YoloConfig &getConfig() const { return config; };
        const std::string getClassName(int class_id) const
        {
            return (static_cast<size_t>(class_id) < config.classNames.size()) ? config.classNames[class_id] : std::to_string(class_id);
        };

    protected:
        const YoloConfig config;

    private:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
        std::vector<Detection> postprocess(const trt::MultiOutput &engineOutputs) override;
    };

    using Yolov8 = Yolo;
    using Yolov11 = Yolo;

    class YoloFactory
    {
    public:
        static std::unique_ptr<Yolo> create(const nlohmann::json &data)
        {
            YoloVersion version = getYoloVersion(data["segmenter"]["name"]);

            auto config = YoloConfig();
            config.loadFromJson(data["segmenter"]);

            switch (version)
            {
            case YoloVersion::YOLOv8:
            {
                return std::make_unique<Yolov8>(config);
            }
            case YoloVersion::YOLOv11:
            {
                return std::make_unique<Yolov11>(config);
            }
            default:
                throw std::runtime_error("Unsupported yolo version");
            }
        }
    };
} // seg