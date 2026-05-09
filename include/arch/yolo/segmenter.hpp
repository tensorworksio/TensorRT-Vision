#pragma once

#include <fstream>
#include <types/detection.hpp>
#include <nlohmann/json.hpp>
#include <arch/yolo/config.hpp>
#include <tasks/segmentation.hpp>

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
        case YoloVersion::YOLOv8:  return "yolov8";
        case YoloVersion::YOLOv11: return "yolov11";
        default: throw std::runtime_error("Unknown yolo version");
        }
    }

    inline YoloVersion getYoloVersion(const std::string &name)
    {
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        for (auto v : {YoloVersion::YOLOv8, YoloVersion::YOLOv11})
            if (lower_name == getYoloVersionString(v))
                return v;
        return YoloVersion::UNKNOWN;
    }

    struct YoloConfig : YoloBaseConfig
    {
        float maskThreshold = 0.5f;

        void loadFromJson(const nlohmann::json &data) override
        {
            YoloBaseConfig::loadFromJson(data);
            if (data.contains("mask_threshold"))
                maskThreshold = data["mask_threshold"].get<float>();
        }
    };

    class Yolo : public Segmenter<trt::MultiOutput>
    {
    public:
        Yolo(const YoloConfig &t_config)
            : Segmenter<trt::MultiOutput>(t_config.engine), config(t_config) {}
        virtual ~Yolo() = default;
        const YoloConfig &getConfig() const { return config; }
        const std::string getClassName(int class_id) const
        {
            return (static_cast<size_t>(class_id) < config.classNames.size()) ? config.classNames[class_id] : std::to_string(class_id);
        }

    protected:
        const YoloConfig config;

    private:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        std::vector<Detection> postprocess(const trt::MultiOutput &engineOutputs) override;
    };

    using Yolov8  = Yolo;
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
            case YoloVersion::YOLOv8:  return std::make_unique<Yolov8>(config);
            case YoloVersion::YOLOv11: return std::make_unique<Yolov11>(config);
            default: throw std::runtime_error("Unsupported yolo version");
            }
        }
    };

} // namespace seg
