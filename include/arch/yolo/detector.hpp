#pragma once

#include <fstream>
#include <types/detection.hpp>
#include <nlohmann/json.hpp>
#include <arch/yolo/config.hpp>
#include <tasks/detection.hpp>

namespace det
{
    enum class YoloVersion
    {
        YOLOv7,
        YOLOv8,
        YOLOv11,
        UNKNOWN
    };

    inline std::string getYoloVersionString(YoloVersion version)
    {
        switch (version)
        {
        case YoloVersion::YOLOv7:  return "yolov7";
        case YoloVersion::YOLOv8:  return "yolov8";
        case YoloVersion::YOLOv11: return "yolov11";
        default: throw std::runtime_error("Unknown yolo version");
        }
    }

    inline YoloVersion getYoloVersion(const std::string &name)
    {
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        for (auto v : {YoloVersion::YOLOv7, YoloVersion::YOLOv8, YoloVersion::YOLOv11})
            if (lower_name == getYoloVersionString(v))
                return v;
        return YoloVersion::UNKNOWN;
    }

    struct YoloConfig : YoloBaseConfig {};

    class Yolo : public Detector<trt::SingleOutput>
    {
    public:
        Yolo(const YoloConfig &t_config)
            : Detector<trt::SingleOutput>(t_config.engine), config(t_config) {}
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
        virtual std::vector<Detection> postprocess(const trt::SingleOutput &featureVector);
    };

    class Yolov7 : public Yolo
    {
    public:
        using Yolo::Yolo;

    private:
        std::vector<Detection> postprocess(const trt::SingleOutput &featureVector) override;
    };

    using Yolov8  = Yolo;
    using Yolov11 = Yolo;

    class YoloFactory
    {
    public:
        static std::unique_ptr<Yolo> create(const nlohmann::json &data)
        {
            YoloVersion version = getYoloVersion(data["detector"]["name"]);
            auto config = YoloConfig();
            config.loadFromJson(data["detector"]);

            switch (version)
            {
            case YoloVersion::YOLOv7:  return std::make_unique<Yolov7>(config);
            case YoloVersion::YOLOv8:  return std::make_unique<Yolov8>(config);
            case YoloVersion::YOLOv11: return std::make_unique<Yolov11>(config);
            default: throw std::runtime_error("Unsupported yolo version");
            }
        }
    };

} // namespace det
