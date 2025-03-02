#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "yolo.hpp"

namespace det
{

    enum class DetectorType
    {
        YOLOv7,
        YOLOv8,
        YOLOv11,
        UNKNOWN
    };

    inline std::string getDetectorName(DetectorType type)
    {
        switch (type)
        {
        case DetectorType::YOLOv7:
            return "yolov7";
        case DetectorType::YOLOv8:
            return "yolov8";
        case DetectorType::YOLOv11:
            return "yolov11";
        default:
            throw std::runtime_error("Unkown detector type");
        }
    };

    inline auto &getDetectors()
    {
        static std::array<DetectorType, 3> detectors{
            DetectorType::YOLOv7,
            DetectorType::YOLOv8,
            DetectorType::YOLOv11};

        return detectors;
    };

    inline DetectorType getDetectorType(const std::string &name)
    {
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        for (const auto &type : getDetectors())
        {
            if (lower_name == getDetectorName(type))
            {
                return type;
            }
        }
        return DetectorType::UNKNOWN;
        return DetectorType::UNKNOWN;
    };

    class YoloFactory
    {
    public:
        static std::unique_ptr<Yolo> create(const std::string &config_file)
        {
            std::ifstream file(config_file);
            auto data = nlohmann::json::parse(file);
            DetectorType detector = getDetectorType(data["detector"]["name"]);

            auto config = YoloConfig();
            config.loadFromJson(data["detector"]);

            switch (detector)
            {
            case DetectorType::YOLOv7:
            {
                return std::make_unique<Yolov7>(config);
            }
            case DetectorType::YOLOv8:
            {
                return std::make_unique<Yolov8>(config);
            }
            case DetectorType::YOLOv11:
            {
                return std::make_unique<Yolov11>(config);
            }
            default:
                throw std::runtime_error("Unknown detector type");
            }
        }
    };
} // det