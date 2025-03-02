#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "yolo.hpp"

namespace det
{
    enum class ModelType
    {
        YOLOv7,
        YOLOv8,
        YOLOv11,
        UNKNOWN
    };

    inline std::string getModelName(ModelType type)
    {
        switch (type)
        {
        case ModelType::YOLOv7:
            return "yolov7";
        case ModelType::YOLOv8:
            return "yolov8";
        case ModelType::YOLOv11:
            return "yolov11";
        default:
            throw std::runtime_error("Unkown model type");
        }
    };

    inline auto &getModels()
    {
        static std::array<ModelType, 3> models{
            ModelType::YOLOv7,
            ModelType::YOLOv8,
            ModelType::YOLOv11};

        return models;
    };

    inline ModelType getModelType(const std::string &name)
    {
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        for (const auto &type : getModels())
        {
            if (lower_name == getModelName(type))
            {
                return type;
            }
        }
        return ModelType::UNKNOWN;
    };

    class YoloFactory
    {
    public:
        static std::unique_ptr<Yolo> create(const std::string &config_file)
        {
            std::ifstream file(config_file);
            auto data = nlohmann::json::parse(file);
            ModelType model = getModelType(data["detector"]["name"]);

            auto config = YoloConfig();
            config.loadFromJson(data["detector"]);

            switch (model)
            {
            case ModelType::YOLOv7:
            {
                return std::make_unique<Yolov7>(config);
            }
            case ModelType::YOLOv8:
            {
                return std::make_unique<Yolov8>(config);
            }
            case ModelType::YOLOv11:
            {
                return std::make_unique<Yolov11>(config);
            }
            default:
                throw std::runtime_error("Unknown model type");
            }
        }
    };

} // det