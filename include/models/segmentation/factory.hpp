#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "yolo.hpp"

namespace seg
{
    enum class ModelType
    {
        YOLO,
        UNKNOWN
    };

    inline std::string getModelName(ModelType type)
    {
        switch (type)
        {
        case ModelType::YOLO:
            return "yolo";
        default:
            throw std::runtime_error("Unkown model type");
        }
    };

    inline auto &getModels()
    {
        static std::array<ModelType, 1> models{
            ModelType::YOLO};

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

    class SegmenterFactory
    {
    public:
        static std::unique_ptr<trt::DetectionProcessor> create(const std::string &config_file)
        {
            std::ifstream file(config_file);
            auto data = nlohmann::json::parse(file);
            ModelType model = getModelType(data["segmenter"]["architecture"]);

            switch (model)
            {
            case ModelType::YOLO:
            {
                return YoloFactory::create(data);
            }
            default:
                throw std::runtime_error("Unknown model architecture");
            }
        }
    };

} // seg